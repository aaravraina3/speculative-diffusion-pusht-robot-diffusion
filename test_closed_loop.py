"""Tests for closed_loop.py's lockstep rollouts and summary, with fake environments.

Run with: .venv/bin/python -m pytest -q test_closed_loop.py
"""
import numpy as np
import pytest
import torch

import closed_loop as cl
import spec_eval as se

GOAL = np.array([100.0, 100.0], np.float32)


class FakeEnv:
    """The agent jumps to whatever position it's told. Success when it's within 1 px of GOAL.
    Seeds set the start position; seed % 3 == 0 starts on the goal and finishes at once."""

    def reset(self, seed):
        self.pos = GOAL.copy() if seed % 3 == 0 else np.array([400.0, 400.0], np.float32)
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        self.pos = np.asarray(action, np.float32)
        self.steps += 1
        cov = float(max(0.0, 1.0 - np.linalg.norm(self.pos - GOAL) / 500.0))
        done = bool(np.linalg.norm(self.pos - GOAL) < 1.0)
        return self._obs(), cov, done, False, {"coverage": cov, "is_success": done}

    def _obs(self):
        return {"agent_pos": self.pos.copy(), "pixels": np.zeros((4, 4, 3), np.uint8)}


class FakeVerifier:
    n_obs, horizon, device = 2, 16, torch.device("cpu")

    def __init__(self):
        self.batches = []

    def chunk(self, states, images, noise, length):
        self.batches.append(states.shape[0])
        return np.tile(GOAL, (states.shape[0], length, 1))  # always plans straight to the goal


def drafts_at(point):
    return [lambda x, p=point: torch.as_tensor(p, dtype=torch.float32).expand(x.shape[0], 2).clone()]


SAMPLER = se.Sampler("DDPM", 20, 8)


def test_verifier_only_reaches_goal_and_pays_per_boundary():
    envs, seeds, v = [FakeEnv() for _ in range(4)], [1, 2, 4, 5], FakeVerifier()
    rows = cl.run_batch(envs, seeds, v, SAMPLER, drafts_at([0, 0]), lambda d, s: se.always("verifier"), 30)
    assert all(r["success"] and r["steps"] == 1 and r["calls"] == 1 for r in rows)
    assert v.batches == [4], "one batched call for all rollouts at the first boundary"


def test_draft_only_never_calls_and_runs_to_the_step_limit():
    envs, v = [FakeEnv() for _ in range(3)], FakeVerifier()
    rows = cl.run_batch(envs, [1, 2, 4], v, None, drafts_at([0, 0]), lambda d, s: se.always("draft"), 20)
    assert all(r["calls"] == 0 and r["steps"] == 20 and not r["success"] for r in rows)
    assert v.batches == []


def test_finished_rollouts_leave_the_batch():
    envs, v = [FakeEnv() for _ in range(3)], FakeVerifier()
    # the draft pushes toward (0, 0) and never succeeds; seed 3's verifier plan lands on the goal at once
    decide = lambda d, s: (lambda b, log: ("verifier", True) if b == 0 else ("draft", False))
    rows = cl.run_batch(envs, [3, 4, 5], v, SAMPLER, drafts_at([0, 0]), decide, 20)
    assert [r["steps"] for r in rows] == [1, 1, 1] and all(r["success"] for r in rows)
    envs, v = [FakeEnv() for _ in range(3)], FakeVerifier()
    only_seed3 = lambda d, s: (lambda b, log: ("verifier", True))
    rows = cl.run_batch(envs, [3, 4, 5], v, SAMPLER, drafts_at([0, 0]), only_seed3, 20)
    assert v.batches == [3], "everyone finishes on step 1, so no later boundary plans for anyone"


def test_verifier_plans_only_for_rollouts_that_call_it():
    envs, v = [FakeEnv() for _ in range(4)], FakeVerifier()
    # ensemble-style gate: trust the draft when spread is low, otherwise call
    def decide(delta_at, spr):
        return lambda b, log: ("draft", False) if spr[b] < 1.0 else ("verifier", True)
    drafts = [lambda x: torch.zeros(x.shape[0], 2),
              lambda x: torch.stack([torch.tensor([0.0, 0.0]), torch.tensor([30.0, 0.0]),
                                     torch.tensor([0.0, 0.0]), torch.tensor([30.0, 0.0])])[: x.shape[0]]]
    rows = cl.run_batch(envs, [1, 2, 4, 5], v, SAMPLER, drafts, decide, 8)
    assert v.batches == [2], "only the two rollouts whose draft copies disagree get planned"
    assert [r["calls"] for r in rows] == [0, 1, 0, 1]


def test_notebook_rule_pays_every_boundary_even_when_serving_the_draft():
    envs, v = [FakeEnv() for _ in range(2)], FakeVerifier()
    decide = lambda d, s: se.gate_needs_verifier(1e9, d)  # always accepts the draft
    rows = cl.run_batch(envs, [1, 2], v, SAMPLER, drafts_at([0, 0]), decide, 24)
    assert all(r["calls"] == 3 and r["draft_chunks"] == 3 for r in rows)  # boundaries 0, 8, 16


def test_gate_that_peeks_without_paying_raises():
    def cheat(delta_at, spread):
        def decide(b, log):
            delta_at(b)
            return "draft", False
        return decide
    with pytest.raises(AssertionError, match="without paying"):
        cl.run_batch([FakeEnv()], [1], FakeVerifier(), SAMPLER, drafts_at([0, 0]), cheat, 16)


def test_previous_agreement_skips_the_call_after_agreement():
    class VerifierToward(FakeVerifier):
        def chunk(self, states, images, noise, length):
            self.batches.append(states.shape[0])
            return np.tile(GOAL + 50, (states.shape[0], length, 1))  # never reaches the goal
    v = VerifierToward()
    decide = lambda d, s: se.gate_previous_agreement(10.0, d)
    # the draft sits on the verifier's plan, so they always agree: call, skip, call
    rows = cl.run_batch([FakeEnv()], [1], v, SAMPLER, drafts_at(GOAL + 50), decide, 24)
    assert rows[0]["calls"] == 2 and rows[0]["draft_chunks"] == 1
    assert v.batches == [1, 1], "no plan is computed at the skipped boundary"


def test_summary_uses_wilson_and_requires_matching_seeds():
    ref = [{"seed": i, "success": True, "max_coverage": 0.97, "steps": 100, "calls": 13} for i in range(10)]
    other = [{"seed": i, "success": i < 5, "max_coverage": 0.5, "steps": 300, "calls": 0} for i in range(10)]
    s = cl.summarize("x", other, ref, call_ms=0.0, draft_ms=0.1, uses_draft=True)
    assert s["success"] == 0.5 and s["success_ci"][0] > 0.2 and s["success_ci"][1] < 0.8
    assert s["success_discordant"] == [0, 5] and s["success_mcnemar_p"] == pytest.approx(2 / 32)
    shuffled = list(reversed(other))
    with pytest.raises(ValueError):
        cl.summarize("x", shuffled, ref, 0.0, 0.1, True)
