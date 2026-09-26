"""Tests for the serving loop, gates, splits and metrics in spec_eval.py.

No models are loaded. Run with: .venv/bin/python -m pytest -q test_spec_eval.py
"""
import numpy as np
import pytest
import torch

import spec_eval as se


# ---------------------------------------------------------------- splits

def test_split_parses_and_is_disjoint():
    s = se.parse_split("0:150,150:170,170:206", 206)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (150, 20, 36)
    assert not set(s["train"]) & set(s["test"])


def test_split_rejects_overlap_and_out_of_range():
    with pytest.raises(ValueError):
        se.parse_split("0:150,140:170,170:206", 206)
    with pytest.raises(ValueError):
        se.parse_split("0:150,150:170,170:300", 206)


# ---------------------------------------------------------------- serving loop

def _chunks(length, chunk, value_fn):
    return {b: np.array([[value_fn(t)] * 2 for t in range(b, min(b + chunk, length))], dtype=float)
            for b in range(0, length, chunk)}


def test_verifier_only_calls_every_boundary_and_serves_every_frame():
    length, chunk = 21, 8
    chunks = _chunks(length, chunk, lambda t: t)
    served, log = se.serve(length, chunk, se.always("verifier"), lambda b: chunks[b], np.zeros((length, 2)))
    assert [e["b"] for e in log] == [0, 8, 16]
    assert all(e["called"] for e in log)
    np.testing.assert_array_equal(served[:, 0], np.arange(length))  # chunk k serves frames b..b+7 in order


def test_draft_only_never_calls():
    length = 10
    draft = np.ones((length, 2))
    served, log = se.serve(length, 4, se.always("draft"), lambda b: pytest.fail("verifier used"), draft)
    assert not any(e["called"] for e in log)
    np.testing.assert_array_equal(served, draft)


def test_serving_verifier_without_paying_is_an_error():
    with pytest.raises(AssertionError):
        se.serve(8, 8, lambda b, log: ("verifier", False), lambda b: np.zeros((8, 2)), np.zeros((8, 2)))


# ---------------------------------------------------------------- gates

def test_notebook_rule_pays_for_the_verifier_at_every_boundary():
    decide = se.gate_needs_verifier(tau=1e9, delta_at=lambda b: 0.0)  # always accepts the draft
    _, log = se.serve(32, 8, decide, lambda b: np.zeros((8, 2)), np.zeros((32, 2)))
    assert all(e["serve"] == "draft" for e in log)
    assert all(e["called"] for e in log), "the rule can't decide without the verifier"


def test_previous_agreement_skips_at_most_one_chunk_and_never_serves_unchecked_first():
    decide = se.gate_previous_agreement(tau=1e9, delta_at=lambda b: 0.0, max_run=1)
    _, log = se.serve(64, 8, decide, lambda b: np.zeros((8, 2)), np.zeros((64, 2)))
    assert log[0]["called"], "nothing has been checked yet at the first boundary"
    pattern = [e["called"] for e in log]
    assert pattern == [True, False] * 4
    for e in log:
        if not e["called"]:
            assert e["serve"] == "draft"


def test_previous_agreement_keeps_calling_when_draft_disagrees():
    decide = se.gate_previous_agreement(tau=1.0, delta_at=lambda b: 5.0)
    _, log = se.serve(32, 8, decide, lambda b: np.zeros((8, 2)), np.zeros((32, 2)))
    assert all(e["called"] for e in log)


def test_ensemble_gate_calls_only_where_members_disagree():
    spread = np.zeros(24)
    spread[8] = 10.0
    decide = se.gate_ensemble(tau=1.0, spread=spread)
    _, log = se.serve(24, 8, decide, lambda b: np.zeros((8, 2)), np.zeros((24, 2)))
    assert [e["called"] for e in log] == [False, True, False]


# ---------------------------------------------------------------- history

def test_history_uses_real_previous_frame_and_repeats_only_at_start():
    ep = se.Episode(0, np.arange(10, dtype=np.float32)[:, None].repeat(2, 1),
                    np.zeros((10, 2), np.float32), torch.arange(10.0).view(10, 1, 1, 1).expand(10, 3, 2, 2))
    st0, im0 = ep.history(0, 2)
    assert st0[:, 0].tolist() == [0.0, 0.0]
    st5, im5 = ep.history(5, 2)
    assert st5[:, 0].tolist() == [4.0, 5.0]
    assert im5[:, 0, 0, 0].tolist() == [4.0, 5.0]


# ---------------------------------------------------------------- metrics

def test_mse_per_frame_matches_notebook_formula():
    served = np.array([[1.0, 2.0], [3.0, 5.0]])
    actions = np.array([[0.0, 0.0], [3.0, 1.0]])
    np.testing.assert_allclose(se.mse_per_frame(served, actions), [2.5, 8.0])


def test_cluster_bootstrap_on_constant_data_has_zero_width():
    est, lo, hi = se.cluster_bootstrap(np.full(5, 7.0), np.arange(1.0, 6.0))
    assert est == lo == hi == 7.0


def test_run_strategy_counts_calls_time_and_error():
    length, chunk = 16, 8
    actions = np.zeros((length, 2), np.float32)
    ep = se.Episode(3, np.zeros((length, 2), np.float32), actions, torch.zeros(length, 3, 2, 2))
    cache = {"chunk": chunk, "chunks": {b: np.ones((1, chunk, 2)) for b in (0, 8)}}  # off by 1 everywhere
    draft = (np.zeros((length, 2)), np.zeros(length))
    rows = se.run_strategy([ep], {3: cache}, {3: draft}, lambda d, s: se.always("verifier"),
                           chunk, 1, 20.0, 0.0, False)
    assert rows[0]["calls"] == 2 and rows[0]["ms"] == 40.0  # 2 calls x 20 ms
    assert rows[0]["mse"][0] == pytest.approx(1.0)
    rows = se.run_strategy([ep], {3: cache}, {3: draft}, lambda d, s: se.always("draft"),
                           chunk, 1, 20.0, 0.5, True)
    assert rows[0]["calls"] == 0 and rows[0]["ms"] == pytest.approx(8.0)  # 0.5 ms x 16 frames
    assert rows[0]["mse"][0] == 0.0


def test_run_strategy_rejects_cache_built_for_another_chunk():
    ep = se.Episode(0, np.zeros((8, 2), np.float32), np.zeros((8, 2), np.float32), torch.zeros(8, 3, 2, 2))
    cache = {"chunk": 12, "chunks": {0: np.zeros((1, 8, 2))}}
    with pytest.raises(ValueError):
        se.run_strategy([ep], {0: cache}, {0: (np.zeros((8, 2)), np.zeros(8))},
                        lambda d, s: se.always("verifier"), 8, 1, 1.0, 0.0, False)


class _FakeVerifier:
    device = torch.device("cpu")
    horizon = 16

    def __init__(self):
        self.calls = []
        self.current = None

    def set_sampler(self, scheduler, steps):
        self.current = (scheduler, steps)

    def chunk(self, states, images, noise, length):
        self.calls.append(self.current)
        return np.zeros((1, length, 2))


def test_latency_benchmark_times_every_sampler_equally_and_rotates_order():
    fake = _FakeVerifier()
    samplers = se.parse_samplers("DDPM:20:8,DDIM:5:8,DDPM:20:12")
    hist = [(torch.zeros(2, 2), torch.zeros(2, 3, 2, 2))]
    times = se.benchmark_latency(fake, samplers, hist, reps=6)
    assert all(len(times[s]) == 6 for s in samplers)
    firsts = [fake.calls[i * len(samplers)] for i in range(6)]
    assert len(set(firsts)) == 2, "each rep should start from a different sampler"  # two share (DDPM, 20)


# ---------------------------------------------------------------- paid access guard

def _one_episode(length=24, chunk=8):
    ep = se.Episode(7, np.zeros((length, 2), np.float32), np.zeros((length, 2), np.float32),
                    torch.zeros(length, 3, 2, 2))
    cache = {"chunk": chunk, "chunks": {b: np.zeros((1, min(chunk, length - b), 2)) for b in range(0, length, chunk)}}
    return ep, {7: cache}, {7: (np.zeros((length, 2)), np.zeros(length))}


def test_open_loop_gate_that_peeks_without_paying_raises():
    ep, caches, drafts = _one_episode()

    def cheat(delta_at, spread):
        def decide(b, log):
            delta_at(b)
            return "draft", False
        return decide
    with pytest.raises(AssertionError, match="without paying"):
        se.run_strategy([ep], caches, drafts, cheat, 8, 1, 1.0, 0.0, True)


def test_open_loop_gate_cannot_read_unpaid_past_plans_or_future_spread():
    ep, caches, drafts = _one_episode()
    peek_past = lambda d, s: (lambda b, log: (d(b - 8) if b else None, ("draft", False))[1])
    with pytest.raises(AssertionError, match="without paying"):
        se.run_strategy([ep], caches, drafts, peek_past, 8, 1, 1.0, 0.0, True)
    look_ahead = lambda d, s: (lambda b, log: (s[b + 1], ("draft", False))[1])
    with pytest.raises(AssertionError, match="ahead"):
        se.run_strategy([ep], caches, drafts, look_ahead, 8, 1, 1.0, 0.0, True)


def test_shipped_gates_pass_the_guard():
    ep, caches, drafts = _one_episode()
    for name, signal, factory in se.GATES:
        for tau in (0.0, 1.0, 1e9):
            rows = se.run_strategy([ep], caches, drafts, se.build_gate(signal, factory, tau), 8, 1, 1.0, 0.0, True)
            assert rows[0]["frames"] == 24
