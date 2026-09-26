"""Closed-loop evaluation in the gym-pusht simulator.

Open-loop agreement with the demonstrator (spec_eval.py) can't say whether a
strategy actually gets the T into place, and it rewards samplers that vary less.
This drives each serving strategy in the simulator and records the two numbers
pusht results are reported in: success (gym-pusht's is_success, block coverage
above 95%) and the best coverage reached.

All rollouts of a strategy run in lockstep, so each plan boundary is one batched
verifier call. Every strategy uses the same environment seeds, so comparisons are
paired by starting position. Once trajectories diverge nothing else is shared.
Gates are the functions from spec_eval.py, wrapped so that a gate which reads the
verifier's plan at a boundary without paying for that call raises.

    .venv/bin/python closed_loop.py --from results/v2_openloop.json --rollouts 100
    .venv/bin/python closed_loop.py --summarize results/v2_closed.json   # recompute stats only
"""
import compat  # noqa: F401  (must precede lerobot)

import argparse
import json
import pathlib
import platform
import time
from collections import deque

import numpy as np
import torch

import spec_eval as se
import stats

ROOT = pathlib.Path(__file__).resolve().parent
QUANTILES = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]  # spec_eval's grid
DRAFT_CHUNK = 8


def obs_to_tensors(obs) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.as_tensor(obs["agent_pos"], dtype=torch.float32)
    image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0  # dataset images are [0, 1]
    return state, image


class _NeedsPlan(Exception):
    """Raised when a gate asks for the verifier's plan at the current boundary."""


def run_batch(envs, seeds, verifier, sampler, drafts, make_decide, max_steps) -> list[dict]:
    """Roll out one strategy on every seed at once. sampler is None for draft-only.

    Plan boundaries fall at the same step for every rollout. At each boundary every
    running rollout's gate decides first. Rollouts that call the verifier, or whose
    gate needs the plan to decide, get planned in one batched call, and the gates
    that needed the plan decide again with it. A gate that reads the plan must pay
    for the call. Each rollout's starting noise is seeded by (seed, boundary); the
    DDPM step noise depends on which rollouts share the batch.
    """
    n = len(envs)
    n_obs = verifier.n_obs
    chunk_len = sampler.chunk if sampler is not None else DRAFT_CHUNK
    states, images = [], []
    for env, seed in zip(envs, seeds):
        obs, _ = env.reset(seed=int(seed))
        st, im = obs_to_tensors(obs)
        states.append(deque([st] * n_obs, maxlen=n_obs))
        images.append(deque([im] * n_obs, maxlen=n_obs))
    alive = np.ones(n, bool)
    best, success, steps = np.zeros(n), np.zeros(n, bool), np.zeros(n, int)
    logs = [[] for _ in range(n)]
    paid = [{} for _ in range(n)]
    draft_at = [{} for _ in range(n)]
    spread = [{} for _ in range(n)]
    queue, mode = [[] for _ in range(n)], [None] * n

    with torch.inference_mode():
        for t in range(max_steps):
            current = torch.stack([states[i][-1] for i in range(n)])
            if t % chunk_len == 0:
                b = t
                running = np.flatnonzero(alive)
                preds = torch.stack([d(current) for d in drafts]).numpy()  # (members, n, 2)
                for i in running:
                    draft_at[i][b] = preds[0, i]
                    spread[i][b] = float(np.linalg.norm(preds[:, i].std(axis=0))) if len(drafts) > 1 else 0.0

                def asker(i, plan):
                    """delta_at for rollout i. plan is this boundary's plan, or None if not computed yet."""
                    peeked = []

                    def delta_at(bb):
                        if bb in paid[i]:
                            return float(np.linalg.norm(paid[i][bb][0] - draft_at[i][bb]))
                        if bb == b:
                            if plan is None:
                                raise _NeedsPlan
                            peeked.append(bb)
                            return float(np.linalg.norm(plan[0] - draft_at[i][bb]))
                        raise AssertionError(f"gate asked about boundary {bb}, where the verifier never ran")
                    return delta_at, peeked

                decisions, need = {}, []
                for i in running:
                    try:
                        decisions[i] = make_decide(asker(i, None)[0], spread[i])(b, logs[i])
                        if decisions[i][1]:
                            need.append(i)
                    except _NeedsPlan:
                        decisions[i] = None
                        need.append(i)
                plans = {}
                if need:
                    if sampler is None:
                        raise AssertionError("gate called a verifier this strategy doesn't have")
                    st = torch.stack([torch.stack(list(states[i])) for i in need])
                    im = torch.stack([torch.stack(list(images[i])) for i in need])
                    noise = torch.cat([torch.randn(1, verifier.horizon, 2, generator=torch.Generator().manual_seed(
                        se.boundary_seed(int(seeds[i]), b))) for i in need])
                    se.seed_all(se.boundary_seed(int(seeds[0]), b), verifier.device)
                    for i, plan in zip(need, verifier.chunk(st, im, noise, chunk_len)):
                        plans[i] = plan
                for i in running:
                    if decisions[i] is None:  # the gate needed the plan; decide again with it
                        delta_at, peeked = asker(i, plans[i])
                        decisions[i] = make_decide(delta_at, spread[i])(b, logs[i])
                        if peeked and not decisions[i][1]:
                            raise AssertionError("gate used the verifier's plan without paying for the call")
                    serve_with, called = decisions[i]
                    if serve_with == "verifier" and not called:
                        raise AssertionError("served the verifier's plan without paying for the call")
                    if called:
                        paid[i][b] = plans[i]
                    logs[i].append({"b": b, "serve": serve_with, "called": called})
                    mode[i] = serve_with
                    queue[i] = list(paid[i][b]) if serve_with == "verifier" else []
            draft_now = drafts[0](current).numpy()
            for i in np.flatnonzero(alive):
                action = queue[i].pop(0) if mode[i] == "verifier" else draft_now[i]
                obs, _, terminated, truncated, info = envs[i].step(action)
                st, im = obs_to_tensors(obs)
                states[i].append(st)
                images[i].append(im)
                best[i] = max(best[i], float(info["coverage"]))
                success[i] = success[i] or bool(info["is_success"])
                steps[i] = t + 1
                if terminated or truncated:
                    alive[i] = False
            if not alive.any():
                break
    return [{"seed": int(s), "success": bool(success[i]), "max_coverage": float(best[i]), "steps": int(steps[i]),
             "calls": int(sum(e["called"] for e in logs[i])),
             "draft_chunks": int(sum(e["serve"] == "draft" for e in logs[i]))} for i, s in enumerate(seeds)]


def summarize(name: str, rows: list[dict], ref: list[dict] | None, call_ms: float, draft_ms: float,
              uses_draft: bool) -> dict:
    succ = np.array([r["success"] for r in rows], bool)
    cov = np.array([r["max_coverage"] for r in rows])
    steps = np.array([r["steps"] for r in rows], float)
    calls = np.array([r["calls"] for r in rows], float)
    ones = np.ones(len(rows))
    p, lo, hi = stats.wilson(int(succ.sum()), len(succ))
    c, clo, chi = stats.bca_cluster(cov, ones)
    out = {
        "strategy": name, "rollouts": len(rows),
        "success": p, "success_ci": [lo, hi],
        "max_coverage": c, "max_coverage_ci": [clo, chi],
        "verifier_calls_per_step": float(calls.sum() / steps.sum()),
        "ms_per_step": float((calls.sum() * call_ms + (steps.sum() * draft_ms if uses_draft else 0.0)) / steps.sum()),
    }
    if ref is not None:
        if [r["seed"] for r in rows] != [r["seed"] for r in ref]:
            raise ValueError(f"{name}: rollouts aren't paired with the reference by seed")
        ref_succ = np.array([r["success"] for r in ref], bool)
        d, dlo, dhi = stats.newcombe_paired(succ, ref_succ)
        out["success_vs_ref"], out["success_vs_ref_ci"] = d, [dlo, dhi]
        out["success_discordant"] = [int(np.sum(succ & ~ref_succ)), int(np.sum(~succ & ref_succ))]
        out["success_mcnemar_p"] = stats.mcnemar_exact(*out["success_discordant"])
        dc, dclo, dchi = stats.bca_cluster(cov - np.array([r["max_coverage"] for r in ref]), ones)
        out["coverage_vs_ref"], out["coverage_vs_ref_ci"] = dc, [dclo, dchi]
    return out


def summarize_all(results: dict, latency: dict, draft_ms: float) -> list[dict]:
    """results: name -> {"rows", "sampler" (name or None), "uses_draft"}; the reference name ends in (reference)."""
    ref_name = next(k for k in results if k.endswith("(reference)"))
    ref_rows = results[ref_name]["rows"]
    out = []
    for name, r in results.items():
        call_ms = latency[r["sampler"]] if r["sampler"] else 0.0
        out.append(summarize(name, r["rows"], None if name == ref_name else ref_rows, call_ms, draft_ms,
                             r["uses_draft"]))
    return out


def print_table(summary: list[dict]) -> None:
    print(f"\n{'strategy':58} {'success [95% CI]':>20} {'Δ vs ref [95% CI]':>22} {'p':>6} "
          f"{'coverage':>9} {'calls/step':>10} {'ms/step':>8}")
    for r in summary:
        s = f"{r['success']:.2f} [{r['success_ci'][0]:.2f},{r['success_ci'][1]:.2f}]"
        ds = (f"{r['success_vs_ref']:+.2f} [{r['success_vs_ref_ci'][0]:+.2f},{r['success_vs_ref_ci'][1]:+.2f}]"
              if "success_vs_ref" in r else "")
        p = f"{r['success_mcnemar_p']:.3f}" if "success_mcnemar_p" in r else ""
        print(f"{r['strategy']:58} {s:>20} {ds:>22} {p:>6} {r['max_coverage']:9.3f} "
              f"{r['verifier_calls_per_step']:10.3f} {r['ms_per_step']:8.1f}")


def git_sha():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # not a git checkout; record that rather than fail
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", help="spec_eval.py results JSON (thresholds, latency, split)")
    ap.add_argument("--summarize", default=None, help="recompute the summary of an existing closed-loop JSON")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rollouts", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=200_000, help="first environment seed")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--gate-quantiles", default="0.1,0.5",
                    help="gate thresholds at these quantiles of the validation signal (spec_eval's grid)")
    ap.add_argument("--samplers", default="DDPM:20:8,DDPM:10:8,DDPM:5:8,DDIM:10:8,DDIM:5:8,DDPM:20:12,DDPM:20:15")
    ap.add_argument("--draft-members", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.summarize:
        path = pathlib.Path(args.summarize)
        rec = json.loads(path.read_text())
        rec["summary"] = summarize_all(rec["results"], rec["latency_ms"], rec["draft_ms_per_frame"])
        rec.setdefault("provenance", {})["summarized_git"] = git_sha()
        path.write_text(json.dumps(rec, indent=2, default=float))
        print_table(rec["summary"])
        print(f"rewrote {path}")
        return
    if not args.source:
        ap.error("--from is required unless --summarize is given")

    import gym_pusht  # noqa: F401  (registers the environment)
    import gymnasium as gym

    src = json.loads(pathlib.Path(args.source).read_text())
    latency = {name: v["median"] for name, v in src["latency_ms"].items()}
    draft_ms = next(iter(src["latency_ms"].values()))["draft_ms_per_frame"]
    quantiles = [float(q) for q in args.gate_quantiles.split(",")]
    grid = src.get("quantiles", QUANTILES)  # the grid spec_eval used for the validation curves
    taus = {q: {name: curve[grid.index(q)]["tau"] for name, curve in src["val_curves"].items()}
            for q in quantiles}

    device = se.pick_device(args.device)
    samplers = se.parse_samplers(args.samplers)
    ref = samplers[0]
    open_ref = se.parse_samplers(src["config"]["samplers"])[0]
    if ref != open_ref:
        raise ValueError(f"reference {ref.name} differs from the open-loop reference {open_ref.name}, "
                         "so the gate thresholds wouldn't mean the same thing")
    missing = [s.name for s in samplers if s.name not in latency]
    if missing:
        raise ValueError(f"no benchmarked latency for {missing} in {args.source}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("lerobot/pusht")
    train = src["splits"]["train"]
    lo = int(dataset.meta.episodes[train[0]]["dataset_from_index"])
    hi = int(dataset.meta.episodes[train[-1]]["dataset_to_index"])
    rows = dataset.hf_dataset.select(range(lo, hi))
    drafts = se.train_drafts(np.asarray(rows["observation.state"], dtype=np.float32),
                             np.asarray(rows["action"], dtype=np.float32), args.draft_members)

    verifier = se.Verifier(device)
    verifier.warmup()
    seeds = list(range(args.seed0, args.seed0 + args.rollouts))
    envs = [gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                     max_episode_steps=args.max_steps) for _ in seeds]

    results = {}

    def run(name, sampler, make_decide, uses_draft):
        if sampler is not None:
            verifier.set_sampler(sampler.scheduler, sampler.steps)
        t0 = time.perf_counter()
        out = run_batch(envs, seeds, verifier, sampler, drafts, make_decide, args.max_steps)
        results[name] = {"rows": out, "sampler": sampler.name if sampler else None, "uses_draft": uses_draft}
        print(f"  {name:58} success {np.mean([r['success'] for r in out]):.2f}  "
              f"coverage {np.mean([r['max_coverage'] for r in out]):.3f}  ({time.perf_counter() - t0:.0f}s)",
              flush=True)

    print(f"device={device}  {args.rollouts} rollouts per strategy, seeds {seeds[0]}..{seeds[-1]}", flush=True)
    use_verifier, use_draft = (lambda d, s: se.always("verifier")), (lambda d, s: se.always("draft"))
    for s in samplers:
        run(f"verifier {s.name}" + (" (reference)" if s == ref else ""), s, use_verifier, False)
    run("draft only", None, use_draft, True)
    for q in quantiles:
        for name, signal, factory in se.GATES:
            run(f"{name} @ tau={taus[q][name]:.2f} (val q={q})", ref,
                se.build_gate(signal, factory, taus[q][name]), True)

    summary = summarize_all(results, latency, draft_ms)
    print_table(summary)

    import gym_pusht as gp
    import lerobot
    record = {
        "config": vars(args),
        "provenance": {"git": git_sha(), "python": platform.python_version(), "torch": torch.__version__,
                       "lerobot": getattr(lerobot, "__version__", None), "device": str(device),
                       "gym_pusht": getattr(gp, "__version__", None), "source": args.source},
        "taus": {str(q): t for q, t in taus.items()},
        "latency_ms": latency,
        "draft_ms_per_frame": draft_ms,
        "summary": summary,
        "results": results,
    }
    out = pathlib.Path(args.out) if args.out else ROOT / "results" / f"closed_loop_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
