"""Speculative serving for lerobot/diffusion_pusht, evaluated properly (v2).

What this fixes over notebook.py:
  - The draft trains only on training episodes, never on evaluation ones.
  - Every strategy is scored on every frame of the same episodes, with the same
    starting diffusion noise wherever plan boundaries coincide, so differences are paired.
  - A real chunked serving loop: decide at each chunk boundary, serve that chunk,
    score every served action.
  - Gates are charged for the verifier calls they actually make, and a gate that
    reads the verifier's plan without paying for it raises. The notebook's rule needs
    the verifier's action to decide, so it pays every time. Two gates that decide
    without the verifier are added.
  - The verifier sees its real two-frame observation history.
  - Thresholds are picked on validation episodes and only reported on test.
  - Intervals are BCa bootstraps over whole episodes.
  - Short episodes and bad splits raise instead of being skipped.

Limits that remain: this is open-loop, with states and images from the recorded
demonstrations, so it measures agreement with the demonstrator and says nothing
about task success. The verifier checkpoint was trained on every pusht episode,
test split included. Only the draft is held out.

    uv run spec_eval.py --max-episodes 2                  # smoke test
    uv run spec_eval.py --out results/repro_openloop.json  # the paper's settings (4 samples)
"""
import compat  # noqa: F401  (patches argparse/pyarrow for lerobot; must come first)

import argparse
import functools
import json
import pathlib
import pickle
import platform
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

import stats

DEADLINE_MS = 100.0  # pusht runs at 10 Hz
# Most open-loop error sits in each episode's first plan, which every sampler shares a hard
# start for (the demonstrator moves fastest there). Frames from LATE_FROM on are past the first
# plan for every chunk length tested. Reported as a sensitivity analysis, added after review.
LATE_FROM = 16
ROOT = pathlib.Path(__file__).resolve().parent


# --------------------------------------------------------------------- splits

def parse_split(spec: str, n_episodes: int) -> dict[str, list[int]]:
    """'0:150,150:170,170:206' -> {'train': [...], 'val': [...], 'test': [...]}."""
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"--split needs train,val,test ranges, got {spec!r}")
    splits, seen = {}, set()
    for name, part in zip(("train", "val", "test"), parts):
        lo, hi = (int(x) for x in part.split(":"))
        if not 0 <= lo < hi <= n_episodes:
            raise ValueError(f"{name} range {lo}:{hi} is outside 0:{n_episodes}")
        eps = list(range(lo, hi))
        if seen & set(eps):
            raise ValueError(f"{name} overlaps an earlier split")
        seen |= set(eps)
        splits[name] = eps
    return splits


# ---------------------------------------------------------------------- draft

class StateMlp(nn.Module):
    """The notebook's draft, unchanged: pusher position -> action, two 64-unit ReLU layers."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_drafts(states: np.ndarray, actions: np.ndarray, members: int,
                 epochs: int = 15, lr: float = 1e-3, batch: int = 256) -> list[StateMlp]:
    """Train `members` independently seeded drafts with the notebook's recipe.

    Member 0 is the draft that gets served. The spread across members is a
    verifier-free uncertainty signal for the ensemble gate.
    """
    X, y = torch.from_numpy(states), torch.from_numpy(actions)
    drafts = []
    for seed in range(members):
        torch.manual_seed(seed)
        order = torch.Generator().manual_seed(seed)
        model = StateMlp()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ in range(epochs):
            perm = torch.randperm(len(X), generator=order)
            for i in range(0, len(X), batch):
                idx = perm[i:i + batch]
                loss = nn.functional.mse_loss(model(X[idx]), y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        model.eval()
        drafts.append(model)
    return drafts


@torch.inference_mode()
def draft_outputs(drafts: list[StateMlp], states: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Served draft actions (member 0), ensemble spread per frame, and ms per frame."""
    x = torch.from_numpy(states)
    t0 = time.perf_counter()
    for i in range(len(x)):  # one frame at a time, the way it would be served
        for d in drafts:
            d(x[i:i + 1])
    ms_per_frame = (time.perf_counter() - t0) * 1000.0 / len(x)
    preds = torch.stack([d(x) for d in drafts]).numpy()  # (members, T, 2)
    spread = np.linalg.norm(preds.std(axis=0), axis=-1) if len(drafts) > 1 else np.zeros(len(x))
    return preds[0], spread, ms_per_frame


# ------------------------------------------------------------------- verifier

def pick_device(name: str) -> torch.device:
    device = torch.device(name) if name != "auto" else (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    if device.type == "cuda":  # deterministic kernels and full fp32, so reruns match each other
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    return device


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)  # also seeds CUDA
    if device.type == "mps":
        torch.mps.manual_seed(seed)


class Verifier:
    """lerobot/diffusion_pusht with explicit history, noise, sampler and chunk length."""

    def __init__(self, device: torch.device):
        from huggingface_hub import hf_hub_download
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from safetensors.torch import load_file

        self.device = device
        self.policy = DiffusionPolicy.from_pretrained("lerobot/diffusion_pusht").to(device).eval()
        self.cfg = self.policy.config
        # lerobot 0.4 no longer normalizes inside the policy; the stats live in the checkpoint.
        ckpt = load_file(hf_hub_download("lerobot/diffusion_pusht", "model.safetensors"))
        keys = {
            "image_mean": "normalize_inputs.buffer_observation_image.mean",
            "image_std": "normalize_inputs.buffer_observation_image.std",
            "state_min": "normalize_inputs.buffer_observation_state.min",
            "state_max": "normalize_inputs.buffer_observation_state.max",
            "action_min": "unnormalize_outputs.buffer_action.min",
            "action_max": "unnormalize_outputs.buffer_action.max",
        }
        self.norm = {k: ckpt[v].to(device) for k, v in keys.items()}
        self.n_obs = self.cfg.n_obs_steps
        self.horizon = self.cfg.horizon
        self.max_chunk = self.horizon - self.n_obs + 1

    def warmup(self) -> None:
        """One untimed call so the first timed call doesn't pay for kernel setup."""
        self.set_sampler("DDPM", 2)
        st = torch.zeros(1, self.n_obs, 2)
        im = torch.zeros(1, self.n_obs, 3, 96, 96)
        self.chunk(st, im, torch.randn(1, self.horizon, 2), 1)
        sync(self.device)

    def set_sampler(self, scheduler: str, steps: int) -> None:
        from diffusers import DDIMScheduler, DDPMScheduler

        c = self.cfg
        kwargs = dict(
            num_train_timesteps=c.num_train_timesteps, beta_start=c.beta_start, beta_end=c.beta_end,
            beta_schedule=c.beta_schedule, clip_sample=c.clip_sample,
            clip_sample_range=c.clip_sample_range, prediction_type=c.prediction_type,
        )
        cls = {"DDPM": DDPMScheduler, "DDIM": DDIMScheduler}[scheduler]
        self.policy.diffusion.noise_scheduler = cls(**kwargs)
        self.policy.diffusion.num_inference_steps = steps

    @torch.inference_mode()
    def chunk(self, states: torch.Tensor, images: torch.Tensor, noise: torch.Tensor,
              length: int) -> np.ndarray:
        """states (B, n_obs, 2) and images (B, n_obs, 3, H, W) in raw units -> (B, length, 2) pixels."""
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        if not 1 <= length <= self.max_chunk:
            raise ValueError(f"chunk length {length} outside 1..{self.max_chunk} for this checkpoint")
        n, dev = self.norm, self.device
        s = 2.0 * (states.to(dev) - n["state_min"]) / (n["state_max"] - n["state_min"]) - 1.0
        im = (images.to(dev) - n["image_mean"]) / n["image_std"]
        batch = {OBS_STATE: s, OBS_IMAGES: im.unsqueeze(2)}  # one camera
        d = self.policy.diffusion
        cond = d._prepare_global_conditioning(batch)
        sample = d.conditional_sample(noise.shape[0], global_cond=cond, noise=noise.to(dev))
        start = self.n_obs - 1
        a = sample[:, start:start + length]
        a = (a + 1.0) / 2.0 * (n["action_max"] - n["action_min"]) + n["action_min"]
        return a.float().cpu().numpy()


# ------------------------------------------------------------------- episodes

@dataclass
class Episode:
    index: int
    states: np.ndarray   # (T, 2) pusher position
    actions: np.ndarray  # (T, 2) demonstrator action
    images: torch.Tensor  # (T, 3, H, W) in [0, 1]

    @property
    def length(self) -> int:
        return len(self.states)

    def history(self, t: int, n_obs: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The n_obs most recent observations ending at t. Only t=0 repeats a frame,
        exactly as a real rollout starts."""
        idx = [max(t - k, 0) for k in reversed(range(n_obs))]
        return torch.from_numpy(self.states[idx]), self.images[idx]


def load_episode(dataset, index: int) -> Episode:
    meta = dataset.meta.episodes[index]
    lo, hi = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    rows = dataset.hf_dataset.select(range(lo, hi))
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    actions = np.asarray(rows["action"], dtype=np.float32)
    images = torch.stack([dataset[i]["observation.image"] for i in range(lo, hi)])
    if not (len(states) == len(actions) == len(images) == hi - lo):
        raise RuntimeError(f"episode {index}: fields disagree on length")
    return Episode(index, states, actions, images)


# ------------------------------------------------------------- verifier calls

@dataclass(frozen=True)
class Sampler:
    scheduler: str
    steps: int
    chunk: int

    @property
    def name(self) -> str:
        return f"{self.scheduler}-{self.steps}/chunk{self.chunk}"


def parse_samplers(spec: str) -> list[Sampler]:
    out = []
    for item in spec.split(","):
        sched, steps, chunk = item.split(":")
        out.append(Sampler(sched.upper(), int(steps), int(chunk)))
    return out


def boundary_seed(episode: int, boundary: int) -> int:
    return episode * 100_003 + boundary


def run_verifier(verifier: Verifier, ep: Episode, sampler: Sampler, samples: int) -> dict:
    """Verifier chunks at every boundary of one episode, all K samples in one batch.

    Initial noise and the DDPM step noise are seeded by (episode, boundary), so
    every sampler sees the same starting noise at the same boundary and every run
    reproduces. Untimed: latency comes from benchmark_latency.
    """
    verifier.set_sampler(sampler.scheduler, sampler.steps)
    chunks = {}
    for b in range(0, ep.length, sampler.chunk):
        n = min(sampler.chunk, ep.length - b)
        st, im = ep.history(b, verifier.n_obs)
        seed = boundary_seed(ep.index, b)
        noise = torch.randn(samples, verifier.horizon, 2, generator=torch.Generator().manual_seed(seed))
        seed_all(seed, verifier.device)
        out = verifier.chunk(st[None].expand(samples, -1, -1), im[None].expand(samples, -1, -1, -1, -1),
                             noise, sampler.chunk)
        chunks[b] = out[:, :n]  # (K, n, 2)
    return {"chunks": chunks, "chunk": sampler.chunk}


def benchmark_latency(verifier: Verifier, samplers: list[Sampler], histories: list, reps: int) -> dict:
    """Time one batch-of-1 call per sampler, round-robin over the same real inputs.

    Interleaving means thermal throttling or background load lands on every
    sampler equally, which the per-episode timing in a long run does not.
    """
    times = {s: [] for s in samplers}
    for r in range(reps):
        k = r % len(samplers)
        for s in samplers[k:] + samplers[:k]:
            st, im = histories[r % len(histories)]
            verifier.set_sampler(s.scheduler, s.steps)
            noise = torch.randn(1, verifier.horizon, 2, generator=torch.Generator().manual_seed(r))
            sync(verifier.device)
            t0 = time.perf_counter()
            verifier.chunk(st[None], im[None], noise, s.chunk)
            sync(verifier.device)
            times[s].append((time.perf_counter() - t0) * 1000.0)
    return {s: np.array(v) for s, v in times.items()}


# ------------------------------------------------------------ serving + gates

class PaidAccess:
    """What a gate may see at boundary b: the verifier's plan at b (and then it must
    pay for the call), plans at earlier boundaries it paid for, and draft spread up
    to b. Anything else raises."""

    def __init__(self, plan_at, draft_actions: np.ndarray, spread: np.ndarray):
        self._plan_at, self._draft, self._spread = plan_at, draft_actions, spread
        self.current, self.peeked, self.paid = None, False, set()
        guard = self

        class _SpreadSoFar:
            def __getitem__(self, i):
                if i > guard.current:
                    raise AssertionError(f"gate looked at draft spread at {i}, ahead of boundary {guard.current}")
                return guard._spread[i]
        self.spread = _SpreadSoFar()

    def delta_at(self, b: int) -> float:
        if b == self.current:
            self.peeked = True
        elif b not in self.paid:
            raise AssertionError(f"gate read the verifier's plan at boundary {b} without paying for it")
        return float(np.linalg.norm(self._plan_at(b)[0] - self._draft[b]))


def serve(length: int, chunk: int, decide, chunk_at, draft_actions: np.ndarray, guard: PaidAccess | None = None):
    """The chunked serving loop over one episode.

    At each boundary b, decide(b, log) returns (serve_with, called_verifier).
    serve_with is "verifier" or "draft". called_verifier says whether this
    boundary paid for a verifier call, which can be true even when the draft is
    served (a gate that needs the verifier to decide). Every frame is served
    exactly once. With a guard, a gate that reads what it didn't pay for raises.
    """
    served = np.full((length, 2), np.nan)
    log = []
    for b in range(0, length, chunk):
        n = min(chunk, length - b)
        if guard is not None:
            guard.current, guard.peeked = b, False
        serve_with, called = decide(b, log)
        if guard is not None:
            if guard.peeked and not called:
                raise AssertionError("gate used the verifier's plan without paying for the call")
            if called:
                guard.paid.add(b)
        if serve_with == "verifier":
            if not called:
                raise AssertionError("served the verifier chunk without paying for the call")
            served[b:b + n] = chunk_at(b)
        elif serve_with == "draft":
            served[b:b + n] = draft_actions[b:b + n]
        else:
            raise ValueError(serve_with)
        log.append({"b": b, "serve": serve_with, "called": called})
    if np.isnan(served).any():
        raise AssertionError("some frames were never served")
    return served, log


def always(serve_with: str):
    return lambda b, log: (serve_with, serve_with == "verifier")


def gate_needs_verifier(tau: float, delta_at):
    """The notebook's rule, charged honestly: it needs the verifier's action to
    compute delta, so it pays for a call at every boundary."""
    def decide(b, log):
        return ("draft" if delta_at(b) < tau else "verifier"), True
    return decide


def gate_previous_agreement(tau: float, delta_at, max_run: int = 1):
    """Verifier-free for the skipped chunk: after a verifier call whose chunk
    agreed with the draft (delta < tau), serve the draft for up to `max_run`
    chunks without calling the verifier, then check again."""
    def decide(b, log):
        run = 0
        for entry in reversed(log):
            if entry["called"]:
                break
            run += 1
        last_called = next((e for e in reversed(log) if e["called"]), None)
        if last_called is not None and run < max_run and delta_at(last_called["b"]) < tau:
            return "draft", False
        return "verifier", True
    return decide


def gate_ensemble(tau: float, spread: np.ndarray):
    """Verifier-free: serve the draft chunk when the draft ensemble agrees with itself."""
    def decide(b, log):
        return ("draft", False) if spread[b] < tau else ("verifier", True)
    return decide


# name, which signal the threshold applies to, gate factory
GATES = [
    ("gate: needs verifier (notebook rule)", "delta", gate_needs_verifier),
    ("gate: previous agreement", "delta", gate_previous_agreement),
    ("gate: draft ensemble", "spread", gate_ensemble),
]


def build_gate(signal: str, factory, tau: float):
    """Adapt a gate factory to run_strategy's make_decide(delta_at, spread)."""
    if signal == "delta":
        return lambda delta_at, spread: factory(tau, delta_at)
    return lambda delta_at, spread: factory(tau, spread)


# -------------------------------------------------------------------- metrics

def mse_per_frame(served: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.mean((served - actions) ** 2, axis=-1)  # same per-frame MSE as the notebook


def cluster_bootstrap(values: np.ndarray, weights: np.ndarray, B: int = 10_000, seed: int = 0):
    """BCa interval for a frame-weighted mean, resampling whole episodes."""
    return stats.bca_cluster(values, weights, B=B, seed=seed)


def evaluate(strategy: str, per_episode: list[dict], ref: dict | None, chunk: int,
             latency: np.ndarray | None = None, margin: float | None = None, max_chunk: int = 15,
             margin_frac: float = 0.05) -> dict:
    """Aggregate one strategy over episodes. Each per_episode entry holds mse (K,),
    bias2, var, frames, calls, draft_frames and ms. `latency` is the benchmarked call
    times of the sampler this strategy calls, or None if it never calls one. `margin`
    is the non-inferiority margin in MSE units."""
    frames = np.array([e["frames"] for e in per_episode], dtype=float)
    mse = np.array([np.mean(e["mse"]) for e in per_episode])
    calls = np.array([e["calls"] for e in per_episode], dtype=float)
    ms = np.array([e["ms"] for e in per_episode])
    est, lo, hi = cluster_bootstrap(mse, frames)
    ms_per_frame = float(ms.sum() / frames.sum())
    called = latency is not None and calls.sum() > 0
    max_call = float(np.max(latency)) if called else 0.0
    out = {
        "strategy": strategy,
        "chunk": chunk,
        "episodes": len(per_episode),
        "frames": int(frames.sum()),
        "mse": est, "mse_ci": [lo, hi],
        # MSE = bias2 + var exactly: error of the sample mean, plus spread across diffusion samples
        "bias2": stats.weighted_mean(np.array([e["bias2"] for e in per_episode]), frames),
        "sample_var": stats.weighted_mean(np.array([e["var"] for e in per_episode]), frames),
        "draft_fraction": float(sum(e["draft_frames"] for e in per_episode) / frames.sum()),
        "per_episode_mse": {int(e["episode"]): float(np.mean(e["mse"])) for e in per_episode},
        "verifier_calls_per_frame": float(calls.sum() / frames.sum()),
        "ms_per_frame": ms_per_frame,
        "p95_call_ms": float(np.percentile(latency, 95)) if called else 0.0,
        "max_call_ms": max_call,
        # averaged model time fits 100 ms per frame; a synchronous loop still stalls at every call
        "budget_per_frame_ok": ms_per_frame <= DEADLINE_MS,
        # A pipelined loop starts the next plan early, from an older observation. A plan made at
        # frame s covers s .. s+max_chunk-1, so it can arrive in time and still cover the next
        # chunk only if the call takes at most (max_chunk - chunk) frames. Checked against the
        # slowest benchmarked call; the pipelined loop itself is not simulated.
        "pipelined_10hz_ok": max_call <= (max_chunk - chunk) * DEADLINE_MS,
    }
    if ref is not None:
        by_ep = {e["episode"]: e for e in ref["per_episode"]}
        if [e["episode"] for e in per_episode] != [e["episode"] for e in ref["per_episode"]]:
            raise ValueError(f"{strategy}: episodes don't line up with the reference")
        diff = mse - np.array([np.mean(by_ep[e["episode"]]["mse"]) for e in per_episode])
        d, dlo, dhi = cluster_bootstrap(diff, frames)
        _, up = stats.bca_cluster(diff, frames, levels=(0.95,))
        out["mse_vs_ref"], out["mse_vs_ref_ci"], out["mse_vs_ref_upper95"] = d, [dlo, dhi], up
        if margin is not None:
            out["noninferior_margin"] = margin
            out["noninferior"] = bool(up <= margin)
        vdiff = np.array([e["var"] for e in per_episode]) - np.array([by_ep[e["episode"]]["var"] for e in per_episode])
        vd, vlo, vhi = cluster_bootstrap(vdiff, frames)
        out["sample_var_vs_ref"], out["sample_var_vs_ref_ci"] = vd, [vlo, vhi]
        out["speedup_vs_ref"] = ref["ms_per_frame"] / out["ms_per_frame"] if out["ms_per_frame"] else float("inf")
        out["episodes_worse"] = int(np.sum(diff > 0))
        late_frames = frames - LATE_FROM
        late = np.array([e["mse_late"] for e in per_episode])
        ref_late = np.array([by_ep[e["episode"]]["mse_late"] for e in per_episode])
        ld, llo, lhi = cluster_bootstrap(late - ref_late, late_frames)
        _, lup = stats.bca_cluster(late - ref_late, late_frames, levels=(0.95,))
        ref_late_mean = stats.weighted_mean(ref_late, late_frames)
        out["late"] = {"from_frame": LATE_FROM, "mse": stats.weighted_mean(late, late_frames),
                       "ref_mse": ref_late_mean, "vs_ref": ld, "vs_ref_ci": [llo, lhi], "vs_ref_upper95": lup,
                       "margin": margin_frac * ref_late_mean, "noninferior": bool(lup <= margin_frac * ref_late_mean),
                       "episodes_worse": int(np.sum(late > ref_late))}
    return out


# --------------------------------------------------------------------- driver

def run_strategy(eps: list[Episode], caches: dict, drafts_out: dict, make_decide, chunk: int,
                 samples: int, call_ms: float, draft_ms: float, uses_draft: bool) -> list[dict]:
    """Serve every episode under one strategy, once per diffusion sample.

    Time is calls x the benchmarked median call latency, plus the draft on every
    frame if the strategy runs it. Calls and draft frames are averaged over samples,
    since gates that look at the verifier's chunk can decide differently per sample.
    """
    results = []
    for ep in eps:
        cache = caches[ep.index]
        if cache["chunk"] != chunk:
            raise ValueError(f"cache was built for chunk {cache['chunk']}, strategy uses {chunk}")
        draft_act, spread = drafts_out[ep.index]
        mses, calls, draft_frames, all_served = [], [], [], []
        for k in range(samples):
            chunk_at = lambda b, k=k: cache["chunks"][b][k]
            guard = PaidAccess(chunk_at, draft_act, spread)
            served, log = serve(ep.length, chunk, make_decide(guard.delta_at, guard.spread), chunk_at,
                                draft_act, guard)
            all_served.append(served)
            mses.append(float(np.mean(mse_per_frame(served, ep.actions))))
            calls.append(sum(e["called"] for e in log))
            draft_frames.append(sum(min(chunk, ep.length - e["b"]) for e in log if e["serve"] == "draft"))
        all_served = np.stack(all_served)  # (K, T, 2)
        centre = all_served.mean(axis=0)
        n_calls = float(np.mean(calls))
        ms = n_calls * call_ms + (draft_ms * ep.length if uses_draft else 0.0)
        results.append({"episode": ep.index, "frames": ep.length, "mse": np.array(mses),
                        "mse_late": float(np.mean([np.mean(mse_per_frame(sv[LATE_FROM:], ep.actions[LATE_FROM:]))
                                                   for sv in all_served])),
                        "bias2": float(np.mean(mse_per_frame(centre, ep.actions))),
                        "var": float(np.mean((all_served - centre) ** 2)),
                        "calls": n_calls, "ms": ms, "draft_frames": float(np.mean(draft_frames))})
    return results


def analyze(args, splits, samplers, v_eps, v_drafts, v_cache, t_eps, t_drafts, t_caches,
            latency, draft_ms, samples, max_chunk: int = 15) -> tuple[dict, list]:
    """Threshold selection on validation, then every strategy on test. Returns the
    results record and the summary rows (for printing)."""
    ref_sampler = samplers[0]
    ref_chunk = ref_sampler.chunk
    call_ms = {s: float(np.median(v)) for s, v in latency.items()}
    ref_ms = call_ms[ref_sampler]
    use_verifier, use_draft = (lambda d, s: always("verifier")), (lambda d, s: always("draft"))
    ev = functools.partial(evaluate, max_chunk=max_chunk, margin_frac=args.max_quality_loss)

    # ---- validation: pick each gate's threshold without looking at test.
    ref_val = ev("reference", run_strategy(v_eps, v_cache, v_drafts, use_verifier, ref_chunk,
                                                 samples, ref_ms, draft_ms, False), None, ref_chunk)
    signals = {
        "delta": np.array([np.linalg.norm(v_cache[e.index]["chunks"][b][0][0] - v_drafts[e.index][0][b])
                           for e in v_eps for b in v_cache[e.index]["chunks"]]),
        "spread": np.array([v_drafts[e.index][1][b] for e in v_eps for b in v_cache[e.index]["chunks"]]),
    }
    budget = ref_val["mse"] * (1 + args.max_quality_loss)
    # Thresholds come from quantiles of each gate's signal on validation, dense at the low end
    # where the draft is trusted rarely. The same grid is reused on test for the curves.
    quantiles = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    grids = {name: np.quantile(signals[signal], quantiles) for name, signal, _ in GATES}
    chosen, val_curves, closest = {}, {}, {}
    for name, signal, factory in GATES:
        best, val_curves[name] = None, []  # fewest verifier calls within the budget, then most draft frames
        for tau in grids[name]:
            r = ev(name, run_strategy(v_eps, v_cache, v_drafts, build_gate(signal, factory, float(tau)),
                                            ref_chunk, samples, ref_ms, draft_ms, True), None, ref_chunk)
            val_curves[name].append({"tau": float(tau), "mse": r["mse"],
                                     "verifier_calls_per_frame": r["verifier_calls_per_frame"],
                                     "draft_fraction": r["draft_fraction"]})
            key = (r["verifier_calls_per_frame"], -r["draft_fraction"])
            if r["mse"] <= budget and (best is None or key < best[1]):
                best = (float(tau), key)
        # report how close the nearest threshold came, so a near miss is visible
        near = min(val_curves[name], key=lambda c: c["mse"])
        closest[name] = {"tau": near["tau"], "val_mse": near["mse"], "budget": budget,
                         "missed_by": near["mse"] - budget,
                         "calls_saved": 1 - near["verifier_calls_per_frame"] / ref_val["verifier_calls_per_frame"]}
        chosen[name] = best[0] if best else 0.0  # tau 0 never trusts the draft: the gate falls back to the reference
        print(f"  {name}: tau={chosen[name]:.3f}" + ("" if best else
              f" (no threshold within budget; closest missed by {closest[name]['missed_by']:.2f})"), flush=True)

    # ---- test: every strategy on the same frames.
    ref_rows = run_strategy(t_eps, t_caches[ref_sampler], t_drafts, use_verifier, ref_chunk,
                            samples, ref_ms, draft_ms, False)
    ref_summary = ev(f"verifier {ref_sampler.name} (reference)", ref_rows, None, ref_chunk,
                           latency[ref_sampler])
    ref_summary["per_episode"] = ref_rows
    margin = args.max_quality_loss * ref_summary["mse"]  # the gates' budget, reused as the non-inferiority margin
    summary = [ref_summary]
    for s in samplers[1:]:
        rows = run_strategy(t_eps, t_caches[s], t_drafts, use_verifier, s.chunk, samples,
                            call_ms[s], draft_ms, False)
        summary.append(ev(f"verifier {s.name}", rows, ref_summary, s.chunk, latency[s], margin))
    rows = run_strategy(t_eps, t_caches[ref_sampler], t_drafts, use_draft, ref_chunk, samples,
                        ref_ms, draft_ms, True)
    summary.append(ev("draft only", rows, ref_summary, ref_chunk, None, margin))
    for name, signal, factory in GATES:
        rows = run_strategy(t_eps, t_caches[ref_sampler], t_drafts, build_gate(signal, factory, chosen[name]),
                            ref_chunk, samples, ref_ms, draft_ms, True)
        r = ev(f"{name} @ tau={chosen[name]:.2f}", rows, ref_summary, ref_chunk, latency[ref_sampler], margin)
        r["tau"] = chosen[name]
        summary.append(r)

    # Test curves over the validation grid: reported, never used to choose anything.
    test_curves = {}
    for name, signal, factory in GATES:
        test_curves[name] = []
        for tau in grids[name]:
            rows = run_strategy(t_eps, t_caches[ref_sampler], t_drafts, build_gate(signal, factory, float(tau)),
                                ref_chunk, samples, ref_ms, draft_ms, True)
            r = ev(name, rows, ref_summary, ref_chunk, latency[ref_sampler], margin)
            test_curves[name].append({k: r[k] for k in (
                "mse", "mse_ci", "mse_vs_ref", "mse_vs_ref_ci", "mse_vs_ref_upper95", "verifier_calls_per_frame",
                "draft_fraction", "ms_per_frame", "speedup_vs_ref")} | {"tau": float(tau)})

    record = {
        "config": vars(args),
        "splits": splits,
        "samples_per_boundary": samples,
        "val_reference_mse": ref_val["mse"],
        "val_draft_mse": stats.weighted_mean(
            np.array([np.mean(mse_per_frame(v_drafts[e.index][0], e.actions)) for e in v_eps]),
            np.array([e.length for e in v_eps], dtype=float)),
        "chosen_tau": chosen,
        "closest_to_budget": closest,
        "noninferiority_margin": margin,
        "quantiles": quantiles.tolist(),
        "max_chunk": max_chunk,
        "error_by_plan_offset": plan_offset_error(t_eps, t_caches, samplers),
        "latency_ms": {s.name: {"median": call_ms[s], "p95": float(np.percentile(v, 95)), "max": float(np.max(v)),
                                "n": len(v), "draft_ms_per_frame": draft_ms}
                       for s, v in latency.items()},
        "summary": [{k: v for k, v in r.items() if k != "per_episode"} for r in summary],
        "val_curves": val_curves,
        "test_curves": test_curves,
        "per_episode_reference": [{"episode": e["episode"], "frames": e["frames"], "mse": e["mse"].tolist()}
                                  for e in ref_rows],
    }
    return record, summary


def plan_offset_error(eps: list, caches: dict, samplers: list) -> dict | None:
    """Pooled error of the longest-chunk sampler's actions by how far into the plan they are,
    for frames from LATE_FROM on. A pipelined loop executes actions further into each plan,
    since the plan was made from an older observation."""
    longest = max(samplers, key=lambda s: s.chunk)
    if longest.chunk <= samplers[0].chunk:
        return None
    sq = np.zeros(longest.chunk)
    n = np.zeros(longest.chunk)
    for ep in eps:
        for b, plans in caches[longest][ep.index]["chunks"].items():
            for o in range(plans.shape[1]):
                if b + o >= LATE_FROM:
                    sq[o] += float(np.mean((plans[:, o] - ep.actions[b + o]) ** 2))
                    n[o] += 1
    per = (sq / np.maximum(n, 1)).tolist()
    return {"sampler": longest.name, "from_frame": LATE_FROM, "mse_by_offset": per,
            "windows": {f"{a}-{a + 7}": float(np.mean(per[a:a + 8])) for a in range(longest.chunk - 7)}}


def print_summary(summary: list) -> None:
    print(f"\n{'strategy':48} {'MSE':>7} {'vs ref [95% CI]':>24} {'1-sided 95%':>11} {'calls/fr':>9} "
          f"{'ms/fr':>7} {'speedup':>8} {'max call':>9}")
    for r in summary:
        vs = (f"{r['mse_vs_ref']:+7.1f} [{r['mse_vs_ref_ci'][0]:+.1f},{r['mse_vs_ref_ci'][1]:+.1f}]"
              if "mse_vs_ref" in r else "")
        up = f"{r['mse_vs_ref_upper95']:+.1f}" if "mse_vs_ref_upper95" in r else ""
        sp = f"{r['speedup_vs_ref']:.2f}x" if "speedup_vs_ref" in r else "1.00x"
        print(f"{r['strategy']:48} {r['mse']:7.1f} {vs:>24} {up:>11} {r['verifier_calls_per_frame']:9.3f} "
              f"{r['ms_per_frame']:7.1f} {sp:>8} {r['max_call_ms']:9.0f}")


MODEL_REPO, DATA_REPO = "lerobot/diffusion_pusht", "lerobot/pusht"


def code_provenance() -> dict:
    """Package versions that change results, hashes of the scripts, and whether the tree was dirty."""
    import hashlib
    from importlib import metadata
    pkgs = {}
    for name in ("torch", "lerobot", "diffusers", "numpy", "gym-pusht", "pymunk", "pygame", "opencv-python",
                 "opencv-python-headless", "torchcodec", "av"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    scripts = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
               for f in sorted(ROOT.glob("*.py")) if not f.name.startswith("test_")}
    try:
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "*.py"],
                                             text=True).strip())
    except Exception:  # not a git checkout
        dirty = None
    revisions = {}
    try:  # which Hugging Face snapshot was loaded
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id in (MODEL_REPO, DATA_REPO):
                revisions[repo.repo_id] = sorted(r.commit_hash for r in repo.revisions)
    except Exception:  # cache not readable; leave it out rather than fail the run
        pass
    return {"packages": pkgs, "script_sha256": scripts, "uncommitted_script_changes": dirty,
            "hf_snapshots": revisions}


def load_from_cache(path: pathlib.Path, samplers_spec: str):
    """Rebuild everything analyze() needs from a previous run's cache, with no diffusion."""
    cache = pickle.loads(path.read_bytes())
    prev = json.loads(path.with_name(path.name.replace(".cache.pkl", ".json")).read_text())
    samplers = parse_samplers(samplers_spec)
    names = [s.name for s in samplers]
    if names != cache["samplers"]:
        raise ValueError(f"cache has samplers {cache['samplers']}, asked for {names}")

    def episodes(split, which):
        eps, drafts_out, caches = [], {}, {s: {} for s in which}
        for i in prev["splits"][split]:
            c = cache[split][i]
            T = len(c["actions"])
            eps.append(Episode(i, np.zeros((T, 2), np.float32), c["actions"], torch.zeros(T, 3, 1, 1)))
            drafts_out[i] = (c["draft"], c["spread"])
            for s in which:
                caches[s][i] = {"chunks": c["chunks"][s.name], "chunk": s.chunk}
        return eps, drafts_out, caches

    v_eps, v_drafts, v_caches = episodes("val", samplers[:1])
    t_eps, t_drafts, t_caches = episodes("test", samplers)
    latency = {s: np.asarray(cache["latency_ms"][s.name]) for s in samplers}
    first = next(iter(cache["test"].values()))["chunks"][samplers[0].name]
    samples = next(iter(first.values())).shape[0]
    return (prev, samplers, v_eps, v_drafts, v_caches[samplers[0]], t_eps, t_drafts, t_caches,
            latency, cache["draft_ms_per_frame"], samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--split", default="0:150,150:170,170:206", help="train,val,test episode ranges")
    ap.add_argument("--max-episodes", type=int, default=None, help="cap val and test episodes (smoke tests)")
    ap.add_argument("--samples", type=int, default=4, help="diffusion samples per boundary (the paper used 4)")
    ap.add_argument("--samplers", default="DDPM:20:8,DDPM:10:8,DDPM:5:8,DDIM:10:8,DDIM:5:8,DDPM:20:12,DDPM:20:15",
                    help="scheduler:steps:chunk, first one is the reference and the verifier for the gates")
    ap.add_argument("--draft-members", type=int, default=5)
    ap.add_argument("--max-quality-loss", type=float, default=0.05,
                    help="gates may raise val MSE by at most this fraction of the reference; also the "
                         "non-inferiority margin on test")
    ap.add_argument("--latency-reps", type=int, default=50, help="timed calls per sampler")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--from-cache", default=None,
                    help="recompute every table and curve from a previous run's .cache.pkl, without diffusion")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    def git_sha():
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        except Exception:  # not a git checkout; record that rather than fail the run
            return None

    if args.from_cache:
        (prev, samplers, v_eps, v_drafts, v_cache, t_eps, t_drafts, t_caches,
         latency, draft_ms, samples) = load_from_cache(pathlib.Path(args.from_cache), args.samplers)
        splits = prev["splits"]
        args.samples = samples
        print(f"recomputing from {args.from_cache}: {len(v_eps)} val, {len(t_eps)} test episodes, "
              f"{samples} samples per boundary", flush=True)
        record, summary = analyze(args, splits, samplers, v_eps, v_drafts, v_cache, t_eps, t_drafts, t_caches,
                                  latency, draft_ms, samples, prev.get("max_chunk", 15))
        record["config"] = dict(prev["config"], max_quality_loss=args.max_quality_loss)
        record["provenance"] = dict(prev.get("provenance", {}), recomputed_from=args.from_cache,
                                    recompute_git=git_sha(), recompute_code=code_provenance())
        print_summary(summary)
        out = pathlib.Path(args.out) if args.out else pathlib.Path(args.from_cache.replace(".cache.pkl", ".json"))
        out.write_text(json.dumps(record, indent=2, default=float))
        print(f"wrote {out}")
        return

    if args.threads:
        torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    samplers = parse_samplers(args.samplers)
    ref_sampler = samplers[0]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset("lerobot/pusht")
    splits = parse_split(args.split, dataset.num_episodes)
    if args.max_episodes:
        splits["val"] = splits["val"][:args.max_episodes]
        splits["test"] = splits["test"][:args.max_episodes]
    print(f"device={device} splits: train {len(splits['train'])}, val {len(splits['val'])}, "
          f"test {len(splits['test'])} episodes", flush=True)

    # Draft: trained on the train split only.
    tr = splits["train"]
    lo = int(dataset.meta.episodes[tr[0]]["dataset_from_index"])
    hi = int(dataset.meta.episodes[tr[-1]]["dataset_to_index"])
    rows = dataset.hf_dataset.select(range(lo, hi))
    drafts = train_drafts(np.asarray(rows["observation.state"], dtype=np.float32),
                          np.asarray(rows["action"], dtype=np.float32), args.draft_members)

    verifier = Verifier(device)
    for s in samplers:
        if s.chunk > verifier.max_chunk:
            raise ValueError(f"{s.name}: chunk {s.chunk} > max {verifier.max_chunk} for this checkpoint")
    verifier.warmup()

    def prepare(split: str, which: list[Sampler]):
        eps, drafts_out, caches, draft_ms = [], {}, {s: {} for s in which}, []
        for i in splits[split]:
            ep = load_episode(dataset, i)
            act, spread, ms = draft_outputs(drafts, ep.states)
            drafts_out[i] = (act, spread)
            draft_ms.append(ms)
            for s in which:
                caches[s][i] = run_verifier(verifier, ep, s, args.samples)
            eps.append(ep)
            print(f"  {split} episode {i}: {ep.length} frames", flush=True)
        return eps, drafts_out, caches, float(np.median(draft_ms))

    print("validation...", flush=True)
    v_eps, v_drafts, v_caches, draft_ms = prepare("val", [ref_sampler])
    print(f"latency benchmark ({args.latency_reps} round-robin reps per sampler)...", flush=True)
    histories = [e.history(b, verifier.n_obs) for e in v_eps for b in range(0, e.length, ref_sampler.chunk)]
    latency = benchmark_latency(verifier, samplers, histories, args.latency_reps)
    for s in samplers:
        print(f"  {s.name:18} median {np.median(latency[s]):7.1f} ms   max {np.max(latency[s]):7.1f} ms", flush=True)
    print("test...", flush=True)
    t_eps, t_drafts, t_caches, draft_ms = prepare("test", samplers)
    record, summary = analyze(args, splits, samplers, v_eps, v_drafts, v_caches[ref_sampler],
                              t_eps, t_drafts, t_caches, latency, draft_ms, args.samples, verifier.max_chunk)
    print_summary(summary)

    import lerobot
    record["provenance"] = {
        "git": git_sha(), "python": platform.python_version(), "torch": torch.__version__,
        "lerobot": getattr(lerobot, "__version__", None), "device": str(device),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
        "threads": torch.get_num_threads(), "machine": platform.machine(),
        "verifier_trained_on_test": True,
        "code": code_provenance(),
    }
    out = pathlib.Path(args.out) if args.out else ROOT / "results" / f"spec_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=float))
    print(f"wrote {out}")

    # Everything needed to recompute any table or curve without running diffusion again (--from-cache).
    cache = {
        "samplers": [s.name for s in samplers],
        "test": {e.index: {"actions": e.actions, "draft": t_drafts[e.index][0], "spread": t_drafts[e.index][1],
                           "chunks": {s.name: t_caches[s][e.index]["chunks"] for s in samplers}} for e in t_eps},
        "val": {e.index: {"actions": e.actions, "draft": v_drafts[e.index][0], "spread": v_drafts[e.index][1],
                          "chunks": {ref_sampler.name: v_caches[ref_sampler][e.index]["chunks"]}} for e in v_eps},
        "latency_ms": {s.name: v for s, v in latency.items()},
        "draft_ms_per_frame": draft_ms,
    }
    out.with_suffix(".cache.pkl").write_bytes(pickle.dumps(cache))
    print(f"wrote {out.with_suffix('.cache.pkl')}")


if __name__ == "__main__":
    main()
