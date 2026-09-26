"""Interval estimates used by spec_eval.py and closed_loop.py.

Open loop: frame-weighted means resampled by whole episode, with BCa intervals,
since per-episode error is right-skewed and plain percentile intervals under-cover.
Closed loop: success is binary with 50 rollouts, so rates get Wilson intervals and
paired comparisons get an exact McNemar test with Newcombe's hybrid-score interval.
"""
import math
from statistics import NormalDist

import numpy as np

_N = NormalDist()


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def bca_cluster(values: np.ndarray, weights: np.ndarray, B: int = 10_000, seed: int = 0,
                levels: tuple = (0.025, 0.975)) -> tuple:
    """BCa bootstrap for a frame-weighted mean, resampling clusters (episodes).

    Returns the estimate followed by one bound per entry in `levels`. Use
    levels=(0.95,) for a one-sided 95% upper bound.
    """
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    est = weighted_mean(values, weights)
    n = len(values)
    if n < 2:
        return (est, *[float("nan")] * len(levels))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = np.sum(values[idx] * weights[idx], axis=1) / np.sum(weights[idx], axis=1)
    if np.allclose(boots, est):
        return (est, *[est] * len(levels))
    below = np.mean(boots < est) + 0.5 * np.mean(boots == est)
    z0 = _N.inv_cdf(min(max(below, 1e-6), 1 - 1e-6))
    keep = np.ones(n, bool)
    jack = np.empty(n)
    for i in range(n):  # leave one episode out
        keep[i] = False
        jack[i] = weighted_mean(values[keep], weights[keep])
        keep[i] = True
    d = jack.mean() - jack
    denom = 6.0 * np.sum(d ** 2) ** 1.5
    a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0
    out = []
    for q in levels:
        zq = _N.inv_cdf(q)
        adj = _N.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(float(np.quantile(boots, adj)))
    return (est, *out)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Wilson score interval for k successes out of n."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo = 0.0 if k == 0 else max(0.0, centre - half)  # the bounds are exactly 0 and 1 at the edges
    hi = 1.0 if k == n else min(1.0, centre + half)
    return p, lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    m = b + c
    if m == 0:
        return 1.0
    tail = sum(math.comb(m, i) for i in range(min(b, c) + 1)) / 2 ** m
    return min(1.0, 2 * tail)


def newcombe_paired(x: np.ndarray, y: np.ndarray, z: float = 1.959964) -> tuple[float, float, float]:
    """Difference in paired success rates, mean(x) - mean(y), with Newcombe's
    hybrid-score interval (method 10, Newcombe 1998)."""
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    n = len(x)
    a = int(np.sum(x & y))
    b = int(np.sum(x & ~y))
    c = int(np.sum(~x & y))
    d = int(np.sum(~x & ~y))
    p1, l1, u1 = wilson(a + b, n, z)
    p2, l2, u2 = wilson(a + c, n, z)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        phi = 0.0
    else:
        num = a * d - b * c
        phi = (max(num - n / 2, 0) if num > 0 else num) / denom  # continuity-corrected, as Newcombe
    delta = p1 - p2
    lo = delta - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    hi = delta + math.sqrt(max((p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2, 0))
    return delta, max(-1.0, lo), min(1.0, hi)
