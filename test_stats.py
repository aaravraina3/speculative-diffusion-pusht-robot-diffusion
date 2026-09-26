"""Tests for stats.py. Run with: .venv/bin/python -m pytest -q test_stats.py"""
import numpy as np
import pytest

import stats


def test_wilson_matches_reference_values():
    _, lo, hi = stats.wilson(0, 50)
    assert lo == 0.0 and hi == pytest.approx(0.0714, abs=1e-3)
    _, lo, hi = stats.wilson(50, 50)
    assert lo == pytest.approx(0.9286, abs=1e-3) and hi == 1.0
    p, lo, hi = stats.wilson(25, 50)
    assert p == 0.5 and lo == pytest.approx(0.3664, abs=1e-3) and hi == pytest.approx(0.6336, abs=1e-3)


def test_mcnemar_exact():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 5) == pytest.approx(2 / 32)
    assert stats.mcnemar_exact(3, 3) == 1.0
    assert stats.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)


def test_newcombe_contains_estimate_and_is_not_degenerate():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.random(50) < rng.random()
        y = rng.random(50) < rng.random()
        d, lo, hi = stats.newcombe_paired(x, y)
        assert d == pytest.approx(x.mean() - y.mean())
        assert -1 <= lo <= d <= hi <= 1
    same = np.ones(50, bool)
    d, lo, hi = stats.newcombe_paired(same, same)
    assert d == 0 and lo < 0 < hi, "identical outcomes still leave uncertainty about the rates"


def test_bca_constant_data_has_zero_width():
    est, lo, hi = stats.bca_cluster(np.full(6, 3.0), np.arange(1.0, 7.0), B=500)
    assert est == lo == hi == 3.0


def test_bca_close_to_normal_interval_for_symmetric_data():
    rng = np.random.default_rng(1)
    v = rng.normal(10, 2, size=200)
    w = np.ones(200)
    est, lo, hi = stats.bca_cluster(v, w, B=4000)
    se = v.std(ddof=1) / np.sqrt(len(v))
    assert lo == pytest.approx(est - 1.96 * se, abs=0.1)
    assert hi == pytest.approx(est + 1.96 * se, abs=0.1)


def test_bca_widens_the_upper_tail_for_right_skewed_data():
    rng = np.random.default_rng(2)
    v = rng.lognormal(3, 1, size=36)
    w = rng.integers(50, 250, size=36).astype(float)
    est, lo, hi = stats.bca_cluster(v, w, B=10_000)
    idx = np.random.default_rng(0).integers(0, 36, size=(10_000, 36))
    boots = np.sum(v[idx] * w[idx], axis=1) / np.sum(w[idx], axis=1)
    assert hi > np.quantile(boots, 0.975), "BCa should push the upper bound out for right skew"
    assert lo < est < hi


def test_one_sided_upper_bound_sits_inside_two_sided():
    rng = np.random.default_rng(3)
    v, w = rng.normal(0, 1, 40), np.ones(40)
    _, lo, hi = stats.bca_cluster(v, w, B=4000)
    _, up = stats.bca_cluster(v, w, B=4000, levels=(0.95,))
    assert lo < up < hi
