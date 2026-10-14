"""Step 2a: paired bootstrap for per-query metric differences.

Contract (hs/stats.py):
    paired_bootstrap(a, b, n_resamples=10000, seed=0, confidence=0.95) -> dict
      a, b: {query_id: score}. Delta = b - a  (b is the 'new' system).
      returns: n, mean_delta, ci_low, ci_high, wins, losses, ties, significant
      significant == CI excludes zero.  A tie is |delta| <= 1e-12.
"""
import random

import pytest
from hs.stats import paired_bootstrap


def scores(vals):
    return {f"q{i}": v for i, v in enumerate(vals)}


def test_identical_systems_are_all_ties_and_not_significant():
    a = scores([0.1, 0.5, 0.9])
    r = paired_bootstrap(a, dict(a))
    assert r["n"] == 3 and r["ties"] == 3 and r["wins"] == 0 and r["losses"] == 0
    assert r["mean_delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0
    assert r["significant"] is False


def test_constant_improvement_is_significant_with_a_degenerate_ci():
    a = scores([0.2, 0.4, 0.6, 0.8])
    b = scores([0.3, 0.5, 0.7, 0.9])
    r = paired_bootstrap(a, b)
    assert r["mean_delta"] == pytest.approx(0.1)
    assert r["ci_low"] == pytest.approx(0.1) and r["ci_high"] == pytest.approx(0.1)
    assert r["wins"] == 4 and r["significant"] is True


def test_exact_mean_and_win_loss_tie_counts():
    a = {"q1": 0.2, "q2": 0.4, "q3": 0.9}
    b = {"q1": 0.5, "q2": 0.4, "q3": 0.6}
    r = paired_bootstrap(a, b)
    assert r["mean_delta"] == pytest.approx((0.3 + 0.0 - 0.3) / 3)
    assert (r["wins"], r["losses"], r["ties"]) == (1, 1, 1)


def test_float_noise_counts_as_a_tie():
    r = paired_bootstrap({"q": 0.1 + 0.2}, {"q": 0.3})
    assert r["ties"] == 1


def test_direction_is_b_minus_a():
    r = paired_bootstrap({"q1": 1.0, "q2": 1.0}, {"q1": 0.0, "q2": 0.0})
    assert r["mean_delta"] == pytest.approx(-1.0) and r["losses"] == 2


def test_same_seed_gives_identical_results_and_ci_contains_the_mean():
    rng = random.Random(1)
    a = scores([rng.random() for _ in range(60)])
    b = scores([min(1.0, v + rng.uniform(-0.2, 0.3)) for v in a.values()])
    r1 = paired_bootstrap(a, b, n_resamples=2000, seed=7)
    r2 = paired_bootstrap(a, b, n_resamples=2000, seed=7)
    assert r1 == r2
    assert r1["ci_low"] <= r1["mean_delta"] <= r1["ci_high"]


def test_symmetric_wins_and_losses_are_not_significant():
    a = scores([0.5] * 100)
    b = scores([0.9, 0.1] * 50)                       # +0.4 / -0.4 alternating
    r = paired_bootstrap(a, b, n_resamples=3000)
    assert r["ci_low"] < 0 < r["ci_high"] and r["significant"] is False


def test_realistic_size_runs_and_a_clear_gain_is_detected():
    rng = random.Random(3)
    a = scores([rng.random() * 0.8 for _ in range(300)])
    b = {k: v + 0.05 + rng.uniform(-0.02, 0.02) for k, v in a.items()}
    r = paired_bootstrap(a, b)                        # default 10k resamples, n=300
    assert r["significant"] is True and r["ci_low"] > 0.03


def test_query_ids_must_match_exactly():
    with pytest.raises(ValueError, match="same query ids"):
        paired_bootstrap({"q1": 1.0, "q2": 1.0}, {"q1": 1.0, "q3": 1.0})


def test_empty_input_raises():
    with pytest.raises(ValueError):
        paired_bootstrap({}, {})


@pytest.mark.parametrize("c", [0, 1, -0.1, 1.5, float("nan")])
def test_confidence_must_be_strictly_between_0_and_1(c):
    with pytest.raises(ValueError):
        paired_bootstrap({"q": 0.0}, {"q": 1.0}, confidence=c)


@pytest.mark.parametrize("n", [0, -5, 1.5, True])
def test_n_resamples_must_be_a_positive_int(n):
    with pytest.raises(ValueError):
        paired_bootstrap({"q": 0.0}, {"q": 1.0}, n_resamples=n)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_scores_raise(bad):
    with pytest.raises(ValueError):
        paired_bootstrap({"q": 0.0}, {"q": bad})
