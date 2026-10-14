"""Statistical utilities for paired per-query retrieval comparisons."""

import math
import random
from collections.abc import Mapping
from numbers import Real


def _quantile(sorted_values: list[float], probability: float) -> float:
    """Calculate a linearly interpolated quantile."""
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return sorted_values[lower]

    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap(
    a: Mapping,
    b: Mapping,
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap paired query-level differences; delta is b minus a."""
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise ValueError("a and b must be mappings of query IDs to scores")

    if set(a.keys()) != set(b.keys()):
        raise ValueError("a and b must have the same query ids")

    if not a:
        raise ValueError("score mappings must not be empty")

    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not math.isfinite(float(confidence))
        or not 0 < confidence < 1
    ):
        raise ValueError("confidence must be strictly between 0 and 1")

    deltas = []

    for query_id in a:
        values = []
        for score in (a[query_id], b[query_id]):
            if isinstance(score, bool) or not isinstance(score, Real):
                raise ValueError("scores must be finite real numbers")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("scores must be finite real numbers")
            values.append(score)

        deltas.append(values[1] - values[0])

    n = len(deltas)
    mean_delta = sum(deltas) / n

    wins = sum(delta > 1e-12 for delta in deltas)
    losses = sum(delta < -1e-12 for delta in deltas)
    ties = n - wins - losses

    rng = random.Random(seed)
    bootstrap_means = []

    for _ in range(n_resamples):
        sample_sum = sum(rng.choices(deltas, k=n))
        bootstrap_means.append(sample_sum / n)

    bootstrap_means.sort()
    alpha = (1.0 - float(confidence)) / 2.0

    ci_low = _quantile(bootstrap_means, alpha)
    ci_high = _quantile(bootstrap_means, 1.0 - alpha)

    return {
        "n": n,
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "significant": ci_low > 0.0 or ci_high < 0.0,
    }
