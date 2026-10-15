"""Compare saved benchmark JSON files with paired bootstrap CIs.

The CLI reports nominal 95% intervals for interpretation and uses
Bonferroni-adjusted intervals for the family-wise significance label.
"""
import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hs.compare import check_consistency, compare, load_results  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="Compare saved retrieval benchmark results with paired bootstrap.")
    parser.add_argument("--a", required=True, help="First results JSON")
    parser.add_argument("--b", help="Optional second results JSON; omitted means compare modes within --a")
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("MODE_A", "MODE_B"),
                        help="Mode pair to compare; repeat for additional pairs")
    parser.add_argument("--metrics", nargs="+", help="Metrics to compare (default: common per-query metrics)")
    parser.add_argument("--resamples", type=int, default=10000, help="Bootstrap resamples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--family-alpha", type=float, default=0.05,
                        help="Family-wise error budget for Bonferroni-adjusted intervals (default: 0.05)")
    return parser


def bonferroni_confidence(family_alpha, n_tests):
    """Return per-comparison confidence for Bonferroni family-wise control."""
    if isinstance(family_alpha, bool) or not isinstance(family_alpha, (int, float)) or not 0 < family_alpha < 1:
        raise ValueError("family_alpha must be between 0 and 1")
    if isinstance(n_tests, bool) or not isinstance(n_tests, int) or n_tests < 1:
        raise ValueError("n_tests must be a positive integer")
    return 1.0 - family_alpha / n_tests


def _available_metrics(result_a, result_b, mode_a, mode_b):
    try:
        sets = []
        for result, mode in ((result_a, mode_a), (result_b, mode_b)):
            rows = result["per_query"][mode]
            aggregate_metrics = set(result["metrics"][mode])
            common = (
                set.intersection(*(
                    {
                        key for key, value in scores.items()
                        if key in aggregate_metrics
                        and isinstance(value, (int, float))
                        and not isinstance(value, bool)
                    }
                    for scores in rows.values()
                ))
                if rows else set()
            )
            sets.append(common)
        return sorted(sets[0] & sets[1])
    except (KeyError, TypeError):
        return []


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not 0 < args.family_alpha < 1:
            raise ValueError("--family-alpha must be between 0 and 1")
        if args.resamples < 1:
            raise ValueError("--resamples must be a positive integer")

        result_a = load_results(args.a)
        result_b = load_results(args.b) if args.b else result_a
        for label, result in (("--a", result_a), ("--b", result_b)):
            issues = check_consistency(result)
            if issues:
                parser.error(f"inconsistent results in {label}: " + " | ".join(issues))

        tests = []
        for mode_a, mode_b in args.pair:
            metric_list = args.metrics or _available_metrics(result_a, result_b, mode_a, mode_b)
            if not metric_list:
                raise ValueError(f"no common per-query metrics for mode pair {mode_a!r}, {mode_b!r}")
            if len(set(metric_list)) != len(metric_list):
                raise ValueError(f"duplicate metric requested for pair {mode_a!r}, {mode_b!r}")
            for metric in metric_list:
                tests.append((mode_a, mode_b, metric))
        if not tests:
            raise ValueError("no comparisons requested")

        family_size = len(tests)
        adjusted_confidence = bonferroni_confidence(args.family_alpha, family_size)
        rows = []
        for mode_a, mode_b, metric in tests:
            nominal = compare(result_a, result_b, mode_a, mode_b, metric,
                              n_resamples=args.resamples, seed=args.seed, confidence=0.95)
            adjusted = compare(result_a, result_b, mode_a, mode_b, metric,
                               n_resamples=args.resamples, seed=args.seed,
                               confidence=adjusted_confidence)
            # Keep the ordinary 95% interval for interpretation, but use the
            # adjusted interval for the significance flag.
            nominal["adjusted_ci_low"] = adjusted["ci_low"]
            nominal["adjusted_ci_high"] = adjusted["ci_high"]
            nominal["adjusted_confidence_level"] = adjusted_confidence
            nominal["significant_unadjusted"] = nominal["significant"]
            nominal["significant"] = adjusted["significant"]
            nominal["family_size"] = family_size
            nominal["family_alpha"] = args.family_alpha
            rows.append((mode_a, mode_b, metric, nominal))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    adjusted_pct = adjusted_confidence * 100.0
    print(f"Comparison family: {family_size} hypotheses; Bonferroni family alpha={args.family_alpha:.3f}; adjusted per-comparison confidence={adjusted_pct:.3f}%")
    print(f"| A | B | Metric | Mean delta (B - A) | 95% CI (nominal) | Bonferroni CI ({adjusted_pct:.3f}%) | W/L/T | Significant after adjustment? |")
    print("|---|---|---|---:|---|---|---:|:---:|")
    for mode_a, mode_b, metric, stats in rows:
        ci = f"[{stats['ci_low']:+.3f}, {stats['ci_high']:+.3f}]"
        adjusted_ci = f"[{stats['adjusted_ci_low']:+.3f}, {stats['adjusted_ci_high']:+.3f}]"
        wlt = f"{stats['wins']}/{stats['losses']}/{stats['ties']}"
        significant = "yes" if stats["significant"] else "no"
        print(f"| {mode_a} | {mode_b} | {metric} | {stats['mean_delta']:+.3f} | {ci} | {adjusted_ci} | {wlt} | {significant} |")
    return rows


if __name__ == "__main__":
    main()
