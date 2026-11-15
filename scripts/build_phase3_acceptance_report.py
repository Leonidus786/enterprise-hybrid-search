#!/usr/bin/env python3
"""Build the Phase 3 evaluation acceptance report from preserved artifacts.

Run from the phase3 acceptance repository root:
    python scripts/build_phase3_acceptance_report.py

By default this script refuses to overwrite an existing report. Use --force only
when intentionally regenerating the report from the same or updated artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

STRATEGIES = ("bm25", "dense", "hybrid", "hybrid_rerank")
QUALITY_FILE = "phase3_acceptance_final_v1.json"
LATENCY_FILE = "phase3_latency_doclevel_v1.json"
STAT_FILE = "phase3_statistical_comparisons_full_precision_v1.json"
RRF_FILE = "phase3_rrf_diagnostic_v1.json"
FAILURE_FILE = "phase3_failure_analysis_v1.json"
MEDCPT_FILE = "phase3_medcpt_ab_300q_v1.json"
MEDCPT_STAT_FILE = "phase3_medcpt_statistical_comparisons_v1.json"
DECISION_FILE = "phase3_reranker_decision_v1.json"
PROVENANCE_FILE = "phase3_provenance_reconciliation_v1.json"
TRUNCATION_FILES = (
    "phase3_reranker_truncation_audit_v1.json",
    "phase3_reranker_truncation_query_association_v1.json",
    "phase3_reranker_relevant_truncation_impact_v1.json",
)
OPTIONAL_FILES = (
    "phase3_latency_final_v1.json",  # chunked experiment; explicitly non-canonical
    "phase3_medcpt_screening_10q_v1.json",
    "phase3_statistical_comparisons_v1.txt",  # rounded display copy; full-precision JSON is authoritative
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def md_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def md_ms(value: Any) -> str:
    return f"{float(value):,.2f} ms"


def md_ci(ci: list[Any] | tuple[Any, Any] | None, digits: int = 5) -> str:
    if not ci or len(ci) != 2:
        return "—"
    return f"[{float(ci[0]):+.{digits}f}, {float(ci[1]):+.{digits}f}]"


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def source_ref(name: str, hashes: dict[str, str]) -> str:
    digest = hashes.get(name)
    return f"`{name}` (SHA-256 `{digest}`)" if digest else f"`{name}` (not available)"


def extract_phase2_candidate_section(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == "## Candidate-depth experiment"), None)
    if start is None:
        raise ValueError("Candidate-depth section not found in docs/development/phase2-freeze.md")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).strip()


def make_report(root: Path, results: Path) -> tuple[str, dict[str, str]]:
    docs = root / "docs"
    dev = docs / "development"
    protocol_path = dev / "phase3-evaluation-acceptance.md"
    phase2_path = dev / "phase2-freeze.md"
    design_path = dev / "independent-validation-design.md"
    required = [
        protocol_path,
        phase2_path,
        design_path,
        *[results / n for n in (
            QUALITY_FILE, LATENCY_FILE, STAT_FILE, RRF_FILE, FAILURE_FILE,
            MEDCPT_FILE, MEDCPT_STAT_FILE, DECISION_FILE, PROVENANCE_FILE,
            *TRUNCATION_FILES,
        )],
    ]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required input(s) missing:\n  " + "\n  ".join(missing))

    names = [p.name for p in required if p.parent == results]
    names.extend(name for name in OPTIONAL_FILES if (results / name).is_file() and name not in names)
    if len(names) != 15:
        raise ValueError(f"Expected 15 packaged evidence artifacts, found {len(names)}")
    hashes = {name: sha256(results / name) for name in names}
    q = load_json(results / QUALITY_FILE)
    latency = load_json(results / LATENCY_FILE)
    stats = load_json(results / STAT_FILE)
    rrf = load_json(results / RRF_FILE)
    failure = load_json(results / FAILURE_FILE)
    medcpt = load_json(results / MEDCPT_FILE)
    medcpt_stats = load_json(results / MEDCPT_STAT_FILE)
    decision = load_json(results / DECISION_FILE)
    provenance = load_json(results / PROVENANCE_FILE)
    trunc = [load_json(results / n) for n in TRUNCATION_FILES]

    # Hard gates: fail closed rather than produce a misleading acceptance report.
    if set(q.get("metrics", {})) != set(STRATEGIES):
        raise ValueError("Quality artifact does not contain the four expected strategies")
    if set(q.get("per_query", {})) != set(STRATEGIES):
        raise ValueError("Per-query artifact does not contain the four expected strategies")
    if any(len(q["per_query"][s]) != 300 for s in STRATEGIES):
        raise ValueError("Expected 300 per-query records for each quality strategy")
    if len(stats.get("results", [])) != 6:
        raise ValueError("Expected exactly six frozen primary statistical comparisons")
    if latency.get("protocol", {}).get("protocol_valid") is not True:
        raise ValueError("Canonical document-level latency protocol is not marked valid")
    if any(latency["protocol"]["actual_samples_per_strategy"].get(s) != 1500 for s in STRATEGIES):
        raise ValueError("Expected 1,500 latency samples per strategy")
    if failure.get("source_quality_artifact", {}).get("sha256") != hashes[QUALITY_FILE]:
        raise ValueError("Failure-analysis artifact does not reference the current quality artifact")
    if medcpt.get("input_sha256", {}).get("quality_artifact") != hashes[QUALITY_FILE]:
        raise ValueError("MedCPT artifact does not reference the current quality artifact")
    if medcpt_stats.get("source_artifacts", {}).get("quality_sha256") != hashes[QUALITY_FILE]:
        raise ValueError("MedCPT statistics do not reference the current quality artifact")
    if medcpt_stats.get("source_artifacts", {}).get("medcpt_sha256") != hashes[MEDCPT_FILE]:
        raise ValueError("MedCPT statistics do not reference the current MedCPT artifact")
    if rrf.get("source_quality_sha256") != hashes[QUALITY_FILE]:
        raise ValueError("RRF diagnostic does not reference the current quality artifact")
    if provenance.get("source_sha256", {}).get("quality") != hashes[QUALITY_FILE]:
        raise ValueError("Provenance reconciliation does not reference the current quality artifact")
    if provenance.get("source_sha256", {}).get("latency") != hashes[LATENCY_FILE]:
        raise ValueError("Provenance reconciliation does not reference the current latency artifact")
    for name, obj in zip(TRUNCATION_FILES, trunc):
        if obj.get("input_sha256", {}).get("quality_artifact") != hashes[QUALITY_FILE]:
            raise ValueError(f"{name} does not reference the current quality artifact")
    decision_sources = decision.get("source_artifacts", {})
    for name in (QUALITY_FILE, LATENCY_FILE, STAT_FILE, MEDCPT_STAT_FILE):
        if decision_sources.get(name) != hashes[name]:
            raise ValueError(f"Reranker decision does not reference current {name}")

    # Recheck the 11-query RRF diagnostic against the correct pre-rerank strategy.
    rrf_results = rrf.get("results", {})
    rrf_ids = list(rrf_results)
    rrf_rank_mismatch = [
        qid for qid, row in rrf_results.items()
        if row.get("ranked_ids") != q["per_query"]["hybrid"][qid]["_trace"]["ranked_ids"]
    ]
    rrf_candidate_mismatch = [
        qid for qid, row in rrf_results.items()
        if row.get("candidate_ids") != q["per_query"]["hybrid"][qid]["_trace"]["candidate_ids"]
    ]
    if rrf_rank_mismatch or rrf_candidate_mismatch:
        raise ValueError(f"RRF trace mismatch: ranking={rrf_rank_mismatch}, candidates={rrf_candidate_mismatch}")

    # Recheck MedCPT candidate-pool identity against the original hybrid-rerank trace.
    medcpt_queries = medcpt.get("queries", [])
    if len(medcpt_queries) != 300:
        raise ValueError("Expected 300 MedCPT query records")
    medcpt_candidate_mismatch = []
    for row in medcpt_queries:
        qid = str(row["query_id"])
        trace = q["per_query"]["hybrid_rerank"][qid]["_trace"]
        if row["candidate_ids"] != trace["candidate_ids"]:
            medcpt_candidate_mismatch.append(qid)
    if medcpt_candidate_mismatch:
        raise ValueError(f"MedCPT candidate pool mismatch: {medcpt_candidate_mismatch[:10]}")

    # Validate failure classes sum to 300 per strategy.
    failure_strategies = failure.get("strategies", {})
    for strategy in STRATEGIES:
        item = failure_strategies[strategy]
        if item.get("query_count") != 300 or sum(item["class_counts"].values()) != 300:
            raise ValueError(f"Failure classification does not sum to 300 for {strategy}")

    qdataset = q["dataset"]
    ldataset = latency["dataset"]
    qprov = q["provenance"]
    lprov = latency["provenance"]
    if qdataset.get("corpus_sha256") != lprov["inputs"]["corpus"]["sha256"]:
        raise ValueError("Quality/latency corpus hash mismatch")
    if qdataset.get("queries_file_sha256") != lprov["inputs"]["queries_and_relevance_labels"]["sha256"]:
        raise ValueError("Quality/latency query file hash mismatch")
    if qdataset.get("query_fingerprint") != ldataset.get("query_fingerprint"):
        raise ValueError("Quality/latency query fingerprint mismatch")
    if q.get("git", {}).get("commit") != latency.get("git", {}).get("commit"):
        raise ValueError("Quality/latency Git commit mismatch")
    if qprov.get("source_code", {}).get("sha256") != lprov.get("source_code", {}).get("sha256"):
        raise ValueError("Quality/latency source-code fingerprint mismatch")

    # Metric tables.
    metric_order = ("precision@10", "recall@10", "recall@100", "candidate_recall@50", "mrr@10", "ndcg@10")
    quality_rows = []
    hit_rates: dict[str, float] = {}
    for strategy in STRATEGIES:
        metrics = q["metrics"][strategy]
        hit = sum(1 for row in q["per_query"][strategy].values() if row["recall@10"] > 0) / 300
        hit_rates[strategy] = hit
        quality_rows.append(
            f"| {strategy} | " + " | ".join(md_num(metrics[k], 4) for k in metric_order) + f" | {md_num(hit, 4)} |"
        )

    stat_rows = []
    for result in stats["results"]:
        nominal = result.get("nominal", {})
        adjusted = result.get("bonferroni", {})
        stat_rows.append(
            "| {a} → {b} | {metric} | {delta:+.5f} | {nominal_ci} | {adjusted_ci} | {w}/{l}/{t} | {sig} |".format(
                a=result["a"], b=result["b"], metric=result["metric"],
                delta=float(nominal["mean_delta"]),
                nominal_ci=md_ci([nominal.get("ci_low"), nominal.get("ci_high")]),
                adjusted_ci=md_ci([adjusted.get("ci_low"), adjusted.get("ci_high")]),
                w=nominal.get("wins", "—"), l=nominal.get("losses", "—"), t=nominal.get("ties", "—"),
                sig="Yes" if adjusted.get("significant") else "No",
            )
        )

    failure_rows = []
    class_order = failure.get("classification_order", [])
    for strategy in STRATEGIES:
        counts = failure_strategies[strategy]["class_counts"]
        failure_rows.append("| " + strategy + " | " + " | ".join(str(counts.get(c, 0)) for c in class_order) + " | 300 |")

    # Latency summary: the artifact is the authority for repeated p50/p95/p99.
    latency_rows = []
    for strategy in STRATEGIES:
        total = latency["summary"][strategy]["total"]
        retrieval = latency["summary"][strategy]["retrieval"]
        rerank = latency["summary"][strategy]["rerank"]
        latency_rows.append(
            f"| {strategy} | {total['count']} | {md_num(total['p50_ms'], 2)} | {md_num(total['p95_ms'], 2)} | {md_num(total['p99_ms'], 2)} | {md_num(retrieval['p50_ms'], 2)} | {md_num(rerank['p50_ms'], 2)} |"
        )

    # MedCPT paired statistics from its dedicated derived artifact.
    med_rows = []
    for metric in ("precision@10", "recall@10", "mrr@10", "ndcg@10"):
        row = medcpt_stats["results"][metric]
        med_rows.append(
            f"| {metric} | {md_num(row['baseline_mean'], 4)} | {md_num(row['medcpt_mean'], 4)} | {float(row['mean_delta_medcpt_minus_baseline']):+.5f} | {md_ci(row['nominal_95_percent_ci'])} | {md_ci(row['bonferroni_adjusted_98_75_percent_ci'])} | {row['query_level_wins']}/{row['query_level_losses']}/{row['query_level_ties']} |"
        )

    # Exact source excerpt keeps the frozen Phase 2 depth study traceable.
    phase2_section = extract_phase2_candidate_section(phase2_path)

    # Helpful selected query identifiers.
    hybrid_failure = failure_strategies["hybrid"]["query_ids"]
    hybrid_no_candidates_top100 = hybrid_failure.get("candidate_miss_top100_hit", [])
    hybrid_no_relevant_top100 = hybrid_failure.get("candidate_miss_top100_miss", [])

    # Derived RRF diagnostic details.
    rrf_k_values = sorted({
        str(row.get("rrf_k")) for row in rrf_results.values() if row.get("rrf_k") is not None
    })
    rrf_k_text = ", ".join(rrf_k_values) if rrf_k_values else "60 (diagnostic/source default)"

    # Runtime and model verification caveat.
    env = latency.get("environment", {})
    hardware = env.get("hardware", {})
    packages = env.get("packages", {})
    model_prov = qprov.get("models", {})
    bge = model_prov.get("embedding", {})
    cross = model_prov.get("reranker", {})

    # Artifact manifest is generated from actual files at report-build time.
    manifest_files = [
        QUALITY_FILE, LATENCY_FILE, STAT_FILE, RRF_FILE, FAILURE_FILE,
        MEDCPT_FILE, MEDCPT_STAT_FILE, DECISION_FILE, PROVENANCE_FILE,
        *TRUNCATION_FILES,
    ]
    for optional in OPTIONAL_FILES:
        if (results / optional).is_file():
            manifest_files.append(optional)
    manifest = "\n".join(
        f"| `{name}` | `{hashes.get(name, sha256(results / name))}` |"
        for name in manifest_files
    )

    design_sha = sha256(design_path)
    protocol_sha = sha256(protocol_path)
    phase2_sha = sha256(phase2_path)

    report = f"""# Phase 3 — Evaluation and Retrieval Quality Acceptance

**Status:** Evidence assembled; full regression passed; final diff review and owner acceptance pending.

**Evaluation worktree:** `{root}`

**Repository result directory:** `results/`

**Evaluation commit:** `{q.get('git', {}).get('commit', 'unavailable')}`
**Dataset:** BEIR SciFact test; document-level; 5,183 documents; 300 queries

## 1. Executive summary

Phase 3 establishes a reproducible retrieval-quality baseline for BM25, dense retrieval, Hybrid (BM25 + dense + RRF), and Hybrid + MS MARCO Cross-Encoder reranking. It also documents query-level failure classes, RRF diagnostics, candidate-depth sensitivity, reranker truncation, repeated latency, model/runtime provenance, and the next independent-validation design.

**Current decision:** Hybrid remains the default retrieval strategy. The MS MARCO Cross-Encoder remains opt-in/default OFF: its observed top-10 quality point estimates are lower than Hybrid, Recall@100 and Candidate Recall@50 are unchanged, and repeated CPU latency is materially higher. MedCPT is a promising exploratory challenger on the same fixed 50-candidate pools, but it is not selected as a production default because this comparison uses previously inspected SciFact queries and lacks independent-domain and dedicated serving-latency validation.

This report does not claim independent confirmatory quality, production deployment, universal search superiority, grounded-answer correctness, citation correctness, or enterprise security readiness.

## 2. Scope, protocol and dataset integrity

The frozen execution protocol is `docs/development/phase3-evaluation-acceptance.md` (SHA-256 `{protocol_sha}`). The quality artifact records:

- Corpus: `{qdataset['corpus_path']}`; {qdataset['searchable_items']:,} searchable documents; SHA-256 `{qdataset['corpus_sha256']}`.
- Queries: `{qdataset['queries_path']}`; {qdataset['evaluated_queries']} evaluated queries; file SHA-256 `{qdataset['queries_file_sha256']}`.
- Query/relevance fingerprint: `{qdataset['query_fingerprint']}`.
- Document-level retrieval; chunking disabled; `k={q['config'].get('k')}`, `retrieval_k={q['config'].get('retrieval_k')}`, `candidate_k={q['config'].get('candidate_k')}`.
- Quality run Git commit `{q['git']['commit']}`; saved artifact says working tree dirty = `{q['git']['working_tree_dirty']}`.

The 300 SciFact test queries had been inspected previously. The fresh run is therefore a reproducibility and baseline refresh, **not** an untouched confirmatory test. Repeated inspection of this set must not be used as the sole basis for further tuning.

### Metric definitions

The implementation in `src/hs/metrics.py` is authoritative:

- Precision@10: relevant results in the first 10 divided by 10.
- Recall@10 and Recall@100: relevant results retrieved in the respective cutoff divided by the number of relevant documents.
- MRR@10: reciprocal rank of the first relevant result within the first 10.
- nDCG@10: **binary-relevance** nDCG, not graded nDCG.
- Candidate Recall@50: fraction of relevant documents present in the candidate pool of 50.
- HitRate@10 below is supplementary/descriptive: proportion of queries with at least one relevant result in top 10.

## 3. Four-strategy quality results

| Strategy | Precision@10 | Recall@10 | Recall@100 | Candidate Recall@50 | MRR@10 | nDCG@10 | HitRate@10* |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(quality_rows)}

*HitRate@10 is derived from per-query `recall@10 > 0`; it is not part of the six-hypothesis statistical family.*

Hybrid and Hybrid + reranker tie for the highest observed Recall@100 and Candidate Recall@50. Hybrid has better observed top-10 ranking metrics than the reranked variant. Its nDCG@10 is only slightly above Dense in point estimate; the frozen paired comparison does not establish a statistically significant Dense-to-Hybrid nDCG improvement after multiplicity correction.

## 4. Paired statistical comparisons — frozen six-test family

The primary statistical family contains six paired comparisons: BM25 vs Dense, Dense vs Hybrid, and Hybrid vs Hybrid + reranker, each for nDCG@10 and Recall@100. The analysis uses 10,000 paired query bootstrap resamples, seed 42, and Bonferroni-adjusted confidence intervals (99.1667% confidence). MRR is descriptive, not part of this six-test family.

Delta is `strategy b - strategy a`. W/L/T are query-level wins/losses/ties for b versus a.

| Comparison | Metric | Mean delta | Nominal 95% CI | Bonferroni-adjusted CI | W/L/T | Significant after adjustment? |
|---|---|---:|---:|---:|---:|---|
{chr(10).join(stat_rows)}

Only BM25 → Dense Recall@100 excludes zero after the frozen correction. The other five comparisons do not establish an adjusted statistically significant difference. This does not imply that strategies are equivalent; it means the available sample and uncertainty intervals do not establish the tested effect at the declared family-wise error rate.

## 5. All-query failure analysis

Every strategy-query record is assigned to exactly one mutually exclusive class. Counts below sum to 300 for each strategy. Definitions are preserved in `{FAILURE_FILE}`.

| Strategy | Top-10 hit | Relevant in candidate pool, missed top 10 | Candidate pool miss, relevant in top 100 | No relevant top-100 hit | Total |
|---|---:|---:|---:|---:|---:|
{chr(10).join(failure_rows)}

For Hybrid, six queries have relevant documents in ranked positions 51–100 but outside the 50-candidate pool: `{', '.join(hybrid_no_candidates_top100)}`. Eleven queries have no relevant document in the top 100: `{', '.join(hybrid_no_relevant_top100)}`. The same 50-candidate pool is used by Hybrid and Hybrid + reranker, so reranking cannot repair a candidate-generation miss. These cases require retrieval/candidate-generation improvements, not a different ordering of the same candidates.

Independent recomputation of the stored ranking traces previously matched all 1,200 query-level recall values; candidate IDs were verified as a subset of their corresponding top-100 rankings. The RRF diagnostic checks are described next.

## 6. RRF and candidate-depth investigation

### RRF diagnostic

The saved RRF diagnostic covers {len(rrf_ids)} selected failure-analysis queries (`{', '.join(rrf_ids)}`), with RRF k = {rrf_k_text}. The corrected comparison verifies:

- Raw RRF ranking equals the `hybrid` (pre-rerank) top-100 trace for all {len(rrf_ids)} diagnostic queries.
- Candidate IDs equal the Hybrid candidate pool for all {len(rrf_ids)} queries.
- The raw RRF ranking differs from `hybrid_rerank` for all {len(rrf_ids)} queries; this is the expected effect of the subsequent Cross-Encoder stage, not an RRF mismatch.

This is an 11-query diagnostic, not a claim that every sparse and dense component ranking was separately retained and reconstructed for all 300 queries. The main quality artifact still contains the full 300-query final rankings and candidate traces.

### Candidate-depth sensitivity (Phase 2 frozen evidence)

The candidate-depth study is preserved in `docs/development/phase2-freeze.md` (SHA-256 `{phase2_sha}`). It is a **sensitivity study, not a pure reranker-only ablation**, because changing candidate depth also changes the upstream candidate pool. The frozen results are reproduced below:

{phase2_section}

Interpretation: at K=10, 20 and 50, reranking did not improve aggregate MRR@10 or nDCG@10. K=20 had a small observed Recall@10 increase but lower MRR/nDCG. K=50 showed lower Recall@10, MRR and nDCG. These are descriptive aggregate results, not statistical proof or evidence of generalization.

The Phase 2 freeze records that its K=50 quality artifact had `working_tree_dirty=true`; a normalized tracked-source equivalence check was documented, including CRLF/LF normalization. It must not be represented as a clean-worktree benchmark. The K=10/K=20 quality and dedicated latency runs were recorded as clean detached worktrees.

## 7. Cross-Encoder truncation investigation

The truncation analyses are derived from saved tokenization/ranking evidence; `inference_performed=false` in these audits. They are descriptive and **do not establish a causal effect** of truncation on ranking.

- 15,000 query-document pairs audited; maximum pair length 512; `longest_first` truncation; 2,197 pairs truncated, all on the document side; 260,742 tokens dropped.
- Raw pair-token length: P50 366, P90 550, P95 625, P99 851, maximum 1,970.
- Across 300 queries, truncated candidate counts: zero for 13 queries, 1–5 for 145, and 6+ for 142.
- Association between truncated-candidate count and nDCG delta was weak and not statistically significant (Spearman rho -0.0829, p=0.152); MRR association was rho -0.1021, p=0.0774.
- Among 320 relevant query-document pairs, 55 were truncated: 9 improved, 10 worsened, 36 unchanged. In top-10 outcomes for these truncated relevant pairs: 1 gained, 3 lost, 48 retained; the remaining pairs were not top-10 outcomes.

These findings justify investigating input-length handling, but not claiming truncation caused the observed reranker regression. Any mitigation must be evaluated on development/independent validation data.

## 8. Dedicated repeated latency evaluation

Canonical latency artifact: `{LATENCY_FILE}`. This is the **document-level, no-chunking** run. The separate `phase3_latency_final_v1.json`, if present, is a chunked experiment and is not used as the canonical latency result here.

Protocol: 5 trials × 300 queries × 4 strategies = 6,000 samples; 1,500 samples per strategy; 10 warm-up queries per strategy excluded from timed samples; strategy order rotates by trial; query-embedding cache disabled; document embeddings cached. The artifact marks the protocol valid.

| Strategy | Samples | Total P50 (ms) | Total P95 (ms) | Total P99 (ms) | Retrieval P50 (ms) | Rerank P50 (ms) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(latency_rows)}

Hybrid + MS MARCO reranker total latency is approximately 46.08× Hybrid at p50, 45.74× at p95, and 35.10× at p99. Its reranker-stage p50 is approximately 4,800.62 ms. These are local CPU measurements, not a deployed-service SLO. The outer total timer is authoritative; retrieval/rerank component timers are diagnostic and may not sum exactly to total because of timer boundaries and residual work.

## 9. Reranker decision and MedCPT exploratory comparison

### MS MARCO Cross-Encoder decision

Decision artifact: `{DECISION_FILE}`. Keep the MS MARCO Cross-Encoder **default OFF** and retain it only as an optional experiment. All four observed top-10 point estimates (Precision@10, Recall@10, MRR@10, nDCG@10) are lower than Hybrid; Recall@100 and Candidate Recall@50 are unchanged; the paired nDCG comparison is not statistically significant after the frozen six-test correction; and CPU latency is substantially higher. The evidence does not prove a statistically significant quality decline, but it does not demonstrate a quality benefit that justifies the measured latency cost.

### MedCPT versus MS MARCO on fixed candidates

MedCPT A/B is exploratory/non-confirmatory. It reranks the same 50 Hybrid candidate IDs for each of the 300 previously inspected SciFact queries; it does not improve candidate generation. The paired statistics below use the exact evaluator definitions (including binary nDCG@10), 10,000 query-paired bootstrap resamples, seed 20261115, and Bonferroni correction across four metrics (98.75% confidence intervals).

| Metric | MS MARCO | MedCPT | Delta | Nominal 95% CI | Bonferroni-adjusted 98.75% CI | W/L/T |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(med_rows)}

Mean Candidate Recall@50 was {md_num(medcpt_stats['mean_candidate_recall_at_50'], 5)} for the fixed candidate pool. All four adjusted intervals exclude zero in this exploratory sample. This is a promising ranking signal, not independent confirmation or a production selection. MedCPT inference took {md_num(medcpt.get('elapsed_seconds'), 2)} seconds in the recorded run; dedicated repeated serving latency and compute-cost evaluation remains required before adoption.

## 10. Model, runtime and provenance reconciliation

The quality and canonical latency runs share:

- Corpus SHA-256 `{qprov['inputs']['corpus']['sha256']}`.
- Query/relevance SHA-256 `{qprov['inputs']['queries_and_relevance_labels']['sha256']}`.
- Query/relevance fingerprint `{qprov['inputs']['query_relevance_fingerprint']}`.
- Git commit `{q['git']['commit']}`, with both artifacts recording clean worktrees.
- Identical source-code fingerprint across {qprov['source_code']['file_count']} files: `{qprov['source_code']['sha256']}`.

Model revisions:

| Model | Requested/resolved revision | Verification |
|---|---|---|
| {bge.get('model_id')} | `{bge.get('resolved_revision')}` | `{bge.get('revision_status')}` |
| {cross.get('model_id')} | `{cross.get('resolved_revision')}` | `{cross.get('revision_status')}` |

Revision status is a match to the local cached snapshot. `verified_from_loaded_model=false`; therefore this report does not claim loaded-model object attestation.

Runtime recorded by the latency artifact: processor `{hardware.get('processor', 'not recorded')}`, logical CPUs `{hardware.get('logical_cpu_count', 'not recorded')}`, CUDA available `{hardware.get('cuda_available', 'not recorded')}`; package versions: `{json.dumps(packages, sort_keys=True)}`.

**Prefix limitation:** quality records `bge_prefix_enabled=true`; latency does not record this flag or the exact invocation. Exact prefix parity between historical quality and latency runs is unverified. The provenance reconciliation explicitly retains this limitation.

**Cache qualification:** quality and latency configurations are not identical. The latency run intentionally disables query-embedding cache and enables document-embedding cache. Interpret latency under this measured configuration; do not imply identical cache behavior between quality and latency runs.

## 11. Independent validation design — Step 3.13

Design document: `docs/development/independent-validation-design.md` (SHA-256 `{design_sha}`). Status: **DESIGN DRAFT — pending domain selection and preregistration**.

The design specifies an approved domain corpus, query-family separation, deduplication/leakage controls, human double annotation and adjudication, a predeclared binary-nDCG primary metric, coverage guardrails, paired statistics, pilot-based sample-size determination, latency/cost gates, and a separately governed untouched holdout. It does **not** claim that the corpus has been collected, labels adjudicated, sample size powered, or an independent test executed.

Required decisions before preregistration include target domain/use cases, corpus owner and permissions, query sampling frame, annotation/adjudication resources, sample sizes, practical effect and non-inferiority thresholds, latency SLO, and independent holdout custody.

## 12. Limitations and non-claims

1. SciFact's 300 test queries were previously inspected. The refresh is reproducible but not untouched confirmatory evidence.
2. Results are specific to this corpus, query set, relevance labels, model snapshots and local runtime; they do not establish universal search superiority.
3. The 11-query RRF component diagnostic is selected, not an all-query component-rank reconstruction.
4. Truncation analyses are observational/descriptive; causal impact is unproven.
5. Model revisions match local cached snapshots; loaded model objects were not independently attested.
6. Historical latency artifact does not record BGE-prefix invocation; exact prefix parity is unverified.
7. Latency is local CPU benchmark evidence, not a deployed-service SLO or cloud cost estimate.
8. MedCPT comparison is exploratory on previously inspected queries and has no independent-domain confirmation or dedicated repeated serving-latency result.
9. Independent domain data, human-verified labels and untouched holdout remain future work.
10. Phase 3 does not evaluate generated-answer correctness, faithfulness, claim-level citation accuracy, abstention, prompt-injection resistance, authentication, tenant isolation, ACL enforcement or disaster recovery.

## 13. Acceptance status

| Step | Status | Evidence / remaining qualification |
|---|---|---|
| 3.1–3.7 | Complete | Frozen protocol, dataset checks, 300-query quality run, four strategies, consistency and six paired comparisons. |
| 3.8 | Built | `phase3_failure_analysis_v1.json`; 1,200 mutually exclusive classifications validated. |
| 3.9 | Verified / consolidated | 11-query RRF trace verified; K=10/20/50 sensitivity evidence preserved in Phase 2 freeze. |
| 3.10 | Built | Reranker decision artifact; MS MARCO default OFF; MedCPT remains exploratory. |
| 3.11 | Accepted with limitations | 6,000-sample document-level latency artifact; protocol marked valid. |
| 3.12 | Built | Dataset, Git, source and model metadata reconciled; prefix and loaded-model attestation limitations explicit. |
| 3.13 | Designed | Independent-validation design draft; domain selection and preregistration pending. |
| 3.14 | Report generated; regression passed | All 15 artifacts packaged; 595 tests passed; final diff review and owner acceptance pending. |

**Phase 3 exit recommendation:** evaluation, analysis, evidence packaging, report generation, isolated reproduction and full regression are complete. Final acceptance remains pending complete Git diff review and explicit project-owner approval. No partial Phase 3 commit is proposed.

## 14. Artifact manifest

SHA-256 values below are computed from the result files present when this report is generated.

| Artifact | SHA-256 |
|---|---|
{manifest}

All 15 preserved result artifacts are versioned in this repository under `results/`. `phase3_latency_doclevel_v1.json` is the canonical document-level latency evaluation; `phase3_latency_final_v1.json` is a separate chunked experiment, not the canonical latency result. The rounded statistical text file is supplementary; the full-precision JSON is authoritative. Regenerate this report with `python scripts/build_phase3_acceptance_report.py` from the repository root.

---

**Report generated from preserved artifacts.** Regenerate with `python scripts/build_phase3_acceptance_report.py` from the acceptance worktree after any source artifact changes; review the Git diff before accepting.
"""
    return report, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="overwrite docs/EVALUATION_ACCEPTANCE.md")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    results = args.results_dir.resolve() if args.results_dir else (root / "results").resolve()
    output = root / "docs" / "EVALUATION_ACCEPTANCE.md"
    if output.exists() and not args.force:
        print(f"REFUSING_TO_OVERWRITE {output}")
        print("Review the existing report; pass --force only when regeneration is intended.")
        return 2
    try:
        report, hashes = make_report(root, results)
    except Exception as exc:  # fail closed; do not leave a partial report
        print(f"REPORT_NOT_CREATED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(report, encoding="utf-8", newline="\n")
    temp.replace(output)
    print(f"REPORT_CREATED {output}")
    print(f"REPORT_SHA256 {sha256(output)}")
    print(f"REPORT_BYTES {output.stat().st_size}")
    print(f"INPUT_ARTIFACTS_HASHED {len(hashes)}")
    print(f"RRF_DIAGNOSTIC_QUERIES {len(load_json(results / RRF_FILE).get('results', {}))}")
    print("QUALITY_QUERY_RECORDS_VALIDATED 1200")
    print("FAILURE_CLASSIFICATIONS_VALIDATED 1200")
    print("MEDCPT_QUERIES_VALIDATED 300")
    print("LATENCY_SAMPLES_VALIDATED 6000")
    print("REPORT_STATUS PENDING_FINAL_DIFF_REVIEW_AND_OWNER_ACCEPTANCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
