import math
import time
from .validate import positive_int
from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank

def percentile(values, p):
    """Linear-interpolated percentile (p in 0..100)."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)

def evaluate(pipeline, queries, modes, k=10, warmup=True, level="doc"):
    """queries: [{id, query, relevant:[ids]}]. level="doc": ids are doc ids, top-k UNIQUE docs;
    level="chunk": ids are chunk ids, top-k chunks. Never mix the two protocols.
    warmup=True runs one untimed search per mode so model load / cache misses
    don't pollute latency percentiles."""
    positive_int("k", k)
    if level not in ("doc", "chunk"):
        raise ValueError('level must be "doc" or "chunk"')
    if not queries:
        raise ValueError("no queries to evaluate")
    known = pipeline.doc_ids if level == "doc" else pipeline.chunk_ids
    for q in queries:
        if not q.get("relevant"):
            raise ValueError(f"query {q.get('id')} has no relevance labels")
        missing = set(q["relevant"]) - known
        if missing:
            raise ValueError(f"query {q.get('id')}: labels not in corpus ({level} level): {sorted(missing)}")
    search = pipeline.search_docs if level == "doc" else pipeline.search
    out = {}
    for mode in modes:
        if warmup:
            search(queries[0]["query"], mode=mode, k=k)
        p = r = m = n = 0.0
        lats = []
        for q in queries:
            rel = set(q["relevant"])
            t0 = time.perf_counter()
            ranked = search(q["query"], mode=mode, k=k)
            lats.append((time.perf_counter() - t0) * 1000)
            p += precision_at_k(ranked, rel, k)
            r += recall_at_k(ranked, rel, k)
            m += reciprocal_rank(ranked, rel)
            n += ndcg_at_k(ranked, rel, k)
        c = len(queries)
        out[mode] = {f"precision@{k}": p / c, f"recall@{k}": r / c, "mrr": m / c,
                     f"ndcg@{k}": n / c,
                     "latency_p50_ms": percentile(lats, 50),
                     "latency_p95_ms": percentile(lats, 95)}
    return out
