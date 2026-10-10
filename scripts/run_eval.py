"""python scripts/run_eval.py [--corpus ..] [--queries ..] [--k 10] [--candidate-k 50] [--level doc|chunk] [--fake]
Real models by default (needs sentence-transformers + HF download). --fake = offline smoke test (NOT evidence)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from hs.evaluate import evaluate
from hs.io import load_chunks, load_queries
from hs.pipeline import HybridPipeline

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus.jsonl")
    ap.add_argument("--queries", default="data/queries.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=50, help="chunks passed to the cross-encoder")
    ap.add_argument("--level", choices=["doc", "chunk"], default="doc")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--no-bge-prefix", action="store_true", help="real models: disable the BGE query instruction")
    return ap

def _fakes():
    import re, zlib
    import numpy as np
    def embed(texts):
        o = np.zeros((len(texts), 128), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                o[i, zlib.crc32(w.encode()) % 128] += 1
        return o
    def score(pairs):
        return [len(set(re.findall(r"\w+", q.lower())) & set(re.findall(r"\w+", d.lower()))) for q, d in pairs]
    return embed, score

def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.fake:
        embed, score = _fakes()
    else:
        from hs.models import cross_encoder_scorer, sentence_embedder
        embed, score = sentence_embedder(), cross_encoder_scorer()
    prefix = "" if (a.fake or a.no_bge_prefix) else BGE_PREFIX
    p = HybridPipeline(load_chunks(a.corpus), embed, score_fn=score, candidate_k=a.candidate_k, query_prefix=prefix)
    res = evaluate(p, load_queries(a.queries), ["bm25", "dense", "hybrid", "hybrid_rerank"], k=a.k, level=a.level)
    cols = list(next(iter(res.values())))
    print(f"{'strategy':<15}" + "".join(f"{c:>17}" for c in cols))
    for mode, m in res.items():
        print(f"{mode:<15}" + "".join(f"{m[c]:>17.3f}" for c in cols))

if __name__ == "__main__":
    main()
