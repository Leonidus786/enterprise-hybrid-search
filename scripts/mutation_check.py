"""Poor-man's mutation testing: break the code on purpose, confirm the test suite notices.
A mutant that SURVIVES (tests still pass) = a hole in the tests.
Run: python scripts/mutation_check.py"""
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "hs"
MUTANTS = [
    ("sparse.py", "1 - self.b + self.b * self.dl[i] / self.avgdl", "1", "drop BM25 length normalisation"),
    ("sparse.py", "math.log(1 + (n - d + 0.5) / (d + 0.5))", "math.log((n - d + 0.5) / (d + 0.5))", "idf without the +1 (zero-idf bug)"),
    ("sparse.py", "(self.k1 + 1)", "1", "drop (k1+1) numerator"),
    ("fusion.py", "1.0 / (k + rank)", "1.0 / rank", "RRF without k constant"),
    ("fusion.py", "enumerate(ranking, 1)", "enumerate(ranking, 0)", "RRF rank off-by-one"),
    ("metrics.py", "math.log2(i + 1)", "math.log2(i + 2)", "nDCG wrong discount"),
    ("metrics.py", "/ len(relevant)", "/ k", "recall wrong denominator"),
    ("chunking.py", "0, 0, size - overlap", "0, 0, size", "chunk overlap ignored"),
    ("pipeline.py", "n = max(k, self.candidate_k)", "n = k", "fusion candidate pool = k only"),
    ("pipeline.py", "fused[:self.candidate_k]", "fused[:k]", "rerank sees only top-k"),
    ("dense.py", "q = _normalize(np.asarray(self.embed_fn([query]), dtype=np.float32))[0]",
     "q = np.asarray(self.embed_fn([query]), dtype=np.float32)[0]", "query not normalised"),
    ("evaluate.py", "if d not in seen:", "if True:", "no chunk->doc dedupe"),
    ("rerank.py", "key=lambda x: -x[0]", "key=lambda x: x[0]", "rerank sorts ascending"),
    ("sparse.py", "if k <= 0:", "if False:", "BM25 accepts k<=0"),
    ("dense.py", "if k <= 0:", "if False:", "dense accepts k<=0"),
    ("pipeline.py", "if len(self._by_id) != len(self.chunks):", "if False:", "duplicate chunk ids allowed"),
    ("metrics.py", "if len(set(ranked)) != len(ranked):", "if False:", "duplicate ranked ids allowed"),
    ("evaluate.py", "if warmup:", "if False:", "no warm-up"),
    ("evaluate.py", "s[lo] + (s[hi] - s[lo]) * (pos - lo)", "s[lo]", "percentile without interpolation"),
    ("evaluate.py", "if not q.get(\"relevant\"):", "if False:", "unlabeled queries accepted"),
    ("dense.py", "if self.matrix is None:", "if False:", "empty-corpus guard removed"),
]

def run():
    # stale .pyc (same size + same mtime second) would hide the mutation, so disable and purge bytecode
    for d in ROOT.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    return r.returncode == 0

if not run():
    sys.exit("baseline suite is red; fix that first")
survivors = []
for fname, old, new, why in MUTANTS:
    f = SRC / fname
    orig = f.read_text()
    if old not in orig:
        print(f"SKIP     {fname}: pattern not found ({why})"); continue
    f.write_text(orig.replace(old, new, 1))
    try:
        killed = not run()
    finally:
        f.write_text(orig)
    print(f"{'killed  ' if killed else 'SURVIVED'} {fname}: {why}")
    if not killed:
        survivors.append(why)
print(f"\n{len(MUTANTS) - len(survivors)}/{len(MUTANTS)} mutants killed")
sys.exit(1 if survivors else 0)
