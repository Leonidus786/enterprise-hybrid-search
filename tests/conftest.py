import re
import zlib
import numpy as np
import pytest

DIM = 128

def fake_embed(texts):
    """Deterministic bag-of-words hashing embedder (no model download)."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in re.findall(r"\w+", t.lower()):
            out[i, zlib.crc32(tok.encode()) % DIM] += 1.0
    return out

def fake_score(pairs):
    """Fake cross-encoder: query-token overlap, tiny bonus for shorter text."""
    res = []
    for q, d in pairs:
        qt = set(re.findall(r"\w+", q.lower()))
        dt = re.findall(r"\w+", d.lower())
        res.append(sum(1 for w in dt if w in qt) + 1.0 / (1 + len(dt)))
    return res

@pytest.fixture
def embed():
    return fake_embed

@pytest.fixture
def score():
    return fake_score
