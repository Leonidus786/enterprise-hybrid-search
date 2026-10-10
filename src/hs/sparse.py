import math
import re
from collections import Counter
from .validate import finite_number, positive_int

def tokenize(text):
    return re.findall(r"\w+", text.lower())

class SparseIndex:
    """BM25 with Lucene-style idf (always > 0, so a term in half the corpus still scores)."""
    def __init__(self, chunks, k1=1.5, b=0.75):
        finite_number("k1", k1)
        finite_number("b", b)
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("need k1 >= 0 and 0 <= b <= 1")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self.dl = [sum(t.values()) for t in self.tf]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        df = Counter(w for t in self.tf for w in t)
        n = len(self.chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def _score(self, i, q_tokens):
        s = 0.0
        for w in q_tokens:
            f = self.tf[i].get(w, 0)
            if f:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                s += self.idf[w] * f * (self.k1 + 1) / denom
        return s

    def search(self, query, k=10):
        positive_int("k", k)
        q = tokenize(query)
        scores = [self._score(i, q) for i in range(len(self.chunks))]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i].chunk_id, scores[i]) for i in order if scores[i] > 0]
