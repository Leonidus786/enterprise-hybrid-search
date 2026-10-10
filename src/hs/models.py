"""Real models. Lazy imports so tests never need torch/HF downloads."""

def sentence_embedder(name="BAAI/bge-small-en-v1.5"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    return lambda texts: m.encode(texts, normalize_embeddings=True, batch_size=64)

def cross_encoder_scorer(name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(name)
    return lambda pairs: ce.predict(pairs).tolist()
