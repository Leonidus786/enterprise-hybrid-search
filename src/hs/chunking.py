from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str

def chunk_text(doc_id, text, size=200, overlap=40):
    """Word-window chunking with overlap."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    words = text.split()
    chunks, i, n, step = [], 0, 0, size - overlap
    while i < len(words):
        chunks.append(Chunk(f"{doc_id}#{n}", doc_id, " ".join(words[i:i + size])))
        n += 1
        if i + size >= len(words):
            break
        i += step
    return chunks
