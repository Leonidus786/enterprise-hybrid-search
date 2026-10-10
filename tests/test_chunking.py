import pytest
from hs.chunking import chunk_text

def test_chunks_overlap_by_requested_words():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_text("d1", text, size=5, overlap=2)
    assert [c.text for c in chunks] == [
        "w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7", "w6 w7 w8 w9"]

def test_chunk_ids_carry_doc_id_and_index():
    chunks = chunk_text("docA", "a b c d e f", size=3, overlap=0)
    assert [c.chunk_id for c in chunks] == ["docA#0", "docA#1"]
    assert all(c.doc_id == "docA" for c in chunks)

def test_short_text_gives_single_chunk():
    assert len(chunk_text("d", "just three words", size=10, overlap=2)) == 1

def test_empty_text_gives_no_chunks():
    assert chunk_text("d", "   ", size=5, overlap=1) == []

def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("d", "a b c", size=3, overlap=3)
