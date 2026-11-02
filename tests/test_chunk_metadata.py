from hs.chunking import Chunk, chunk_text


def test_legacy_chunk_constructor_keeps_three_positional_arguments():
    chunk = Chunk("d1#0", "d1", "legacy text")

    assert (chunk.chunk_id, chunk.doc_id, chunk.text) == (
        "d1#0", "d1", "legacy text",
    )
    assert chunk.source is None
    assert chunk.title is None
    assert chunk.metadata == {}


def test_metadata_is_propagated_to_every_word_window():
    chunks = chunk_text(
        "d1",
        "one two three four five",
        size=3,
        overlap=1,
        source="handbook.pdf",
        title="Security handbook",
        metadata={"department": "security"},
    )

    assert len(chunks) == 2
    assert all(chunk.source == "handbook.pdf" for chunk in chunks)
    assert all(chunk.title == "Security handbook" for chunk in chunks)
    assert all(chunk.metadata == {"department": "security"} for chunk in chunks)


def test_metadata_is_propagated_in_no_chunk_mode():
    chunks = chunk_text(
        "d1",
        " one   two\nthree ",
        size=None,
        source="source.txt",
        title="Title",
        metadata={"priority": 2},
    )

    assert len(chunks) == 1
    assert chunks[0].text == "one two three"
    assert chunks[0].metadata == {"priority": 2}


def test_optional_metadata_does_not_break_legacy_chunk_equality_or_hash():
    legacy = Chunk("d1#0", "d1", "same text")
    enriched = Chunk(
        "d1#0", "d1", "same text",
        source="source.txt",
        title="title",
        metadata={"department": "security"},
    )

    assert legacy == enriched
    assert hash(legacy) == hash(enriched)
