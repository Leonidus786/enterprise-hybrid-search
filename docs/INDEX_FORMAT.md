# Persistent Index Format — Schema v1

This document freezes the Step 4.1 serialization contract, the Step 4.2
snapshot-builder behavior, and the Step 4.3 validated-loader behavior. Active
snapshot switching, incremental updates, deletion and recovery remain later
tasks; this step does not claim those capabilities are implemented.

## Snapshot artifacts

Every index snapshot contains four files:

- `documents.jsonl`: one source `Document` record per ingested document.
- `chunks.jsonl`: one searchable `Chunk` record per chunk, in vector-row order.
- `vectors.npy`: a two-dimensional, C-contiguous `float32` matrix. Row `i`
  corresponds to record `i` in `chunks.jsonl`; vectors are L2-normalized.
- `manifest.json`: counts, model/config identity, corpus fingerprint and the
  byte-level SHA-256/size of each of the three payload artifacts.

`manifest.json` is written last and is not included in its own checksums.
`CURRENT.json` is a separate active-version pointer planned for the atomic
activation step.

## Record serialization

JSONL is UTF-8, canonical JSON (sorted keys, compact separators, Unicode retained),
one object per line, and a final LF for each record. Blank lines, malformed JSON,
non-object rows, invalid UTF-8, NaN, and Infinity are rejected.

Document records store `doc_id`, exact source `text`, normalized `source`, `title`,
JSON-compatible `metadata`, `content_sha256`, `revision_sha256`, and
`record_schema_version=1`.

Chunk records store `chunk_id`, `doc_id`, chunk `text`, source/title/metadata,
`document_revision_sha256`, `chunking_config`, content/revision fingerprints and
`record_schema_version=1`.

Deserializers recompute and verify the supplied fingerprints. Existing `chunk_id`
values remain unchanged; revision fingerprints distinguish changed content or
configuration.

## Manifest schema v1

The manifest contains:

- `format`, `schema_version`, safe `index_id`, UTC `created_at_utc`.
- `corpus`: document count, chunk count and SHA-256 corpus fingerprint.
- `embedding`: model ID, immutable model revision and vector-affecting config.
- `vectors`: row count, dimension, `float32` dtype and normalization flag.
- `chunking`: chunking algorithm/version/settings.
- `sparse`: BM25 algorithm/version and ranking/tokenization settings.
- `artifacts`: checksum and byte size for every payload; JSONL record counts and
  NumPy matrix shape/dtype are also recorded.

Manifest validation is strict. Unknown schema versions, missing/extra artifacts,
unsafe index IDs, incompatible model identities, inconsistent row counts,
invalid dimensions, bad checksums, unsupported dtypes and invalid timestamps are
rejected. Schema changes require a schema-version change and explicit migration
policy; old snapshots must not be silently interpreted as a new schema.

## Step 4.2 builder contract

`build_index_snapshot(output_dir, ...)` creates a new snapshot directory. The
destination must not already exist; existing snapshots are never overwritten.
The builder:

1. validates unique document/chunk IDs and parent-document revision links;
2. checks that the supplied ordered chunks exactly match the supported,
   deterministic chunker and the documents in the same order;
3. embeds each chunk exactly once in that order;
4. validates a finite two-dimensional matrix with the same number of rows as
   chunks, converts it to `float32`, and L2-normalizes non-zero rows;
5. preserves zero-vector rows as zeros, matching the existing `DenseIndex`
   normalization policy;
6. writes the two JSONL files and `vectors.npy`, computes their checksums, then
   writes `manifest.json` last.

The manifest requires an immutable hexadecimal embedding revision and explicit
non-empty embedding/sparse configuration. A mutable branch name such as `main`
is not accepted. The caller is responsible for supplying the actual loaded
model identity/configuration; the builder does not download or inspect models.

The vector matrix row order is exactly the input `chunks` order. The builder
reproduces the deterministic chunking result before it calls the embedder, so
missing, reordered, stale or metadata-mismatched chunk sets are rejected.

A failed in-process build removes the incomplete output directory. A hard
process/host termination can leave an orphan directory, but since the manifest
is written last and no `CURRENT.json` pointer is changed here, it is not an
active index. The future loader and activation operation must independently
validate all artifacts before activation.


## Step 4.3 loader contract

`load_index_snapshot(snapshot_dir, expected_embedding=...)` validates a snapshot
before exposing any records or vectors:

1. requires the snapshot directory to contain exactly `manifest.json` and the
   three declared payload artifacts, all as regular non-symlink files;
2. strictly parses the manifest and validates its schema, UTC timestamp,
   dimensions, artifact metadata and optional expected model ID/revision/config;
3. recomputes the SHA-256 and byte size of every payload before parsing it;
4. reads JSONL incrementally, rejects duplicate object keys, non-standard JSON
   constants, blank lines, invalid UTF-8 and non-LF line endings;
5. verifies document/chunk record fingerprints, ordered corpus identity,
   parent-document revisions and reproducible chunking configuration;
6. loads `vectors.npy` with `allow_pickle=False`, then verifies its shape, exact
   `float32` dtype, C-contiguous layout, finite values and unit norms for every
   non-zero row (zero rows remain valid);
7. returns immutable document/chunk tuples and a read-only vector matrix.

The loader does not accept or invoke an embedding function. It therefore does
not re-embed source chunks. Dense query embedding remains the caller's
responsibility and will be connected to the existing `DenseIndex` separately.

SHA-256 checks detect accidental artifact corruption, but they are not a digital
signature: someone able to rewrite both payloads and the manifest can produce a
new internally consistent snapshot. Do not treat this format as tamper-proof
against an attacker with write access to the index directory.

## Identity and integrity boundaries

SHA-256 values identify exact serialized artifacts and canonical document/chunk
identity payloads. The loader compares artifact hashes and byte sizes with the
manifest before exposing a snapshot to retrieval. This detects accidental
corruption; it does not authenticate who created the snapshot.

BM25 postings are intentionally not serialized in v1. The loader will rebuild
`SparseIndex` from the persisted chunks using the manifest-declared sparse config.

## Known follow-up

The current `Document` model does not yet expose a first-class timestamp field.
If ingestion requires source publication/update timestamps, add a backwards-
compatible typed field and identity semantics before declaring the full ingestion
schema complete. Do not silently treat an arbitrary metadata string as a validated
UTC timestamp.

## Step 4.4 DenseIndex integration

`DenseIndex` can now initialize from a validated, precomputed normalized matrix
without calling the document `embed_fn`. The matrix row count must match the
ordered chunks; every value must be finite; each non-zero row must already have
unit L2 norm; zero rows remain valid. The matrix is not normalized again, and a
read-only, C-contiguous loader array remains read-only when reused. A compatible
query embedder is still required for search.

Example after the Step 4.3 loader validates the snapshot:

```python
loaded = load_index_snapshot(
    snapshot_dir,
    expected_embedding={
        "model_id": model_id,
        "model_revision": model_revision,
        "config": embedding_config,
    },
)
dense = DenseIndex.from_snapshot(
    loaded,
    query_embed_fn=query_embedder,
    query_prefix=query_prefix,
)
```

This constructor preserves the original embedding-based initialization path for
existing callers. Full `HybridPipeline` construction from a loaded snapshot is a
separate Step 4.5 integration.
