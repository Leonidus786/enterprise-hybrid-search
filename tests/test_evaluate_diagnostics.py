from hs.evaluate import evaluate


class DiagnosticPipeline:
    candidate_k = 1
    doc_ids = {"relevant_doc", "other_doc"}
    chunk_ids = {"relevant_chunk", "other_chunk"}

    def __init__(self):
        self.last_candidate_ids = []
        self.last_timing_ms = {
            "retrieval": 0.0,
            "rerank": 0.0,
        }

    def doc_of(self, chunk_id):
        mapping = {
            "relevant_chunk": "relevant_doc",
            "other_chunk": "other_doc",
        }
        return mapping[chunk_id]

    def search_docs(self, query, mode="hybrid", k=10):
        if mode == "hybrid":
            self.last_candidate_ids = ["relevant_chunk"]
            self.last_timing_ms = {
                "retrieval": 3.0,
                "rerank": 0.0,
            }
            return ["relevant_doc", "other_doc"][:k]

        if mode == "hybrid_rerank":
            # Relevant document was not in the pre-rerank pool.
            self.last_candidate_ids = ["other_chunk"]
            self.last_timing_ms = {
                "retrieval": 4.0,
                "rerank": 8.0,
            }
            return ["other_doc", "relevant_doc"][:k]

        raise ValueError(f"Unexpected mode: {mode}")


def test_candidate_recall_and_component_latency_are_reported():
    pipeline = DiagnosticPipeline()

    results = evaluate(
        pipeline,
        [
            {
                "id": "q1",
                "query": "test query",
                "relevant": ["relevant_doc"],
            }
        ],
        ["hybrid", "hybrid_rerank"],
        k=1,
        retrieval_k=2,
        warmup=False,
        level="doc",
    )

    assert results["hybrid"]["candidate_recall@1"] == 1.0
    assert results["hybrid_rerank"]["candidate_recall@1"] == 0.0

    assert results["hybrid"]["retrieval_latency_p50_ms"] == 3.0
    assert results["hybrid"]["rerank_latency_p50_ms"] == 0.0

    assert (
        results["hybrid_rerank"]["retrieval_latency_p50_ms"]
        == 4.0
    )
    assert (
        results["hybrid_rerank"]["rerank_latency_p50_ms"]
        == 8.0
    )

    assert "latency_p50_ms" in results["hybrid_rerank"]
    assert "latency_p95_ms" in results["hybrid_rerank"]
