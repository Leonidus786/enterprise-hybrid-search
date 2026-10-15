"""Per-query retrieval metrics used by evaluation and run comparisons."""
from collections.abc import Mapping, Sequence

from .metrics import (
    graded_ndcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .validate import positive_int


def _score_one_ranking(
    ranking,
    query_item,
    k=10,
    retrieval_k=100,
    *,
    level="doc",
    candidate_ranking=None,
    candidate_k=None,
):
    """Score one ranked query using the evaluator's relevance semantics.

    The optional candidate ranking adds Candidate Recall before reranking.
    Graded nDCG is used only for document-level queries with a non-empty
    ``grades`` mapping, matching ``evaluate.py``'s historical behavior.
    """
    positive_int("k", k)
    positive_int("retrieval_k", retrieval_k)
    if retrieval_k < k:
        raise ValueError("retrieval_k must be greater than or equal to k")
    if level not in ("doc", "chunk"):
        raise ValueError('level must be "doc" or "chunk"')
    if not isinstance(query_item, Mapping):
        raise ValueError("query must be a mapping")

    qid = query_item.get("id")
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError("each query must have a non-empty string id")

    relevant_values = query_item.get("relevant")
    if not isinstance(relevant_values, (list, tuple, set)) or not relevant_values:
        raise ValueError(f"query {qid} has no relevance labels")
    if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in relevant_values):
        raise ValueError(f"query {qid} has invalid relevance labels")
    relevant = set(relevant_values)

    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)):
        raise ValueError(f"ranking for query {qid} must be a sequence of document IDs")
    if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in ranking):
        raise ValueError(f"ranking for query {qid} contains invalid document IDs")
    if len(set(ranking)) != len(ranking):
        raise ValueError(f"ranking for query {qid} contains duplicate IDs")

    top_k = list(ranking[:k])
    grades = query_item.get("grades")
    if level == "doc" and grades:
        ndcg = graded_ndcg_at_k(top_k, grades, k)
    else:
        ndcg = ndcg_at_k(top_k, relevant, k)

    scores = {
        f"precision@{k}": precision_at_k(top_k, relevant, k),
        f"recall@{k}": recall_at_k(top_k, relevant, k),
        f"recall@{retrieval_k}": recall_at_k(
            list(ranking[:retrieval_k]), relevant, retrieval_k
        ),
        "mrr": reciprocal_rank(top_k, relevant),
        f"ndcg@{k}": ndcg,
    }

    if candidate_ranking is not None:
        if candidate_k is None:
            raise ValueError("candidate_k is required with candidate_ranking")
        positive_int("candidate_k", candidate_k)
        if not isinstance(candidate_ranking, Sequence) or isinstance(candidate_ranking, (str, bytes)):
            raise ValueError("candidate_ranking must be a sequence")
        if len(set(candidate_ranking)) != len(candidate_ranking):
            raise ValueError("candidate ranking contains duplicate IDs")
        scores[f"candidate_recall@{candidate_k}"] = recall_at_k(
            list(candidate_ranking), relevant, candidate_k
        )

    return scores


def score_rankings(rankings, queries, k=10, retrieval_k=100):
    """Score each query independently using binary relevance.

    ``rankings`` maps query IDs to ranked document/chunk IDs (best first).
    Precision, Recall@k, MRR and nDCG use the top-k ranking; Recall@retrieval_k
    uses the requested deeper cutoff. Extra ranking IDs are ignored.
    """
    positive_int("k", k)
    positive_int("retrieval_k", retrieval_k)
    if retrieval_k < k:
        raise ValueError("retrieval_k must be greater than or equal to k")
    if not isinstance(rankings, Mapping):
        raise ValueError("rankings must be a mapping of query IDs to ranked IDs")
    if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)) or not queries:
        raise ValueError("queries must be a non-empty sequence")

    output = {}
    seen_qids = set()
    for item in queries:
        if not isinstance(item, Mapping):
            raise ValueError("each query must be a mapping")
        qid = item.get("id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError("each query must have a non-empty string id")
        if qid in seen_qids:
            raise ValueError(f"duplicate query id {qid!r}")
        seen_qids.add(qid)

        if qid not in rankings:
            raise ValueError(f"missing ranking for query {qid}")
        # This public utility intentionally uses binary relevance, as required
        # by the Step 3 test contract. The integrated evaluator uses the same
        # helper with graded nDCG enabled when query grades are supplied.
        output[qid] = _score_one_ranking(
            rankings[qid], item, k, retrieval_k, level="chunk"
        )
    return output
