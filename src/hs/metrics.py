import math
from .validate import positive_int


def _check_k(k):
    positive_int("k", k)


def _check_unique(ranked):
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranking contains duplicate ids")


def precision_at_k(ranked, relevant, k):
    _check_k(k)
    _check_unique(ranked)
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked, relevant, k):
    _check_k(k)
    _check_unique(ranked)
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranked, relevant):
    _check_unique(ranked)
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """Binary-relevance nDCG@k."""
    _check_k(k)
    _check_unique(ranked)
    dcg = sum(1 / math.log2(i + 1)
              for i, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(i + 1)
                for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def graded_ndcg_at_k(ranked, grades, k):
    """Calculate nDCG@k using graded relevance scores."""
    _check_k(k)
    _check_unique(ranked)

    def gain(grade):
        return (2**grade) - 1 if grade > 0 else 0

    dcg = sum(
        gain(grades.get(doc_id, 0)) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
    )

    ideal_grades = sorted(
        (grade for grade in grades.values() if grade > 0),
        reverse=True,
    )[:k]

    ideal_dcg = sum(
        gain(grade) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    return dcg / ideal_dcg if ideal_dcg else 0.0
