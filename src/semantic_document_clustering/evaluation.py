"""Small, transparent retrieval and clustering evaluation helpers."""

from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .engine import SemanticDocumentEngine
from .io import document_text


def evaluate_retrieval(
    engine: SemanticDocumentEngine,
    documents: Sequence[dict[str, str]],
    queries: Sequence[dict[str, str]],
    document_embeddings: np.ndarray,
    k: int = 5,
) -> dict:
    """Evaluate category-level relevance on the curated demonstration queries."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    relevant_by_category = defaultdict(set)
    for index, document in enumerate(documents):
        category = str(document.get("category", "")).strip()
        if category:
            relevant_by_category[category].add(index)

    per_query = []
    for query in queries:
        category = query["category"].strip()
        relevant = relevant_by_category.get(category, set())
        if not relevant:
            raise ValueError(f"No corpus documents match query category {category!r}.")
        hits = engine.search(query["query"], documents, document_embeddings, top_k=k)
        rank_by_id = {str(doc["doc_id"]): index for index, doc in enumerate(documents)}
        retrieved_indexes = [rank_by_id[hit.doc_id] for hit in hits]
        relevant_ranks = [rank + 1 for rank, index in enumerate(retrieved_indexes) if index in relevant]
        precision_at_k = len(relevant_ranks) / k
        recall_at_k = len(relevant_ranks) / len(relevant)
        reciprocal_rank = 1 / relevant_ranks[0] if relevant_ranks else 0.0
        per_query.append({
            "query_id": query["query_id"],
            "query": query["query"],
            "expected_category": category,
            f"precision_at_{k}": precision_at_k,
            f"recall_at_{k}": recall_at_k,
            f"reciprocal_rank_at_{k}": reciprocal_rank,
        })

    metric_names = (f"precision_at_{k}", f"recall_at_{k}", f"reciprocal_rank_at_{k}")
    summary = {name: float(np.mean([row[name] for row in per_query])) for name in metric_names}
    summary["query_count"] = len(per_query)
    summary["per_query"] = per_query
    return summary


def evaluate_clustering(
    engine: SemanticDocumentEngine,
    documents: Sequence[dict[str, str]],
    document_embeddings: np.ndarray,
) -> dict[str, float | int]:
    """Compare unsupervised clusters with sample categories, used only for evaluation."""
    categories = [str(document.get("category", "")).strip() for document in documents]
    if not all(categories):
        raise ValueError("Every sample document needs a category for clustering evaluation.")
    unique_categories = sorted(set(categories))
    if len(unique_categories) < 2:
        raise ValueError("Clustering evaluation requires at least two known categories.")

    labels, chosen_k, silhouette = engine.cluster(
        documents,
        document_embeddings,
        n_clusters=len(unique_categories),
    )
    category_ids = {category: index for index, category in enumerate(unique_categories)}
    reference_labels = [category_ids[category] for category in categories]
    return {
        "cluster_count": chosen_k,
        "silhouette_cosine": silhouette,
        "adjusted_rand_index": float(adjusted_rand_score(reference_labels, labels)),
        "normalized_mutual_information": float(normalized_mutual_info_score(reference_labels, labels)),
    }
