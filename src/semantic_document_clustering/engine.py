"""Sentence embedding, semantic search, and clustering."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .io import document_text


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SearchHit:
    """A document ranked for a query by cosine similarity."""

    doc_id: str
    title: str
    category: str
    score: float
    text: str


class SemanticDocumentEngine:
    """Encode documents once, then cluster them or search by meaning."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, documents: Sequence[dict[str, str]]) -> np.ndarray:
        """Create normalized sentence embeddings for a document collection."""
        if not documents:
            raise ValueError("The document collection is empty.")
        texts = [document_text(document) for document in documents]
        embeddings = self.model.encode_document(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def search(
        self,
        query: str,
        documents: Sequence[dict[str, str]],
        document_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Return the most semantically similar documents for a natural-language query."""
        if not query.strip():
            raise ValueError("Query text cannot be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if len(documents) != len(document_embeddings):
            raise ValueError("Document and embedding counts do not match.")

        query_embedding = self.model.encode_query(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = document_embeddings @ np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(documents))]
        return [
            SearchHit(
                doc_id=str(documents[index]["doc_id"]),
                title=str(documents[index]["title"]),
                category=str(documents[index].get("category", "")),
                score=float(scores[index]),
                text=str(documents[index]["text"]),
            )
            for index in order
        ]

    def cluster(
        self,
        documents: Sequence[dict[str, str]],
        document_embeddings: np.ndarray,
        n_clusters: int | None = None,
        max_clusters: int = 8,
        random_state: int = 42,
    ) -> tuple[np.ndarray, int, float]:
        """Cluster embeddings; optionally choose k by the cosine silhouette score."""
        n_documents = len(documents)
        if n_documents != len(document_embeddings):
            raise ValueError("Document and embedding counts do not match.")
        if n_documents < 3:
            raise ValueError("At least three documents are required for clustering.")

        if n_clusters is None:
            upper_k = min(max_clusters, n_documents - 1)
            candidates = range(2, upper_k + 1)
            best = None
            for candidate_k in candidates:
                labels = self._fit_kmeans(document_embeddings, candidate_k, random_state)
                score = float(silhouette_score(document_embeddings, labels, metric="cosine"))
                if best is None or score > best[0]:
                    best = (score, candidate_k, labels)
            if best is None:
                raise ValueError("Automatic cluster selection needs at least four documents.")
            silhouette, chosen_k, labels = best
            return labels, chosen_k, silhouette

        if not 2 <= n_clusters < n_documents:
            raise ValueError("n_clusters must be at least 2 and less than the document count.")
        labels = self._fit_kmeans(document_embeddings, n_clusters, random_state)
        score = float(silhouette_score(document_embeddings, labels, metric="cosine"))
        return labels, n_clusters, score

    @staticmethod
    def _fit_kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        return model.fit_predict(embeddings)
