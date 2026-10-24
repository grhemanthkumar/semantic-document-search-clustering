"""Input validation and CSV helpers."""

from pathlib import Path

import pandas as pd

DOCUMENT_COLUMNS = {"doc_id", "title", "text"}
QUERY_COLUMNS = {"query_id", "query", "category"}


def load_documents(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV corpus and validate the fields used by search and clustering."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = DOCUMENT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Corpus is missing required columns: {', '.join(sorted(missing))}")
    if frame["doc_id"].duplicated().any():
        duplicates = frame.loc[frame["doc_id"].duplicated(), "doc_id"].tolist()
        raise ValueError(f"Document IDs must be unique; duplicates: {duplicates[:5]}")
    if (frame["title"].str.strip() == "").any() or (frame["text"].str.strip() == "").any():
        raise ValueError("Every document must have a non-empty title and text.")
    if "category" not in frame.columns:
        frame["category"] = ""
    return frame.to_dict(orient="records")


def load_queries(path: str | Path) -> list[dict[str, str]]:
    """Load labeled query examples used for the demonstration retrieval evaluation."""
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = QUERY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Query file is missing required columns: {', '.join(sorted(missing))}")
    if (frame["query"].str.strip() == "").any() or (frame["category"].str.strip() == "").any():
        raise ValueError("Every evaluation query must include text and a relevance category.")
    return frame.to_dict(orient="records")


def document_text(document: dict[str, str]) -> str:
    """Combine the title and body so both contribute to the embedding."""
    return f"{document['title']}. {document['text']}".strip()
