"""Command-line interface for semantic search and clustering."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import DEFAULT_MODEL, SemanticDocumentEngine
from .evaluation import evaluate_clustering, evaluate_retrieval
from .io import load_documents, load_queries


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "sample_documents.csv"
DEFAULT_QUERIES = ROOT / "data" / "search_queries.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semantic-docs",
        description="Cluster a text corpus and search it using sentence embeddings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Find documents similar in meaning to a query.")
    search_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--model", default=DEFAULT_MODEL)
    search_parser.add_argument("--device", default=None)

    cluster_parser = subparsers.add_parser("cluster", help="Group documents by embedding similarity.")
    cluster_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cluster_parser.add_argument("--clusters", type=int, default=None, help="Set k; default selects k by silhouette.")
    cluster_parser.add_argument("--max-clusters", type=int, default=8)
    cluster_parser.add_argument("--output", type=Path, default=ROOT / "results" / "clusters.csv")
    cluster_parser.add_argument("--model", default=DEFAULT_MODEL)
    cluster_parser.add_argument("--device", default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate search and clustering on the labeled demo corpus.")
    eval_parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    eval_parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--output", type=Path, default=ROOT / "results" / "evaluation.json")
    eval_parser.add_argument("--model", default=DEFAULT_MODEL)
    eval_parser.add_argument("--device", default=None)
    return parser


def _engine(args: argparse.Namespace) -> tuple[SemanticDocumentEngine, list[dict[str, str]], np.ndarray]:
    documents = load_documents(args.corpus)
    engine = SemanticDocumentEngine(model_name=args.model, device=args.device)
    embeddings = engine.encode_documents(documents)
    return engine, documents, embeddings


def run_search(args: argparse.Namespace) -> int:
    engine, documents, embeddings = _engine(args)
    hits = engine.search(args.query, documents, embeddings, top_k=args.top_k)
    print(f"Query: {args.query}\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. {hit.title}  [cosine={hit.score:.4f}]")
        print(f"   ID: {hit.doc_id} | Category: {hit.category or 'unlabeled'}")
        print(f"   {hit.text}\n")
    return 0


def run_cluster(args: argparse.Namespace) -> int:
    engine, documents, embeddings = _engine(args)
    labels, chosen_k, silhouette = engine.cluster(
        documents,
        embeddings,
        n_clusters=args.clusters,
        max_clusters=args.max_clusters,
    )
    output_rows = []
    for document, label in zip(documents, labels):
        output_rows.append({
            "doc_id": document["doc_id"],
            "title": document["title"],
            "category": document.get("category", ""),
            "cluster": int(label),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(f"Selected {chosen_k} clusters; cosine silhouette={silhouette:.4f}")
    print(f"Assignments saved to {args.output}")
    for cluster_id in sorted(set(labels.tolist())):
        members = [row["title"] for row in output_rows if row["cluster"] == cluster_id]
        print(f"Cluster {cluster_id} ({len(members)} documents):")
        for title in members[:5]:
            print(f"  - {title}")
    return 0


def run_evaluation(args: argparse.Namespace) -> int:
    engine, documents, embeddings = _engine(args)
    queries = load_queries(args.queries)
    retrieval = evaluate_retrieval(engine, documents, queries, embeddings, k=args.top_k)
    clustering = evaluate_clustering(engine, documents, embeddings)
    report = {
        "model": args.model,
        "corpus_size": len(documents),
        "evaluation_query_count": retrieval.pop("query_count"),
        "retrieval": retrieval,
        "clustering": clustering,
        "notes": [
            "The sample corpus and category labels are curated for demonstration, not a benchmark.",
            "Clustering metrics use category labels only for evaluation; labels are not model inputs.",
            "Five clusters are supplied for this labeled-corpus comparison; unsupervised cluster quality can depend on k.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "retrieval"}, indent=2))
    print("Retrieval summary:")
    for key, value in report["retrieval"].items():
        if key != "per_query":
            print(f"  {key}: {value:.3f}")
    print(f"Full evaluation saved to {args.output}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "search":
            return run_search(args)
        if args.command == "cluster":
            return run_cluster(args)
        return run_evaluation(args)
    except (FileNotFoundError, ValueError, StopIteration) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
