# Semantic Document Clustering & Similarity Search

A small, reproducible Python project for exploring a text collection with sentence embeddings. It supports semantic nearest-neighbour search, KMeans clustering, and a transparent evaluation on a curated sample corpus.

> **Scope:** this is a portfolio-scale demonstration, not a production search service or a scientific benchmark. The sample corpus is small and curated; evaluation scores illustrate the workflow and should not be generalized.

## What it does

- Encodes document titles and text with the pretrained `sentence-transformers/all-MiniLM-L6-v2` sentence encoder.
- Searches by meaning with cosine similarity between normalized query and document embeddings.
- Clusters documents with KMeans. You can supply the number of clusters or let the program choose a value using cosine silhouette score.
- Evaluates semantic retrieval with Precision@k, Recall@k, and reciprocal rank on labeled sample queries.
- Evaluates clusters with cosine silhouette, Adjusted Rand Index, and Normalized Mutual Information. Category labels are used only to evaluate the sample clusters, not as model input.
- Offers a command line interface and a beginner-friendly notebook.

## Why this is semantic search

The original notebook used TF-IDF. TF-IDF represents text through word overlap, so different wording can lead to a zero similarity score. This version uses a pretrained sentence-transformer to map complete passages into dense vectors. Related sentences can receive similar vectors even when they do not share every keyword. Sentence Transformers documents this approach for semantic similarity, search, and clustering. See the [Sentence Transformers usage guide](https://www.sbert.net/docs/sentence_transformer/usage/usage.html) and the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

Cosine similarity is a ranking score, not a probability or a guarantee that the top result is correct.

## Requirements

- Python 3.10 or later (the Sentence Transformers project currently lists Python 3.13 support).
- Internet access for the first run, to download the Python packages and pretrained model.
- A CPU is sufficient for the small example corpus. A supported GPU can be selected with `--device cuda`.

## Setup

From this project folder, create and activate a virtual environment, then install the package.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If the `python` command is not recognized in your PowerShell, use the full Python path for your installation to create the environment and install the package. For example:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -e .
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The first command that loads the model downloads it into the Hugging Face cache. Later runs reuse the cached model.

## Quick start

Run semantic search on the bundled sample corpus:

```powershell
semantic-docs search --query "How can I protect passwords if a website is breached?" --top-k 5
```

If the console command is unavailable before reinstalling the project, use:

```powershell
python -m semantic_document_clustering search --query "How can I protect passwords if a website is breached?" --top-k 5
```

Cluster documents and save the assignments:

```powershell
semantic-docs cluster --clusters 5 --output results/clusters.csv
```

Omit `--clusters` to select k from a small candidate range using cosine silhouette score. This is a heuristic; it does not prove that the selected grouping is the true or best taxonomy.

Run the bundled retrieval and clustering evaluation:

```powershell
semantic-docs evaluate --top-k 5
```

The command writes a detailed JSON result to `results/evaluation.json` and prints the main metrics. Results are ignored by Git so local run outputs do not clutter the repository.

## Use from Python

```python
from semantic_document_clustering.engine import SemanticDocumentEngine
from semantic_document_clustering.io import load_documents

documents = load_documents("data/sample_documents.csv")
engine = SemanticDocumentEngine()
embeddings = engine.encode_documents(documents)

hits = engine.search(
    "How can a system find useful passages before answering a question?",
    documents,
    embeddings,
    top_k=3,
)

for hit in hits:
    print(f"{hit.score:.3f}  {hit.title}")
```

## Bring your own corpus

Use a UTF-8 CSV with these columns:

| Column | Required | Description |
|---|---|---|
| `doc_id` | Yes | Unique document identifier |
| `title` | Yes | Short descriptive title |
| `text` | Yes | Document content to embed |
| `category` | No for search/clustering; yes for evaluation | Optional human-assigned category used for evaluation only |

The loader checks required columns, unique IDs, and empty titles or text. The provided sample has 30 short documents across five themes. It is curated to demonstrate the pipeline and is too small to support meaningful claims about real-world performance.

## Evaluation details

- **Precision@k:** fraction of the top-k retrieved documents that match the query's category.
- **Recall@k:** fraction of all corpus documents in that category that appear in the top-k results.
- **Reciprocal rank:** inverse rank of the first category-matching result (zero if none appears in the top k).
- **Silhouette (cosine):** internal separation/cohesion score calculated from embedding distances.
- **Adjusted Rand Index and Normalized Mutual Information:** compare cluster assignments with the sample's reference categories. Cluster IDs are arbitrary; these metrics account for that.

The evaluation labels and query examples are hand-curated for this demonstration. They are not an independent benchmark, and the model was not tuned on an external test set. Use a larger, independently labeled corpus and human relevance judgments for a serious evaluation.

## Repository layout

```text
.
├── data/
│   ├── README.md
│   ├── sample_documents.csv
│   └── search_queries.csv
├── notebooks/
│   └── semantic_clustering_demo.ipynb
├── results/
├── src/semantic_document_clustering/
│   ├── cli.py
│   ├── engine.py
│   ├── evaluation.py
│   └── io.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Limitations and next research steps

- The sample corpus is intentionally small and curated; reported metrics are demonstrations only.
- Search quality depends on the pretrained model, document length, and domain. This repository does not fine-tune the encoder.
- Automatic k selection uses silhouette as a heuristic; different k values can produce different but plausible clusterings.
- The current implementation re-encodes the corpus on each command and uses exact vector search. It is suitable for small collections, not large-scale production indexing.
- A stronger study would use a larger public corpus, independent human relevance judgments, repeated evaluation, and a comparison against a lexical baseline such as TF-IDF or BM25.

## AI assistance

This repository was prepared with AI assistance based on a starter notebook. Review the code, run the commands, verify the outputs, and follow any rules about AI use before presenting or publishing it as your work.
