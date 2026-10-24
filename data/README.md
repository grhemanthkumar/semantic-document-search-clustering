# Demo data

`sample_documents.csv` is a small, manually curated corpus created for this repository. It contains 30 short documents across five broad subject categories. The `category` column is reference metadata for evaluation and display; the semantic encoder and unsupervised clustering code use only each document's title and text.

`search_queries.csv` contains 10 natural-language queries with expected subject categories. These labels support a transparent demonstration of Precision@k, Recall@k, and reciprocal rank. Because the corpus and queries are small and hand-curated, evaluation scores are illustrative, not a benchmark of general semantic-search quality.

To evaluate this system seriously, replace the demo with a larger, independently labeled corpus and queries judged by people who did not build the system.
