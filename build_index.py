"""
build_index.py
---------------
Run this once (or whenever the dataset changes) to build the search index:

    python build_index.py path/to/spotify_millsongdata.csv
    python build_index.py path/to/spotify_millsongdata.csv --sample 2000   # quick smoke test

It chunks every song's lyrics into stanza-based pieces, embeds each chunk
with a sentence-transformer (all-mpnet-base-v2, 768-dim), and writes
everything into a local Milvus (Milvus Lite) collection under ./data/.
The FastAPI app then just connects to that file at startup instead of
rebuilding anything.

On the full ~57.6k-track dataset this produces on the order of a few
hundred thousand chunks. mpnet-base-v2 is a larger model than MiniLM, so
encoding them on CPU takes longer -- budget more time than you would for
a small model (much faster on GPU). Use --sample while you're iterating
on the pipeline, then drop it for the real build.

By default this also writes data/eval_embeddings.npz -- a cache of every
chunk's raw vector, used by evaluate.py to run an exact brute-force
search baseline to compare against Milvus's HNSW index. At 768 dimensions
this cache is large (roughly 3KB/chunk, i.e. ~1GB+ for the full dataset).
Pass --no-eval-cache to skip it if you don't need the evaluation script.
"""
import argparse
from search_engine import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Milvus vibe-search index.")
    parser.add_argument("csv_path", nargs="?", default="spotify_millsongdata.csv")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only index a random sample of N songs (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--no-eval-cache", action="store_true",
        help="Skip writing data/eval_embeddings.npz (saves disk space; disables evaluate.py).",
    )
    args = parser.parse_args()
    build_index(args.csv_path, sample=args.sample, save_eval_cache=not args.no_eval_cache)
