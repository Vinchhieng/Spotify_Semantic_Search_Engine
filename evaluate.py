"""
evaluate.py
-----------
Compares Milvus's HNSW (approximate) search against an exact brute-force
cosine-similarity baseline, over this project's own indexed chunks --
measuring Recall@K and per-query latency for both.

HNSW is an *approximate* nearest-neighbor index: by design it trades some
recall for large latency gains over scanning every vector. This script
measures that trade-off empirically on the vectors this project actually
indexes, rather than asserting a number. It does not print or assume any
result -- every value below comes from running the search calls below.

This is a standalone evaluation tool, separate from the core search
engine (search_engine.py) and the FastAPI demo (app.py): it doesn't
change how either of those work, it just measures one of search_engine.py's
components (the HNSW index) against a ground-truth baseline.

Requires data/eval_embeddings.npz, written by build_index.py by default
(skip with --no-eval-cache at build time, in which case this script has
nothing to compare against).

Usage:
    python evaluate.py
    python evaluate.py --k 10 20 50 --queries 5
"""
import argparse
import time

import numpy as np

from search_engine import SemanticSearchEngine, EVAL_CACHE_PATH

# A fixed, hand-written set of vibe-style queries representative of how
# the app is actually used -- not lyric lines pulled from the index, so
# this doesn't just test "can it retrieve an exact substring."
DEFAULT_QUERIES = [
    "melancholic acoustic ballad for a rainy morning",
    "aggressive high-energy hype music for the gym",
    "heartbreak and lost love",
    "christmas holiday cheer with family",
    "summer beach party dance vibes",
    "driving alone at night feeling nostalgic",
    "slow romantic love song for a wedding",
    "angry rebellious punk energy",
    "peaceful meditation and calm",
    "triumphant victory anthem",
]


def load_eval_cache():
    if not EVAL_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"No eval cache found at {EVAL_CACHE_PATH}. Rebuild the index with "
            "`python build_index.py` (without --no-eval-cache) first."
        )
    data = np.load(EVAL_CACHE_PATH)
    return data["ids"], data["vectors"]


def exact_search(query_vec: np.ndarray, vectors: np.ndarray, ids: np.ndarray, k: int):
    """Ground-truth brute-force search: since every vector is L2-normalized,
    cosine similarity is just a dot product against the full matrix."""
    scores = vectors @ query_vec
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return ids[top_idx]


def hnsw_search(engine: SemanticSearchEngine, query_vec: np.ndarray, k: int):
    hits = engine.client.search(
        collection_name=engine.collection,
        data=[query_vec.tolist()],
        limit=k,
        search_params={"params": {"ef": 96}},
    )[0]
    return np.array([hit["id"] for hit in hits])


def run_evaluation(k_values: list[int], num_queries: int) -> None:
    print("Loading exact-search cache ...")
    ids, vectors = load_eval_cache()
    print(f"  {len(ids):,} chunk vectors loaded (dim={vectors.shape[1]})")

    print("Connecting to Milvus + loading embedding model (this can take a bit) ...")
    engine = SemanticSearchEngine()

    queries = DEFAULT_QUERIES[:num_queries]
    print(f"Running {len(queries)} queries x {len(k_values)} K value(s) ...\n")

    per_k = {k: {"recall": [], "hnsw_ms": [], "exact_ms": []} for k in k_values}

    for q in queries:
        query_vec = engine.model.encode(
            [q], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype(np.float32)

        for k in k_values:
            t0 = time.perf_counter()
            exact_ids = exact_search(query_vec, vectors, ids, k)
            exact_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            hnsw_ids = hnsw_search(engine, query_vec, k)
            hnsw_ms = (time.perf_counter() - t0) * 1000

            overlap = len(set(exact_ids.tolist()) & set(hnsw_ids.tolist()))
            recall = overlap / k

            per_k[k]["recall"].append(recall)
            per_k[k]["hnsw_ms"].append(hnsw_ms)
            per_k[k]["exact_ms"].append(exact_ms)

    engine.close()

    print(f"{'K':>4} | {'Recall@K':>10} | {'HNSW ms':>10} | {'Exact ms':>10} | {'Speedup':>8}")
    print("-" * 55)
    for k in k_values:
        recall = float(np.mean(per_k[k]["recall"]))
        hnsw_ms = float(np.mean(per_k[k]["hnsw_ms"]))
        exact_ms = float(np.mean(per_k[k]["exact_ms"]))
        speedup = exact_ms / hnsw_ms if hnsw_ms > 0 else float("nan")
        print(f"{k:>4} | {recall:>10.3f} | {hnsw_ms:>10.2f} | {exact_ms:>10.2f} | {speedup:>7.1f}x")

    print(
        "\nRecall@K = fraction of the exact top-K chunks that HNSW also returned "
        "in its top-K, averaged across queries. 1.000 = HNSW matched the exact "
        "baseline exactly for that query set; lower values quantify the "
        "approximation cost paid for HNSW's speed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Milvus HNSW search against an exact cosine baseline."
    )
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20, 50],
                         help="K values to evaluate Recall@K for.")
    parser.add_argument("--queries", type=int, default=len(DEFAULT_QUERIES),
                         help="Number of queries from DEFAULT_QUERIES to run.")
    args = parser.parse_args()
    run_evaluation(args.k, args.queries)
