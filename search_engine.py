"""
search_engine.py
-----------------
Core semantic search engine for the Spotify Million Song Dataset.

Architecture (see README.md for the full design write-up):
  * Chunking      : each song's lyrics are split by STANZA -- the blank
                    lines that already separate verses/choruses in the raw
                    text -- rather than embedded as one giant document or
                    cut at an arbitrary character count. A verse/chorus is
                    the natural unit where a song's "vibe" actually lives.
                    Oversized stanzas are sub-split by line-grouping;
                    undersized stanzas (stray one-word ad-libs, etc.) are
                    merged into a neighboring chunk so no near-empty,
                    low-signal vector gets indexed.
  * Embeddings    : sentence-transformers "all-MiniLM-L6-v2" -> 384-dim
                    vectors, L2-normalized at encode time.
  * Vector store  : Milvus (via pymilvus's embedded "Milvus Lite" client -
                    a single local .db file, no server to run).
  * Index type    : HNSW, metric_type="COSINE" -- an *approximate*
                    nearest-neighbor index. It trades a small amount of
                    recall for a large speed gain over scanning every
                    vector; see evaluate.py for the measured recall/
                    latency trade-off on this project's own data, rather
                    than an assumed number.
  * Aggregation   : a query searches over CHUNKS, then results are grouped
                    back up to their parent song, keeping each song's
                    single best-matching chunk as both its score and its
                    displayed excerpt.
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = str(DATA_DIR / "spotify_vibes.db")
EVAL_CACHE_PATH = DATA_DIR / "eval_embeddings.npz"
COLLECTION = "song_chunks"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
ENCODE_BATCH_SIZE = 128  # MiniLM processes 128 lyric chunks per batch
INSERT_BATCH_SIZE = 2000
# Checkpoint settings
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

EMBEDDING_CHECKPOINT_PATH = CHECKPOINT_DIR / "embeddings.dat"
PROGRESS_CHECKPOINT_PATH = CHECKPOINT_DIR / "progress.txt"

# Save after every 100 encoding batches
CHECKPOINT_BATCHES = 100
# Chunking thresholds, tuned against the real dataset (see README):
#   - a typical stanza runs ~150-200 chars, so anything much larger than
#     that is probably a song formatted with few/no blank lines and needs
#     sub-splitting rather than being indexed as one oversized chunk.
#   - anything under ~30 chars is a stray ad-lib / formatting artifact
#     ("Oh!", "Yeah yeah") that carries almost no semantic signal on its
#     own, so it gets folded into a neighboring chunk instead.
STANZA_MAX_CHARS = 400
STANZA_MIN_CHARS = 30
SUBSPLIT_TARGET_CHARS = 220


# --------------------------------------------------------------------------
# Text cleaning & chunking
# --------------------------------------------------------------------------

def split_into_stanzas(raw_text: str) -> list[list[str]]:
    """Split the raw CSV lyric field into stanzas (verses/choruses), using
    blank lines as the boundary -- exactly how the source lyrics are
    already formatted. Returns each stanza as its own list of cleaned
    lines (line boundaries are kept so oversized stanzas can later be
    sub-split without re-parsing)."""
    raw_lines = re.split(r"\r\n|\n", raw_text)
    stanzas: list[list[str]] = []
    current: list[str] = []
    for line in raw_lines:
        stripped = re.sub(r"\s+", " ", line).strip()
        if stripped == "":
            if current:
                stanzas.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        stanzas.append(current)
    return stanzas


def group_lines(lines: list[str], target_chars: int = SUBSPLIT_TARGET_CHARS) -> list[str]:
    """Fallback splitter: group consecutive lines up to ~target_chars.
    Used only to sub-split a stanza that's larger than STANZA_MAX_CHARS."""
    chunks, buffer = [], ""
    for line in lines:
        candidate = f"{buffer} {line}".strip() if buffer else line
        if len(candidate) > target_chars and buffer:
            chunks.append(buffer)
            buffer = line
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_lyrics(raw_text: str) -> list[str]:
    """Turn one song's raw lyrics into a list of chunk strings: split by
    stanza, sub-split anything oversized, then merge anything undersized
    into a neighbor so every indexed chunk carries real signal."""
    stanzas = split_into_stanzas(raw_text)
    if not stanzas:
        return []

    # Pass 1: stanza is the primary unit; sub-split only if it's too big.
    chunks: list[str] = []
    for lines in stanzas:
        joined = " ".join(lines)
        if len(joined) > STANZA_MAX_CHARS:
            chunks.extend(group_lines(lines))
        else:
            chunks.append(joined)

    # Pass 2: fold any too-short chunk into the next one (or the previous
    # one, if it happens to be the last chunk in the song).
    merged: list[str] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if len(c) < STANZA_MIN_CHARS and len(chunks) > 1:
            if i + 1 < len(chunks):
                chunks[i + 1] = f"{c} {chunks[i + 1]}".strip()
                i += 1
                continue
            elif merged:
                merged[-1] = f"{merged[-1]} {c}".strip()
                i += 1
                continue
        merged.append(c)
        i += 1
    return merged


# --------------------------------------------------------------------------
# Index building
# --------------------------------------------------------------------------

def build_index(
    csv_path: str,
    sample: int | None = None,
    db_path: str = DB_PATH,
    collection: str = COLLECTION,
    save_eval_cache: bool = True,
) -> None:
    t0 = time.time()
    print(f"[1/6] Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    df = df.dropna(subset=["artist", "song", "text"]).reset_index(drop=True)
    df["song_id"] = df.index

    print(f"[2/6] Chunking {len(df):,} songs into stanza-based lyric chunks ...")
    records = []
    for row in df.itertuples(index=False):
        for chunk in chunk_lyrics(row.text):
            records.append(
                {
                    "song_id": int(row.song_id),
                    "artist": row.artist,
                    "song": row.song,
                    "link": row.link,
                    "chunk_text": chunk,
                }
            )
    print(f"      -> {len(records):,} chunks ({len(records) / len(df):.1f} chunks/song avg)")

    print(f"[3/6] Loading sentence-transformer model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    assert model.get_sentence_embedding_dimension() == EMBEDDING_DIM, (
        f"Model '{MODEL_NAME}' produced "
        f"{model.get_sentence_embedding_dimension()}-dim embeddings, "
        f"but EMBEDDING_DIM is set to {EMBEDDING_DIM}. Update EMBEDDING_DIM to match."
    )

    print(f"[4/6] Encoding {len(records):,} chunks (batch_size={ENCODE_BATCH_SIZE}) ...")

    texts = [r["chunk_text"] for r in records]
    total_chunks = len(texts)

    # Check if an old checkpoint exists
    if (
        EMBEDDING_CHECKPOINT_PATH.exists()
        and PROGRESS_CHECKPOINT_PATH.exists()
    ):
        completed_chunks = int(
            PROGRESS_CHECKPOINT_PATH.read_text().strip()
        )

        print(
            f"      Checkpoint found: "
            f"{completed_chunks:,}/{total_chunks:,} chunks completed"
        )
        print("      Resuming encoding...")

        embeddings = np.memmap(
            EMBEDDING_CHECKPOINT_PATH,
            dtype=np.float32,
            mode="r+",
            shape=(total_chunks, EMBEDDING_DIM),
        )

    else:
        completed_chunks = 0

        print("      No checkpoint found. Starting from beginning.")

        embeddings = np.memmap(
            EMBEDDING_CHECKPOINT_PATH,
            dtype=np.float32,
            mode="w+",
            shape=(total_chunks, EMBEDDING_DIM),
        )


    # 128 chunks/batch × 100 batches = 12,800 chunks/checkpoint
    checkpoint_size = ENCODE_BATCH_SIZE * CHECKPOINT_BATCHES

    for start in range(
        completed_chunks,
        total_chunks,
        checkpoint_size,
    ):
        end = min(
            start + checkpoint_size,
            total_chunks,
        )

        print(
            f"\n      Encoding chunks "
            f"{start:,} -> {end:,} "
            f"of {total_chunks:,}"
        )

        block_embeddings = model.encode(
            texts[start:end],
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Save embeddings into checkpoint
        embeddings[start:end] = block_embeddings
        embeddings.flush()

        # Save current progress
        PROGRESS_CHECKPOINT_PATH.write_text(str(end))

        print(
            f"      Checkpoint saved: "
            f"{end:,}/{total_chunks:,} chunks"
        )

    print("      All embeddings encoded.")

    print(f"[5/6] Building Milvus collection '{collection}' at {db_path} ...")
    client = MilvusClient(db_path)
    if client.has_collection(collection):
        # Dropping + recreating means switching EMBEDDING_DIM (e.g. after a
        # model change) is handled automatically -- no separate migration
        # step needed.
        client.drop_collection(collection)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",       # approximate nearest-neighbor: see evaluate.py
        metric_type="COSINE",    # embeddings are L2-normalized -> cosine similarity
        params={"M": 16, "efConstruction": 64},
    )
    client.create_collection(
        collection_name=collection,
        dimension=EMBEDDING_DIM,
        index_params=index_params,
    )

    rows = [
        {
            "id": i,
            "vector": embeddings[i].tolist(),
            "song_id": r["song_id"],
            "artist": r["artist"],
            "song": r["song"],
            "link": r["link"],
            "chunk_text": r["chunk_text"],
        }
        for i, r in enumerate(records)
    ]
    for start in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[start : start + INSERT_BATCH_SIZE]
        client.insert(collection_name=collection, data=batch)
        print(f"      inserted {min(start + INSERT_BATCH_SIZE, len(rows)):,} / {len(rows):,}")

    client.close()

    if save_eval_cache:
        print(f"[6/6] Saving exact-search evaluation cache to {EVAL_CACHE_PATH} ...")
        ids = np.arange(len(records), dtype=np.int64)
        song_ids = np.array([r["song_id"] for r in records], dtype=np.int64)
        np.savez(EVAL_CACHE_PATH, ids=ids, song_ids=song_ids, vectors=embeddings)
    else:
        print("[6/6] Skipping evaluation cache (save_eval_cache=False)")

    print(
        f"Done in {time.time() - t0:.1f}s | {len(df):,} songs | "
        f"{len(records):,} chunks indexed in Milvus"
    )


# --------------------------------------------------------------------------
# Query-time search
# --------------------------------------------------------------------------

class SemanticSearchEngine:
    """Loads the model + connects to the persisted Milvus Lite db once,
    then serves chunk-level ANN search aggregated back up to song level."""

    def __init__(self, db_path: str = DB_PATH, collection: str = COLLECTION):
        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"No Milvus index found at {db_path}. Run `python build_index.py` first."
            )
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = MilvusClient(db_path)
        self.collection = collection
        self.client.load_collection(collection_name=self.collection)

    def search(self, query: str, top_k: int = 10, candidate_chunks: int = 200) -> list[dict]:
        if not query or not query.strip():
            return []

        query_vec = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()

        hits = self.client.search(
            collection_name=self.collection,
            data=[query_vec],
            limit=candidate_chunks,          # search wide over CHUNKS ...
            search_params={"params": {"ef": 96}},
            output_fields=["song_id", "artist", "song", "link", "chunk_text"],
        )[0]

        # ... then collapse to each song's single best-matching chunk.
        best_per_song: dict[int, dict] = {}
        for hit in hits:
            entity = hit["entity"]
            song_id = entity["song_id"]
            score = float(hit["distance"])  # COSINE metric -> higher is more similar
            if song_id not in best_per_song or score > best_per_song[song_id]["score"]:
                best_per_song[song_id] = {
                    "artist": entity["artist"],
                    "song": entity["song"],
                    "link": entity["link"],
                    "score": score,
                    "excerpt": entity["chunk_text"],
                }

        ranked = sorted(best_per_song.values(), key=lambda r: -r["score"])
        return ranked[:top_k]

    def close(self):
        self.client.close()
