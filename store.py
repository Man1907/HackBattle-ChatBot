"""
store.py — pluggable vector store
==================================

One interface, three backends. Swap with the VECTOR_STORE env var; nothing
else in the app changes.

    VECTOR_STORE=memory    (default) NumPy array, rebuilt at startup.
                           Zero setup. Right for a few hundred chunks.
    VECTOR_STORE=chroma    Persistent local directory (or a Chroma server).
                           Right for thousands to low millions of chunks.
    VECTOR_STORE=faiss     FAISS index + JSON metadata sidecar. Fastest at
                           large scale; the sidecar is the trade-off.
    VECTOR_STORE=pgvector  Postgres + the pgvector extension.
                           Right when vectors live beside relational data.

Interface:
    store.upsert(ids, texts, embeddings, metadatas)
    store.query(embedding, k) -> list[(id, text, metadata, score)]
    store.count()

Scores are cosine similarity in [-1, 1], higher is better, for every backend.

Install per backend:
    chroma    pip install chromadb
    pgvector  pip install "psycopg[binary]" pgvector
"""

from __future__ import annotations

import os
import numpy as np

BACKEND = os.environ.get("VECTOR_STORE", "memory").lower()
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_store")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "hackbattle")
FAISS_DIR = os.environ.get("FAISS_DIR", "./faiss_store")
PG_DSN = os.environ.get("PG_DSN", "")          # e.g. postgresql://user:pw@host/db
PG_TABLE = os.environ.get("PG_TABLE", "chunks")


class BaseStore:
    def upsert(self, ids, texts, embeddings, metadatas): ...
    def query(self, embedding, k: int): ...
    def count(self) -> int: ...


# ---------------------------------------------------------------------------
class MemoryStore(BaseStore):
    """NumPy array + dot product. No dependency, no persistence."""

    def __init__(self):
        self.ids, self.texts, self.metas = [], [], []
        self.emb = None

    def upsert(self, ids, texts, embeddings, metadatas):
        e = np.asarray(embeddings, dtype="float32")
        self.ids, self.texts, self.metas = list(ids), list(texts), list(metadatas)
        self.emb = e                       # assumed L2-normalised by the caller

    def query(self, embedding, k: int):
        if self.emb is None or not len(self.ids):
            return []
        q = np.asarray(embedding, dtype="float32").ravel()
        sims = self.emb @ q                # cosine, vectors are unit-norm
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], self.texts[i], self.metas[i], float(sims[i]))
                for i in order]

    def count(self):
        return len(self.ids)


# ---------------------------------------------------------------------------
class ChromaStore(BaseStore):
    """Persistent local Chroma. Set CHROMA_DIR to control where it lives."""

    def __init__(self, path: str = CHROMA_DIR, collection: str = CHROMA_COLLECTION):
        import chromadb
        # Chroma requires 3-512 chars, alphanumeric start/end.
        if len(collection) < 3:
            collection = f"col-{collection}"
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})

    def upsert(self, ids, texts, embeddings, metadatas):
        # upsert (not add) so re-running the ingest updates rather than errors
        self.col.upsert(ids=list(ids), documents=list(texts),
                        embeddings=np.asarray(embeddings, "float32").tolist(),
                        metadatas=list(metadatas))

    def query(self, embedding, k: int):
        k = max(1, min(k, max(1, self.count())))
        res = self.col.query(
            query_embeddings=[np.asarray(embedding, "float32").ravel().tolist()],
            n_results=k, include=["documents", "metadatas", "distances"])
        out = []
        for i, _id in enumerate(res["ids"][0]):
            dist = res["distances"][0][i]      # cosine distance = 1 - similarity
            out.append((_id, res["documents"][0][i], res["metadatas"][0][i],
                        float(1.0 - dist)))
        return out

    def count(self):
        return self.col.count()


# ---------------------------------------------------------------------------
class PgVectorStore(BaseStore):
    """Postgres + pgvector. Requires PG_DSN and the vector extension."""

    def __init__(self, dsn: str = PG_DSN, table: str = PG_TABLE, dim: int = 384):
        if not dsn:
            raise RuntimeError("PG_DSN is not set")
        import psycopg
        from pgvector.psycopg import register_vector
        self.table = table
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id        text PRIMARY KEY,
                content   text NOT NULL,
                metadata  jsonb DEFAULT '{{}}'::jsonb,
                embedding vector({dim})
            )""")
        # Approximate-NN index. Cosine ops match our normalised vectors.
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {table}_emb_idx ON {table}
            USING hnsw (embedding vector_cosine_ops)""")

    def upsert(self, ids, texts, embeddings, metadatas):
        import json
        embs = np.asarray(embeddings, "float32")
        with self.conn.cursor() as cur:
            for _id, text, emb, meta in zip(ids, texts, embs, metadatas):
                cur.execute(
                    f"""INSERT INTO {self.table} (id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                          content = EXCLUDED.content,
                          metadata = EXCLUDED.metadata,
                          embedding = EXCLUDED.embedding""",
                    (str(_id), text, json.dumps(meta), np.asarray(emb)))

    def query(self, embedding, k: int):
        q = np.asarray(embedding, "float32").ravel()
        with self.conn.cursor() as cur:
            cur.execute(
                f"""SELECT id, content, metadata, 1 - (embedding <=> %s) AS score
                    FROM {self.table}
                    ORDER BY embedding <=> %s
                    LIMIT %s""", (q, q, k))
            return [(r[0], r[1], r[2] or {}, float(r[3])) for r in cur.fetchall()]

    def count(self):
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.table}")
            return int(cur.fetchone()[0])


class FaissStore(BaseStore):
    """FAISS index plus a JSON sidecar.

    FAISS stores ONLY vectors — no ids, text, or metadata — so everything else
    lives in a parallel sidecar file keyed by row position. That extra moving
    part is the main cost of choosing FAISS over Chroma at small scale; it pays
    off at hundreds of thousands of vectors, where FAISS is markedly faster.

    Uses IndexFlatIP: exact inner product, which equals cosine similarity on
    the L2-normalised vectors this project produces.
    """

    def __init__(self, path: str = FAISS_DIR):
        import faiss
        self.faiss = faiss
        self.dir = path
        os.makedirs(path, exist_ok=True)
        self.index_path = os.path.join(path, "index.faiss")
        self.meta_path = os.path.join(path, "meta.json")
        self.index = None
        self.ids, self.texts, self.metas = [], [], []
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            import json
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, encoding="utf-8") as f:
                side = json.load(f)
            self.ids = side["ids"]
            self.texts = side["texts"]
            self.metas = side["metas"]

    def upsert(self, ids, texts, embeddings, metadatas):
        import json
        embs = np.asarray(embeddings, dtype="float32")
        # Rebuild wholesale: the corpus is small and always written in full,
        # so incremental updates would add complexity for no benefit.
        self.index = self.faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.ids, self.texts, self.metas = list(ids), list(texts), list(metadatas)
        self.faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "texts": self.texts,
                       "metas": self.metas}, f)

    def query(self, embedding, k: int):
        if self.index is None or not self.ids:
            return []
        q = np.asarray(embedding, dtype="float32").reshape(1, -1)
        k = max(1, min(k, len(self.ids)))
        scores, idx = self.index.search(q, k)
        out = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0:
                continue
            # inner product on unit vectors IS cosine similarity, so this is
            # already on the same scale the other backends return
            out.append((self.ids[i], self.texts[i], self.metas[i], float(score)))
        return out

    def count(self):
        return len(self.ids)


# ---------------------------------------------------------------------------
def get_store(backend: str | None = None, **kwargs) -> BaseStore:
    b = (backend or BACKEND).lower()
    if b == "chroma":
        return ChromaStore(**kwargs)
    if b == "faiss":
        return FaissStore(**kwargs)
    if b in ("pgvector", "postgres", "pg"):
        return PgVectorStore(**kwargs)
    return MemoryStore()
