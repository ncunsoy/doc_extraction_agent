"""
vector_store.py
Embeds chunks via Gemini, stores them in a FAISS index, and retrieves by similarity.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

from preprocessor import DocumentChunk

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
EMBEDDING_MODEL = "gemini-embedding-001"


def _embed(texts: list[str]) -> np.ndarray:
    """Embed a list of texts and return L2-normalized vectors (cosine-ready)."""
    all_vectors = []
    for i in range(0, len(texts), 100):
        batch  = texts[i : i + 100]
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        all_vectors.extend([e.values for e in result.embeddings])
    vectors = np.array(all_vectors, dtype="float32")
    norms   = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class VectorStore:
    def __init__(self, store_dir: str | Path = "store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self.index:  faiss.Index | None    = None
        self.chunks: list[DocumentChunk]   = []

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Embed chunks and add them to the FAISS index."""
        if not chunks:
            return
        vectors = _embed([c.content for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def save(self, doc_id: str) -> None:
        """Persist the index and chunk list to disk."""
        faiss.write_index(self.index, str(self.store_dir / f"{doc_id}.faiss"))
        with open(self.store_dir / f"{doc_id}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, doc_id: str) -> bool:
        """Load a previously saved index. Returns True on success."""
        index_path  = self.store_dir / f"{doc_id}.faiss"
        chunks_path = self.store_dir / f"{doc_id}.pkl"
        if not index_path.exists() or not chunks_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        return True

    def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """Return the top-k most similar chunks to the query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        vector          = _embed([query])
        scores, indices = self.index.search(vector, top_k)
        return [self.chunks[idx] for idx in indices[0] if idx != -1]
