"""
vector_store.py
Chunk'ları embed eder, FAISS'e yazar ve arama yapar.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import faiss
import numpy as np
from google import genai
from dotenv import load_dotenv
import os

from preprocessor import DocumentChunk

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
    )
EMBEDDING_MODEL = "gemini-embedding-001"


def _embed(texts: list[str]) -> np.ndarray:
    # Gemini API tek seferde max 100 metin kabul ettiğinden
    all_vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=batch)
        all_vectors.extend([e.values for e in result.embeddings])
    vectors = np.array(all_vectors, dtype="float32")
    # cosine için normalize et
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class VectorStore:
    def __init__(self, store_dir: str | Path = "store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)

        self.index: faiss.Index | None = None
        self.chunks: list[DocumentChunk] = []

    # Chunk'ları embed edip index'e ekler.
    def add(self, chunks: list[DocumentChunk]) -> None:
        texts = [c.content for c in chunks]
        vectors = _embed(texts)

        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product (normalized = cosine)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    # Index ve chunk listesini diske yazar.
    def save(self, doc_id: str) -> None:
        faiss.write_index(self.index, str(self.store_dir / f"{doc_id}.faiss"))
        with open(self.store_dir / f"{doc_id}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    # Daha önce kaydedilmiş index'i yükler. Başarılıysa True döner.
    def load(self, doc_id: str) -> bool:
        index_path = self.store_dir / f"{doc_id}.faiss"
        chunks_path = self.store_dir / f"{doc_id}.pkl"

        if not index_path.exists() or not chunks_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        return True

    # Soru cümlesini embed edip index'te arama yapar, en ilgili top_k chunk'ı döndürür.
    def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        if self.index is None or self.index.ntotal == 0:
            return []

        vector = _embed([query])

        scores, indices = self.index.search(vector, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunks[idx])
        return results