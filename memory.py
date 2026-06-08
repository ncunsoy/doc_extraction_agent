"""
memory.py
Cross-session learning.

Saves successfully resolved questions to JSON. When a new question arrives, finds the most
similar past examples via embedding similarity and passes them to the Reformer as few-shot
examples. Shares the Gemini-based _embed from vector_store to keep a single embedding mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vector_store import _embed


class Memory:
    def __init__(self, path: str | Path = "memory.json"):
        self.path    = Path(path)
        self.records: list[dict] = self._load()

    def add(self, question: str, clean_query: str, modality: str) -> None:
        """Persist a successfully resolved question to memory."""
        self.records.append({
            "question":    question,
            "clean_query": clean_query,
            "modality":    modality,
        })
        self._save()

    def recall(self, question: str, k: int = 2) -> list[dict]:
        """Return the k most similar past questions based on embedding similarity."""
        if not self.records:
            return []
        questions = [r["question"] for r in self.records]
        vectors   = _embed(questions)
        query_vec = _embed([question])[0]
        scores    = np.dot(vectors, query_vec)
        top_idx   = np.argsort(scores)[::-1][:k]
        return [self.records[i] for i in top_idx]

    def _load(self) -> list[dict]:
        """Load records from disk, returning an empty list on failure."""
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self) -> None:
        """Write current records to disk."""
        self.path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
