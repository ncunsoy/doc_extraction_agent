import sys
import pickle
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessor import DocumentChunk
from vector_store import VectorStore

DIM = 8


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic fake embeddings: L2-normalized sequential floats."""
    vecs = []
    for i, text in enumerate(texts):
        v = np.full(DIM, float(i + 1), dtype="float32")
        v /= np.linalg.norm(v)
        vecs.append(v)
    return np.array(vecs, dtype="float32")


def make_chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        content=content,
        metadata={"source": "test"},
    )


# add() indexes chunks without error
print("[TEST 1] add() indexes chunks")
with patch("vector_store._embed", side_effect=_fake_embed), \
     tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    chunks = [make_chunk(f"c{i}", f"text {i}") for i in range(3)]
    vs.add(chunks)
    assert vs.index is not None
    assert vs.index.ntotal == 3
    assert len(vs.chunks) == 3
    print(f"  index.ntotal={vs.index.ntotal}  len(chunks)={len(vs.chunks)}")
    print(f"  PASS\n")

# add() on empty list is a no-op
print("[TEST 2] add() with empty list is no-op")
with patch("vector_store._embed", side_effect=_fake_embed), \
     tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    vs.add([])
    assert vs.index is None
    assert vs.chunks == []
    print(f"  index={vs.index}  chunks={vs.chunks}")
    print(f"  PASS\n")

# save() + load() round-trip preserves chunks and index
print("[TEST 3] save() and load() round-trip")
with patch("vector_store._embed", side_effect=_fake_embed), \
     tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    chunks = [make_chunk("a", "hello"), make_chunk("b", "world")]
    vs.add(chunks)
    vs.save("testdoc")
    assert (Path(tmp) / "testdoc.faiss").exists()
    assert (Path(tmp) / "testdoc.pkl").exists()

    vs2 = VectorStore(store_dir=tmp)
    ok = vs2.load("testdoc")
    assert ok is True
    assert vs2.index.ntotal == 2
    assert [c.id for c in vs2.chunks] == ["a", "b"]
    print(f"  load() returned {ok}  ntotal={vs2.index.ntotal}  ids={[c.id for c in vs2.chunks]}")
    print(f"  PASS\n")

# load() returns False when no files exist
print("[TEST 4] load() returns False when no saved store")
with tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    ok = vs.load("missing_doc")
    assert ok is False
    print(f"  load() returned {ok}")
    print(f"  PASS\n")

# search() returns the right chunk via cosine similarity
print("[TEST 5] search() returns nearest chunk")
call_count = [0]

def embed_for_search(texts: list[str]) -> np.ndarray:
    vecs = []
    for text in texts:
        if text == "apple":
            v = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype="float32")
        elif text == "orange":
            v = np.array([0, 1.0, 0, 0, 0, 0, 0, 0], dtype="float32")
        else:
            v = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype="float32")  # query matches apple
        vecs.append(v)
    return np.array(vecs, dtype="float32")

with patch("vector_store._embed", side_effect=embed_for_search), \
     tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    apple  = make_chunk("apple_chunk", "apple")
    orange = make_chunk("orange_chunk", "orange")
    vs.add([apple, orange])
    results = vs.search("query about apple", top_k=1)
    assert len(results) == 1
    assert results[0].id == "apple_chunk", f"Expected apple_chunk, got {results[0].id}"
    print(f"  top-1 result: {results[0].id}")
    print(f"  PASS\n")

# search() on empty store returns []
print("[TEST 6] search() on empty index returns []")
with patch("vector_store._embed", side_effect=_fake_embed), \
     tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    results = vs.search("query", top_k=3)
    assert results == []
    print(f"  results={results}")
    print(f"  PASS\n")

# add() accumulates across multiple calls
print("[TEST 7] add() accumulates across multiple calls")
call_n = [0]
def counting_embed(texts: list[str]) -> np.ndarray:
    vecs = []
    for text in texts:
        v = np.full(DIM, float(call_n[0] + 1), dtype="float32")
        v /= np.linalg.norm(v)
        call_n[0] += 1
        vecs.append(v)
    return np.array(vecs, dtype="float32")

with patch("vector_store._embed", side_effect=counting_embed), \
     tempfile.TemporaryDirectory() as tmp:
    vs = VectorStore(store_dir=tmp)
    vs.add([make_chunk("x1", "first")])
    vs.add([make_chunk("x2", "second"), make_chunk("x3", "third")])
    assert vs.index.ntotal == 3
    assert len(vs.chunks) == 3
    print(f"  ntotal={vs.index.ntotal}  len(chunks)={len(vs.chunks)}")
    print(f"  PASS\n")

print("All tests passed.")
