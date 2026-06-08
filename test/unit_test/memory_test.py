import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from memory import Memory

# ADD AND PERSISTENCE
print("[TEST 1] add() saves to disk")
with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(path=Path(tmp) / "memory.json")
    mem.add("How many encoder layers?", "encoder layer count", "text")
    mem.add("What is d_model?", "d_model dimension", "text")

    mem2 = Memory(path=Path(tmp) / "memory.json")
    assert len(mem2.records) == 2, f"Expected 2 records, got: {len(mem2.records)}"
    print(f"  {len(mem2.records)} records persisted")
    print(f"  PASS\n")

# RECALL SIMILARITY
print("[TEST 2] recall() returns semantically similar questions")
with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(path=Path(tmp) / "memory.json")
    mem.add("How many encoder layers does the Transformer have?", "encoder layer count", "text")
    mem.add("What is the learning rate used in training?", "learning rate", "text")
    mem.add("What optimizer was used?", "optimizer name", "text")

    results = mem.recall("How many layers are in the encoder?", k=1)
    assert len(results) == 1, f"Expected 1 result, got: {len(results)}"
    assert "encoder" in results[0]["question"].lower(), \
        f"Top result should be encoder-related, got: {results[0]['question']}"
    print(f"  Query   : 'How many layers are in the encoder?'")
    print(f"  Recalled: '{results[0]['question']}'")
    print(f"  PASS\n")

# EMPTY MEMORY
print("[TEST 3] recall() on empty memory returns []")
with tempfile.TemporaryDirectory() as tmp:
    mem     = Memory(path=Path(tmp) / "memory.json")
    results = mem.recall("anything")
    assert results == [], f"Expected [], got: {results}"
    print(f"  PASS\n")

# K LIMIT
print("[TEST 4] recall() respects k limit")
with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(path=Path(tmp) / "memory.json")
    for i in range(5):
        mem.add(f"Question {i}", f"query {i}", "text")

    results = mem.recall("Question 0", k=2)
    assert len(results) == 2, f"Expected 2 results, got: {len(results)}"
    print(f"  k=2 - {len(results)} results returned")
    print(f"  PASS\n")

print("All tests passed.")
