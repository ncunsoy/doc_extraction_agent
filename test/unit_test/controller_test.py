from unittest.mock import MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from controller import Controller
from agents.reformer import ReformerOutput
from agents.validator import ValidationResult
from agents.retriever import RetrieverOutput

REFORMER_OUT = ReformerOutput(clean_query="clean question", sub_queries=["clean question"], modality="text")
RETRIEVER_OUT = RetrieverOutput(chunks=[], answer_draft="draft answer")


# NORMAL QUESTION
print("[TEST 1] First Attempt PASS")
reformer  = MagicMock()
retriever = MagicMock()
validator = MagicMock()
reformer.run.return_value  = REFORMER_OUT
retriever.run.return_value = RETRIEVER_OUT
validator.run.return_value = ValidationResult(verdict=True)

ctrl   = Controller(reformer, retriever, validator, max_iter=3)
result = ctrl.run("question")
assert result.solved is True, "solved must be True"
assert result.iterations == 1, f"Expected 1 iteration, got: {result.iterations}"
assert reformer.run.call_count == 1
assert retriever.run.call_count == 1
print(f"  iterations : {result.iterations}")
print(f"  PASS\n")

# GROUNDING ERROR - REFORMER RETRY
print("[TEST 2] Grounding Error - Reformer Retry")
reformer  = MagicMock()
retriever = MagicMock()
validator = MagicMock()
reformer.run.return_value  = REFORMER_OUT
retriever.run.return_value = RETRIEVER_OUT
validator.run.side_effect  = [
    ValidationResult(verdict=False, failure_type="grounding", reason="x"),
    ValidationResult(verdict=True),
]

ctrl   = Controller(reformer, retriever, validator, max_iter=3)
result = ctrl.run("question")
assert result.solved is True
assert reformer.run.call_count == 2, f"Reformer must be called 2 times, got: {reformer.run.call_count}"
print(f"  reformer.call_count : {reformer.run.call_count}")
print(f"  PASS\n")

# COVERAGE ERROR - RETRIEVER RETRY, NO REFORMER RETRY
print("[TEST 3] Coverage Error - Retriever Retry, No Reformer Retry")
reformer  = MagicMock()
retriever = MagicMock()
validator = MagicMock()
reformer.run.return_value  = REFORMER_OUT
retriever.run.return_value = RETRIEVER_OUT
validator.run.side_effect  = [
    ValidationResult(verdict=False, failure_type="coverage", reason="x"),
    ValidationResult(verdict=True),
]

ctrl   = Controller(reformer, retriever, validator, max_iter=3)
result = ctrl.run("question")
assert result.solved is True
assert reformer.run.call_count == 1, f"Reformer must be called only once, got: {reformer.run.call_count}"
assert retriever.run.call_count == 2, f"Retriever must be called 2 times, got: {retriever.run.call_count}"
print(f"  reformer.call_count  : {reformer.run.call_count}")
print(f"  retriever.call_count : {retriever.run.call_count}")
print(f"  PASS\n")

# MAX_ITER
print("[TEST 4] Max Iteration Limit")
reformer  = MagicMock()
retriever = MagicMock()
validator = MagicMock()
reformer.run.return_value  = REFORMER_OUT
retriever.run.return_value = RETRIEVER_OUT
validator.run.side_effect  = [
    ValidationResult(verdict=False, failure_type="grounding", reason="x"),
    ValidationResult(verdict=False, failure_type="grounding", reason="x"),
    ValidationResult(verdict=False, failure_type="grounding", reason="x"),
]

ctrl   = Controller(reformer, retriever, validator, max_iter=3)
result = ctrl.run("question")
assert result.solved is False, "solved must be False"
assert result.iterations == 3, f"Expected 3 iterations, got: {result.iterations}"
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer[:60]}...")
print(f"  PASS\n")

print("All tests passed.")
