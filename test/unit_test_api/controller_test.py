import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.reformer import ReformerAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from controller import Controller
from vector_store import VectorStore

PDF_PATH = Path(__file__).resolve().parents[1] / "test_examples" / "attention_is_all_you_need.pdf"
DOC_ID   = "attention"

# LOAD VECTOR STORE
store = VectorStore(store_dir="store")
if not store.load(DOC_ID):
    print("Index not found. Run preprocessing_test.py first.")
    exit(1)
print(f"Index loaded: {store.index.ntotal} chunks\n")

# BUILD CONTROLLER
controller = Controller(
    reformer  = ReformerAgent(),
    retriever = RetrieverAgent(vector_store=store, pdf_path=PDF_PATH),
    validator = ValidatorAgent(),
    max_iter  = 3,
)

# NORMAL QUESTION
print("TEST 1: Normal question (expect solved=True)")
result = controller.run("How many encoder layers does the Transformer model have?")
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer}\n")
assert result.solved, "Expected solved=True"
assert result.iterations >= 1

# OUT-OF-SCOPE QUESTION
print("TEST 2: Out-of-scope question")
result            = controller.run("What is the GDP of Turkey in 2023?")
answer_lower      = result.answer.lower()
not_found_phrases = ["not contain", "not found", "no information", "not available", "not mention", "not in the"]
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer}\n")
assert result.iterations <= controller.max_iter, "Controller exceeded max_iter"
if result.solved:
    assert any(p in answer_lower for p in not_found_phrases), \
        "If solved=True for out-of-scope question, answer must acknowledge missing info"

# RETRY MECHANISM
print("TEST 3: Retry mechanism (expect no infinite loop)")
result = controller.run(
    "What is the exact BLEU score improvement on WMT 2014 English-to-French "
    "compared to all previous single models, and what was the training cost in FLOPs?"
)
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer[:200]}\n")
assert result.iterations <= controller.max_iter, "Controller exceeded max_iter"

print("All controller tests completed.")
