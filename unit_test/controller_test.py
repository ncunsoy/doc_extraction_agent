from vector_store import VectorStore
from agents.reformer import ReformerAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from controller import Controller

PDF_PATH = "D:\\Desktop\\doc_extraction_agent\\test_examples\\attention_is_all_you_need.pdf"
DOC_ID   = "attention"

# --- LOAD VECTOR STORE ---
store = VectorStore(store_dir="store")
if not store.load(DOC_ID):
    print("Index not found. Run preprocessing_test.py first.")
    exit(1)
print(f"Index loaded: {store.index.ntotal} chunks\n")

# --- BUILD CONTROLLER ---
controller = Controller(
    reformer  = ReformerAgent(),
    retriever = RetrieverAgent(vector_store=store, pdf_path=PDF_PATH),
    validator = ValidatorAgent(),
    max_iter  = 3,
)

# --- TEST 1: Normal question ---
print("=" * 60)
print("TEST 1: Normal question (expect solved=True, 1 iteration)")

result = controller.run("How many encoder layers does the Transformer model have?")
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer}")

assert result.solved, "Expected solved=True"
assert result.iterations >= 1

# --- TEST 2: Out-of-scope question ---
print("\n" + "=" * 60)
print("TEST 2: Out-of-scope question (expect no info or solved=False)")

result = controller.run("What is the GDP of Turkey in 2023?")
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer}")

not_found_phrases = ["not contain", "not found", "no information", "not available", "not mention", "not in the"]
answer_lower = result.answer.lower()

assert result.iterations <= controller.max_iter, "Controller exceeded max_iter"
if result.solved:
    assert any(p in answer_lower for p in not_found_phrases), \
        "If solved=True for out-of-scope question, answer must acknowledge info is not in document"

# --- TEST 3: Retry mechanism ---
print("\n" + "=" * 60)
print("TEST 3: Retry mechanism (expect no infinite loop)")

result = controller.run(
    "What is the exact BLEU score improvement on WMT 2014 English-to-French "
    "compared to all previous single models, and what was the training cost in FLOPs?"
)
print(f"  solved     : {result.solved}")
print(f"  iterations : {result.iterations}")
print(f"  answer     : {result.answer[:200]}")

assert result.iterations <= controller.max_iter, "Controller exceeded max_iter"

print("\n" + "=" * 60)
print("All controller tests completed.")
