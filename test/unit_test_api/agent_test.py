import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.reformer import ReformerAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from vector_store import VectorStore

DOC_ID = "attention"

# LOAD VECTOR STORE
store  = VectorStore(store_dir="store")
loaded = store.load(DOC_ID)
if not loaded:
    print("Index not found. Run preprocessing_test.py first.")
    exit(1)
print(f"Index loaded: {store.index.ntotal} chunks\n")

# AGENTS
reformer  = ReformerAgent()
retriever = RetrieverAgent(vector_store=store)
validator = ValidatorAgent()

# NORMAL QUESTIONS
questions = [
    "How many attention layers are there?",
    "How many encoder layers are in the Transformer model and what does each contain?",
]

for question in questions:
    print(f"\nQUESTION: {question}")

    ref_out = reformer.run(question)
    print(f"\nReformer - clean_query : {ref_out.clean_query}")
    print(f"           sub_queries : {ref_out.sub_queries}")
    print(f"           modality    : {ref_out.modality}")

    ret_out = retriever.run(ref_out)
    print(f"\nRetriever - {len(ret_out.chunks)} chunks retrieved")
    print(f"\nANSWER:\n{ret_out.answer_draft}")

    val_out     = validator.run(
        question=question,
        reformer_out=ref_out,
        chunks=ret_out.chunks,
        answer_draft=ret_out.answer_draft,
    )
    verdict_str = "PASS" if val_out.verdict else "FAIL"
    print(f"\nValidator - {verdict_str}")
    if not val_out.verdict:
        print(f"  failure_type : {val_out.failure_type}")
        print(f"  reason       : {val_out.reason}")
    print()

# ANOTHER LANGUAGE
print("\n[TEST] Non-English question\n")
question_turkish = "Transformer'da kaç encoder katmanı vardır?"
ref_turkish      = reformer.run(question_turkish)
print(f"Original     : {question_turkish}")
print(f"clean_query  : {ref_turkish.clean_query}")
print(f"sub_queries  : {ref_turkish.sub_queries}")
ret_turkish      = retriever.run(ref_turkish)
val_turkish      = validator.run(
    question=question_turkish,
    reformer_out=ref_turkish,
    chunks=ret_turkish.chunks,
    answer_draft=ret_turkish.answer_draft,
)
print(f"Answer       : {ret_turkish.answer_draft[:120]}")
print(f"Verdict      : {'PASS' if val_turkish.verdict else 'FAIL'}")
if not val_turkish.verdict:
    print(f"failure_type : {val_turkish.failure_type}")
    print(f"reason       : {val_turkish.reason}")


# FAIL CASES
print("\nFAIL CASE TESTS\n")

# GROUNDING FAILURE
print("\n[TEST 1] Grounding failure - hallucinated answer")
ref_grounding = reformer.run("How many encoder layers does the Transformer have?")
ret_grounding = retriever.run(ref_grounding)
hallucinated_answer = (
    "The Transformer model uses 24 encoder layers, each containing "
    "a cross-attention mechanism and a convolutional layer. "
    "This was later confirmed by LeCun in 2019."
)
val_grounding = validator.run(
    question="How many encoder layers does the Transformer have?",
    reformer_out=ref_grounding,
    chunks=ret_grounding.chunks,
    answer_draft=hallucinated_answer,
)
print(f"Injected answer : {hallucinated_answer[:80]}...")
print(f"Verdict         : {'PASS' if val_grounding.verdict else 'FAIL'}")
print(f"failure_type    : {val_grounding.failure_type}")
print(f"reason          : {val_grounding.reason}")

# COVERAGE FAILURE
print("\n[TEST 2] Coverage failure - out-of-scope sub-question")
question_mixed  = "How many encoder layers does the Transformer have and who won the 2024 Nobel Prize in Physics?"
ref_coverage    = reformer.run(question_mixed)
ret_coverage    = retriever.run(ref_coverage)
val_coverage    = validator.run(
    question=question_mixed,
    reformer_out=ref_coverage,
    chunks=ret_coverage.chunks,
    answer_draft=ret_coverage.answer_draft,
)
print(f"Sub-queries  : {ref_coverage.sub_queries}")
print(f"Verdict      : {'PASS' if val_coverage.verdict else 'FAIL'}")
print(f"failure_type : {val_coverage.failure_type}")
print(f"reason       : {val_coverage.reason}")



