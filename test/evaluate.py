"""
evaluate.py
Measures system accuracy against a ground truth test set.

Hybrid evaluation:
  1. Keyword match (fast, no API) — does the answer contain the expected keyword?
  2. LLM-judge (Gemini) — does the answer semantically match the expected answer?
"""

from __future__ import annotations

import os
from google import genai
from dotenv import load_dotenv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessor import DocumentPreprocessor
from vector_store import VectorStore
from agents.reformer import ReformerAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from controller import Controller

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
    )

TEST_EXAMPLES = Path(__file__).resolve().parent / "test_examples"
PDF_PATH      = TEST_EXAMPLES / "attention_is_all_you_need.pdf"
DOC_ID        = "attention"


# TEST SET
TEST_SET = [
    {
        "question": "How many encoder layers does the Transformer have?",
        "expected": "6 encoder layers",
        "keywords": ["six"],
    },
    {
        "question": "What is the dimension of the model (d_model)?",
        "expected": "512",
        "keywords": ["512"],
    },
    {
        "question": "What type of attention mechanism does the Transformer use?",
        "expected": "Scaled dot-product attention (and multi-head attention)",
        "keywords": ["scaled dot-product", "multi-head", "self-attention"],
    },
]


# EVALUATION METHODS
def keyword_match(answer: str, keywords: list[str]) -> bool:
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def llm_judge(question: str, expected: str, answer: str) -> bool:
    prompt = (
        f"Question: {question}\n"
        f"Expected answer: {expected}\n"
        f"System answer: {answer}\n\n"
        f"Does the system answer contain the same information as the expected answer? "
        f"Reply with only YES or NO."
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return "YES" in response.text.strip().upper()


def evaluate_one(question: str, expected: str, keywords: list[str], answer: str) -> tuple[bool, str]:
    if keyword_match(answer, keywords):
        return True, "keyword"
    if llm_judge(question, expected, answer):
        return True, "llm-judge"
    return False, "llm-judge"


# MAIN
def main():
    store = VectorStore()
    if not store.load(DOC_ID):
        print("Index not found, processing document...")
        pre = DocumentPreprocessor()
        chunks, _ = pre.preprocess_pdf(PDF_PATH)
        store.add(chunks)
        store.save(DOC_ID)

    controller = Controller(
        reformer  = ReformerAgent(),
        retriever = RetrieverAgent(vector_store=store, pdf_path=PDF_PATH),
        validator = ValidatorAgent(),
    )

    correct = 0
    print("=" * 60)
    print(f"EVALUATION — {len(TEST_SET)} questions")
    print("=" * 60)

    for i, case in enumerate(TEST_SET, 1):
        result = controller.run(case["question"])
        is_correct, method = evaluate_one(
            case["question"], case["expected"], case["keywords"], result.answer
        )
        correct += int(is_correct)

        status = "CORRECT" if is_correct else "WRONG"
        print(f"\n[{i}] {case['question']}")
        print(f"    Expected : {case['expected']}")
        print(f"    Answer   : {result.answer[:120]}")
        print(f"    Result   : {status}  ({method})")

    print("\n" )
    print(f"SCORE: {correct}/{len(TEST_SET)}  ({100 * correct / len(TEST_SET):.0f}%)\n")


if __name__ == "__main__":
    main()
