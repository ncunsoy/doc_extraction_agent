"""
validator.py
Üretilen cevabın belgeye bağlılığını (grounding) ve alt soruların karşılanıp karşılanmadığını (coverage) denetler.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from google import genai
from dotenv import load_dotenv

from preprocessor import DocumentChunk
from agents.reformer import ReformerOutput

MODEL_NAME = "gemini-2.5-flash"
load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
    )


@dataclass
class ValidationResult:
    verdict:      bool         # True=pass, False=fail
    failure_type: str = ""     # "grounding" - "coverage" - ""
    reason:       str = ""


SYSTEM_PROMPT = """
You are an answer validation agent. You will check two things:
1. GROUNDING: Is every claim in the answer actually present in the provided document chunks?
    - If the answer contains information not found in the document → grounding error
2. COVERAGE: Was every sub-question substantively answered?
    - A sub-question is NOT answered if the answer says "not found", "not in the document", "cannot be determined", or gives no relevant information for that sub-question.
    - If any sub-question is unanswered or deflected → coverage error
Return your response ONLY in the following JSON format:
{
  "verdict": true or false,
  "failure_type": "grounding" or "coverage" or "",
  "reason": "short explanation"
}
"""


class ValidatorAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def run(
        self,
        question: str,
        reformer_out: ReformerOutput,
        chunks: list[DocumentChunk],
        answer_draft: str,
    ) -> ValidationResult:
        prompt = self._build_prompt(question, reformer_out, chunks, answer_draft)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        return self._parse(response.text)

    # Validator'ın anlayabileceği şekilde prompt'u düzenle
    def _build_prompt(
        self,
        question: str,
        reformer_out: ReformerOutput,
        chunks: list[DocumentChunk],
        answer_draft: str,
    ) -> str:
        context = "\n\n---\n\n".join(
            f"[Page {c.metadata.get('page', '?')}]\n{c.content}"
            for c in chunks
        )
        sub_queries = "\n".join(f"- {q}" for q in reformer_out.sub_queries) \
                      or f"- {reformer_out.clean_query}"

        return (
            f"Question: {question}\n\n"
            f"Sub-questions:\n{sub_queries}\n\n"
            f"Document chunks:\n{context}\n\n"
            f"Generated answer:\n{answer_draft}"
        )

    def _parse(self, text: str) -> ValidationResult:
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text.strip())
            return ValidationResult(
                verdict      = bool(data.get("verdict", False)),
                failure_type = data.get("failure_type", ""),
                reason       = data.get("reason", ""),
            )
        except Exception:
            # If parsing fails, return a safe failure
            return ValidationResult(verdict=False, failure_type="grounding", reason="Parse error")