"""
reformer.py
Ham kullanıcı sorusunu temizler, sub-query'lere böler ve modalite tahmini yapar.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
    )

MODEL_NAME = "gemini-2.5-flash"

@dataclass
class ReformerOutput:
    clean_query:  str
    sub_queries:  list[str] = field(default_factory=list)
    modality:     str = "text"   # "text" - "visual" - "both"
    reasoning:    str = ""


SYSTEM_PROMPT = """
You are a question analysis agent. You will analyze the user's question and produce a response in JSON format.

Your tasks:
1. Clean the question from unnecessary noise (conversational language, meta expressions)
2. Write clean_query and sub_queries in the SAME LANGUAGE as the document (if the document is in English, use English; if Turkish, use Turkish) — regardless of the language of the user's question.
3. If it's a multi-step question, break it into sub-queries; otherwise, create a single-element list
4. Estimate where the answer is located in the document:
   - "text": plain text, paragraph
   - "visual": graph, table, figure
   - "both": could be in both

Provide your response ONLY in the following JSON format, nothing else:
{
  "clean_query": "cleaned question",
  "sub_queries": ["sub-question 1", "sub-question 2"],
  "modality": "text",
  "reasoning": "why this modality"
}
"""


class ReformerAgent:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def run(
        self,
        question: str,
        outline: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> ReformerOutput:
        """Soruyu reformüle eder.
        outline: belge yapısı (opsiyonel, modalite kararına yardımcı olur)
        history: önceki başarısız denemeler (few-shot için)
        """

        prompt = self._build_prompt(question, outline, history)
        response = None
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"system_instruction": SYSTEM_PROMPT},
            )
        
        except Exception as e:
            print(f"Error during reformulation: {e}")
            return ReformerOutput(
                clean_query=question,
                sub_queries=[question],
                modality="text",
                reasoning="Reformulation failed due to an error."
            )
        return self._parse(response.text)

    # Prompt'u oluşturur. Outline ve history varsa ekler.
    def _build_prompt(
        self,
        question: str,
        outline: list[dict] | None,
        history: list[dict] | None,
    ) -> str:
        parts = []

        if outline:
            titles = [f"- {n['title']}" for n in outline[:10]]  
            parts.append("Belge bölümleri:\n" + "\n".join(titles))

        if history:
            parts.append("Önceki başarısız denemeler:")
            for h in history:
                parts.append(f"  - Sorgu: '{h['clean_query']}' → Hata: {h['reason']}")

        parts.append(f"Soru: {question}")
        return "\n\n".join(parts)

    # Gemini'nin JSON çıktısını parse eder.
    def _parse(self, text: str) -> ReformerOutput:
        try:
            # ```json ... ``` formatındaki cevaplara karşı esnek davranma
            text = text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            data = json.loads(text.strip())
            return ReformerOutput(
                clean_query = data.get("clean_query", ""),
                sub_queries = data.get("sub_queries", []),
                modality    = data.get("modality", "text"),
                reasoning   = data.get("reasoning", ""),
            )
        except Exception:
            # Parse başarısız olursa ham soruyu temiz soru olarak kullanma
            return ReformerOutput(clean_query=text, sub_queries=[text])