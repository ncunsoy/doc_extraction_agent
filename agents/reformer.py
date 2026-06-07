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
Sen bir soru analiz ajanısın. Kullanıcının sorusunu analiz edip JSON formatında yanıt üreteceksin.

Görevlerin:
1. Soruyu gereksiz gürültüden temizle (konuşma dili, meta ifadeler)
2. Çok adımlı soruysa alt sorulara böl, değilse tek elemanlı liste yap
3. Cevabın belgede nerede olduğunu tahmin et:
   - "text": düz metin, paragraf
   - "visual": grafik, tablo, şekil
   - "both": her ikisinde de olabilir

Yanıtını SADECE şu JSON formatında ver, başka hiçbir şey yazma:
{
  "clean_query": "temizlenmiş soru",
  "sub_queries": ["alt soru 1", "alt soru 2"],
  "modality": "text",
  "reasoning": "neden bu modalite"
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
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT},
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