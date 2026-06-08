"""
controller.py
Tüm ajanları yöneten ana döngü.

Routing kuralları (deterministik):
  grounding fail → Reformer'a dön (yanlış yeri çektik)
  coverage fail  → Retriever'a dön (doğru yer ama eksik)
  max_iter       → "bulunamadı" döndür
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from preprocessor import DocumentChunk
from agents.reformer import ReformerAgent, ReformerOutput
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent, ValidationResult


@dataclass
class PipelineResult:
    answer:       str
    chunks:       list[DocumentChunk] = field(default_factory=list)
    iterations:   int = 0
    solved:       bool = False


class Controller:

    def __init__(
        self,
        reformer:  ReformerAgent,
        retriever: RetrieverAgent,
        validator: ValidatorAgent,
        max_iter:  int = 3,
    ):
        self.reformer  = reformer
        self.retriever = retriever
        self.validator = validator
        self.max_iter  = max_iter

    def run(
        self,
        question: str,
        outline: list[dict] | None = None,
    ) -> PipelineResult:

        history:      list[dict] = []
        reformer_out: ReformerOutput | None = None
        last_chunks:  list[DocumentChunk] = []
        last_answer:  str = ""
        validation:   ValidationResult | None = None

        for iteration in range(self.max_iter):

            # grounding fail veya ilk iterasyon → Reformer'ı çalıştır
            if iteration == 0 or (validation and validation.failure_type == "grounding"):
                reformer_out = self.reformer.run(
                    question = question,
                    outline  = outline,
                    history  = history if history else None,
                )

            # Retriever
            retriever_out = self.retriever.run(reformer_out)
            last_chunks   = retriever_out.chunks
            last_answer   = retriever_out.answer_draft

            # Validation
            validation = self.validator.run(
                question     = question,
                reformer_out = reformer_out,
                chunks       = last_chunks,
                answer_draft = last_answer,
            )

            # Sonuç kontrolü 
            if validation.verdict:
                return PipelineResult(
                    answer     = last_answer,
                    chunks     = last_chunks,
                    iterations = iteration + 1,
                    solved     = True,
                )

            # Başarısız ise history'e ekleme, sonraki iterasyon için hazırlama
            history.append({
                "clean_query": reformer_out.clean_query,
                "failure_type": validation.failure_type,
                "reason": validation.reason,
            })

        # max_iter dolma durumu
        return PipelineResult(
            answer     = "No sufficient information could be found in the document for this question.",
            chunks     = last_chunks,
            iterations = self.max_iter,
            solved     = False,
        )