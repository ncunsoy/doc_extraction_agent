"""
controller.py
Orchestrates the agent pipeline with a deterministic ReAct loop.

Routing rules:
  grounding fail : re-run Reformer (wrong retrieval target)
  coverage fail  : re-run Retriever only (right target, incomplete results)
  max_iter       : return "not found"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.reformer import ReformerAgent, ReformerOutput
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent, ValidationResult
from memory import Memory
from preprocessor import DocumentChunk


@dataclass
class PipelineResult:
    answer:     str
    chunks:     list[DocumentChunk] = field(default_factory=list)
    iterations: int  = 0
    solved:     bool = False


class Controller:
    def __init__(
        self,
        reformer:  ReformerAgent,
        retriever: RetrieverAgent,
        validator: ValidatorAgent,
        max_iter:  int = 3,
        memory:    Memory | None = None,
    ):
        self.reformer  = reformer
        self.retriever = retriever
        self.validator = validator
        self.max_iter  = max_iter
        self.memory    = memory

    def run(
        self,
        question: str,
        outline: list[dict] | None = None,
    ) -> PipelineResult:
        """Run the ReAct loop until the answer passes validation or max_iter is reached."""
        memory_examples = self.memory.recall(question) if self.memory else None
        history:      list[dict]               = []
        reformer_out: ReformerOutput | None    = None
        last_chunks:  list[DocumentChunk]      = []
        last_answer:  str                      = ""
        validation:   ValidationResult | None  = None

        for iteration in range(self.max_iter):
            if iteration == 0 or validation:
                reformer_out = self.reformer.run(
                    question=question,
                    outline=outline,
                    history=history if history else None,
                    memory_examples=memory_examples,
                )

            retriever_out = self.retriever.run(reformer_out)
            last_chunks   = retriever_out.chunks
            last_answer   = retriever_out.answer_draft

            validation = self.validator.run(
                question=question,
                reformer_out=reformer_out,
                chunks=last_chunks,
                answer_draft=last_answer,
            )

            if validation.verdict:
                if self.memory:
                    self.memory.add(question, reformer_out.clean_query, reformer_out.modality)
                return PipelineResult(
                    answer=last_answer,
                    chunks=last_chunks,
                    iterations=iteration + 1,
                    solved=True,
                )

            history.append({
                "clean_query":  reformer_out.clean_query,
                "failure_type": validation.failure_type,
                "reason":       validation.reason,
            })

        return PipelineResult(
            answer="No sufficient information could be found in the document for this question.",
            chunks=last_chunks,
            iterations=self.max_iter,
            solved=False,
        )
