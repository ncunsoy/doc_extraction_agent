"""
retriever.py
Retrieves relevant chunks and generates a draft answer using Gemini.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from PIL import Image

from agents.reformer import ReformerOutput
from preprocessor import DocumentChunk
from vector_store import VectorStore

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"

# Note: answers are currently forced to English so the Validator evaluates consistently.
# To support multilingual answers, replace "Provide your answer in English." with
# "Answer in the same language as the question." — the Reformer already rewrites
# sub-queries to the document's language for retrieval, so retrieval quality is unaffected.
SYNTHESIS_PROMPT = """
Use the following document excerpts to answer the question.
Only use the provided content. Do not invent information that is not in the documents.
Provide your answer in English.

Document excerpts:
{context}

Question: {question}
"""


@dataclass
class RetrieverOutput:
    chunks:       list[DocumentChunk] = field(default_factory=list)
    answer_draft: str = ""


class RetrieverAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        model_name: str = MODEL_NAME,
    ):
        self.store      = vector_store
        self.top_k      = top_k
        self.model_name = model_name

    def run(self, reformer_out: ReformerOutput) -> RetrieverOutput:
        """Retrieve chunks for each sub-query, deduplicate, and synthesize a draft answer."""
        queries  = reformer_out.sub_queries or [reformer_out.clean_query]
        modality = reformer_out.modality

        seen_ids:   set[str]            = set()
        all_chunks: list[DocumentChunk] = []
        for query in queries:
            for chunk in self.store.search(query, top_k=self.top_k):
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        images: list[Image.Image] = []
        if modality in ("visual", "both"):
            for chunk in all_chunks:
                if chunk.image_bytes:
                    images.append(Image.open(io.BytesIO(chunk.image_bytes)))

        answer = self._synthesize(reformer_out.clean_query, all_chunks, images)
        return RetrieverOutput(chunks=all_chunks, answer_draft=answer)

    def _synthesize(
        self,
        question: str,
        chunks: list[DocumentChunk],
        images: list[Image.Image],
    ) -> str:
        """Generate a draft answer from retrieved chunks and optional page images."""
        if not chunks and not images:
            return "No relevant content found."

        context = "\n\n---\n\n".join(
            f"[Page {c.metadata.get('page', '?')} / {c.metadata.get('title', '')}]\n{c.content}"
            for c in chunks
        )
        prompt  = SYNTHESIS_PROMPT.format(context=context, question=question)
        content = [prompt] + images if images else prompt

        try:
            response = client.models.generate_content(model=self.model_name, contents=content)
        except Exception as e:
            print(f"Error during synthesis: {e}")
            return f"Response generation failed: {e}"

        return response.text

