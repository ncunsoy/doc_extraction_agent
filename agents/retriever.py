"""
retriever.py
Retrieves relevant chunks and generates a draft answer using Gemini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai
from PIL import Image

from agents.reformer import ReformerOutput
from preprocessor import DocumentChunk
from vector_store import VectorStore

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL_NAME = "gemini-2.5-flash"

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
        pdf_path: str | Path | None = None,
        top_k: int = 5,
        model_name: str = MODEL_NAME,
    ):
        self.store      = vector_store
        self.pdf_path   = Path(pdf_path) if pdf_path else None
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

        visual_chunks = [c for c in all_chunks if c.metadata.get("type") != "text"]
        images: list[Image.Image] = []
        if self.pdf_path and (modality in ("visual", "both") or visual_chunks):
            images = self._get_page_images(visual_chunks)

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

    def _get_page_images(self, visual_chunks: list[DocumentChunk]) -> list[Image.Image]:
        """Extract embedded images from the PDF pages referenced by visual chunks."""
        if not self.pdf_path or not self.pdf_path.exists():
            return []

        pages  = {c.metadata.get("page", 0) for c in visual_chunks}
        images = []
        doc    = pymupdf.open(str(self.pdf_path))

        for page_num in sorted(pages):
            if page_num >= doc.page_count:
                continue
            page = doc[page_num]
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 1:  # 0 = text, 1 = embedded image
                    continue
                rect = pymupdf.Rect(block["bbox"])
                pix  = page.get_pixmap(clip=rect, dpi=150)
                images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))

        doc.close()
        return images
