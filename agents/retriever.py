"""
retriever.py

Reformer'ın çıktısına göre chunk çeker ve taslak cevap üretir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from PIL import Image
import pymupdf

from preprocessor import DocumentChunk
from agents.reformer import ReformerOutput
from vector_store import VectorStore

load_dotenv()
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
    )


@dataclass
class RetrieverOutput:
    chunks:       list[DocumentChunk] = field(default_factory=list)
    answer_draft: str = ""

MODEL_NAME = "gemini-2.5-flash"

SYNTHESIS_PROMPT = """
Use the following document excerpts to answer the question.
Only use the provided content. Do not invent information that is not in the documents.
Provide your answer in English.

Document excerpts:
{context}

Question: {question}
"""


class RetrieverAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        pdf_path: str | Path | None = None,  # görsel retrieval için
        top_k: int = 5,
        model_name: str = MODEL_NAME,
    ):
        self.store      = vector_store
        self.pdf_path   = Path(pdf_path) if pdf_path else None
        self.top_k      = top_k
        self.model_name = model_name

    def run(self, reformer_out: ReformerOutput) -> RetrieverOutput:
        """
        Reformer çıktısına göre chunk çeker ve cevap üretir.
        Sub-query varsa her biri için ayrı arama yapar, sonuçları birleştirir.
        """
        queries = reformer_out.sub_queries or [reformer_out.clean_query]
        modality = reformer_out.modality

        # Her sub-query için arama yapma, duplicateleri temizleme
        seen_ids: set[str] = set()
        all_chunks: list[DocumentChunk] = []

        for query in queries:
            results = self.store.search(query, top_k=self.top_k)
            for chunk in results:
                if chunk.id not in seen_ids:
                    all_chunks.append(chunk)
                    seen_ids.add(chunk.id)

        # Preprocessor'ın metadata'sında type="text" olmayanlar görsel chunk'tır.
        visual_chunks = [c for c in all_chunks if c.metadata.get("type") != "text"]

        images: list[Image.Image] = []
        if self.pdf_path and (modality in ("visual", "both") or visual_chunks):
            images = self._get_page_images(visual_chunks)

        answer = self._synthesize(reformer_out.clean_query, all_chunks, images)
        return RetrieverOutput(chunks=all_chunks, answer_draft=answer)

    # Chunk'ları ve görselleri Gemini'ye verip cevap üretir.
    def _synthesize(
        self,
        question: str,
        chunks: list[DocumentChunk],
        images: list[Image.Image],
    ) -> str:
        if not chunks and not images:
            return "İlgili içerik bulunamadı."

        context = "\n\n---\n\n".join(
            f"[Sayfa {c.metadata.get('page', '?')} / {c.metadata.get('title', '')}]\n{c.content}"
            for c in chunks
        )
        
        prompt = SYNTHESIS_PROMPT.format(context=context, question=question)
        response = None
        # Görseller varsa multimodal call
        if images:
            content = [prompt] + images
        else:
            content = prompt

        try:
            response = client.models.generate_content(model=self.model_name, contents=content)
        except Exception as e:
            print(f"Error during synthesis: {e}")
            return f"Response generation failed due to an error: {e}"
        
        return response.text

    # Visual chunk'ların metadata'sındaki sayfa numaralarına göre görsel blokları döndürür.
    def _get_page_images(self, visual_chunks: list[DocumentChunk]) -> list[Image.Image]:
        if not self.pdf_path or not self.pdf_path.exists():
            return []

        pages = {c.metadata.get("page", 0) for c in visual_chunks}
        images = []
        doc = pymupdf.open(str(self.pdf_path))
        for page_num in sorted(pages):
            if page_num >= doc.page_count:
                continue
            page = doc[page_num]
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 1:  # PyMuPDF: 0=metin, 1=gömülü görsel
                    continue
                rect = pymupdf.Rect(block["bbox"])
                pix = page.get_pixmap(clip=rect, dpi=150)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        doc.close()
        return images