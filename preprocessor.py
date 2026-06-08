"""
preprocessor.py
PDF ve Markdown belgelerini okur, outline çıkarır, bölüm-hizalı (structure-aware) chunk'lara böler.

Ana kural: Önce veri yapılarına (Tablolar) göre sonra Langchain ile recursive splitting yaparak böl.
Fallback:  bölüm çok uzunsa (> max_tokens) token sınırında böl.
"""

from __future__ import annotations

from pydoc import doc
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymupdf
import markdown

from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: dict

class DocumentPreprocessor:
    def __init__(self, max_tokens=1000, ocr_char_threshold=30, languages: list[str] | None = None):
        self.max_tokens = max_tokens
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.max_tokens,
            chunk_overlap=int(self.max_tokens * 0.12),
            separators=["\n\n", "\n", " ", ""],
        )
        self.ocr_char_threshold = ocr_char_threshold
        self.languages = languages or []


    # PDF'leri bölümlere göre böler, tablo ve metin ayrımı yaparak ve recursive chunk'lara dönüştürür
    def preprocess_pdf(self, file_path: Path) -> tuple[list[DocumentChunk], list[dict]]:
        try:
            # TOC ve Sayfa Sayısı Çıkarımı
            doc = pymupdf.open(file_path)
            outline = doc.get_toc()
            page_count = doc.page_count
            page_to_section = self._map_pages_to_sections(outline)
            if not page_to_section:
                page_to_section = self._extract_sections_from_fonts(doc)
            
            #  Bölümlere göre PDF bölme - tablo ve metin ayrımı 
            elements = partition_pdf(
                filename=str(file_path),
                strategy="fast", 
                infer_table_structure=True 
            )

            #  OCR fallback
            total_text = sum(len(str(e).strip()) for e in elements)
            if not elements or total_text < doc.page_count * 30:
                elements = partition_pdf(
                    filename=str(file_path),
                    strategy="ocr_only",
                    infer_table_structure=True,
                    languages=self.languages or None,
                )

            doc.close()

            chunks = []
            current_section_title = "Full Document"
            current_section_level = 0
            text_buffer = []

            # PDF'deki her bir element için işlemler
            for element in elements:
                page_num = element.metadata.page_number or 1
                
                # Yeni bir bölüme geçiş kontrolü (Sayfa numarasına göre)
                if page_num in page_to_section:
                    # Yeni bölüme geçerken eski bölümde biriken metni LangChain ile bölüp ekleme
                    if text_buffer:
                        full_text = "\n\n".join(text_buffer)
                        chunks.extend(self._split_into_chunks(
                            full_text, 
                            {"title": current_section_title, "level": current_section_level, "page": page_num}
                        ))
                        text_buffer = [] # Buffer'ı sıfırla
                    
                    current_section_title, current_section_level = page_to_section[page_num]

                element_type = type(element).__name__  # "NarrativeText", "Title", "Table", "Image", vb.

                if element_type in ("NarrativeText", "Text", "Title", "ListItem", "Header"):
                    text_buffer.append(str(element))

                else:
                    if text_buffer:
                        full_text = "\n\n".join(text_buffer)
                        chunks.extend(self._split_into_chunks(
                            full_text,
                            {"title": current_section_title, "level": current_section_level, "page": page_num}
                        ))
                        text_buffer = []

                    html_content = element.metadata.text_as_html or str(element)
                    metadata = {
                        "title": current_section_title,
                        "level": current_section_level,
                        "page": page_num,
                        "type": element_type.lower(),  # "table", "image", "figuredcaption", vb.
                    }
                    chunk_id = hashlib.md5(html_content.encode()).hexdigest()
                    chunks.append(DocumentChunk(id=chunk_id, content=html_content, metadata=metadata))
                
            # Kalan metni ekleme
            if text_buffer:
                full_text = "\n\n".join(text_buffer)
                chunks.extend(self._split_into_chunks(
                    full_text, 
                    {"title": current_section_title, "level": current_section_level, "page": page_count}
                ))

            outline_list = [
                {"title": title, "level": level, "page": page}
                for page, (title, level) in sorted(page_to_section.items())
            ]
            return chunks, outline_list

        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return [], []

    # Markdown belgelerini başlıklara göre böler ve recursive chunk'lara dönüştürür
    def preprocess_markdown(self, file_path: Path) -> tuple[list[DocumentChunk], list[dict]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()
            html = markdown.markdown(md_content)
        except Exception as e:
            print(f"Error processing Markdown {file_path}: {e}")
            return [], []
        
        # Markdown'da başlıkları kullanarak bölümlere ayırma
        sections = re.split(r'^(#{1,3}\s+.+)$', md_content, flags=re.MULTILINE)
        chunks = []
        for i in range(1, len(sections), 2):
            title = sections[i]
            content = sections[i + 1]
            chunks.extend(self._split_into_chunks(content, {"title": title, "type": "text"}))
        
        outline_list = [
            {"title": re.sub(r'^#{1,3}\s+', '', sections[i]).strip(), 
            "level": len(re.match(r'^(#+)', sections[i]).group(1)),
            "page": 0}
            for i in range(1, len(sections), 2)
        ]
        return chunks, outline_list

    # Metni token sınırına göre bölerek DocumentChunk'lara dönüştürür, metadata'yı günceller
    def _split_into_chunks(self, text: str, metadata: dict) -> list[DocumentChunk]:
        if not text.strip():
            return []

        text_chunks = self.text_splitter.split_text(text)
        doc_chunks = []
        
        for index, chunk_text in enumerate(text_chunks):
            # Metadata'yı kopyalayarak güncelleme
            chunk_meta = metadata.copy()
            chunk_meta["chunk_index"] = index
            if "type" not in chunk_meta:
                chunk_meta["type"] = "text"
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            doc_chunks.append(DocumentChunk(id=chunk_id, content=chunk_text, metadata=chunk_meta))

        return doc_chunks

    # PyMuPDF'den gelen TOC'u sayfa numaralarına göre haritalar
    def _map_pages_to_sections(self, outline: list) -> dict:
        """PyMuPDF'den gelen TOC'u sayfa numaralarına göre haritalar."""
        page_mapping = {}
        if not outline:
            return page_mapping
            
        for item in outline:
            level, title, page = item
            # Eğer aynı sayfada birden fazla başlık varsa ilkini/ana başlığı tutar
            if page not in page_mapping:
                page_mapping[page] = (title, level)
        return page_mapping
    
    # TOC yoksa veya güvenilir değilse, font boyutlarına göre bölümleri tahmin eder
    def _extract_sections_from_fonts(self, doc) -> dict:
        size_counts = {}
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        s = round(span["size"], 1)
                        size_counts[s] = size_counts.get(s, 0) + 1

        if not size_counts:
            return {}

        body_size = max(size_counts, key=lambda s: size_counts[s])
        heading_sizes = sorted(
            {s for s in size_counts if s > body_size + 1},
            reverse=True
        )[:3]
        size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}

        page_mapping = {}
        for page_num, page in enumerate(doc, start=1):
            for block in page.get_text("dict", sort=True)["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    first_span = line["spans"][0] if line["spans"] else None
                    if not first_span:
                        continue
                    sz = round(first_span["size"], 1)
                    if sz in size_to_level:
                        title = " ".join(s["text"] for s in line["spans"]).strip()
                        if title and page_num not in page_mapping:
                            page_mapping[page_num] = (title, size_to_level[sz])
        return page_mapping