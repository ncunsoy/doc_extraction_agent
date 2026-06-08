"""
main.py
CLI arayüzü.

Kullanım:
    python main.py --doc  --question  --top_k  --max_iter
"""

from __future__ import annotations

from pathlib import Path

import typer

from preprocessor import DocumentPreprocessor
from vector_store import VectorStore
from agents.reformer import ReformerAgent
from agents.retriever import RetrieverAgent
from agents.validator import ValidatorAgent
from controller import Controller

app = typer.Typer()


@app.command()
def main(
    doc:      Path = typer.Option(..., help="PDF veya Markdown dosya yolu"),
    question: str  = typer.Option(..., help="Sorulacak soru"),
    top_k:    int  = typer.Option(5,   help="Retrieval chunk sayisi"),
    max_iter: int  = typer.Option(3,   help="Maksimum iterasyon"),
):
    if not doc.exists():
        print(f"Dosya bulunamadi: {doc}")
        raise typer.Exit(1)

    doc_id = doc.stem
    store  = VectorStore()

    if store.load(doc_id):
        print(f"Index yuklendi: {doc_id}")
        outline = []
    else:
        print(f"Belge isleniyor: {doc}")
        preprocessor = DocumentPreprocessor()
        if doc.suffix.lower() == ".pdf":
            chunks, outline = preprocessor.preprocess_pdf(doc)
        else:
            chunks, outline = preprocessor.preprocess_markdown(doc)

        if not chunks:
            print("Belge islenemedi.")
            raise typer.Exit(1)

        print(f"{len(chunks)} chunk olusturuldu, indexe yaziliyor...")
        store.add(chunks)
        store.save(doc_id)

    controller = Controller(
        reformer  = ReformerAgent(),
        retriever = RetrieverAgent(vector_store=store, pdf_path=doc, top_k=top_k),
        validator = ValidatorAgent(),
        max_iter  = max_iter,
    )

    print(f"\nSoru: {question}\n")
    result = controller.run(question=question, outline=outline)

    status = "Solved" if result.solved else "Unsolved"
    print(f"{status} - {result.iterations} iteration(s)\n")
    print(f"Cevap:\n{result.answer}\n")

    if result.chunks:
        print("Kaynaklar:")
        seen: set[str] = set()
        for chunk in result.chunks:
            title = chunk.metadata.get("title", "")
            page  = chunk.metadata.get("page", "?")
            key   = f"{title}_{page}"
            if key not in seen:
                print(f"  Sayfa {page} - {title}")
                seen.add(key)


if __name__ == "__main__":
    app()
