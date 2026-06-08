"""
main.py
CLI interface.

Usage:
    python main.py --doc <PDF_OR_MARKDOWN_FILE> --question <QUESTION> --top_k <TOP_K> --max_iter <MAX_ITER>
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
from memory import Memory

app = typer.Typer()
memory = Memory()


@app.command()
def main(
    doc:      Path = typer.Option(..., help="PDF or Markdown file path"),
    question: str  = typer.Option(..., help="Question to ask"),
    top_k:    int  = typer.Option(5,   help="Number of retrieval chunks"),
    max_iter: int  = typer.Option(3,   help="Maximum iterations"),
):
    if not doc.exists():
        print(f"File not found: {doc}")
        raise typer.Exit(1)

    doc_id = doc.stem
    store  = VectorStore()

    if store.load(doc_id):
        print(f"Index loaded: {doc_id}")
        outline = []
    else:
        print(f"Document being processed: {doc}")
        preprocessor = DocumentPreprocessor()
        if doc.suffix.lower() == ".pdf":
            chunks, outline = preprocessor.preprocess_pdf(doc)
        else:
            chunks, outline = preprocessor.preprocess_markdown(doc)

        if not chunks:
            print("Document could not be processed.")
            raise typer.Exit(1)

        print(f"{len(chunks)} chunk created, saving to index.")
        store.add(chunks)
        store.save(doc_id)

    controller = Controller(
        reformer  = ReformerAgent(),
        retriever = RetrieverAgent(vector_store=store, pdf_path=doc, top_k=top_k),
        validator = ValidatorAgent(),
        max_iter  = max_iter,
        memory    = memory
    )

    print(f"\nQuestion: {question}\n")
    result = controller.run(question=question, outline=outline)

    status = "Solved" if result.solved else "Unsolved"
    print(f"{status} - {result.iterations} iteration(s)\n")
    print(f"Answer:\n{result.answer}\n")

    if result.chunks:
        print("Sources:")
        seen: set[str] = set()
        for chunk in result.chunks:
            title = chunk.metadata.get("title", "")
            page  = chunk.metadata.get("page", "?")
            key   = f"{title}_{page}"
            if key not in seen:
                print(f"  Page {page} - {title}")
                seen.add(key)


if __name__ == "__main__":
    app()
