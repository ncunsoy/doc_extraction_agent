# Document Extraction Agent

An agentic RAG system that processes PDF and Markdown documents and answers questions using a multi-agent pipeline with a ReAct loop.

## Features

- Structure-aware chunking: section boundaries detected from TOC or font size heuristics
- Multimodal retrieval: figures and tables stored with pixel data and passed to the LLM during synthesis
- Multi-language support: questions in any language are answered in the same language; retrieval always operates in the document's language
- ReAct loop: Reformer -> Retriever -> Validator with deterministic retry routing
- Cross-session memory: past resolved queries improve future reformulation via few-shot injection
- OCR fallback: image-based PDFs automatically processed with Tesseract

## Architecture

```
PDF / Markdown
     |
     v
DocumentPreprocessor
  - TOC / font-based section detection
  - unstructured element extraction (text, table, image)
  - RecursiveCharacterTextSplitter (1000 tokens, cl100k_base)
  - image_bytes stored per chunk
     |
     v
VectorStore (FAISS IndexFlatIP)
  - gemini-embedding-001, batch embed, L2-normalized
     |
     v
Controller (ReAct loop, max 3 iterations)
     |
     +-> Reformer
     |     - clean query + sub-queries in document language
     |     - modality estimation (text / visual / both)
     |     - few-shot examples from Memory
     |
     +-> Retriever
     |     - FAISS search per sub-query, deduplicated
     |     - image_bytes decoded and passed to Gemini if visual
     |     - Gemini 2.5 Flash synthesis
     |
     +-> Validator
           - grounding check (hallucination)
           - coverage check (unanswered sub-queries)
```

## Setup

See [SETUP.md](SETUP.md) for system dependencies (Poppler, Tesseract) and environment variables.

```
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY
```

## Usage

```
python main.py --doc path/to/document.pdf --question "Your question here"
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--doc` | required | PDF or Markdown file |
| `--question` | required | Question to ask |
| `--top_k` | 5 | Number of chunks retrieved per sub-query |
| `--max_iter` | 3 | Maximum ReAct loop iterations |

The index is saved to `store/<doc_name>.faiss` after the first run and reused on subsequent runs.

## Project Structure

```
doc_extraction_agent/
  agents/
    reformer.py       query reformulation and sub-query decomposition
    retriever.py      chunk retrieval and answer synthesis
    validator.py      grounding and coverage validation
  test/
    unit_test/        offline unit tests (no API calls)
    unit_test_api/    integration tests (real API)
    evaluate.py       accuracy evaluation with keyword + LLM judge
  controller.py       ReAct loop orchestration
  preprocessor.py     document parsing and chunking
  vector_store.py     FAISS index and embedding
  memory.py           cross-session query memory
  main.py             CLI entry point
```

## Models Used

| Purpose | Model |
|---|---|
| Embedding | gemini-embedding-001 |
| Synthesis & agents | gemini-2.5-flash |
| Evaluation judge | gemini-2.5-flash |

## Design

See [DESIGN.md](DESIGN.md) for detailed rationale behind chunking strategy, embedding model selection, agent design, and known limitations.
