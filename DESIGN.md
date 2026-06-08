# Design Document

## Overview

An agentic RAG (Retrieval-Augmented Generation) system that processes PDF and Markdown documents and answers questions through a multi-agent pipeline. The system is built around a ReAct loop: the Controller iterates through Reformer -> Retriever -> Validator until the answer passes validation or the iteration limit is reached.

```
User Question
     |
     v
 Controller
     |
     v
  Reformer  ->  sub-queries + modality
     |
     v
  Retriever  ->  FAISS search -> chunks -> draft answer
     |
     v
  Validator  ->  grounding check + coverage check
     |
     +-- PASS ---------------------> final answer
     |
     +-- FAIL (grounding) ---------> back to Reformer
     |
     +-- FAIL (coverage) ----------> back to Retriever
     |
     +-- max_iter reached ---------> "not found"
```

---

## Chunking Strategy

### Why structure-aware chunking

Fixed-size chunking ignores document structure: a chunk boundary can split a paragraph mid-sentence or merge unrelated sections, degrading retrieval precision. Instead, the system uses PyMuPDF's table of contents (TOC) to detect section boundaries. When no TOC is available, font size heuristics infer headings (the most frequent font size is treated as body text; larger sizes are treated as heading levels).

Within each section, `RecursiveCharacterTextSplitter` (tiktoken `cl100k_base`, 1000 tokens, 12% overlap) further splits long sections. The overlap preserves context across boundaries.

### Non-text elements

Tables and figures are extracted separately via `unstructured`. Each becomes its own `DocumentChunk` with `type` metadata (`table`, `image`). Image chunks also carry `image_bytes` (PNG, extracted by PyMuPDF) and `image_index` (position on the page, to distinguish multiple figures per page). The actual pixel data travels with the chunk through the pipeline — no need to re-open the PDF at retrieval time.

---

## Embedding & Vector Store

### Model choice: `gemini-embedding-001`

`gemini-embedding-001` was chosen for the following reasons:

- **Batch support**: up to 100 texts per API call, making full-document indexing fast
- **Multilingual quality**: strong performance on cross-lingual retrieval
- **Stability**: production-grade quota and reliability

`gemini-embedding-2` was evaluated during development. It supports multimodal embedding (text + images in the same vector space), which would allow images to be searched semantically rather than by caption text. However, it has no batch API — each input requires a separate request. At current API quotas this makes indexing a 300+ chunk document impractical. The note in `vector_store.py` marks the migration path when a batch API becomes available.

For private or sensitive documents, a locally hosted embedding model (e.g. a `sentence-transformers` model) should replace the Gemini API entirely.

### FAISS index

`IndexFlatIP` with L2-normalized vectors computes exact cosine similarity. Approximate indexes (HNSW, IVF) would be faster at scale but are unnecessary for document-scale workloads.

---

## Agent Pipeline

### Reformer

The most critical factor in RAG quality is retrieving the right chunk. A verbatim user question is often suboptimal as a search query — too conversational, ambiguous, or in a different language than the document. The Reformer uses Gemini to:

- Rewrite the question as a clean retrieval query in the document's language
- Decompose compound questions into sub-queries (one per semantic intent)
- Estimate modality (`text`, `visual`, or `both`) to decide whether image chunks should be included in synthesis

Memory examples from past successful queries are injected into the Reformer prompt as few-shot context, improving reformulation quality over time.

### Retriever

For each sub-query, the Retriever searches FAISS and deduplicates results across sub-queries. If modality is `visual` or `both`, `image_bytes` from retrieved chunks are decoded into PIL images and passed alongside the text context to Gemini for multimodal synthesis.

The synthesis prompt instructs Gemini to answer in the same language as the question, so a Turkish question receives a Turkish answer even though retrieval operates on document-language queries.

### Validator

Two checks run sequentially:

1. **Grounding check**: detects hallucination — does the answer contain claims not supported by the retrieved chunks?
2. **Coverage check**: detects incompleteness — are all sub-queries from the Reformer addressed in the answer?

Both failure types re-run the Reformer with the updated history, which now contains the previous `failure_type` and `reason`. The Reformer uses this to generate different sub-queries rather than repeating the same mistake. Re-running only the Retriever on coverage failure would be pointless — deterministic FAISS search returns the same chunks for the same query.

### Controller

The Controller runs the ReAct loop for up to `max_iter` iterations (default 3). Each failed iteration appends `{clean_query, failure_type, reason}` to the history, which the Reformer uses on the next attempt to avoid repeating the same mistake. On success, the resolved question is written to memory.

---

## Memory

Cross-session learning via a JSON file. Each successfully resolved question is stored with its `clean_query` and `modality`. On a new question, the `k` most similar past entries are retrieved by embedding cosine similarity and injected into the Reformer as few-shot examples. This gradually improves reformulation accuracy for recurring query patterns.

---

## Known Limitations & Future Work

| Limitation | Details | Mitigation path |
|---|---|---|
| Image embedding is text-based | `gemini-embedding-001` cannot embed images; image chunks are indexed by their HTML/caption content | Migrate to `gemini-embedding-2` when its batch API is available |
| Cross-page figure/caption | A figure on page N and its caption on page N-1 are separate chunks with no explicit link | Multimodal embedding would make the figure semantically searchable regardless of page; a post-processing pass could also match captions to figures by proximity |
| Private data | Gemini API sends document content to Google servers | Replace embedding and synthesis models with locally hosted alternatives |
| Single-document index | The vector store holds one document at a time | Extend `VectorStore` with a `doc_id` filter per chunk for multi-document retrieval |
