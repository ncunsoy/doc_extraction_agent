# Technical Note

This note covers the two design decisions with the most significant trade-offs.

## Decision 1 — Separating diagnosis (Validator) from treatment (Controller)

The validation loop could have been a single agent that both judges the answer
and decides what to do next. Instead, the responsibility is split: the
**Validator only diagnoses**, returning a `failure_type` (`grounding` or
`coverage`) without proposing a fix, and the **Controller decides the route**
with a fixed, deterministic rule.

Both failure types route back to the Reformer. The retriever is deterministic,
so re-running it on the same query returns the same chunks. Only the Reformer can
unlock progress — by producing a sharper query, or by switching modality (e.g.
from text to the visual space) where the missing information might actually live.

**Gain:** A fixed routing rule makes the loop predictable, debuggable, and
unit-testable; `controller_test.py` verifies the exact retry behavior with mocked
agents and no API calls. Agents never call each other — each reports to the
Controller — yielding a loosely coupled architecture.

**Trade-off:** An LLM-based router could be smarter and give context-specific
advice, but that flexibility was given up for determinism and testability.

## Decision 2 — Reverting from gemini-embedding-2 to gemini-embedding-001

`gemini-embedding-2` supports multimodal embedding — text and images share one
vector space, which would let figures and tables be retrieved by visual semantics
rather than just their caption text. The model was integrated and tested, "but was reverted for a practical reason: it has no batch API.

Each input requires a separate request, so indexing a 300-chunk document needs
300+ calls. At free-tier quota (150 requests/minute) this takes minutes per
document, and any 429 error interrupts it. The decision was to revert to
`gemini-embedding-001`, which accepts batches of up to 100 texts per call and has
stable quotas.

**Gain:** Fast, batched indexing with predictable quota behavior, making the
ingest step reliable enough for everyday use.

**Trade-off:** Visual-semantic retrieval is lost; figures are reachable only via
their caption text. The multimodal migration path is documented in
`vector_store.py` for when a batch API becomes available.
