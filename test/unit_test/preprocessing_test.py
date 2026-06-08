import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessor import DocumentPreprocessor, DocumentChunk

SAMPLE_CONTENT = (
    "# Introduction\n"
    "This is an introduction section. It explains the system purpose.\n\n"
    "# Method\n"
    "This section explains the method. It contains multiple sentences. Details are here.\n\n"
    "# Conclusion\n"
    "Results are summarized in this section.\n"
)

pre = DocumentPreprocessor()

# MARKDOWN CHUNK
print("[TEST 1] Markdown chunk generation")
with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
    f.write(SAMPLE_CONTENT)
    tmp_path = Path(f.name)

chunks, outline = pre.preprocess_markdown(tmp_path)
assert len(chunks) > 0, "No chunks were generated"
assert all(isinstance(c, DocumentChunk) for c in chunks), "Some elements are not DocumentChunk"
print(f"  {len(chunks)} chunks generated")
print(f"  PASS\n")

# TITLE METADATA
print("[TEST 2] Metadata Titles")
titles = {c.metadata.get("title") for c in chunks}
assert any("Introduction" in (t or "") for t in titles), f"'Introduction' title not found: {titles}"
print(f"  Found titles: {titles}")
for c in chunks:
    if c.metadata.get("title"):
        # titles should be marked as plain text, not sentence
        assert c.metadata.get("type") == "text", f"Title metadata type is not 'text': {c.metadata}"
print(f"  PASS\n")

# REQUIRED FIELDS
print("[TEST 3] Required metadata fields")
for chunk in chunks:
    assert chunk.id, f"id missing: {chunk}"
    assert chunk.content.strip(), f"content empty: {chunk}"
    assert "type" in chunk.metadata, f"'type' missing in metadata: {chunk.metadata}"
print(f"  All {len(chunks)} chunks have required fields")
print(f"  PASS\n")

# EMPTY CONTENT
print("[TEST 4] Empty content handling")
result = pre._split_into_chunks("", {"title": "x"})
assert result == [], f"Chunks were generated from empty text: {result}"
print(f"  Result: {result}")
print(f"  PASS\n")

tmp_path.unlink(missing_ok=True)
print("All tests passed.")
