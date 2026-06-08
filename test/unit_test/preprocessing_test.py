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
print("[TEST 1] Markdown chunk uretimi")
with tempfile.NamedTemporaryFile(suffix=".md", mode="w", encoding="utf-8", delete=False) as f:
    f.write(SAMPLE_CONTENT)
    tmp_path = Path(f.name)

chunks, outline = pre.preprocess_markdown(tmp_path)
assert len(chunks) > 0, "Hic chunk uretilmedi"
assert all(isinstance(c, DocumentChunk) for c in chunks), "Bazi elemanlar DocumentChunk degil"
print(f"  {len(chunks)} chunk uretildi")
print(f"  PASS\n")

# TITLE METADATA
print("[TEST 2] Bolum basligi metadataya tasiniyor mu")
titles = {c.metadata.get("title") for c in chunks}
assert any("Giris" in (t or "") for t in titles), f"'Giris' basligi bulunamadi: {titles}"
print(f"  Bulunan basliklar: {titles}")
print(f"  PASS\n")

# REQUIRED FIELDS
print("[TEST 3] Zorunlu metadata alanlari")
for chunk in chunks:
    assert chunk.id, f"id eksik: {chunk}"
    assert chunk.content.strip(), f"content bos: {chunk}"
    assert "type" in chunk.metadata, f"'type' eksik metadata'da: {chunk.metadata}"
print(f"  Tum {len(chunks)} chunk zorunlu alanlara sahip")
print(f"  PASS\n")

# EMPTY CONTENT
print("[TEST 4] Bos metin chunk uretmemeli")
result = pre._split_into_chunks("", {"title": "x"})
assert result == [], f"Bos metin icin chunk uretildi: {result}"
print(f"  Sonuc: {result}")
print(f"  PASS\n")

tmp_path.unlink(missing_ok=True)
print("Tum testler gecti.")
