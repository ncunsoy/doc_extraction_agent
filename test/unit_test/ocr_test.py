import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessor import DocumentPreprocessor

TEST_EXAMPLES  = Path(__file__).resolve().parents[1] / "test_examples"
RASTERIZED_PDF = TEST_EXAMPLES / "attention_is_all_you_need_page1.pdf"

pre = DocumentPreprocessor()

# OCR FALLBACK
print("[TEST 1] PDF - OCR fallback should be triggered")
chunks_ocr, _ = pre.preprocess_pdf(RASTERIZED_PDF)
assert len(chunks_ocr) > 0, "No chunks produced from PDF — OCR fallback failed"
print(f"  {len(chunks_ocr)} chunks produced via OCR")
print(f"  PASS\n")

# OCR OUTPUT QUALITY
print("[TEST 2] OCR output contains meaningful text")
all_text = " ".join(c.content for c in chunks_ocr)
assert len(all_text.strip()) > 100, "OCR output too short — likely empty"
print(f"  Total characters extracted: {len(all_text)}")
print(f"  Sample: {all_text[:120].strip()}")
print(f"  PASS\n")

print("All tests passed.")
