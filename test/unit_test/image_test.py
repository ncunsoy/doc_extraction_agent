import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pymupdf
from PIL import Image

from preprocessor import DocumentPreprocessor

PDF_PATH = Path(__file__).resolve().parents[1] / "test_examples" / "attention_is_all_you_need.pdf"

pre = DocumentPreprocessor.__new__(DocumentPreprocessor)


def _find_image_page(doc: pymupdf.Document) -> int | None:
    """Return the first 1-indexed page number that contains an embedded image block."""
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 1:
                return page_num
    return None


doc            = pymupdf.open(str(PDF_PATH))
image_page_num = _find_image_page(doc)
doc.close()

if image_page_num is None:
    print("No image blocks found in PDF — skipping image extraction tests")
else:
    # _extract_image_bytes: returns valid PNG bytes for a page with an image
    print(f"[TEST 1] _extract_image_bytes returns PNG bytes (page {image_page_num})")
    doc    = pymupdf.open(str(PDF_PATH))
    result = pre._extract_image_bytes(doc, page_num=image_page_num, image_index=0)
    doc.close()
    assert result is not None, "Expected bytes, got None"
    assert isinstance(result, bytes)
    img = Image.open(io.BytesIO(result))
    assert img.size[0] > 0
    print(f"  {len(result)} bytes  size={img.size}")
    print(f"  PASS\n")

    # _extract_image_bytes: default image_index=0
    print("[TEST 2] _extract_image_bytes default image_index=0")
    doc    = pymupdf.open(str(PDF_PATH))
    result = pre._extract_image_bytes(doc, page_num=image_page_num)
    doc.close()
    assert result is not None
    print(f"  {len(result)} bytes")
    print(f"  PASS\n")

# _extract_image_bytes: out-of-bounds page returns None
print("[TEST 3] Out-of-bounds page returns None")
doc    = pymupdf.open(str(PDF_PATH))
result = pre._extract_image_bytes(doc, page_num=9999)
doc.close()
assert result is None, f"Expected None, got {type(result)}"
print(f"  result: {result}")
print(f"  PASS\n")

# _extract_image_bytes: out-of-bounds image_index returns None
print("[TEST 4] Out-of-bounds image_index returns None")
doc    = pymupdf.open(str(PDF_PATH))
result = pre._extract_image_bytes(doc, page_num=1, image_index=999)
doc.close()
assert result is None, f"Expected None, got {type(result)}"
print(f"  result: {result}")
print(f"  PASS\n")

# image chunks from preprocess_pdf carry image_bytes
print("[TEST 5] image chunks from preprocess_pdf carry image_bytes")
pre_full     = DocumentPreprocessor()
chunks, _    = pre_full.preprocess_pdf(PDF_PATH)
image_chunks = [c for c in chunks if c.metadata.get("type") == "image"]
if image_chunks:
    for chunk in image_chunks:
        assert chunk.image_bytes is not None, f"image chunk missing image_bytes: {chunk.id}"
        assert "image_index" in chunk.metadata, f"image chunk missing image_index: {chunk.metadata}"
    print(f"  {len(image_chunks)} image chunks found, all have image_bytes and image_index")
else:
    print(f"  No image-type chunks detected by unstructured in this PDF (strategy=fast)")
print(f"  PASS\n")

print("All tests passed.")
