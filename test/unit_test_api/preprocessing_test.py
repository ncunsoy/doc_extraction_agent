import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocessor import DocumentPreprocessor
from vector_store import VectorStore

TEST_EXAMPLES = Path(__file__).resolve().parents[1] / "test_examples"
MARKDOWN_PATH = TEST_EXAMPLES / "test_belgesi.md"
PDF_PATH      = TEST_EXAMPLES / "attention_is_all_you_need.pdf"

processor = DocumentPreprocessor(max_tokens=50)

# MARKDOWN
print("Processing Markdown\n")
created_chunks, outline = processor.preprocess_markdown(MARKDOWN_PATH)
for i, chunk in enumerate(created_chunks):
    print(f"CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   TITLE  : {chunk.metadata.get('title')}")
    print(f"   TYPE   : {chunk.metadata.get('type')}")
    print(f"   CONTENT: {chunk.content.strip()[:100]}...\n")

# PDF
pdf_chunks, outline = processor.preprocess_pdf(PDF_PATH)
print("\nProcessing PDF\n")
for i, chunk in enumerate(pdf_chunks[:20]):
    print(f"CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   TITLE  : {chunk.metadata.get('title')}")
    print(f"   TYPE   : {chunk.metadata.get('type')}")
    print(f"   CONTENT: {chunk.content.strip()[:100]}...\n")

# VECTOR STORE
print("Adding to Vector Store\n")
store = VectorStore(store_dir="store")
store.add(pdf_chunks)
store.save("attention")
print(f"Total {store.index.ntotal} chunks added to index.")
print("Index saved as 'store/attention.faiss'.\n")

# SEARCH
print("Search test:\n")
query   = "What is the attention mechanism?"
results = store.search(query, top_k=3)
print(f"Query: '{query}'\n")
for i, chunk in enumerate(results):
    print(f"RESULT {i+1} | Section: {chunk.metadata.get('title')}")
    print(f"   {chunk.content.strip()[:150]}...\n")
