
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocessor import DocumentPreprocessor
from vector_store import VectorStore

PATH = Path(__file__).resolve().parent.parent
MARKDOWN_PATH = PATH / "test_examples" / "test_belgesi.md"
PDF_PATH      = PATH / "test_examples" / "attention_is_all_you_need.pdf"

processor = DocumentPreprocessor(max_tokens=50)

print("Processing Markdown...\n")
created_chunks, outline = processor.preprocess_markdown(MARKDOWN_PATH)

# DISPLAYING RESULTS
for i, chunk in enumerate(created_chunks):
    print(f"CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   TITLE: {chunk.metadata.get('title')}")
    print(f"   TYPE: {chunk.metadata.get('type')}")
    print(f"   CONTENT: {chunk.content.strip()[:100]}...") # Display first 100 characters
    print("-" * 50)


# TESTING CODE
pdf_chunks, outline = processor.preprocess_pdf(PDF_PATH)

print("\nProcessing PDF...\n")
for i, chunk in enumerate(pdf_chunks[:20]):  # First 20 chunks
    print(f"CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   TITLE: {chunk.metadata.get('title')}")
    print(f"   TYPE: {chunk.metadata.get('type')}")
    print(f"   CONTENT: {chunk.content.strip()[:100]}...")
    print("-" * 50)

# VECTOR STORE TEST
print("\n" + "="*50)
print("Adding to Vector Store...\n")

store = VectorStore(store_dir="store")
store.add(pdf_chunks)
store.save("attention")

print(f"\nTotal {store.index.ntotal} chunks added to index.")
print("Index saved as 'store/attention.faiss'.\n")

print("="*50)
print("Search test:\n")

query = "What is the attention mechanism?"
results = store.search(query, top_k=3)

print(f"Query: '{query}'\n")
for i, chunk in enumerate(results):
    print(f"RESULT {i+1} | Section: {chunk.metadata.get('title')}")
    print(f"   {chunk.content.strip()[:150]}...")
    print("-" * 50)

    