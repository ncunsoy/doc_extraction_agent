import os
from preprocessor import DocumentPreprocessor
from vector_store import VectorStore

# --- 2. TESTING OUR CODE ---
# We deliberately set max_tokens to a low value (50) to quickly see results.
# This way we can see how LangChain intelligently splits the texts.
islemci = DocumentPreprocessor(max_tokens=50) 

print("Processing Markdown...\n")
markdown = "D:\\Desktop\\doc_extraction_agent\\test_examples\\test_belgesi.md"
olusan_chunklar, outline = islemci.preprocess_markdown(markdown)

# --- 3. DISPLAYING RESULTS ---
for i, chunk in enumerate(olusan_chunklar):
    print(f"CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   TITLE: {chunk.metadata.get('title')}")
    print(f"   TYPE: {chunk.metadata.get('type')}")
    print(f"   CONTENT: {chunk.content.strip()[:100]}...") # Display first 100 characters
    print("-" * 50)


# --- 2. TESTING OUR CODE ---
pdf_file = "D:\\Desktop\\doc_extraction_agent\\test_examples\\attention_is_all_you_need.pdf"
with open(pdf_file, "rb") as f:
    pdf_chunklar, outline = islemci.preprocess_pdf(pdf_file)

print("\nProcessing PDF...\n")
for i, chunk in enumerate(pdf_chunklar[:20]):  # First 20 chunks
    print(f"CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   TITLE: {chunk.metadata.get('title')}")
    print(f"   TYPE: {chunk.metadata.get('type')}")
    print(f"   CONTENT: {chunk.content.strip()[:100]}...")
    print("-" * 50)

# --- VECTOR STORE TEST ---
print("\n" + "="*50)
print("Adding to Vector Store...\n")

store = VectorStore(store_dir="store")
store.add(pdf_chunklar)
store.save("attention")

print(f"\nTotal {store.index.ntotal} chunks added to index.")
print("Index saved as 'store/attention.faiss'.\n")

print("="*50)
print("Search test:\n")

sorgu = "What is the attention mechanism?"
sonuclar = store.search(sorgu, top_k=3)

print(f"Query: '{sorgu}'\n")
for i, chunk in enumerate(sonuclar):
    print(f"RESULT {i+1} | Section: {chunk.metadata.get('title')}")
    print(f"   {chunk.content.strip()[:150]}...")
    print("-" * 50)

    