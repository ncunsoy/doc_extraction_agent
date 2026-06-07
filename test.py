import os
from preprocessor import DocumentPreprocessor
from vector_store import VectorStore

# --- 2. KODUMUZU TEST ETME ---
# Testi hızla görmek için max_tokens değerini bilerek çok düşük (50) tutuyoruz.
# Böylece LangChain'in metinleri nasıl acımasızca ama akıllıca böldüğünü görebileceğiz.
islemci = DocumentPreprocessor(max_tokens=50) 

print("Markdown işleniyor...\n")
markdown = "D:\\Desktop\\doc_extraction_agent\\test_examples\\test_belgesi.md"
olusan_chunklar, outline = islemci.preprocess_markdown(markdown)

# --- 3. SONUÇLARI GÖRÜNTÜLEME ---
for i, chunk in enumerate(olusan_chunklar):
    print(f"📦 CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   🏷️ BAŞLIK: {chunk.metadata.get('title')}")
    print(f"   ⚙️ TÜR: {chunk.metadata.get('type')}")
    print(f"   📝 İÇERİK: {chunk.content.strip()[:100]}...") # İçeriğin ilk 100 karakterini göster
    print("-" * 50)


# --- 2. KODUMUZU TEST ETME ---
pdf_file = "D:\\Desktop\\doc_extraction_agent\\test_examples\\attention_is_all_you_need.pdf"
with open(pdf_file, "rb") as f:
    pdf_chunklar, outline = islemci.preprocess_pdf(pdf_file)

print("\nPDF işleniyor...\n")
for i, chunk in enumerate(pdf_chunklar[:20]):  # İlk 20 chunk'
    print(f"📦 CHUNK {i+1} | ID: {chunk.id[:8]}...")
    print(f"   🏷️ BAŞLIK: {chunk.metadata.get('title')}")
    print(f"   ⚙️ TÜR: {chunk.metadata.get('type')}")
    print(f"   📝 İÇERİK: {chunk.content.strip()[:100]}...")
    print("-" * 50)

# --- VECTOR STORE TESTİ ---
print("\n" + "="*50)
print("Vector Store'a ekleniyor...\n")

store = VectorStore(store_dir="store")
store.add(pdf_chunklar)
store.save("attention")

print(f"\nToplam {store.index.ntotal} chunk index'e eklendi.")
print("Index 'store/attention.faiss' olarak kaydedildi.\n")

print("="*50)
print("Arama testi:\n")

sorgu = "What is the attention mechanism?"
sonuclar = store.search(sorgu, top_k=3)

print(f"Sorgu: '{sorgu}'\n")
for i, chunk in enumerate(sonuclar):
    print(f"🔍 SONUÇ {i+1} | Bölüm: {chunk.metadata.get('title')}")
    print(f"   📝 {chunk.content.strip()[:150]}...")
    print("-" * 50)

    