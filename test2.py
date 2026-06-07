from agents.reformer import ReformerAgent
from agents.retriever import RetrieverAgent
from vector_store import VectorStore

PDF_PATH = "D:\\Desktop\\doc_extraction_agent\\test_examples\\attention_is_all_you_need.pdf"
DOC_ID   = "attention"

# --- VECTOR STORE YÜKLE ---
store = VectorStore(store_dir="store")
loaded = store.load(DOC_ID)
if not loaded:
    print("Index bulunamadı. Önce test.py çalıştır.")
    exit(1)
print(f"Index yüklendi: {store.index.ntotal} chunk\n")

# --- AJANLAR ---
reformer  = ReformerAgent()
retriever = RetrieverAgent(vector_store=store, pdf_path=PDF_PATH)

# --- TEST SORULARI ---
questions = [
    "attention layer sayısı kaçtır?",
    "Transformer modelinde kaç encoder katmanı var ve her birinde ne var?",
]

for soru in questions:
    print("=" * 60)
    print(f"SORU: {soru}")

    ref_out = reformer.run(soru)
    print(f"\nReformer → clean_query : {ref_out.clean_query}")
    print(f"           sub_queries : {ref_out.sub_queries}")
    print(f"           modality    : {ref_out.modality}")

    ret_out = retriever.run(ref_out)
    print(f"\nRetriever → {len(ret_out.chunks)} chunk çekildi")
    print(f"\nCEVAP:\n{ret_out.answer_draft}")
    print()
