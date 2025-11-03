import streamlit as st
import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from pypdf import PdfReader
import google.generativeai as genai

# ======================
# 🔐 تنظیمات اولیه
# ======================
st.set_page_config(page_title="AI Book Translator", page_icon="📘", layout="wide")
st.title("📘 AI Book Translator (Gemini + FAISS)")
st.write("Translate English book pages into Persian using Google Gemini.")

# گرفتن API Key از secrets
if "GOOGLE_API_KEY" in st.secrets:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("❌ لطفاً GOOGLE_API_KEY را در بخش Streamlit Secrets تنظیم کنید.")
    st.stop()

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "models/gemini-2.5-pro"
ARTIFACTS_DIR = Path("rag_artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ======================
# 📘 توابع کمکی
# ======================

def extract_pages_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []
    filename = Path(pdf_file.name).name
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"filename": filename, "page": i + 1, "text": text.strip()})
    return pages

def embed_texts_google(texts, model_name=EMBEDDING_MODEL, batch_size=5):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        result = genai.embed_content(model=model_name, content=batch, task_type="retrieval_document")
        embeddings.extend(result["embedding"])
    return np.array(embeddings, dtype=np.float32)

def build_index_from_pages(pages):
    texts = [p["text"] for p in pages if p["text"]]
    if not texts:
        st.warning("هیچ متنی برای embedding پیدا نشد.")
        return
    st.info(f"در حال تولید embedding برای {len(texts)} صفحه ...")
    vectors = embed_texts_google(texts)
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(ARTIFACTS_DIR / "index.faiss"))
    with open(ARTIFACTS_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(pages, f)
    st.success("✅ ایندکس ساخته و ذخیره شد.")

def get_page_text(filename, page_num):
    meta_path = ARTIFACTS_DIR / "metadata.pkl"
    if not meta_path.exists():
        st.warning("ابتدا ایندکس را بسازید.")
        return None
    with open(meta_path, "rb") as f:
        pages = pickle.load(f)
    for p in pages:
        if p["filename"] == filename and p["page"] == int(page_num):
            return p["text"]
    return None

def translate_with_gemini(text):
    if not text.strip():
        return "صفحه خالی است."
    system_prompt = (
        "شما یک مترجم حرفه‌ای در زمینه یادگیری ماشین و مهندسی هستید. "
        "اگر قطعه کدی در متن وجود دارد، آن را همان‌طور که هست نگه دارید. "
        "متن انگلیسی را به فارسی روان و دقیق ترجمه کنید."
    )
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content([system_prompt, text])
    return response.text.strip()

# ======================
# 🎨 رابط کاربری Streamlit
# ======================

uploaded_pdf = st.file_uploader("📤 یک فایل PDF آپلود کن", type=["pdf"])

if uploaded_pdf:
    pages = extract_pages_from_pdf(uploaded_pdf)
    st.success(f"✅ {len(pages)} صفحه از فایل استخراج شد.")
    
    # ساخت ایندکس
    if st.button("🔍 ساخت ایندکس برای فایل"):
        build_index_from_pages(pages)

    # انتخاب صفحه
    page_numbers = [p["page"] for p in pages]
    selected_page = st.number_input("شماره صفحه مورد نظر:", min_value=1, max_value=len(pages), value=1)

    if st.button("🌐 ترجمه صفحه"):
        page_text = get_page_text(Path(uploaded_pdf.name).name, selected_page)
        if not page_text:
            st.warning("متن صفحه یافت نشد.")
        else:
            st.subheader("📄 متن اصلی (انگلیسی):")
            st.text_area("Original Text", page_text, height=200)

            st.subheader("🇮🇷 ترجمه فارسی:")
            translated_text = translate_with_gemini(page_text)
            st.text_area("Persian Translation", translated_text, height=300)

