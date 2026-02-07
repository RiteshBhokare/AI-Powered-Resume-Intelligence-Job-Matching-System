# ========== IMPORTS ==========
import os
from pathlib import Path
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()

# ========== STREAMLIT CONFIG ==========
st.set_page_config(
    page_title="Company Profiling – TRUE RAG",
    page_icon="🏢",
    layout="wide"
)

# ========== CUSTOM STYLES ==========
st.markdown(
    """
    <style>
    body{font-family:Inter,system-ui,Segoe UI,Arial;background:#000;color:#e6eef8}
    .title{font-size:28px;font-weight:700}
    .subtitle{color:#9aa6bf}
    .stButton>button{background:#0b5cff;color:#fff;border-radius:8px;font-weight:600}
    .stTextArea textarea{background:#04101a;color:#e6eef8;border-radius:8px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>🏢 Company Profiling – TRUE RAG System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Cross-platform RAG powered company analysis</div>", unsafe_allow_html=True)

# ========== PATHS (CROSS PLATFORM) ==========
BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_COMPANY_DOCS_FOLDER = BASE_DIR / "resume_samples"
CHROMA_PATH = BASE_DIR / "chroma_db_company"

SIMILARITY_THRESHOLD = 0.7

# ========== DYNAMIC FOLDER INPUT ==========
st.divider()
st.subheader("📂 Company Documents Folder")

user_docs_path = st.text_input(
    "Enter company PDF folder path (Windows / Linux)",
    value=str(DEFAULT_COMPANY_DOCS_FOLDER),
    help="Example: C:\\Users\\Admin\\company_pdfs OR /home/user/company_pdfs"
)

COMPANY_DOCS_FOLDER = Path(user_docs_path).expanduser().resolve()

# ========== LOAD EMBEDDING MODEL ==========
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ========== GROQ CLIENT ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ========== CHROMA DB ==========
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

collection = chroma_client.get_or_create_collection(
    name="company_profiles",
    metadata={"hnsw:space": "cosine"}
)

# ========== EMBED & STORE COMPANY DOCS ==========
def embed_and_store_company_docs():
    if not COMPANY_DOCS_FOLDER.exists() or not COMPANY_DOCS_FOLDER.is_dir():
        st.error("❌ Invalid company documents folder path")
        return

    pdf_files = list(COMPANY_DOCS_FOLDER.glob("*.pdf"))

    if not pdf_files:
        st.warning("⚠️ No company PDFs found.")
        return

    existing_ids = set(collection.get()["ids"])
    added, skipped, failed = 0, 0, []

    for pdf_path in pdf_files:
        file_name = pdf_path.name

        if file_name in existing_ids:
            skipped += 1
            continue

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)

            if not text.strip():
                failed.append(file_name)
                continue

            embedding = embedder.encode(text).tolist()

            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[file_name]
            )
            added += 1

        except Exception:
            failed.append(file_name)

    if added:
        st.success(f"✅ {added} company profiles embedded.")
    if skipped:
        st.info(f"ℹ️ {skipped} profiles already exist.")
    if failed:
        st.error(f"❌ Failed files: {failed}")

# ========== SEARCH ==========
def search_top_k_companies(query_text, top_k=3):
    query_vector = embedder.encode(query_text).tolist()

    return collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "distances"]
    )

# ========== RAG PROMPT ==========
def build_rag_prompt(query, company_ids, company_texts):
    context_blocks = []

    for idx, (cid, text) in enumerate(zip(company_ids, company_texts), start=1):
        context_blocks.append(
            f"""
[Candidate {idx}]
Document ID: {cid}
Document Content:
{text}
"""
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""
You are a STRICT ranking and evaluation engine.

Rules:
- Use ONLY provided content
- No assumptions
- Score between 0–100
- Rank BEST to LEAST relevant

QUERY:
{query}

CONTEXT:
{context}

OUTPUT FORMAT:

Rank X:
- Document ID:
- Query Match Score: XX%
- Reasons:
  - Skill overlap
  - Experience alignment
  - Domain suitability
  - Missing areas
"""

# ========== LLM GENERATION ==========
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are an expert company analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ========== STREAMLIT UI ==========
st.divider()
with st.expander("Step 1: Embed Company Documents"):
    st.code(str(COMPANY_DOCS_FOLDER))
    if st.button("Embed All Company PDFs"):
        embed_and_store_company_docs()

st.divider()
user_query = st.text_area(
    "Step 2: Enter your company-related query",
    height=200
)

if user_query.strip():
    top_k = st.number_input("Top K Companies", 1, 10, 3)

    if st.button("Analyze Companies"):
        results = search_top_k_companies(user_query, top_k)

        distances = results["distances"][0]
        docs = results["documents"][0]
        ids = results["ids"][0]

        if not docs or min(distances) > SIMILARITY_THRESHOLD:
            st.warning("❌ No relevant company profiles found.")
        else:
            prompt = build_rag_prompt(user_query, ids, docs)
            answer = generate_answer(prompt)
            st.subheader("✅ Company Analysis")
            st.write(answer)
else:
    st.info("Enter a query to begin company profiling.")
