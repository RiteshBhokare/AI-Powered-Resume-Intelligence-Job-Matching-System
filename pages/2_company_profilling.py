# ========== IMPORTS ==========
import os
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
    layout="centered"
)

# ========== PATHS ==========
COMPANY_DOCS_FOLDER = "/home/uadmin/Desktop/ResumeAnalyzerProject/AI-Powered-Resume-Intelligence-Job-Matching-System/resume_samples"
CHROMA_PATH = "chroma_db_company"

SIMILARITY_THRESHOLD = 0.7  # cosine distance

# ========== LOAD EMBEDDING MODEL ==========
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ========== GROQ CLIENT ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ========== CHROMA DB ==========
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="company_profiles",
    metadata={"hnsw:space": "cosine"}
)

# ========== EMBED & STORE COMPANY DOCS ==========
def embed_and_store_company_docs():
    pdf_files = [f for f in os.listdir(COMPANY_DOCS_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        st.warning("⚠️ No company PDFs found.")
        return

    existing_ids = set(collection.get()["ids"])
    added, skipped, failed = 0, 0, []

    for file in pdf_files:
        if file in existing_ids:
            skipped += 1
            continue

        try:
            path = os.path.join(COMPANY_DOCS_FOLDER, file)

            with pdfplumber.open(path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)

            if not text.strip():
                failed.append(file)
                continue

            embedding = embedder.encode(text).tolist()

            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[file]
            )
            added += 1

        except Exception:
            failed.append(file)

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

MANDATORY RULES:
- Use ONLY the provided document content.
- Use ONLY the given query (job description).
- Do NOT add assumptions or external knowledge.
- If information is missing, write exactly: "Information not available".
- Match Score must be between 0 and 100 (percentage).
- Rank candidates from BEST to LEAST relevant.
- Output ONLY in the format defined below.
- Do NOT include explanations outside the defined structure.

========================
QUERY / JOB DESCRIPTION
========================
{query}

========================
CANDIDATE CONTEXT
========================
{context}

========================
OUTPUT FORMAT (STRICT)
========================

TOP {len(company_ids)} CANDIDATES (RANKED)

For EACH candidate, follow this EXACT format:

Rank X:
- Document ID:
- Query Match Score: XX%
- Reasons:
  - Skill or requirement overlap with query
  - Relevant experience alignment
  - Role or domain suitability
  - Missing or weak areas (if any)

========================
Generate the final ranked result now.
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
st.title("🏢 Company Profiling – TRUE RAG System")

st.markdown(
    "This module retrieves relevant company documents from a vector database "
    "and generates insights using Retrieval-Augmented Generation (RAG)."
)

# ---- STEP 1: EMBED COMPANY DOCS ----
with st.expander("Step 1: Embed Company Documents"):
    st.code(COMPANY_DOCS_FOLDER)
    if st.button("Embed All Company PDFs"):
        embed_and_store_company_docs()

# ---- STEP 2: USER QUERY ----
st.divider()
user_query = st.text_area(
    "Step 2: Enter your company-related query",
    height=200
)

# ---- STEP 3: RAG SEARCH ----
if user_query.strip():
    st.divider()
    top_k = st.number_input("Top K Companies", 1, 10, 3)

    if st.button("Analyze Companies"):
        results = search_top_k_companies(user_query, top_k)

        distances = results["distances"][0]
        docs = results["documents"][0]
        ids = results["ids"][0]

        if not docs or min(distances) > SIMILARITY_THRESHOLD:
            st.warning("❌ No relevant company profiles found. LLM not triggered.")
        else:
            prompt = build_rag_prompt(
                query=user_query,
                company_ids=ids,
                company_texts=docs
            )

            answer = generate_answer(prompt)

            st.subheader("✅ Company Analysis")
            st.write(answer)
else:
    st.info("Enter a query to begin company profiling.")




