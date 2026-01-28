# ========== IMPORTS ==========
import os
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# ========== LOAD ENV ==========
load_dotenv()  # 🔴 VERY IMPORTANT

# ========== STREAMLIT CONFIG ==========
st.set_page_config(
    page_title="Resume Matcher - TRUE RAG",
    layout="centered"
)

# ========== PATHS ==========
RESUME_FOLDER = "/home/uadmin/Desktop/ResumeAnalyzerProject/Resume_Ananlysis_project_using_llm_langchain/resume_samples"
CHROMA_PATH = "chroma_db"

SIMILARITY_THRESHOLD = 0.7  # cosine distance

# ========== LOAD EMBEDDING MODEL ==========
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ========== OPENAI CLIENT ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY not found. Please set it in .env file.")
    st.stop()


client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# ========== CHROMA DB ==========
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(
    name="resumes",
    metadata={"hnsw:space": "cosine"}
)

# ========== EMBED & STORE RESUMES ==========
def embed_and_store_resumes():
    pdf_files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        st.warning("⚠️ No PDF resumes found.")
        return

    existing_ids = set(collection.get()["ids"])
    added, skipped, failed = 0, 0, []

    for file in pdf_files:
        if file in existing_ids:
            skipped += 1
            continue

        try:
            path = os.path.join(RESUME_FOLDER, file)

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

        except Exception as e:
            failed.append(file)

    if added:
        st.success(f"✅ {added} resumes embedded successfully.")
    if skipped:
        st.info(f"ℹ️ {skipped} resumes already exist.")
    if failed:
        st.error(f"❌ Failed resumes: {failed}")

# ========== SEARCH ==========
def search_top_k_resumes(jd_text, top_k=3):
    jd_vector = embedder.encode(jd_text).tolist()

    return collection.query(
        query_embeddings=[jd_vector],
        n_results=top_k,
        include=["documents", "distances"]
    )

# ========== RAG PROMPT ==========
def build_rag_prompt(jd, resume_ids, resume_texts):
    context_blocks = []

    for i, (rid, text) in enumerate(zip(resume_ids, resume_texts), start=1):
        context_blocks.append(
            f"Candidate {i} (Resume ID: {rid}):\n{text}"
        )

    combined_context = "\n\n---\n\n".join(context_blocks)

    return f"""
You are an expert HR assistant.

Use ONLY the resume context below.
If information is missing, say "Information not available".

Resume Context:
{combined_context}

Job Description:
{jd}

Question:
For EACH candidate:
1. Explain suitability for the job
2. Mention key matching skills
3. Mention major gaps (if any)

Respond in a clear, structured format.
"""


# ========== LLM GENERATION ==========
def generate_answer(prompt):
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "You are a professional recruiter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content

# ========== STREAMLIT UI ==========
st.title("📄 Resume Matcher – TRUE RAG System")

st.markdown(
    "This system retrieves relevant resumes from a vector database and generates an explanation using an LLM (RAG)."
)

# ---- STEP 1: EMBED RESUMES ----
with st.expander("Step 1: Embed Resumes"):
    st.code(RESUME_FOLDER)
    if st.button("Embed All Resumes"):
        embed_and_store_resumes()

# ---- STEP 2: JOB DESCRIPTION ----
st.divider()
jd_input = st.text_area(
    "Step 2: Paste Job Description",
    height=220
)

# ---- STEP 3: RAG SEARCH ----
if jd_input.strip():
    st.divider()
    top_k = st.number_input("Top K Resumes", 1, 10, 3)

    if st.button("Analyze Top Candidates"):
        results = search_top_k_resumes(jd_input, top_k)

        distances = results["distances"][0]
        docs = results["documents"][0]
        ids = results["ids"][0]

        if not docs or min(distances) > SIMILARITY_THRESHOLD:
            st.warning("❌ No relevant resumes found. LLM not triggered.")
        else:
            prompt = build_rag_prompt(
                jd=jd_input,
                resume_ids=ids,
                resume_texts=docs
            )

            answer = generate_answer(prompt)

            st.subheader("✅ Top Candidates Analysis")
            st.write(answer)

else:
    st.info("Paste a job description to begin.")

