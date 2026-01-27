# ========== Import all required libraries ==========

import os
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb

# ========== Setup Streamlit and file paths ==========

st.set_page_config(
    page_title="Company Profiling - Resume Matcher",
    layout="centered"
)

# Folder where all resume PDFs are stored
RESUME_FOLDER = "/home/uadmin/Desktop/ResumeAnalyzerProject/Resume_Ananlysis_project_using_llm_langchain/resume_samples"

# Folder where ChromaDB will store the embeddings
CHROMA_PATH = "chroma_db"

# ========== Load models and setup ChromaDB ==========

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# Initialize ChromaDB (persistent)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Explicit cosine similarity (IMPORTANT)
collection = chroma_client.get_or_create_collection(
    name="resumes",
    metadata={"hnsw:space": "cosine"}
)

# ========== Function to Embed Resumes from PDF ==========

def embed_and_store_resumes():
    """
    Reads all PDF resumes from folder, creates embeddings,
    and stores them in ChromaDB with proper status messages.
    """
    pdf_files = [f for f in os.listdir(RESUME_FOLDER) if f.lower().endswith(".pdf")]

    if not pdf_files:
        st.warning("⚠️ No PDF files found in the resume folder.")
        return

    existing_ids = set(collection.get()["ids"])
    added_count = 0
    skipped_count = 0
    failed_files = []

    for file in pdf_files:
        if file in existing_ids:
            skipped_count += 1
            continue

        full_path = os.path.join(RESUME_FOLDER, file)

        try:
            with pdfplumber.open(full_path) as pdf:
                text = " ".join(page.extract_text() or "" for page in pdf.pages)

            if not text.strip():
                failed_files.append(file)
                continue

            embedding = embedder.encode(text).tolist()

            collection.add(
                documents=[text],
                embeddings=[embedding],
                ids=[file]
            )

            added_count += 1

        except Exception:
            failed_files.append(file)

    # -------- FINAL STATUS MESSAGE --------

    total = len(pdf_files)

    if added_count == total:
        st.success("✅ All resumes were embedded successfully!")
    elif added_count > 0 and skipped_count > 0:
        st.success(f"✅ {added_count} resumes embedded successfully.")
        st.info(f"ℹ️ {skipped_count} resumes were already embedded.")
    elif skipped_count == total:
        st.info("ℹ️ All resumes were already embedded. No new embeddings needed.")
    else:
        st.warning("⚠️ Some resumes could not be processed.")

    if failed_files:
        st.error(f"❌ Failed to process {len(failed_files)} resumes:")
        for f in failed_files:
            st.write(f"• {f}")


# ========== Function to Search Top Matching Resumes ==========

def search_top_k_resumes(jd_text, top_k=5):
    """
    Takes a job description and returns top_k matching resumes.
    """
    jd_vector = embedder.encode(jd_text).tolist()

    results = collection.query(
        query_embeddings=[jd_vector],
        n_results=top_k
    )

    return results

# ========== Streamlit UI Starts ==========

st.title("Company Profiling - Resume Matcher (RAG + AI)")
st.markdown(
    "Upload resumes into the database and search for the best matches based on the job description."
)

# --- STEP 1: Embed Resumes ---
with st.expander("Step 1: Embed Resumes from Folder"):
    st.write(f"📂 Resume folder path:")
    st.code(RESUME_FOLDER)
    st.warning("Make sure the folder contains only `.pdf` files.")

    if st.button("Embed All Resumes"):
        embed_and_store_resumes()

# --- STEP 2: Job Description Input ---
st.divider()
jd_input = st.text_area(
    "Step 2: Paste the Job Description",
    height=220,
    placeholder="Paste the job description here..."
)

# --- STEP 3: Search ---
if jd_input.strip():
    st.divider()
    st.subheader("Step 3: Get Top Matching Resumes")

    top_k = st.number_input(
        "Number of top resumes to fetch",
        min_value=1,
        max_value=50,
        value=5
    )

    if st.button("Find Top Matching Resumes"):
        results = search_top_k_resumes(jd_input, top_k)

        if results and results.get("ids"):
            st.markdown("### 🔍 Top Matching Resumes")

            for i, (doc_id, distance) in enumerate(
                zip(results["ids"][0], results["distances"][0])
            ):
                match_percent = (1 - distance) * 100

                st.markdown(
                    f"""
                    **{i+1}. {doc_id}**  
                    🔹 Match Score: **{match_percent:.2f}%**
                    """
                )
                st.markdown("---")
        else:
            st.warning("No matching resumes found.")

else:
    st.info("Paste a job description to search for matching resumes.")

