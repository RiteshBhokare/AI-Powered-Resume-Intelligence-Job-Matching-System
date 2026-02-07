import os
import io
import pdfplumber
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# ================== Page Config ==================
st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styles matching preview.html
st.markdown(
    """
    <style>
    body{font-family:Inter,system-ui,Segoe UI,Arial;margin:0;background:#000;color:#e6eef8}
    .container{max-width:1100px;margin:28px auto;padding:20px}
    .title{font-size:28px;font-weight:700}
    .subtitle{color:#9aa6bf;margin-top:6px}
    .card{background:#071229;padding:18px;border-radius:10px;box-shadow:0 8px 30px rgba(0,0,0,0.6)}
    .small{font-size:13px;color:#9aa6bf}
    .uploader{height:110px;border:2px dashed #1f2a38;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#9aa6bf;background:linear-gradient(180deg,rgba(255,255,255,0.02),transparent)}
    textarea{width:100%;border:1px solid #1f2a38;border-radius:8px;padding:10px;resize:vertical;background:#04101a;color:#e6eef8}
    .stButton>button{background:#0b5cff;color:#fff;border:none;padding:10px 14px;border-radius:8px;cursor:pointer;font-weight:600}
    .stButton>button:hover{opacity:0.9}
    .metric{font-size:20px;font-weight:600}
    .score{font-size:34px;font-weight:700;color:#7cc0ff}
    .progress{height:12px;background:#071229;border-radius:8px;overflow:hidden}
    .progress > i{display:block;height:100%;background:linear-gradient(90deg,#0b5cff,#47a7ff);}
    details{background:linear-gradient(180deg,#061426,#04101a);padding:12px;border-radius:8px;border:1px solid #123047}
    a{color:#9fd1ff}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='container'><div class='title'>📄 ATS Resume vs Job Description Analyzer</div><div class='subtitle'>Upload a resume and paste the Job Description to get an ATS-style match, category breakdown and actionable improvements.</div></div>", unsafe_allow_html=True)

# ================ Groq/OpenAI Client ==================
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


def call_llm(prompt, model="openai/gpt-oss-20b"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# ================== Helper Functions ==================
def extract_text_from_pdf(uploaded_file):
    # accept file-like or path
    if hasattr(uploaded_file, "read"):
        uploaded_file.seek(0)
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    else:
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)


model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


def get_similarity_score(text1, text2):
    emb = model.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]


def extract_category(text, keywords):
    text = (text or "").lower()
    return "\n".join(line for line in text.split("\n") if any(word in line for word in keywords))


# ================== Prompt Templates ==================
score_explainer_prompt = PromptTemplate.from_template("""
You are an advanced ATS system used by top tech companies.

ATS Score: {score}%

Resume:
{resume}

Job Description:
{jd}

Explain:
1. Skill gaps
2. Missing tools or technologies
3. Experience mismatch
4. How to improve ATS score

Keep it concise and actionable.
""")


improvement_prompt = PromptTemplate.from_template("""
You are a resume improvement expert.

Resume:
{resume}

Job Description:
{jd}

Give category-wise suggestions:

1. Technical Skills
2. Soft Skills
3. Experience
4. Location
5. Grammar & Formatting
""")


# ================== Sidebar ==================
st.sidebar.header("How to use")
st.sidebar.write("- Upload a PDF resume (one file).\n- Paste the target Job Description.\n- Click Analyze to view results and suggestions.")

# try load sample JD if present
sample_jd = ""
try:
    root = os.getcwd()
    sample_path = os.path.join(root, "PythonJD.txt")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            sample_jd = f.read()
            st.sidebar.markdown("**Example Job Description (sidebar)**")
            st.sidebar.text(sample_jd[:800] + ("..." if len(sample_jd) > 800 else ""))
except Exception:
    pass


# ================== UI Inputs (form) ==================
with st.form(key="analyze_form"):
    c1, c2 = st.columns([1, 2])
    with c1:
        uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"], help="Single PDF file")
        st.markdown("<div class='small'>Accepted: PDF resumes. We'll parse text and compare to JD.</div>", unsafe_allow_html=True)
    with c2:
        jd_text = st.text_area("Paste Job Description", height=260, value=sample_jd)

    analyze_btn = st.form_submit_button("🔍 Analyze Resume")


# ================== Analysis ==================
if analyze_btn:
    if not uploaded_resume or not jd_text or jd_text.strip() == "":
        st.warning("Please upload a resume and paste a job description.")
    else:
        with st.spinner("Analyzing resume and computing similarity..."):
            resume_text = extract_text_from_pdf(uploaded_resume)

            overall_score = get_similarity_score(resume_text, jd_text)
            formatted_score = float(f"{overall_score * 100:.2f}")

            categories = {
                "Technical Skills": ["python", "java", "c++", "sql", "nosql", "javascript", "flask", "fastapi", "django", "machine learning", "deep learning", "nlp", "etl", "data pipeline", "data engineering", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch"],
                "Soft Skills": ["communication", "teamwork", "collaboration", "leadership", "problem solving", "critical thinking", "time management", "adaptability"],
                "Experience": ["experience", "work experience", "projects", "internship", "industrial training", "freelance", "research", "capstone", "production"],
                "Tools & Platforms": ["git", "github", "docker", "kubernetes", "aws", "azure", "gcp", "linux", "jira", "postman"],
                "Databases": ["mysql", "postgresql", "mongodb", "redis", "oracle", "sqlite", "cassandra"],
                "Location": ["remote", "hyderabad", "bangalore", "pune", "chennai", "mumbai", "india", "onsite", "hybrid"]
            }

            category_scores = {}
            for cat, keywords in categories.items():
                rs = extract_category(resume_text, keywords)
                js = extract_category(jd_text, keywords)
                score = get_similarity_score(rs, js) if rs and js else 0
                category_scores[cat] = float(f"{score * 100:.2f}")

            explanation = call_llm(
                score_explainer_prompt.format(resume=resume_text, jd=jd_text, score=f"{formatted_score:.2f}")
            )

            suggestions = call_llm(improvement_prompt.format(resume=resume_text, jd=jd_text))

        # ================ Output ==================
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        left, right = st.columns([1, 2])

        with left:
            st.markdown("<div class='metric'>Overall Match</div>", unsafe_allow_html=True)
            st.metric(label="ATS Match Score", value=f"{formatted_score:.2f}%")
            st.progress(min(max(formatted_score / 100.0, 0.0), 1.0))
            st.markdown("---")
            st.markdown("<div class='small'>Downloadable Results</div>", unsafe_allow_html=True)
            result_txt = f"Overall Score: {formatted_score:.2f}%\n\nSuggestions:\n{suggestions}\n\nExplanation:\n{explanation}"
            st.download_button("Download Suggestions", result_txt, file_name="ats_suggestions.txt")

        with right:
            st.subheader("📊 Category-wise Scores")
            # show scores in two columns for compactness
            cats = list(category_scores.items())
            for i in range(0, len(cats), 2):
                cols = st.columns(2)
                for j, (cat, val) in enumerate(cats[i:i+2]):
                    with cols[j]:
                        st.write(f"**{cat}**")
                        st.progress(min(max(val / 100.0, 0.0), 1.0))
                        st.write(f"{val:.2f}%")

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🧠 ATS Explanation", expanded=False):
            st.write(explanation)

        with st.expander("🛠 Improvement Suggestions", expanded=True):
            st.write(suggestions)

        st.markdown("---")
        st.info("Tip: Tailor the top technical skills and keywords in your resume to the Job Description to increase your ATS match.")
