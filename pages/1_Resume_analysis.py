import os

import pdfplumber
import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

# ================== Page Config ==================
st.set_page_config(
    page_title="ATS Resume Analyzer",
    layout="wide"
)

st.title("📄 ATS Resume vs Job Description Analyzer")

# ================== Groq Client ==================



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
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)
      
def get_similarity_score(text1, text2):
    emb = model.encode([text1, text2])
    return cosine_similarity([emb[0]], [emb[1]])[0][0]

def extract_category(text, keywords):
    text = text.lower()
    return "\n".join(
        line for line in text.split("\n")
        if any(word in line for word in keywords)
    )

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

# ================== UI Inputs ==================
uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text = st.text_area("Paste Job Description", height=250)

analyze_btn = st.button("🔍 Analyze Resume")

# ================== Analysis ==================
if analyze_btn and uploaded_resume and jd_text:

    with st.spinner("Analyzing resume..."):
        resume_text = extract_text_from_pdf(uploaded_resume)

        overall_score = get_similarity_score(resume_text, jd_text)
        formatted_score = f"{overall_score * 100:.2f}"

        categories = {
    "Technical Skills": [
        "python","java","c++","sql","nosql","javascript",
        "flask","fastapi","django",
        "machine learning","deep learning","nlp",
        "etl","data pipeline","data engineering",
        "pandas","numpy","scikit-learn","tensorflow","pytorch"
    ],

    "Soft Skills": [
        "communication","teamwork","collaboration",
        "leadership","problem solving","critical thinking",
        "time management","adaptability"
    ],

    "Experience": [
        "experience","work experience","projects",
        "internship","industrial training","freelance",
        "research","capstone","production"
    ],

    "Tools & Platforms": [
        "git","github","docker","kubernetes",
        "aws","azure","gcp",
        "linux","jira","postman"
    ],

    "Databases": [
        "mysql","postgresql","mongodb","redis",
        "oracle","sqlite","cassandra"
    ],

    "Location": [
        "remote","hyderabad","bangalore","pune",
        "chennai","mumbai","india","onsite","hybrid"
    ]
        }

        category_scores = {}
        for cat, keywords in categories.items():
            rs = extract_category(resume_text, keywords)
            js = extract_category(jd_text, keywords)
            score = get_similarity_score(rs, js) if rs and js else 0
            category_scores[cat] = f"{score * 100:.2f}"


        explanation = call_llm(
            score_explainer_prompt.format(
                resume=resume_text,
                jd=jd_text,
                score=formatted_score
            )
        )

        suggestions = call_llm(
            improvement_prompt.format(
                resume=resume_text,
                jd=jd_text
            )
        )

    # ================== Output ==================
    st.success(f"Overall Match Score: **{formatted_score}%**")

    st.subheader("📊 Category-wise Scores")
    for cat, val in category_scores.items():
        st.progress(float(val) / 100)
        st.write(f"**{cat}: {val}%**")

    st.subheader("🧠 ATS Explanation")
    st.write(explanation)

    st.subheader("🛠 Improvement Suggestions")
    st.write(suggestions)

elif analyze_btn:
    st.warning("Please upload a resume and paste a job description.")
