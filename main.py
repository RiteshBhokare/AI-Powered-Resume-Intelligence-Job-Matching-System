import streamlit as st

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# ================== GLOBAL STYLES ==================
st.markdown(
    """
    <style>
    body, .main {
        background: radial-gradient(1200px 600px at 50% -200px, #0b1c3d 0%, #000 60%);
        color: #e6eef8;
        font-family: Inter, system-ui, Segoe UI, Arial;
    }

    /* HERO */
    .hero {
        text-align: center;
        margin-top: 40px;
        margin-bottom: 40px;
    }
    .hero-icon {
        font-size: 48px;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 36px;
        font-weight: 800;
        margin: 6px 0;
    }
    .hero-subtitle {
        color: #9aa6bf;
        font-size: 16px;
        margin-bottom: 12px;
    }
    .hero-desc {
        max-width: 820px;
        margin: 0 auto;
        color: #bfcfe6;
        font-size: 15px;
        line-height: 1.6;
    }

    /* CARDS */
    .card {
        background: linear-gradient(180deg, #071229, #050d1c);
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.65);
        height: 100%;
    }
    .card-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .card-desc {
        font-size: 13px;
        color: #9aa6bf;
        margin-bottom: 18px;
        line-height: 1.5;
    }

    /* LIGHT BLUE BUTTON */
    div.stButton > button {
        background: linear-gradient(180deg, #4da3ff, #1f7cff);
        color: white;
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 600;
        border: none;
        box-shadow: 0 6px 18px rgba(77,163,255,0.35);
    }
    div.stButton > button:hover {
        background: linear-gradient(180deg, #66b2ff, #2f8cff);
        box-shadow: 0 8px 22px rgba(77,163,255,0.45);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== HERO ==================
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">📄</div>
        <div class="hero-title">Resume Matcher</div>
        <div class="hero-subtitle">AI-Powered Talent Matcher</div>
        <div class="hero-desc">
            Upload resumes or query company profiles to get ATS-style matches,
            category breakdowns and concise improvement suggestions —
            built for hiring teams and candidates.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ================== ACTION CARDS ==================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">📄 Resume Analysis</div>
            <div class="card-desc">
                Compare a resume against a Job Description and get an ATS match score
                with actionable improvement suggestions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Open", key="open_resume"):
        st.switch_page("pages/1_Resume_analysis.py")

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-title">🏢 Company Profiling</div>
            <div class="card-desc">
                Search company documents stored in the vector database and generate
                concise RAG-based analysis and ranked insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Open", key="open_company"):
        st.switch_page("pages/2_company_profilling.py")
