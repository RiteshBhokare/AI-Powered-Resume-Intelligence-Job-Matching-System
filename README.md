# 📄 AI Powered Resume Intelligence Job Matching System

An end-to-end **AI-powered Resume Intelligence platform** that simulates a modern **ATS (Applicant Tracking System)** and implements **TRUE RAG (Retrieval-Augmented Generation)** for **company profiling**.

This project demonstrates a **clean separation between deterministic ATS scoring and hallucination-safe RAG**, making it **interview-ready and production-aligned**.

---

## 🚀 Features

### 📄 Part A – ATS Resume Analyzer

* Upload resume (PDF)
* Paste job description
* Text extraction using **pdfplumber**
* Semantic embeddings using **Sentence Transformers**
* **Cosine similarity–based ATS score**
* Category-wise similarity
* AI-generated explanation and improvement suggestions
* LLM is **not used for scoring**

---

### 🏢 Part B – TRUE RAG Company Profiling

* Offline document ingestion
* Vector storage using **ChromaDB**
* Top-K semantic retrieval
* **STRICT RAG** (LLM uses only retrieved documents)
* Ranked companies with insights

---

## 🧠 Architecture Overview

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/85231c39-752f-4e4a-a350-c1a8c112271c" />

---

## 🛠 Technology Stack

* Python
* Streamlit
* Sentence Transformers
* ChromaDB
* Groq LLM (GPT-OSS-20B)
* pdfplumber
* scikit-learn
* langchain-core
* python-dotenv

---

## 📁 Project Structure

```
AI-Powered-Resume-Intelligence-Job-Matching-System/
├── main.py
├── pages/
│   ├── 1_Resume_analysis.py
│   ├── 2_company_profilling.py
│   └── .env
├── resume_samples/
├── chroma_db/
├── chroma_db_company/
├── PythonJD.txt
├── requirements.txt
└── README.md
```
### 🔐 Create Groq API Key

1. Visit 👉 [https://console.groq.com/keys](https://console.groq.com/keys)
2. Log in
3. Create an API key
4. Copy the key

---

### 🧾 Add API Key to `.env`

```env
GROQ_API_KEY=your_groq_api_key_here
```

⚠️ Do not commit `.env` to GitHub.

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

---

## 🎯 Use Cases

* ATS score simulation
* Resume optimization
* Skill gap analysis
* Company & job fit analysis
* HR & recruitment intelligence

---
## 👤 Author

* **Ritesh Bhokare**
* **Pranav Shintre**
