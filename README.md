# PapLex AI — Intelligent Document Assistant

![Python](https://img.shields.io/badge/Python-3.8+-3572A5?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-brightgreen?style=flat-square)
![LLaMA](https://img.shields.io/badge/LLaMA-3.3--70B-purple?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Vector--DB-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=flat-square)
![Status](https://img.shields.io/badge/Status-Production--Ready-success?style=flat-square)

> **A production-ready RAG-powered Document Intelligence Platform** —
> Chat with your documents like ChatGPT. Upload any file, ask anything, get answers with source citations.

---

## What is PapLex AI?

PapLex AI is a full-stack Retrieval-Augmented Generation (RAG) system that lets you
upload documents and have intelligent conversations with them. Built with LLaMA 3.3 70B
via Groq API, FAISS vector database, and LangChain — it delivers fast, accurate,
cited answers from your own documents.

---

## Features

| Feature | Details |
|--------|---------|
| 6 File Formats | PDF, CSV, Excel, JSON, TXT, DOCX |
| 8 Languages | English, Hindi, Spanish, French, German, Arabic, Bengali, Tamil |
| Smart Q&A | Multi-document Q&A with source citations and chat history |
| 5 Smart Actions | Summarize, Quiz Generator, Entity Extraction, Document Comparison, Question Suggestion |
| Data Visualization | Bar, line, scatter, pie charts, heatmaps for CSV/Excel files |
| File Converter | 6 format-to-format conversions |
| Voice Input | Ask questions via microphone using SpeechRecognition |

---

## System Architecture
User Upload (PDF/CSV/Excel/JSON/TXT/DOCX)
↓
Document Processor (PyPDF, python-docx, pandas)
↓
Text Chunking + HuggingFace Embeddings
↓
FAISS Vector Database (Semantic Search)
↓
LangChain RAG Pipeline
↓
LLaMA 3.3 70B via Groq API (Ultra-fast inference)
↓
Answer + Source Citations → Streamlit UI
---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | LLaMA 3.3 70B (via Groq API) |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embeddings | HuggingFace Transformers |
| Frontend | Streamlit |
| Visualization | Plotly |
| File Processing | PyPDF, Python-docx, Pandas |
| Voice Input | SpeechRecognition |
| Export | FPDF2 |
| Language | Python 3.8+ |

---

## Smart Actions

- **Summarize** — Get a concise summary of any uploaded document
- **Quiz Generator** — Auto-generate MCQs from document content
- **Entity Extraction** — Extract people, places, dates, organizations
- **Document Comparison** — Compare two documents side by side
- **Question Suggestion** — AI suggests relevant questions to ask

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/shyama230706/paplex-ai.git
cd paplex-ai

# Install dependencies
pip install -r requirements.txt

# Add your Groq API key
# Create a .env file and add:
# GROQ_API_KEY=your_api_key_here

# Run the app
streamlit run app.py
```

---

## Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Generate your API key
4. Paste it in the app when prompted

---

## Project Structure
User Upload (PDF/CSV/Excel/JSON/TXT/DOCX)
↓
Document Processor (PyPDF, python-docx, pandas)
↓
Text Chunking + HuggingFace Embeddings
↓
FAISS Vector Database (Semantic Search)
↓
LangChain RAG Pipeline
↓
LLaMA 3.3 70B via Groq API (Ultra-fast inference)
↓
Answer + Source Citations → Streamlit UI

---


---

## Why PapLex AI Stands Out

- **Production-ready** — not just a tutorial project, fully functional end-to-end
- **Multi-format** — handles 6 different file types in one system
- **Multilingual** — responds in 8 languages based on user preference
- **Fast** — Groq API delivers LLaMA 3.3 70B at lightning speed
- **Cited answers** — every answer shows which document/page it came from

---

## Author

**Shyama Mishra** — B.Tech CSE (Data Science), Galgotias College of Engineering & Technology

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/shyama-mishra-50980628)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat-square&logo=github)](https://github.com/shyama230706)

---

> Built with passion as a flagship GenAI project | Open to Data Scientist / ML Engineer / GenAI Developer roles
