import streamlit as st
import os
import json
import io
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.embeddings import Embeddings
from typing import List
import docx

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PapLex AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%); }
    .hero-box {
        background: linear-gradient(135deg, #2d1b69 0%, #1a0533 50%, #0d1b4b 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #4a2d8a;
    }
    .hero-title { font-size: 3rem; font-weight: 800; color: #a78bfa; margin: 0; }
    .hero-sub { font-size: 1.1rem; color: #c4b5fd; margin: 0.5rem 0; }
    .hero-tags { color: #8b7cf8; font-size: 0.9rem; }
    .feature-card {
        background: linear-gradient(135deg, #1e1b4b, #1a1a3e);
        border: 1px solid #4c1d95;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .chat-user {
        background: linear-gradient(135deg, #2d1b69, #1e1b4b);
        border-radius: 12px 12px 4px 12px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #7c3aed;
        color: #e2d9f3;
    }
    .chat-ai {
        background: linear-gradient(135deg, #0d1b4b, #1a1a2e);
        border-radius: 12px 12px 12px 4px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #06b6d4;
        color: #e2e8f0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6d28d9, #4338ca);
    }
    .sidebar .stRadio label { color: #c4b5fd; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1a, #1a1a2e); }
</style>
""", unsafe_allow_html=True)

# ── Groq Embeddings (Lightweight — no torch needed) ───────────────────────────
class GroqEmbeddings(Embeddings):
    """Lightweight embeddings using Groq API — no torch/sentence-transformers needed."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def _embed(self, text: str) -> List[float]:
        # Use a simple TF-IDF style hash embedding as fallback
        # This is lightweight and works without heavy ML libraries
        import hashlib
        words = text.lower().split()
        vec = [0.0] * 384
        for i, word in enumerate(words[:384]):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % 384] += 1.0
        norm = sum(x**2 for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

# ── Helper: Extract text from files ───────────────────────────────────────────
def extract_text(file) -> str:
    name = file.name.lower()
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(file)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        elif name.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".csv"):
            df = pd.read_csv(file)
            return df.to_string()
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
            return df.to_string()
        elif name.endswith(".json"):
            data = json.load(file)
            return json.dumps(data, indent=2)
        elif name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        else:
            return file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Error reading file: {e}"

# ── Build FAISS vector store ───────────────────────────────────────────────────
def build_vectorstore(texts: List[str], api_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    embeddings = GroqEmbeddings(api_key=api_key)
    return FAISS.from_texts(chunks, embeddings)

# ── Groq LLM Call ─────────────────────────────────────────────────────────────
def groq_chat(messages: list, api_key: str, model="llama-3.3-70b-versatile") -> str:
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0.7
    )
    return resp.choices[0].message.content

# ── Language map ──────────────────────────────────────────────────────────────
LANGUAGES = {
    "English": "English",
    "Hindi": "Hindi",
    "French": "French",
    "Spanish": "Spanish",
    "German": "German",
    "Arabic": "Arabic",
    "Chinese": "Chinese (Simplified)",
    "Japanese": "Japanese"
}

# ── Session State Init ────────────────────────────────────────────────────────
for key, val in {
    "chat_history": [],
    "vectorstore": None,
    "file_texts": [],
    "file_names": [],
    "dataframes": {},
    "page": "Chat"
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 PapLex AI")
    st.markdown('<span style="background:#4c1d95;color:#c4b5fd;padding:2px 10px;border-radius:20px;font-size:0.8rem;">Powered by LLaMA 3.3</span>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🧭 Navigation")
    page = st.radio("", ["💬 Chat", "🔄 Convert Files", "📊 Visualize"], label_visibility="collapsed")
    st.session_state.page = page

    st.markdown("---")
    st.markdown("### 🌐 Language")
    lang = st.selectbox("Answer language:", list(LANGUAGES.keys()))

    st.markdown("---")
    st.markdown("### 📁 Upload Files")
    st.caption("PDF • CSV • Excel • JSON • TXT • Word")
    uploaded = st.file_uploader(
        "Upload your files",
        type=["pdf", "txt", "csv", "xlsx", "xls", "json", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded:
        if st.button("🚀 Process Files"):
            with st.spinner("Processing files..."):
                st.session_state.file_texts = []
                st.session_state.file_names = []
                st.session_state.dataframes = {}
                for f in uploaded:
                    text = extract_text(f)
                    st.session_state.file_texts.append(text)
                    st.session_state.file_names.append(f.name)
                    if f.name.endswith((".csv", ".xlsx", ".xls")):
                        f.seek(0)
                        try:
                            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
                            st.session_state.dataframes[f.name] = df
                        except:
                            pass
                api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
                if api_key:
                    st.session_state.vectorstore = build_vectorstore(st.session_state.file_texts, api_key)
                    st.success(f"✅ {len(uploaded)} file(s) processed!")
                else:
                    st.error("❌ GROQ_API_KEY not found!")

    if st.session_state.file_names:
        st.markdown("### 📄 Loaded Files")
        for name in st.session_state.file_names:
            st.markdown(f"✅ `{name}`")

    st.markdown("---")
    if st.button("🗑️ Clear All"):
        for key in ["chat_history", "vectorstore", "file_texts", "file_names", "dataframes"]:
            st.session_state[key] = [] if key != "vectorstore" and key != "dataframes" else ({} if key == "dataframes" else None)
        st.rerun()

# ── Get API Key ───────────────────────────────────────────────────────────────
api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
if "Chat" in st.session_state.page:

    # Hero
    st.markdown("""
    <div class="hero-box">
        <div class="hero-title">🧠 PapLex AI</div>
        <div class="hero-sub">Your Intelligent Document Assistant</div>
        <div class="hero-tags">Chat • Convert • Visualize • Quiz • Voice • Multilingual</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.file_texts:
        st.markdown("""
        <div style="text-align:center; padding:2rem;">
            <h3 style="color:#a78bfa;">👆 Upload files from sidebar to get started!</h3>
            <p style="color:#8b7cf8;">Supports PDF • CSV • Excel • JSON • TXT • Word</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="feature-card"><h4 style="color:#a78bfa;">💬 Smart Q&A</h4><p style="color:#c4b5fd;">Ask anything about your documents with AI-powered answers</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="feature-card"><h4 style="color:#a78bfa;">📊 Visualize Data</h4><p style="color:#c4b5fd;">Auto-generate charts and graphs from CSV/Excel files</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="feature-card"><h4 style="color:#a78bfa;">🌐 Multilingual</h4><p style="color:#c4b5fd;">Get answers in 8 different languages</p></div>', unsafe_allow_html=True)

    else:
        # Smart Action Buttons
        st.markdown("### ⚡ Smart Actions")
        col1, col2, col3, col4, col5 = st.columns(5)

        action = None
        with col1:
            if st.button("📝 Summarize"):
                action = "summarize"
        with col2:
            if st.button("❓ Quiz Me"):
                action = "quiz"
        with col3:
            if st.button("🔍 Extract Entities"):
                action = "ner"
        with col4:
            if st.button("💡 Suggest Questions"):
                action = "suggest"
        with col5:
            if st.button("🔄 Compare Docs"):
                action = "compare"

        if action and api_key:
            full_text = "\n\n".join(st.session_state.file_texts)[:6000]
            lang_name = LANGUAGES[lang]

            prompts = {
                "summarize": f"Summarize the following document(s) in {lang_name}. Be comprehensive but concise:\n\n{full_text}",
                "quiz": f"Generate 5 multiple choice quiz questions from this document in {lang_name}. Include answers:\n\n{full_text}",
                "ner": f"Extract all named entities (people, places, organizations, dates, numbers) from this document in {lang_name}:\n\n{full_text}",
                "suggest": f"Suggest 8 insightful questions a user could ask about this document in {lang_name}:\n\n{full_text}",
                "compare": f"Compare and contrast the main themes across these documents in {lang_name}:\n\n{full_text}"
            }

            with st.spinner("🤖 AI is thinking..."):
                result = groq_chat([
                    {"role": "system", "content": f"You are PapLex AI, an intelligent document assistant. Always respond in {lang_name}."},
                    {"role": "user", "content": prompts[action]}
                ], api_key)
                st.session_state.chat_history.append(("🤖 Smart Action", result))

        st.markdown("---")

        # Chat History
        if st.session_state.chat_history:
            st.markdown("### 💬 Conversation")
            for q, a in st.session_state.chat_history:
                st.markdown(f'<div class="chat-user">👤 <strong>{q}</strong></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-ai">🤖 {a}</div>', unsafe_allow_html=True)

        # Chat Input
        st.markdown("### 💬 Ask Anything")
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_q = st.text_input("", placeholder="Ask a question about your documents...", label_visibility="collapsed")
        with col_btn:
            ask = st.button("Send 🚀")

        if ask and user_q and api_key:
            with st.spinner("🤖 Thinking..."):
                # Get context from vectorstore
                context = ""
                if st.session_state.vectorstore:
                    docs = st.session_state.vectorstore.similarity_search(user_q, k=4)
                    context = "\n\n".join(d.page_content for d in docs)

                lang_name = LANGUAGES[lang]
                messages = [
                    {"role": "system", "content": f"You are PapLex AI. Answer based on the document context provided. Always respond in {lang_name}. Context:\n{context}"},
                ]
                for q, a in st.session_state.chat_history[-3:]:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user", "content": user_q})

                answer = groq_chat(messages, api_key)
                st.session_state.chat_history.append((user_q, answer))
                st.rerun()

        elif ask and not api_key:
            st.error("❌ GROQ_API_KEY not configured!")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CONVERT FILES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Convert" in st.session_state.page:
    st.markdown("## 🔄 File Converter")
    st.markdown("Convert between different file formats easily.")

    conv_file = st.file_uploader(
        "Upload file to convert",
        type=["pdf", "txt", "csv", "xlsx", "json", "docx"]
    )

    if conv_file:
        name = conv_file.name
        st.success(f"✅ Loaded: `{name}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**From:** `{name.split('.')[-1].upper()}`")
        with col2:
            target = st.selectbox("**To:**", ["TXT", "CSV", "JSON"])

        if st.button("🔄 Convert Now"):
            text = extract_text(conv_file)

            if target == "TXT":
                buf = io.BytesIO(text.encode())
                st.download_button("⬇️ Download TXT", buf, file_name=name.rsplit(".", 1)[0] + ".txt", mime="text/plain")

            elif target == "CSV":
                rows = [[line] for line in text.split("\n") if line.strip()]
                df = pd.DataFrame(rows, columns=["Content"])
                buf = io.BytesIO()
                df.to_csv(buf, index=False)
                buf.seek(0)
                st.download_button("⬇️ Download CSV", buf, file_name=name.rsplit(".", 1)[0] + ".csv", mime="text/csv")

            elif target == "JSON":
                lines = [line for line in text.split("\n") if line.strip()]
                data = {"filename": name, "content": lines}
                buf = io.BytesIO(json.dumps(data, indent=2).encode())
                st.download_button("⬇️ Download JSON", buf, file_name=name.rsplit(".", 1)[0] + ".json", mime="application/json")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUALIZE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Visualize" in st.session_state.page:
    st.markdown("## 📊 Data Visualizer")

    if not st.session_state.dataframes:
        st.info("📁 Upload a CSV or Excel file from the sidebar to visualize data.")
    else:
        file_choice = st.selectbox("Select file:", list(st.session_state.dataframes.keys()))
        df = st.session_state.dataframes[file_choice]

        st.markdown(f"**Shape:** `{df.shape[0]} rows × {df.shape[1]} columns`")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📈 Create Chart")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram", "Heatmap"])
        with col2:
            x_col = st.selectbox("X Axis", all_cols)
        with col3:
            y_col = st.selectbox("Y Axis", numeric_cols) if numeric_cols else st.selectbox("Y Axis", all_cols)

        if st.button("📊 Generate Chart"):
            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=["#7c3aed"])
                elif chart_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=["#06b6d4"])
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=["#a78bfa"])
                elif chart_type == "Pie":
                    fig = px.pie(df, names=x_col, values=y_col)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, color_discrete_sequence=["#7c3aed"])
                elif chart_type == "Heatmap":
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, color_continuous_scale="Purples")

                fig.update_layout(
                    paper_bgcolor="#1a1a2e",
                    plot_bgcolor="#0f0f1a",
                    font_color="#c4b5fd",
                    title=f"{chart_type} Chart: {x_col} vs {y_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#6b7280;font-size:0.85rem;">🧠 PapLex AI • Built by <strong style="color:#a78bfa;">Shyama Mishra</strong> • B.Tech CSE Data Science, Galgotias 2027 • Powered by LLaMA 3.3 70B • Groq • LangChain • FAISS</div>',
    unsafe_allow_html=True
)
