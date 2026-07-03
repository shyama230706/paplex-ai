import streamlit as st
import os
import json
from io import BytesIO

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PapLex AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e2e8f0;
}

section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.97) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}

.main-header {
    background: linear-gradient(135deg, #4c1d95, #6d28d9, #7c3aed);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(124, 58, 237, 0.4);
}
.main-header h1 { font-size: 2.8rem; font-weight: 700; color: white; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0 0; }

.ftag {
    background: rgba(255,255,255,0.15);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    border: 1px solid rgba(255,255,255,0.2);
    margin: 0.2rem;
    display: inline-block;
}

.chat-user {
    background: linear-gradient(135deg, #4c1d95, #6d28d9);
    border-radius: 15px 15px 5px 15px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: white;
    margin-left: 15%;
    box-shadow: 0 4px 15px rgba(109,40,217,0.3);
}
.chat-bot {
    background: rgba(30, 27, 75, 0.85);
    border: 1px solid rgba(139, 92, 246, 0.35);
    border-radius: 15px 15px 15px 5px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
    margin-right: 8%;
}

.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(109,40,217,0.4) !important;
}

.info-card {
    background: rgba(30, 27, 75, 0.6);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 0.8rem 0;
    text-align: center;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(30, 27, 75, 0.85) !important;
    border: 1px solid rgba(139, 92, 246, 0.45) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

div[data-testid="stFileUploader"] {
    background: rgba(30, 27, 75, 0.5) !important;
    border: 2px dashed rgba(139, 92, 246, 0.5) !important;
    border-radius: 15px !important;
    padding: 0.5rem !important;
}

.stSelectbox > div > div {
    background: rgba(30, 27, 75, 0.85) !important;
    border: 1px solid rgba(139, 92, 246, 0.45) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

hr { border-color: rgba(139, 92, 246, 0.25) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
for key, val in {
    "chat_history": [],
    "raw_texts": [],
    "doc_names": [],
    "page": "Chat",
    "voice_text": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── API Key ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── File Extraction ────────────────────────────────────────────────────────────
def extract_text(f):
    name = f.name.lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(f.read())).pages)
        elif name.endswith(".txt"):
            return f.read().decode("utf-8", errors="ignore")
        elif name.endswith(".csv"):
            import pandas as pd
            return pd.read_csv(f).to_string()
        elif name.endswith((".xlsx", ".xls")):
            import pandas as pd
            return pd.read_excel(f).to_string()
        elif name.endswith(".json"):
            return json.dumps(json.load(f), indent=2)
        elif name.endswith(".docx"):
            from docx import Document
            return "\n".join(p.text for p in Document(BytesIO(f.read())).paragraphs)
    except Exception as e:
        return f"Error reading file: {e}"
    return ""

# ── Context Search ─────────────────────────────────────────────────────────────
def get_context(query, texts):
    words = query.lower().split()
    chunks = []
    for t in texts:
        for i in range(0, len(t), 600):
            chunk = t[i:i+900]
            score = sum(chunk.lower().count(w) for w in words)
            chunks.append((score, chunk))
    chunks.sort(reverse=True)
    return "\n\n".join(c for _, c in chunks[:5])

# ── Groq LLM ──────────────────────────────────────────────────────────────────
def ask_llm(prompt, context, language, history=None):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    lang_note = f"Always respond in {language}." if language != "English" else ""
    system = f"""You are PapLex AI, an intelligent document assistant powered by LLaMA 3.3 70B.
Answer accurately based on the document context. Be helpful, detailed, and clear.
{lang_note}
If answer is not in context, say so honestly."""

    messages = [{"role": "system", "content": system}]
    if history:
        for h in history[-5:]:
            messages += [
                {"role": "user", "content": h["user"]},
                {"role": "assistant", "content": h["bot"]}
            ]
    messages.append({"role": "user", "content": f"Context:\n{context[:4000]}\n\nQuestion: {prompt}"})

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1500,
        temperature=0.7
    )
    return resp.choices[0].message.content

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 PapLex AI")
    st.markdown(
        '<span style="background:linear-gradient(135deg,#6d28d9,#7c3aed);'
        'color:white;padding:0.25rem 0.75rem;border-radius:20px;font-size:0.72rem;">'
        'Powered by LLaMA 3.3</span>',
        unsafe_allow_html=True
    )
    st.divider()

    # Navigation
    st.markdown("### 🧭 Navigation")
    for pg in ["💬 Chat", "🔄 Convert Files", "📊 Visualize"]:
        if st.button(pg, use_container_width=True, key=f"nav_{pg}"):
            st.session_state.page = pg.split(" ", 1)[1]

    st.divider()

    # Language
    st.markdown("### 🌍 Language")
    st.caption("Answer language:")
    language = st.selectbox(
        "lang", ["English","Hindi","Spanish","French","German","Japanese","Chinese","Arabic"],
        label_visibility="collapsed"
    )

    st.divider()

    # Upload
    st.markdown("### 📁 Upload Files")
    st.caption("PDF, CSV, Excel, JSON, TXT, Word")
    uploaded = st.file_uploader(
        "files", type=["pdf","txt","csv","xlsx","xls","json","docx"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded:
        if st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("Processing..."):
                texts, names = [], []
                for f in uploaded:
                    t = extract_text(f)
                    if t.strip():
                        texts.append(t)
                        names.append(f.name)
                if texts:
                    st.session_state.raw_texts = texts
                    st.session_state.doc_names = names
                    st.success(f"✅ {len(names)} file(s) ready!")
                else:
                    st.error("Could not extract text.")

    if st.session_state.doc_names:
        st.divider()
        st.markdown("### 📄 Loaded Documents")
        for n in st.session_state.doc_names:
            st.markdown(f"✅ `{n}`")
        if st.button("🗑️ Clear All", use_container_width=True):
            for k in ["raw_texts","doc_names","chat_history"]:
                st.session_state[k] = [] if isinstance(st.session_state[k], list) else ""
            st.rerun()

    if not GROQ_API_KEY:
        st.divider()
        st.markdown("### 🔑 Groq API Key")
        k = st.text_input("key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if k:
            GROQ_API_KEY = k

# ── MAIN PAGES ─────────────────────────────────────────────────────────────────
page = st.session_state.page

# ════════════════════════════════════════════════════════════════════════════════
# CHAT PAGE
# ════════════════════════════════════════════════════════════════════════════════
if "Chat" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 PapLex AI</h1>
        <p>Your Intelligent Document Assistant</p>
        <div style="margin-top:1rem">
            <span class="ftag">💬 Chat</span>
            <span class="ftag">🔄 Convert</span>
            <span class="ftag">📊 Visualize</span>
            <span class="ftag">🌍 Multilingual</span>
            <span class="ftag">🎙️ Voice</span>
            <span class="ftag">📁 6 Formats</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_names:
        st.markdown("""
        <div class="info-card">
            <h2>👋 Welcome to PapLex AI</h2>
            <h4>Your Intelligent Document Assistant</h4>
            <p style="color:rgba(255,255,255,0.7)">👆 Upload files from the sidebar to get started!</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="info-card"><h3>📄 6 Formats</h3><p>PDF • CSV • Excel<br>JSON • TXT • Word</p></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="info-card"><h3>🤖 Smart Actions</h3><p>Q&A • Summary • Quiz<br>NER • Suggestions</p></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="info-card"><h3>🌍 8 Languages</h3><p>English • Hindi • Spanish<br>French • German • More</p></div>', unsafe_allow_html=True)

    else:
        st.markdown(f"### 💬 Chat with your documents")
        st.caption(f"📄 Loaded: {', '.join(st.session_state.doc_names)}")

        # Smart Actions
        st.markdown("#### ⚡ Smart Actions")
        a1, a2, a3, a4, a5 = st.columns(5)
        action = None
        with a1:
            if st.button("📝 Summarize", use_container_width=True): action = "summarize"
        with a2:
            if st.button("❓ Quiz Me", use_container_width=True): action = "quiz"
        with a3:
            if st.button("🏷️ Entities", use_container_width=True): action = "ner"
        with a4:
            if st.button("💡 Suggest", use_container_width=True): action = "suggest"
        with a5:
            if st.button("🔍 Compare", use_container_width=True): action = "compare"

        if action and GROQ_API_KEY:
            action_prompts = {
                "summarize": "Provide a comprehensive, well-structured summary of the document(s) with key points.",
                "quiz":      "Generate 5 multiple choice quiz questions with 4 options each and mark correct answers.",
                "ner":       "Extract and categorize all named entities: Persons, Organizations, Locations, Dates, Numbers, Products.",
                "suggest":   "Suggest 8 insightful and thought-provoking questions someone could ask about this document.",
                "compare":   "If multiple documents, compare and contrast them in detail. Otherwise, analyze key themes and insights."
            }
            ctx = get_context(action_prompts[action], st.session_state.raw_texts)
            with st.spinner("🤖 AI is thinking..."):
                try:
                    resp = ask_llm(action_prompts[action], ctx, language)
                    st.session_state.chat_history.append({"user": f"[{action.upper()}]", "bot": resp})
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

        st.divider()

        # Chat History
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="chat-user">👤 {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bot">🧠 {chat["bot"]}</div>', unsafe_allow_html=True)

        # Voice Input (Browser JS based)
        st.markdown("---")
        st.markdown("""
        <div id="voice-container">
            <button onclick="startVoice()" style="
                background: linear-gradient(135deg, #6d28d9, #7c3aed);
                color: white; border: none; border-radius: 10px;
                padding: 0.5rem 1.2rem; cursor: pointer; font-size: 0.9rem;
                font-weight: 600; margin-bottom: 0.5rem;">
                🎙️ Start Voice Input
            </button>
            <div id="voice-result" style="color: #a78bfa; font-size: 0.85rem; margin-top: 0.3rem;"></div>
        </div>

        <script>
        function startVoice() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                document.getElementById('voice-result').innerText = '⚠️ Voice not supported in this browser. Use Chrome.';
                return;
            }
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            const rec = new SR();
            rec.lang = 'en-US';
            rec.interimResults = false;
            document.getElementById('voice-result').innerText = '🎙️ Listening...';
            rec.onresult = function(e) {
                const txt = e.results[0][0].transcript;
                document.getElementById('voice-result').innerText = '✅ ' + txt;
                // Fill the text input
                const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                if (inputs.length > 0) {
                    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    nativeInputValueSetter.call(inputs[inputs.length-1], txt);
                    inputs[inputs.length-1].dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            rec.onerror = function(e) {
                document.getElementById('voice-result').innerText = '❌ Error: ' + e.error;
            };
            rec.start();
        }
        </script>
        """, unsafe_allow_html=True)

        # Chat Input
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "Ask", placeholder="Ask anything about your documents...",
                label_visibility="collapsed", key="chat_input"
            )
        with c2:
            send = st.button("Send 🚀", use_container_width=True)

        if send and user_input.strip() and GROQ_API_KEY:
            ctx = get_context(user_input, st.session_state.raw_texts)
            with st.spinner("🤖 Thinking..."):
                try:
                    resp = ask_llm(user_input, ctx, language, st.session_state.chat_history)
                    st.session_state.chat_history.append({"user": user_input, "bot": resp})
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

        if not GROQ_API_KEY:
            st.warning("⚠️ Add your Groq API key in the sidebar.")

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# CONVERT PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif "Convert" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🔄 File Converter</h1>
        <p>Convert between PDF, CSV, Excel, JSON, TXT, Word</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cf = st.file_uploader("Upload file to convert",
                              type=["pdf","txt","csv","xlsx","json","docx"],
                              key="conv_file")
    with c2:
        fmt = st.selectbox("Convert to:", ["TXT", "CSV", "JSON", "Excel (XLSX)"])

    if cf and st.button("🔄 Convert Now", use_container_width=True):
        with st.spinner("Converting..."):
            text = extract_text(cf)
            if fmt == "TXT":
                st.download_button("⬇️ Download TXT", text.encode(),
                                   file_name=f"{cf.name}.txt", mime="text/plain")
            elif fmt == "JSON":
                out = json.dumps({"filename": cf.name, "content": text}, indent=2)
                st.download_button("⬇️ Download JSON", out.encode(),
                                   file_name=f"{cf.name}.json", mime="application/json")
            elif fmt == "CSV":
                import pandas as pd
                df = pd.DataFrame({"line": [l for l in text.split("\n") if l.strip()]})
                st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
                                   file_name=f"{cf.name}.csv", mime="text/csv")
            elif fmt == "Excel (XLSX)":
                import pandas as pd
                df = pd.DataFrame({"line": [l for l in text.split("\n") if l.strip()]})
                buf = BytesIO()
                df.to_excel(buf, index=False)
                st.download_button("⬇️ Download Excel", buf.getvalue(),
                                   file_name=f"{cf.name}.xlsx")
            st.success("✅ Conversion complete!")

# ════════════════════════════════════════════════════════════════════════════════
# VISUALIZE PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif "Visualize" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Visualizer</h1>
        <p>Upload CSV or Excel to create interactive charts</p>
    </div>
    """, unsafe_allow_html=True)

    vf = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"], key="viz_file")

    if vf:
        try:
            import pandas as pd
            import plotly.express as px

            df = pd.read_csv(vf) if vf.name.endswith(".csv") else pd.read_excel(vf)
            st.success(f"✅ {df.shape[0]} rows × {df.shape[1]} columns")

            tab1, tab2 = st.tabs(["📋 Data Preview", "📈 Charts"])

            with tab1:
                st.dataframe(df, use_container_width=True)
                st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
                st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")

            with tab2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    chart = st.selectbox("Chart Type", ["Bar","Line","Scatter","Pie","Histogram","Heatmap"])
                with c2:
                    x_col = st.selectbox("X Axis", df.columns.tolist())
                with c3:
                    y_col = st.selectbox("Y Axis", df.columns.tolist())

                if st.button("📊 Generate Chart", use_container_width=True):
                    fig = None
                    purple = ["#7c3aed","#6d28d9","#5b21b6","#8b5cf6","#a78bfa"]
                    if chart == "Bar":
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", color_discrete_sequence=purple)
                    elif chart == "Line":
                        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", color_discrete_sequence=purple)
                    elif chart == "Scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}", color_discrete_sequence=purple)
                    elif chart == "Pie":
                        fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} Distribution", color_discrete_sequence=purple)
                    elif chart == "Histogram":
                        fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}", color_discrete_sequence=purple)
                    elif chart == "Heatmap":
                        num_df = df.select_dtypes(include="number")
                        if not num_df.empty:
                            fig = px.imshow(num_df.corr(), title="Correlation Heatmap", color_continuous_scale="Purples")
                        else:
                            st.warning("No numeric columns for heatmap.")

                    if fig:
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(30,27,75,0.5)",
                            font_color="#e2e8f0",
                            title_font_color="#a78bfa"
                        )
                        st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.35);font-size:0.78rem;padding:0.8rem">
    🧠 PapLex AI • Powered by <strong style="color:#7c3aed">LLaMA 3.3 70B</strong> •
    Groq API • LangChain • FAISS
</div>
""", unsafe_allow_html=True)
