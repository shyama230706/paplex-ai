import streamlit as st
import os
import json
from io import BytesIO

st.set_page_config(
    page_title="PapLex AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: rgba(15,12,41,0.97) !important; border-right: 1px solid rgba(139,92,246,0.3); }
.main-header { background: linear-gradient(135deg, #4c1d95, #6d28d9, #7c3aed); border-radius: 20px; padding: 2.5rem; text-align: center; margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(124,58,237,0.4); }
.main-header h1 { font-size: 2.8rem; font-weight: 700; color: white; margin: 0; }
.main-header p { color: rgba(255,255,255,0.85); font-size: 1.1rem; margin: 0.5rem 0 0 0; }
.ftag { background: rgba(255,255,255,0.15); color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.78rem; border: 1px solid rgba(255,255,255,0.2); margin: 0.2rem; display: inline-block; }
.chat-user { background: linear-gradient(135deg, #4c1d95, #6d28d9); border-radius: 15px 15px 5px 15px; padding: 0.9rem 1.2rem; margin: 0.5rem 0; color: white; margin-left: 15%; }
.chat-bot { background: rgba(30,27,75,0.85); border: 1px solid rgba(139,92,246,0.35); border-radius: 15px 15px 15px 5px; padding: 0.9rem 1.2rem; margin: 0.5rem 0; color: #e2e8f0; margin-right: 8%; }
.stButton > button { background: linear-gradient(135deg, #6d28d9, #7c3aed) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
.info-card { background: rgba(30,27,75,0.6); border: 1px solid rgba(139,92,246,0.3); border-radius: 15px; padding: 1.5rem; margin: 0.8rem 0; text-align: center; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: rgba(30,27,75,0.85) !important; border: 1px solid rgba(139,92,246,0.45) !important; color: #e2e8f0 !important; border-radius: 10px !important; }
div[data-testid="stFileUploader"] { background: rgba(30,27,75,0.5) !important; border: 2px dashed rgba(139,92,246,0.5) !important; border-radius: 15px !important; }
hr { border-color: rgba(139,92,246,0.25) !important; }
.history-item { background: rgba(30,27,75,0.4); border: 1px solid rgba(139,92,246,0.2); border-radius: 10px; padding: 0.8rem; margin: 0.3rem 0; cursor: pointer; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "raw_texts": [],
    "doc_names": [],
    "page": "Chat",
    "voice_result": "",
    "all_sessions": [],
    "current_session_name": "Session 1"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
        return f"Error: {e}"
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
Answer accurately based on the document context. Be helpful, detailed, and clear. {lang_note}
If answer is not in context, say so honestly."""
    messages = [{"role": "system", "content": system}]
    if history:
        for h in history[-5:]:
            messages += [{"role": "user", "content": h["user"]}, {"role": "assistant", "content": h["bot"]}]
    messages.append({"role": "user", "content": f"Context:\n{context[:4000]}\n\nQuestion: {prompt}"})
    resp = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=messages, max_tokens=1500, temperature=0.7)
    return resp.choices[0].message.content

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 PapLex AI")
    st.markdown('<span style="background:linear-gradient(135deg,#6d28d9,#7c3aed);color:white;padding:0.25rem 0.75rem;border-radius:20px;font-size:0.72rem;">Powered by LLaMA 3.3</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🧭 Navigation")
    for pg in ["💬 Chat", "📜 History", "🔄 Convert Files", "📊 Visualize"]:
        if st.button(pg, use_container_width=True, key=f"nav_{pg}"):
            st.session_state.page = pg.split(" ", 1)[1]

    st.divider()
    st.markdown("### 🌍 Language")
    language = st.selectbox("lang", ["English","Hindi","Spanish","French","German","Japanese","Chinese","Arabic"], label_visibility="collapsed")

    st.divider()
    st.markdown("### 📁 Upload Files")
    st.caption("PDF, CSV, Excel, JSON, TXT, Word")
    uploaded = st.file_uploader("files", type=["pdf","txt","csv","xlsx","xls","json","docx"], accept_multiple_files=True, label_visibility="collapsed")

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
        st.markdown("### 📄 Loaded")
        for n in st.session_state.doc_names:
            st.markdown(f"✅ `{n}`")
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.raw_texts = []
            st.session_state.doc_names = []
            st.session_state.chat_history = []
            st.rerun()

    if not GROQ_API_KEY:
        st.divider()
        st.markdown("### 🔑 Groq API Key")
        k = st.text_input("key", type="password", placeholder="gsk_...", label_visibility="collapsed")
        if k:
            GROQ_API_KEY = k

# ── PAGES ──────────────────────────────────────────────────────────────────────
page = st.session_state.page

# ══════════════════════════════════════════════════════════════
# CHAT PAGE
# ══════════════════════════════════════════════════════════════
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
        st.markdown('<div class="info-card"><h2>👋 Welcome to PapLex AI</h2><h4>Your Intelligent Document Assistant</h4><p style="color:rgba(255,255,255,0.7)">👆 Upload files from the sidebar to get started!</p></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown('<div class="info-card"><h3>📄 6 Formats</h3><p>PDF • CSV • Excel<br>JSON • TXT • Word</p></div>', unsafe_allow_html=True)
        with c2: st.markdown('<div class="info-card"><h3>🤖 Smart Actions</h3><p>Q&A • Summary • Quiz<br>NER • Suggestions</p></div>', unsafe_allow_html=True)
        with c3: st.markdown('<div class="info-card"><h3>🌍 8 Languages</h3><p>English • Hindi • Spanish<br>French • German • More</p></div>', unsafe_allow_html=True)

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
            prompts = {
                "summarize": "Provide a comprehensive, well-structured summary with key points.",
                "quiz": "Generate 5 multiple choice quiz questions with 4 options each and mark correct answers.",
                "ner": "Extract and categorize all named entities: Persons, Organizations, Locations, Dates, Numbers.",
                "suggest": "Suggest 8 insightful questions someone could ask about this document.",
                "compare": "Compare and contrast documents in detail, or analyze key themes if only one."
            }
            ctx = get_context(prompts[action], st.session_state.raw_texts)
            with st.spinner("🤖 AI is thinking..."):
                try:
                    resp = ask_llm(prompts[action], ctx, language)
                    st.session_state.chat_history.append({"user": f"[{action.upper()}]", "bot": resp})
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

        st.divider()

        # Chat History Display
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="chat-user">👤 {chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bot">🧠 {chat["bot"]}</div>', unsafe_allow_html=True)

        # Voice Input using streamlit component
        st.divider()
        st.markdown("#### 🎙️ Voice Input")
        st.components.v1.html("""
        <div style="font-family: Inter, sans-serif;">
            <button id="voiceBtn" onclick="toggleVoice()" style="
                background: linear-gradient(135deg, #6d28d9, #7c3aed);
                color: white; border: none; border-radius: 10px;
                padding: 0.5rem 1.5rem; cursor: pointer;
                font-size: 0.9rem; font-weight: 600; margin-bottom: 8px;">
                🎙️ Start Voice Input
            </button>
            <div id="status" style="color: #a78bfa; font-size: 0.82rem; margin: 4px 0;"></div>
            <input id="voiceOutput" type="text" readonly placeholder="Voice text will appear here..."
                style="width:100%; padding:8px; border-radius:8px; border:1px solid #7c3aed;
                background:rgba(30,27,75,0.9); color:#e2e8f0; font-size:0.85rem; box-sizing:border-box;">
            <button onclick="copyText()" style="
                background: rgba(109,40,217,0.4); color: #a78bfa;
                border: 1px solid #7c3aed; border-radius: 8px;
                padding: 0.3rem 1rem; cursor: pointer; font-size: 0.8rem; margin-top:6px;">
                📋 Copy Text
            </button>
        </div>
        <script>
        let recognition = null;
        let isListening = false;

        function toggleVoice() {
            if (isListening) {
                stopVoice();
            } else {
                startVoice();
            }
        }

        function startVoice() {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                document.getElementById('status').innerText = '❌ Not supported. Please use Chrome browser.';
                return;
            }
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = function() {
                isListening = true;
                document.getElementById('voiceBtn').innerText = '⏹️ Stop Listening';
                document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #dc2626, #ef4444)';
                document.getElementById('status').innerText = '🎙️ Listening... Speak now!';
            };

            recognition.onresult = function(e) {
                let interim = '';
                let final = '';
                for (let i = e.resultIndex; i < e.results.length; i++) {
                    if (e.results[i].isFinal) {
                        final += e.results[i][0].transcript;
                    } else {
                        interim += e.results[i][0].transcript;
                    }
                }
                document.getElementById('voiceOutput').value = final || interim;
                if (final) {
                    document.getElementById('status').innerText = '✅ Done! Copy text and paste in chat.';
                }
            };

            recognition.onerror = function(e) {
                document.getElementById('status').innerText = '❌ Error: ' + e.error + '. Try again.';
                stopVoice();
            };

            recognition.onend = function() {
                stopVoice();
            };

            recognition.start();
        }

        function stopVoice() {
            isListening = false;
            if (recognition) recognition.stop();
            document.getElementById('voiceBtn').innerText = '🎙️ Start Voice Input';
            document.getElementById('voiceBtn').style.background = 'linear-gradient(135deg, #6d28d9, #7c3aed)';
            if (!document.getElementById('status').innerText.includes('✅')) {
                document.getElementById('status').innerText = '⏹️ Stopped.';
            }
        }

        function copyText() {
            const txt = document.getElementById('voiceOutput').value;
            if (txt) {
                navigator.clipboard.writeText(txt).then(() => {
                    document.getElementById('status').innerText = '📋 Copied! Now paste in chat box below.';
                });
            }
        }
        </script>
        """, height=160)

        # Chat Input
        st.markdown("#### 💬 Type or Paste your question:")
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input("Ask", placeholder="Type here or paste voice text...", label_visibility="collapsed", key="chat_input")
        with c2:
            send = st.button("Send 🚀", use_container_width=True)

        if send and user_input.strip() and GROQ_API_KEY:
            ctx = get_context(user_input, st.session_state.raw_texts)
            with st.spinner("🤖 Thinking..."):
                try:
                    resp = ask_llm(user_input, ctx, language, st.session_state.chat_history)
                    st.session_state.chat_history.append({"user": user_input, "bot": resp})
                    # Save to sessions
                    if st.session_state.chat_history:
                        session = {
                            "name": st.session_state.current_session_name,
                            "docs": st.session_state.doc_names.copy(),
                            "history": st.session_state.chat_history.copy()
                        }
                        # Update or add session
                        found = False
                        for i, s in enumerate(st.session_state.all_sessions):
                            if s["name"] == session["name"]:
                                st.session_state.all_sessions[i] = session
                                found = True
                                break
                        if not found:
                            st.session_state.all_sessions.append(session)
                except Exception as e:
                    st.error(f"Error: {e}")
            st.rerun()

        if not GROQ_API_KEY:
            st.warning("⚠️ Add your Groq API key in the sidebar.")

        if st.session_state.chat_history:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                if st.button("💾 Save Session"):
                    session = {
                        "name": f"Session {len(st.session_state.all_sessions)+1}",
                        "docs": st.session_state.doc_names.copy(),
                        "history": st.session_state.chat_history.copy()
                    }
                    st.session_state.all_sessions.append(session)
                    st.session_state.current_session_name = session["name"]
                    st.success(f"✅ Saved as {session['name']}!")

# ══════════════════════════════════════════════════════════════
# HISTORY PAGE
# ══════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown("""
    <div class="main-header">
        <h1>📜 Chat History</h1>
        <p>All your previous document conversations</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.all_sessions:
        st.markdown('<div class="info-card"><h3>No history yet</h3><p>Start chatting with documents and save sessions to see them here.</p></div>', unsafe_allow_html=True)
    else:
        for i, session in enumerate(reversed(st.session_state.all_sessions)):
            with st.expander(f"📁 {session['name']} — {', '.join(session['docs'])}", expanded=(i==0)):
                for chat in session["history"]:
                    st.markdown(f'<div class="chat-user">👤 {chat["user"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-bot">🧠 {chat["bot"]}</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear All History", type="secondary"):
            st.session_state.all_sessions = []
            st.rerun()

# ══════════════════════════════════════════════════════════════
# CONVERT PAGE
# ══════════════════════════════════════════════════════════════
elif "Convert" in page:
    st.markdown("""
    <div class="main-header">
        <h1>🔄 File Converter</h1>
        <p>Convert between PDF, CSV, Excel, JSON, TXT, Word</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cf = st.file_uploader("Upload file to convert", type=["pdf","txt","csv","xlsx","json","docx"], key="conv_file")
    with c2:
        fmt = st.selectbox("Convert to:", ["TXT", "CSV", "JSON", "Excel (XLSX)"])

    if cf and st.button("🔄 Convert Now", use_container_width=True):
        with st.spinner("Converting..."):
            text = extract_text(cf)
            if fmt == "TXT":
                st.download_button("⬇️ Download TXT", text.encode(), file_name=f"{cf.name}.txt", mime="text/plain")
            elif fmt == "JSON":
                st.download_button("⬇️ Download JSON", json.dumps({"filename": cf.name, "content": text}, indent=2).encode(), file_name=f"{cf.name}.json", mime="application/json")
            elif fmt == "CSV":
                import pandas as pd
                df = pd.DataFrame({"line": [l for l in text.split("\n") if l.strip()]})
                st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), file_name=f"{cf.name}.csv", mime="text/csv")
            elif fmt == "Excel (XLSX)":
                import pandas as pd
                buf = BytesIO()
                pd.DataFrame({"line": [l for l in text.split("\n") if l.strip()]}).to_excel(buf, index=False)
                st.download_button("⬇️ Download Excel", buf.getvalue(), file_name=f"{cf.name}.xlsx")
            st.success("✅ Done!")

# ══════════════════════════════════════════════════════════════
# VISUALIZE PAGE
# ══════════════════════════════════════════════════════════════
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
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Rows", df.shape[0])
                with c2: st.metric("Columns", df.shape[1])
                with c3: st.metric("Missing Values", df.isnull().sum().sum())

            with tab2:
                c1, c2, c3 = st.columns(3)
                with c1: chart = st.selectbox("Chart Type", ["Bar","Line","Scatter","Pie","Histogram","Heatmap"])
                with c2: x_col = st.selectbox("X Axis", df.columns.tolist())
                with c3: y_col = st.selectbox("Y Axis", df.columns.tolist())

                if st.button("📊 Generate Chart", use_container_width=True):
                    purple = ["#7c3aed","#6d28d9","#5b21b6","#8b5cf6","#a78bfa"]
                    fig = None
                    if chart == "Bar": fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=purple)
                    elif chart == "Line": fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=purple)
                    elif chart == "Scatter": fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=purple)
                    elif chart == "Pie": fig = px.pie(df, names=x_col, values=y_col, color_discrete_sequence=purple)
                    elif chart == "Histogram": fig = px.histogram(df, x=x_col, color_discrete_sequence=purple)
                    elif chart == "Heatmap":
                        num = df.select_dtypes(include="number")
                        fig = px.imshow(num.corr(), color_continuous_scale="Purples") if not num.empty else None
                    if fig:
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,27,75,0.5)", font_color="#e2e8f0")
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div style="text-align:center;color:rgba(255,255,255,0.35);font-size:0.78rem;padding:0.8rem">🧠 PapLex AI • Powered by <strong style="color:#7c3aed">LLaMA 3.3 70B</strong> • Groq API • LangChain</div>', unsafe_allow_html=True)
