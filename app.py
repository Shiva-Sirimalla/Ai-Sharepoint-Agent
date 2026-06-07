import os
import streamlit as st

from langchain_groq import ChatGroq

from config import GROQ_API_KEY
from document_loader import DocumentLoader
from rag_pipeline import RAGPipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI SharePoint Agent",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.main { padding-top: 1rem; }

.stChatMessage {
    border-radius: 12px;
    padding: 10px;
}

[data-testid="stSidebar"] {
    width: 350px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 AI SharePoint Agent")
st.caption("Enterprise Document Intelligence System")

# --------------------------------------------------
# OBJECTS
# --------------------------------------------------

loader = DocumentLoader()
rag = RAGPipeline()

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# STATS
# --------------------------------------------------

try:
    stats = loader.get_stats()
except:
    stats = {"files": 0, "pages": 0}

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 AI SharePoint")
    st.caption("Enterprise Knowledge Assistant")

    st.divider()

    # Upload Section
    st.subheader("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs("documents", exist_ok=True)

        for file in uploaded_files:
            save_path = os.path.join("documents", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())

        st.success(f"{len(uploaded_files)} file(s) uploaded")

    # Index Button
    if st.button("🔄 Load & Index Documents", use_container_width=True):

        try:
            with st.spinner("Loading documents..."):
                docs = loader.load_documents()

            if not docs:
                st.error("No documents found.")
            else:
                with st.spinner("Creating vector database..."):
                    rag.create_vector_db(docs)

                st.success(f"{len(docs)} pages indexed")

        except Exception as e:
            st.exception(e)

    st.divider()

    # Dashboard
    st.subheader("📊 Dashboard")

    st.metric("📄 Docs", stats["files"])
    st.metric("📑 Pages", stats["pages"])
    st.metric("💬 Chats", len(st.session_state.history))

    st.divider()

    # Documents List
    st.subheader("📁 Documents")

    if os.path.exists("documents"):
        files = os.listdir("documents")

        if files:
            for file in files:
                st.caption(f"📄 {file}")
        else:
            st.caption("No documents uploaded")
    else:
        st.caption("No documents folder")

    st.divider()

    st.subheader("⚙️ System")
    st.success("🟢 Online")
    st.caption("Model: Llama 3.3 70B")
    st.caption("Vector DB: FAISS")
    st.caption("Embeddings: BGE Small")

    st.divider()

    st.subheader("💡 Examples")
    st.caption("• What is leave policy?")
    st.caption("• Summarize handbook")
    st.caption("• Password rules")
    st.caption("• Incident response")

# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

query = st.chat_input("Ask a question about your documents...")

# --------------------------------------------------
# RAG PROCESSING
# --------------------------------------------------

if query:

    st.session_state.history.append(("user", query))

    with st.chat_message("user"):
        st.write(query)

    if not os.path.exists("vector_db"):
        st.error("Please load and index documents first.")
        st.stop()

    try:
        db = rag.load_db()

        results = db.max_marginal_relevance_search(
            query,
            k=3,
            fetch_k=10
        )

    except Exception as e:
        st.exception(e)
        st.stop()

    context = ""
    citations = []

    for doc in results:
        context += doc.page_content + "\n\n"

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        citations.append(f"📄 {source} (Page {page})")

    history_text = "\n".join(
        [f"{r}: {m}" for r, m in st.session_state.history[-6:]]
    )

    prompt = f"""
You are an AI SharePoint Assistant.

Answer ONLY using the context.

Conversation:
{history_text}

Context:
{context}

Question:
{query}
"""

    with st.spinner("Thinking..."):
        response = llm.invoke(prompt)

    answer = response.content

    source_text = "\n".join(sorted(set(citations)))

    final_answer = f"""
{answer}

### 📚 Sources
{source_text}
"""

    with st.chat_message("assistant"):
        st.markdown(final_answer)

    st.session_state.history.append(("assistant", final_answer))

# --------------------------------------------------
# DOWNLOAD CHAT
# --------------------------------------------------

if st.session_state.history:

    chat_text = "\n\n".join(
        [f"{r}: {m}" for r, m in st.session_state.history]
    )

    st.download_button(
        label="⬇ Download Chat",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )