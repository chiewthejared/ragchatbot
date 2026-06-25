import streamlit as st
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not set. Please add it to Streamlit Secrets.")
    st.stop()
CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
PERSIST_DIR = "./faiss_db_v6"

# Loaders
def load_documents_from_folder(folder_path="."):
    docs = []
    folder = Path(folder_path)

    # Process .txt files
    for file_path in folder.glob("*.txt"):
        if file_path.name == "requirements.txt":
            continue
        try:
            loader = TextLoader(str(file_path), encoding="cp1252")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = file_path.name
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {e}")

    # Process .pdf files
    for file_path in folder.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(file_path))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = file_path.name
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {e}")

    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# Vector Store
def get_vector_store(docs=None):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    if docs:
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectorstore.save_local(PERSIST_DIR)
    else:
        vectorstore = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

# Answer Function
def ask(question, vectorstore):
    retrieved = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in retrieved])

    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context.

**Instructions:**
- If the question asks for a specific fact (e.g., GPA, date, list), output that fact concisely and **nothing else**.
- If the question asks about a project, work experience, or any descriptive content, respond with a short bullet‑point list (2‑4 bullet points) that includes the name, what it involved (tools, purpose, outcomes), and the time period if available.
- **Format:** Use a dash (`-`) at the start of each bullet point, and put each bullet point on a new line.
- **Do not add any extra text** like "I don't have that information" unless the answer is truly not present.

**Preferred projects to describe when the question is generic:**
- For a **data project** → use "COVID-19 Analytics Dashboard (Power BI)"
- For a **software project** → use "Sturge-Weber Foundation App"
- For an **AI project** → use "Resume RAG Chatbot with Gemini"

**Examples of good answers:**

For a data project:
"COVID-19 Analytics Dashboard (Power BI) (April 2026 – May 2026)
- Built an interactive dashboard using Power BI and COVID-19 data from Our World in Data
- Featured KPI cards, trend visualizations, and country‑level filtering
- Performed data cleaning and transformation using Power Query"

For a software project:
"Sturge-Weber Foundation App (August 2025 – May 2026)
- Developed a mobile app for children with neurological conditions
- Implemented features like timers, educational content, and medical analytics
- Worked on UI design and navigation workflows"

For an AI project:
"Resume RAG Chatbot with Gemini (June 2026)
- Built a retrieval‑augmented generation (RAG) chatbot using Google Gemini
- Indexed multiple resumes and enabled natural language querying
- Deployed with a Streamlit web interface for interactive conversations"

Now, using the context below, answer the question in the same style.

Context:
{context}

Question: {question}
Answer:"""

    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    response = llm.invoke(prompt)
    answer = response.content
    return answer

# Streamlit UI
st.set_page_config(page_title="Resume RAG Chatbot", page_icon="📄")

with st.sidebar:
    st.markdown("### Menu")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_vectorstore():
    if not Path(PERSIST_DIR).exists():
        with st.spinner("Indexing resumes... This may take a minute on the first run."):
            docs = load_documents_from_folder()
            if not docs:
                raise FileNotFoundError("No text files found in the current directory.")
            chunks = split_documents(docs)
            vectorstore = get_vector_store(chunks)
        return vectorstore, True
    else:
        with st.spinner("Loading existing vector store..."):
            vectorstore = get_vector_store()
        return vectorstore, False

try:
    vectorstore, freshly_indexed = load_vectorstore()
    if freshly_indexed:
        st.toast("✅ Indexing complete! Ready to answer questions.", icon="🎉")
    else:
        st.toast("📂 Vector store loaded.", icon="✅")
except FileNotFoundError as e:
    st.warning(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.title("📄 Jared's Resume Chatbot")
    st.markdown("Ask anything about Jared Chiew's experience, skills, education, and projects.")
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; margin: 10px 0 20px 0;">
        <div style="font-size: 80px; animation: float 2s ease-in-out infinite;">
            🧑‍💻
        </div>
        <p style="font-size: 16px; color: #888; margin-top: -10px;">Ask away!</p>
        <p style="font-size: 16px; color: #888; margin-top: -10px;">"List me all the programming languages Jared is familiar with."</p>
    </div>
    <style>
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-12px); }
            100% { transform: translateY(0px); }
        }
    </style>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about Jared's resumes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking..."):
            answer = ask(prompt, vectorstore)
            placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()