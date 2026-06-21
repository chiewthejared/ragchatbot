import sys
import ollama
import time, threading
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Configuration
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:1b"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
PERSIST_DIR = "./chroma_db"

# Loading all text files in the current directory
def load_documents_from_folder(folder_path="."):
    docs=[]
    folder=Path(folder_path)

    # Loading each text file
    for file_path in folder.glob("*.txt"):
        if file_path.name == "requirements.txt":
            continue
        print(f"Loading: {file_path.name}")
        loader = TextLoader(str(file_path), encoding="cp1252")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = file_path.name
        docs.extend(loaded)

    return docs

# Splitting documents into chunks
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators =["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

# Create/Load vector store
def get_vector_store(docs=None):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    if docs:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
    else:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )
    return vectorstore

# Animated loading dots
def animate_dots(stop_event):
    frames = ["   ", ".  ", ".. ", "...", "....", "....."]
    i = 0
    while not stop_event.is_set():
        sys.stdout.write(f"\rRetrieving info{frames[i % len(frames)]}")
        sys.stdout.flush()
        time.sleep(0.4)
        i += 1
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()

# Answer questions using the vector store
def ask(question, vectorstore):
    retrieved = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in retrieved])
    sources = set([doc.metadata.get("source", "Unknown") for doc in retrieved])

    prompt = f"""You are an assistant that answers questions based ONLY on the provided context. 
If the answer is not in the context, say "I don't have that information."

Instructions:
- Be thorough and extract all relevant details.
- If the question asks for a list (e.g., semesters, skills), list them explicitly.
- Cite the source file if possible.

Context:
{context}

Question: {question}
Answer:"""
    
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]

    print(f"\nSources: {', '.join(sources)}")
    return answer

def main():
    import os
    if not os.path.exists(PERSIST_DIR):
        print("Loading and indexing all resumes in this folder...")
        docs = load_documents_from_folder()
        if not docs:
            print("No .txt files found in this current folder!")
            return
        chunks = split_documents(docs)
        vectorstore = get_vector_store(chunks)
        print("Indexing complete. Ask away!")
    else:
        print("Loading existing vector database...")
        vectorstore = get_vector_store()

    print("\nHello! Ask me anything about Jared Chiew's resumes.")
    print("(Type 'exit', 'quit' or 'bye' to stop.)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break
        if not question:
            continue

        # Start loading animation
        stop_event = threading.Event()
        t = threading.Thread(target=animate_dots, args=(stop_event,))
        t.start()
        # Get answer
        answer = ask(question, vectorstore)
        # Stop loading animation
        stop_event.set()
        t.join()

        print("Answer: ")
        print(answer)
        print("-" * 60)

if __name__ == "__main__":
    main()