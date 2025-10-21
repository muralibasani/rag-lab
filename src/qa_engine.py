from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM



from .loader import load_and_split
import os
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------
# Configuration
# --------------------------------------------
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
llm_model = os.getenv("LLM_MODEL", "llama3")
vector_store = os.getenv("VECTOR_EMBEDDINGS_STORAGE", "inmemory")  # "inmemory" or "chroma"
vector_persist_dir = os.getenv("VECTOR_PERSIST_DIR", "./chroma_store")

# Use resources directory for configuration files
# url_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "source_urls.txt")
# files = []  # add local files here if needed


# --------------------------------------------
# Build QA System
# --------------------------------------------
def build_qa_system(source_urls, source_files, assistant_name, assistant_source_info_type):
    print("📚 Loading and splitting documents...")
    chunks = load_and_split(source_urls, source_files, assistant_source_info_type, assistant_name)

    print(f"🔍 Using embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if(len(chunks)==0):
        print("📚 No chunks available...")
        exit
    # ----------------------------------------
    # Choose Vector Store Backend
    # ----------------------------------------
    if vector_store.lower() == "inmemory":
        print("⚡ Using in-memory FAISS vector store")
        vectorstore = FAISS.from_documents(chunks, embeddings)

    elif vector_store.lower() == "vectordb":
        print(f"🧠 Using Chroma vector store (persisted at {vector_persist_dir})")

        # if assistant_name is given, use a subdirectory
        assistant_vector_dir = vector_persist_dir
        if assistant_name:
            assistant_vector_dir = os.path.join(
                vector_persist_dir,
                assistant_name.replace(" ", "_").lower()
            )

        # Ensure directory exists
        os.makedirs(assistant_vector_dir, exist_ok=True)

        # Check for valid Chroma DB (look for 'chroma.sqlite3' file)
        chroma_db_file = os.path.join(assistant_vector_dir, "chroma.sqlite3")
        if os.path.exists(chroma_db_file):
            print("🔄 Loading existing Chroma vector store...", assistant_vector_dir)
            vectorstore = Chroma(
                persist_directory=assistant_vector_dir,
                embedding_function=embeddings
            )
        else:
            print("✨ Building new Chroma vector store...", assistant_vector_dir)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=assistant_vector_dir
            )

    else:
        raise ValueError(f"❌ Unknown VECTOR_EMBEDDINGS_STORAGE type: {assistant_vector_dir}")

    # ----------------------------------------
    # Build Retriever and QA Chain
    # ----------------------------------------
    if vector_store.lower() == "vectordb":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        # retriever = vectorstore.as_retriever(
        #     search_type="similarity_score_threshold",
        #     search_kwargs={"score_threshold": 0.8, "k": 8}
        # )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8, "fetch_k": 16, "lambda_mult": 0.5})
    llm = OllamaLLM(model=llm_model)

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    print("✅ QA system initialized successfully.")

    return qa, retriever
