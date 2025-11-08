from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from src.models import EventType
import os
from src.data_loaders import load_and_split

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def get_vector_store(event_type : EventType) -> FAISS:
    chunks = load_and_split(event_type)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  
    # embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    assistant_vector_dir = os.path.join("vector_store", event_type.value.lower())
    os.makedirs(assistant_vector_dir, exist_ok=True)
    chroma_db_file = os.path.join(assistant_vector_dir, "chroma.sqlite3")

    if os.path.exists(chroma_db_file):
        print("🔄 Loading existing Chroma vector store...")
        return Chroma(persist_directory=assistant_vector_dir, embedding_function=embeddings)
    else:
        print("✨ Building new Chroma vector store...")
        return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=assistant_vector_dir)


def init_retrievers():
    retrievers = {}
    for event_type in EventType:
        print('Retriever ', event_type)
        vectorstore = get_vector_store(event_type)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        retrievers[event_type] = vector_retriever

    return retrievers