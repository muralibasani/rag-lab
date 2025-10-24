import json
import os
import uuid
import random
from pathlib import Path
from rich.console import Console
from dotenv import load_dotenv
import warnings

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# LangGraph
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.loader import load_and_split

# ------------------------
# Environment & Console
# ------------------------
warnings.filterwarnings("ignore", message=".*tokenizers before the fork.*")
load_dotenv()
console = Console()

# ------------------------
# Configs
# ------------------------
APP_NAME = os.getenv("APP_NAME")
QUESTION_PREFIX = os.getenv("QUESTION_PREFIX", "")
QUESTION_SUFFIX = os.getenv("QUESTION_SUFFIX", "")
MODEL_KEEP_ALIVE = os.getenv("MODEL_KEEP_ALIVE", "-1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
VECTOR_STORE = os.getenv("VECTOR_EMBEDDINGS_STORAGE", "inmemory")
VECTOR_PERSIST_DIR = os.getenv("VECTOR_PERSIST_DIR", "./chroma_store")
DEFAULT_ASSISTANT = "Domain_1"

FRIENDLY_RESPONSES = [
    "👍 No problem! What else would you like to ask? or type 'exit' to quit.",
    "✨ Sure thing! Feel free to ask another question when you're ready or type 'exit' to quit.",
    "😄 Got it! Fire away with your next question or type 'exit' to quit.",
    "👌 Okay! I’m here whenever you’re ready for the next question or type 'exit' to quit.",
    "🧐 Hmm, okay! What should we explore next? or type 'exit' to quit.",
]

# ------------------------
# Helper Functions
# ------------------------
def print_banner(file_path="resources/banner.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            console.print(f.read())
    except FileNotFoundError:
        console.print(f"🧠 {APP_NAME} 🧠")


def build_vector_store(chunks, assistant_name):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if VECTOR_STORE.lower() == "inmemory":
        console.print("⚡ Using in-memory FAISS vector store")
        return FAISS.from_documents(chunks, embeddings)
    elif VECTOR_STORE.lower() == "vectordb":
        assistant_vector_dir = os.path.join(VECTOR_PERSIST_DIR, assistant_name.replace(" ", "_").lower())
        os.makedirs(assistant_vector_dir, exist_ok=True)
        chroma_db_file = os.path.join(assistant_vector_dir, "chroma.sqlite3")
        if os.path.exists(chroma_db_file):
            console.print("🔄 Loading existing Chroma vector store...")
            return Chroma(persist_directory=assistant_vector_dir, embedding_function=embeddings)
        else:
            console.print("✨ Building new Chroma vector store...")
            return Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=assistant_vector_dir)
    else:
        raise ValueError(f"Unknown VECTOR_EMBEDDINGS_STORAGE: {VECTOR_STORE}")


def build_qa_system(assistant_meta):
    console.print("📚 Loading and splitting documents...")
    chunks = load_and_split(assistant_meta["source_urls"], assistant_meta["source_files"], assistant_meta["source_info_type"], assistant_meta["assistant_name"])
    if not chunks:
        console.print("📚 No chunks available...")
        exit()
    vectorstore = build_vector_store(chunks, assistant_meta["assistant_name"])
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    console.print("✅ QA system initialized successfully.")
    return retriever


def load_assistants(file_path="resources/assistants.json"):
    assistant_map = {}
    with open(file_path, "r", encoding="utf-8") as f:
        assistants = json.load(f)
    for assistant in assistants:
        console.print(f"----------- Loading assistant: {assistant['assistant_name']} --------------")
        retriever = build_qa_system(assistant)
        assistant_map[assistant["assistant_name"]] = {"retriever": retriever, "metadata": assistant}
    return assistant_map


def extract_source_docs(result=None, retriever=None, full_query=None, source_info_type=None, source_urls=[]):
    source_docs = result.get("source_documents") if isinstance(result, dict) else []
    if not source_docs and retriever:
        try:
            source_docs = retriever._get_relevant_documents(full_query, run_manager=None)
        except Exception:
            source_docs = []
    unique_sources = sorted({d.metadata.get("source", "Unknown") for d in source_docs})
    if unique_sources:
        if source_info_type == "github" and source_urls:
            prefix = source_urls[0].rstrip("/") + "/tree/main"
            unique_sources = [f"{prefix}/{src.lstrip('/')}" for src in unique_sources]
        console.print("\n🙏 Thanks to these sources:")
        for src in unique_sources:
            console.print(f"- {src}")


# ------------------------
# Chat Functionality
# ------------------------
def start_chat(assistant_qa_map):
    assistant_name = DEFAULT_ASSISTANT
    assistant_meta = assistant_qa_map[assistant_name]["metadata"]
    retriever = assistant_qa_map[assistant_name]["retriever"]
    llm_model_name = assistant_meta["model"]
    source_info_type = assistant_meta["source_info_type"]
    source_urls = assistant_meta["source_urls"]

    console.print(f"💬 Ready to chat about {assistant_name}. Ask anything, or type 'exit' to quit.")

    llm = ChatOllama(model=llm_model_name, model_kwargs={"keep_alive": "-1", "options": {"temperature": 0.8}})
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        question = state["messages"][-1].content
        docs = retriever._get_relevant_documents(question, run_manager=None)
        context = "\n\n".join([d.page_content for d in docs])
        prompt_text = f"Chat history:\n{''.join([m.content for m in state['messages']])}\n\nContext:\n{context}\n\nQuestion:\n{question}"
        answer = llm.invoke(prompt_text)
        if isinstance(answer, AIMessage):
            answer_text = answer.content
        elif isinstance(answer, dict):
            answer_text = answer.get("content", str(answer))
        else:
            answer_text = str(answer)
        return {"messages": state["messages"] + [AIMessage(content=answer_text)]}

    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    app = workflow.compile(checkpointer=memory)
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        query = input("🧠 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            console.print(f"🙏 Thank you for using {APP_NAME}! 🚀")
            break

        input_message = HumanMessage(content=query)
        for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
            answer_message = event["messages"][-1]
            console.print("\n🤖 Assistant:")
            console.print(answer_message.content)

        extract_source_docs(
            result={"source_documents": retriever._get_relevant_documents(query, run_manager=None)},
            retriever=retriever,
            full_query=query,
            source_info_type=source_info_type,
            source_urls=source_urls
        )

        # Optional follow-ups
        follow_up_prompt = f"Based on '{query}' and answer '{answer_message.content}', suggest 3 short, relevant follow-up questions (1-3)."
        follow_ups = llm.invoke(follow_up_prompt)
        suggestions = [line.strip() for line in follow_ups.content.split("\n") if line.strip() and any(ch.isdigit() for ch in line)]
        if suggestions:
            console.print("\n💡 Next steps you might explore:")
            for s in suggestions:
                number, text = s.split(".", 1) if "." in s else ("•", s)
                console.print(f"   [cyan]{number.strip()}.[/cyan] {text.strip()}")
    console.print(random.choice(FRIENDLY_RESPONSES))


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    print_banner()
    assistant_qa_map = load_assistants()
    start_chat(assistant_qa_map)
