import json
from pathlib import Path
import random
from rich.console import Console
from dotenv import load_dotenv
import os
import warnings

# ✅ Updated LangChain imports for v1+
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaLLM

# ✅ LCEL components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ Conversational memory replacement
from langchain_core.messages import HumanMessage, AIMessage
import uuid


from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from src.loader import load_and_split

from dotenv import load_dotenv

import os
import warnings

warnings.filterwarnings("ignore", message=".*tokenizers before the fork.*")

load_dotenv()

app_name = os.getenv("APP_NAME")

question_prefix = os.getenv("QUESTION_PREFIX")
question_suffix = os.getenv("QUESTION_SUFFIX")
model_keep_alive = os.getenv("MODEL_KEEP_ALIVE", "-1")
embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
llm_model = os.getenv("LLM_MODEL", "llama3")
vector_store = os.getenv("VECTOR_EMBEDDINGS_STORAGE", "inmemory")  # "inmemory" or "chroma"
vector_persist_dir = os.getenv("VECTOR_PERSIST_DIR", "./chroma_store")

friendly_responses = [
    "👍 No problem! What else would you like to ask? or type 'exit' to quit.",
    "✨ Sure thing! Feel free to ask another question when you're ready or type 'exit' to quit.",
    "😄 Got it! Fire away with your next question or type 'exit' to quit.",
    "👌 Okay! I’m here whenever you’re ready for the next question or type 'exit' to quit.",
    "🧐 Hmm, okay! What should we explore next? or type 'exit' to quit.",
]   
default_assistant_name = "Domain_1"


# --------------------------------------------
# 1. Config: Files + URL Sources
# --------------------------------------------
files = [
    # "../data/anyfile.txt",
]

# Use resources directory for configuration files
console = Console()

# Print banner
def print_banner(file_path=None):
    if file_path is None:
        # Use resources directory for banner file
        file_path = os.path.join("resources", "banner.txt")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            banner = f.read()
        print(banner)
    except FileNotFoundError:
        print("🧠 "+ app_name +" 🧠")  # fallback if file not found

print_banner()


def build_qa_system(source_urls, source_files, assistant_name, assistant_source_info_type):
    print("📚 Loading and splitting documents...")
    chunks = load_and_split(source_urls, source_files, assistant_source_info_type, assistant_name)

    print(f"🔍 Using embedding model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if(len(chunks)==0):
        print("📚 No chunks available...")
        exit()
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

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}
    """)

    qa_chain = (
        {
            "context": lambda x: "\n\n".join(
                [d.page_content for d in retriever._get_relevant_documents(x["question"])]
            ),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )


    print("✅ QA system initialized successfully.")

    return qa_chain, retriever


# --------------------------------------------
# 2. Build vector store, retriever + QA chain
# --------------------------------------------
assistant_qa_map = {}

def load_assistants():
    resources_path = Path(__file__).parent / "resources" / "assistants.json"
    with open(resources_path, "r", encoding="utf-8") as f:
        assistants_data = json.load(f)
        # Dictionary to store QA systems per assistant
    
    for assistant in assistants_data:
        name = assistant["assistant_name"]
        print('-----------  Loading assistant --------------  :: ', name)
        # Initialize QA system and retriever for this assistant
        qa, retriever = build_qa_system(assistant["source_urls"], assistant["source_files"], name, assistant["source_info_type"])
        # Store in dictionary
        assistant_qa_map[name] = {
            "qa": qa,
            "retriever": retriever,
            "metadata": assistant
        }

load_assistants()

# --------------------------------------------
# 9. Interactive Q&A
# --------------------------------------------

def extract_source_docs(result=None, retriever=None, full_query=None, source_info_type=None, source_urls=[]):
    # 6️⃣ Sources
        source_docs = []
        if isinstance(result, dict):
            # common key used by LangChain: "source_documents" or "source_documents"
            # print('result is ----- ', result)
            source_docs = result.get("source_documents") or result.get("source_doc") or []
        
        # Fallback: if chain didn't return sources, call retriever with same full_query (not raw query)
        if not source_docs:
            try:
                source_docs = retriever._get_relevant_documents(full_query, run_manager=None)  # <- fixed here
            except Exception:
                try:
                    source_docs = retriever.invoke(full_query)
                except Exception:
                    source_docs = []


        unique_sources = sorted({d.metadata.get("source", "Unknown") for d in source_docs})
        if unique_sources:
            if source_info_type == "github" and source_urls and len(source_urls) > 0:
                prefix = source_urls[0].rstrip("/") + "/tree/main"
                unique_sources = [f"{prefix}/{src.lstrip('/')}" for src in unique_sources]
            print("\n🙏 Thanks to these sources:")
            for src in unique_sources:
                print(f"- {src}")


prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the following conversation and context to answer.

Chat history:
{chat_history}

Context:
{context}

Question:
{question}
""")

def conversational_chain(input_dict):
    question = input_dict["question"]
    chat_history = input_dict.get("chat_history", "")
    docs = retriever._get_relevant_documents(question, run_manager=None)  # <- fixed here
    context = "\n\n".join([d.page_content for d in docs])

    result = (
        {"context": context, "chat_history": chat_history, "question": question}
        | prompt
        | llm
        | StrOutputParser()
    ).invoke({})

    return {"answer": result, "source_documents": docs}



# --------------------------------------------
# 9. Interactive Q&A (refactored for LangGraph)
# --------------------------------------------
def start_chat():
    assistant_name = default_assistant_name
    assistant_meta = assistant_qa_map[assistant_name]["metadata"]
    retriever = assistant_qa_map[assistant_name]["retriever"]
    
    llm_model_name = assistant_meta["model"]
    source_info_type = assistant_meta["source_info_type"]
    source_urls = assistant_meta["source_urls"]

    console.print(f"💬 Hey there! I’m ready to chat about {assistant_name}. Ask me anything, or type 'exit' to quit.")

    # Initialize LLM
    llm = ChatOllama(model=llm_model_name, model_kwargs={
        "keep_alive": "-1",
        "options": {"temperature": 0.8},
    })

    # --- LangGraph workflow ---
    workflow = StateGraph(state_schema=MessagesState)

    def call_model(state: MessagesState):
        question = state["messages"][-1].content

        # Retrieve relevant documents
        docs = retriever._get_relevant_documents(question, run_manager=None)
        context = "\n\n".join([d.page_content for d in docs])

        # Build prompt
        prompt_text = f"""
    Chat history:
    {''.join([m.content for m in state['messages']])}

    Context:
    {context}

    Question:
    {question}
    """
        # Call LLM
        answer = llm.invoke(prompt_text)
        if isinstance(answer, AIMessage):
            answer_text = answer.content
        elif isinstance(answer, dict):
            answer_text = answer.get("content", str(answer))
        else:
            answer_text = str(answer)

        # Return new messages list
        return {"messages": state["messages"] + [AIMessage(content=answer_text)]}



    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Use MemorySaver to persist conversation per user
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # Unique thread_id per session
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}

    # --- Conversation loop ---
    while True:
        query = input("🧠 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            console.print(f"🙏 Thank you for using {app_name}! 🚀")
            break

        # Create HumanMessage for input
        input_message = HumanMessage(content=query)

        # Stream responses (stateful memory automatically handled)
        for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
            answer_message = event["messages"][-1]
            print("\n🤖 Assistant:")
            print(answer_message.content)

        # Optional: extract sources from retriever
        extract_source_docs(
            result={"source_documents": retriever._get_relevant_documents(query, run_manager=None)},
            retriever=retriever,
            full_query=query,
            source_info_type=source_info_type,
            source_urls=source_urls
        )

        # Optional: follow-up suggestions
        follow_up_prompt = f"""
        Based on the user's question '{query}' and your answer '{answer_message.content}',
        suggest 3 short, relevant follow-up questions (numbered 1-3).
        """
        follow_ups = llm.invoke(follow_up_prompt)
        suggestions = [
            line.strip() for line in follow_ups.content.split("\n")
            if line.strip() and any(ch.isdigit() for ch in line)
        ]

        if suggestions:
            console.print("\n💡 [bold yellow]Next steps you might explore:[/bold yellow]")
            for s in suggestions:
                number, text = s.split(".", 1) if "." in s else ("•", s)
                console.print(f"   [cyan]{number.strip()}.[/cyan] {text.strip()}")

    # Friendly prompt for next question
    console.print(random.choice(friendly_responses))



start_chat()