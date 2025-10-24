from typing import List
import uuid
from langchain_community.document_loaders import (
    PyPDFLoader
)
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage


def get_docs() -> List[Document]:
    file_path='resources/F2510149287.pdf'
    loader = PyPDFLoader(str(file_path))
    docs=[]
    loaded_docs = loader.load()
    print(f"✅ Loaded {len(loaded_docs)} document(s) from: {file_path}")
    docs.extend(loaded_docs)
    return docs

def get_chunks() -> List[Document]:
    all_docs = []
    docs = get_docs()
    all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    try:
        chunks = splitter.split_documents(all_docs)
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return []

    print(f"✂️ Split into {len(chunks)} chunks.")
    return chunks

def get_vector_store() -> FAISS:
    chunks = get_chunks()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  
    return FAISS.from_documents(chunks, embeddings)


retriever = None
llm = None

def init_llm():
    vectorstore = get_vector_store()
    global retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    global llm
    llm = ChatOllama(model="llama3", model_kwargs={"keep_alive": "-1", "options": {"temperature": 0.8}})


def call_model(state: MessagesState):
    question = state["messages"][-1].content
    docs = retriever._get_relevant_documents(question, run_manager=None)
    context = "\n\n".join([d.page_content for d in docs])
    # Build chat history as a single string
    chat_history = "\n".join(m.content for m in state["messages"])

    # Format the full prompt
    prompt_text = f"""
    Chat history:
    {chat_history}

    Context:
    {context}

    Question:
    {question}
    """.strip()

    
    answer = llm.invoke(prompt_text)

    print('Answer is ', answer)
    
    if isinstance(answer, AIMessage):
        answer_text = answer.content
    elif isinstance(answer, dict):
        answer_text = answer.get("content", str(answer))
    else:
        answer_text = str(answer)
    return {"messages": state["messages"] + [AIMessage(content=answer_text)]}


def get_app():
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    app = workflow.compile(checkpointer=memory)
    
    print(app.get_graph().draw_ascii())
    return app


def start_chat():
    app = get_app()
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        query = input("🧠 You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print(f"🙏 Thank you for using ai assistant! ")
            break
        input_message = HumanMessage(content=query)

        for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
            answer_message = event["messages"][-1]
            print("\n🤖 Assistant:")
            print(answer_message.content)

init_llm()
start_chat()