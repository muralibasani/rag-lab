from typing import List
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


def read_file():
    file_path = '<PDF_FILE_PATH>'
    loader = PyPDFLoader(file_path)
    docs = []
    loaded_docs = loader.load()
    docs.extend(loaded_docs)
    print('Document data:', docs)
    return docs


def get_chunks() -> List[Document]:
    docs = read_file()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print('Chunks of documents:', chunks)
    return chunks


def get_vector_store() -> FAISS:
    chunks = get_chunks()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)


retriever = None  # Fixed typo: retreiver -> retriever
llm = None


def init_llm():
    vector_store = get_vector_store()
    global retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})
    global llm
    # ✅ Fixed: keep_alive and temperature are direct parameters
    llm = ChatOllama(
                    model="llama3:8b",
                    temperature=0.7,
                    num_ctx=2048,  # instead of default 8192
                    num_predict=256)
init_llm()


def call_model(state: MessagesState):
    question = state["messages"][-1].content
    docs = retriever.invoke(question)  # ✅ Fixed: use invoke() instead of _get_relevant_documents()
    context = "\n\n".join([d.page_content for d in docs])
    
    # Get only the last few messages to avoid context overflow
    chat_history = "\n".join([f"{m.type}: {m.content}" for m in state["messages"][-5:]])

    prompt_text = f"""Chat history:
{chat_history}

Context:
{context}

Question: {question}

Answer the question based on the context provided above.""".strip()
    
    answer = llm.invoke(prompt_text)
    return {"messages": [AIMessage(content=answer.content)]}


def get_app():
    memory = MemorySaver()
    workflow = StateGraph(state_schema=MessagesState)
    
    # ✅ Fixed: Correct order - node first, then edges
    workflow.add_node("workshop_llm_rag", call_model)
    workflow.add_edge(START, "workshop_llm_rag")
    workflow.add_edge("workshop_llm_rag", END)  # ✅ Added missing edge to END
    
    app = workflow.compile(checkpointer=memory)
    print(app.get_graph().draw_ascii())
    return app


chat_app = get_app()
thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}


def start_chat():
    print("Resume Assistant started. Type 'exit' or 'quit' to end.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Thank you for using AI chat")
            break
        
        if not query:
            continue
            
        input_message = HumanMessage(content=query)
        try:
            for event in chat_app.stream({"messages": [input_message]}, config, stream_mode="values"):
                answer_message = event["messages"][-1]
                if isinstance(answer_message, AIMessage):
                    print(f"\nAssistant: {answer_message.content}\n")
        except Exception as e:
            print(f"Error: {e}\n")


# ✅ Fixed: Call start_chat() OUTSIDE the function definition
start_chat()
