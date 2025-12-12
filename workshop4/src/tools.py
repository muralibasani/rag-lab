
from src.llm import LlmModel
from src.log_prompt import _run_message_classifier, make_log_prompt
from src.models import EventType, MessageClassifier
from src.retrievers import init_retrievers
from langchain_core.messages import HumanMessage, AIMessage
import os
from langchain.chat_models import init_chat_model

LOAD_APP_FROM_WORKFLOW = os.getenv("LOAD_APP_FROM_WORKFLOW")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL_PAID", "gpt-4o-mini")

retrievers = init_retrievers()

llm = LlmModel.get_llm()

# -------------------------------------------------------------------------
# 🔹 Unified helper to handle all retrieval + response logic
# -------------------------------------------------------------------------
def _handle_tool_request(user_message: str, event_type: EventType, **kwargs):
    question = user_message.strip()
    print(f'Into {event_type.value} tool - Original question: "{question}"')

    # Retrieve context documents
    retriever = retrievers.get(event_type)
    docs = retriever._get_relevant_documents(question, run_manager=None) if retriever else []
    return respond_with_answer(question, docs, event_type)

# -------------------------------------------------------------------------
# 🔹 Individual tool wrappers
# -------------------------------------------------------------------------
def kafka_logs_tool(user_message: str, **kwargs):
    """Tool for Kafka logs."""
    return _handle_tool_request(user_message, EventType.KAFKA, **kwargs)

def schema_registry_logs_tool(user_message: str, **kwargs):
    """Tool for Schema Registry logs."""
    return _handle_tool_request(user_message, EventType.SCHEMA_REGISTRY, **kwargs)

def kafka_docs_tool(user_message: str, **kwargs):
    """Tool for Apache Kafka documentation."""
    return _handle_tool_request(user_message, EventType.KAFKA_DOCS, **kwargs)

# -------------------------------------------------------------------------
# 🔹 LLM response function
# -------------------------------------------------------------------------
def respond_with_answer(question, docs, event_type: EventType):
    context = "\n\n".join([d.page_content for d in docs])
   
    formatted_prompt = make_log_prompt(event_type).format(
        chat_history=None,
        context=context,
        question=question
    )

    answer = llm.invoke([HumanMessage(content=formatted_prompt)])
    answer_text = (
        answer.content if isinstance(answer, AIMessage)
        else answer.get("content", str(answer)) if isinstance(answer, dict)
        else str(answer)
    )

    # Normalize state messages
    # current_messages = state.get("messages", [])
    # if current_messages and isinstance(current_messages[0], dict):
    #     normalized = []
    #     for m in current_messages:
    #         if isinstance(m, dict):
    #             role = m.get("type", m.get("role", "user"))
    #             content = m.get("content", "")
    #             if role in ("human", "user"):
    #                 normalized.append(HumanMessage(content=content))
    #             elif role in ("ai", "assistant"):
    #                 normalized.append(AIMessage(content=content))
    #         else:
    #             normalized.append(m)
    #     return {"messages": normalized + [AIMessage(content=answer_text)]}

    return {"messages": [AIMessage(content=answer_text)]}

# -------------------------------------------------------------------------
# 🔹 Message classifier
# -------------------------------------------------------------------------
def classify_message_tool(user_message: str):
    """Classify a message into one of the known types."""
    message_type = _run_message_classifier(user_message, llm)
    print(f"Routed to: {message_type.value}")
    return {"message_type": message_type.value}
