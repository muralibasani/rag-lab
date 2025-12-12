from langchain_core.prompts import ChatPromptTemplate

from src.models import EventType, MessageClassifier

def make_log_prompt(event_type: EventType) -> ChatPromptTemplate:
    # ... (setup remains the same)

    base_rules = [
        "Fix any spelling and grammar mistakes in the user's question before answering."
        "Treat word topic or topics as a kafka topic"
    ]

    # Determine type-specific rules
    extra_rules = []
    if event_type in [EventType.KAFKA, EventType.SCHEMA_REGISTRY]:
        base_rules.append("Base your answer ONLY on the given logs. Do not assume anything beyond them.")
        base_rules.append("If you refer to a specific event, mention the timestamp or log line for verification.") # New
        if event_type == EventType.SCHEMA_REGISTRY:
             extra_rules.append("Look for POST calls on subjects in schema registry logs.")
    elif event_type == EventType.KAFKA_DOCS:
        base_rules.append("Answer using Kafka documentation and concepts. Do not assume anything beyond official docs.")
    else:
        base_rules.append(
            "Answer using general reasoning and knowledge. No log files are provided in this context."
        )
    
    # Add the "I don't know" and tone rules near the end
    base_rules.append("Start your answer directly without preamble. Provide a concise, professional, and easy-to-read explanation.")
    base_rules.append("If the Context does not contain enough information to answer the question, you MUST reply: 'I don't have enough information in the provided context to answer that.'")
    
    all_rules = "\n".join(f"- {rule}" for rule in base_rules + extra_rules)

    prompt_name = event_type.value.replace('_', ' ')

    return ChatPromptTemplate.from_template(f"""
You are an expert system assistant for {prompt_name}.

Use the following context and chat history to answer the user's question accurately.

Rules:
{all_rules}

Chat history:
{{chat_history}}

Context:
{{context}}

Question:
{{question}}

Answer:
""".strip())


def _run_message_classifier(input_text: str, llm):
    """Internal helper to classify a message using the structured LLM."""
    classifier_llm = llm.with_structured_output(MessageClassifier)
    system_prompt = """
        You are a classifier that decides whether a user query is about *Kafka logs*, 
        *Schema Registry logs*, or *Kafka documentation*.
        
        Follow these steps carefully:

        1. **Fix any spelling or grammar errors** in the user's message before classification.
        2. **Classify the intent** of the corrected message into **one** of the following categories:


        - `"kafka-logs"` → when the query refers to **Kafka runtime events** such as:
            - topic creation, deletion, partition changes
            - broker startup/shutdown
            - consumer group assignments or offsets
            - consumer rebalances
            - ACL or authorization events
            - error messages from Kafka servers

        - `"schema-registry-logs"` → when the query refers to **Schema Registry events**, such as:
            - subjects or schema versions
            - schema registration or deletion
            - API calls like `POST /subjects/...`
            - compatibility checks or errors

        - `"kafka-docs"` → when the query asks about **Kafka concepts or documentation**, such as:
            - how to create a topic
            - what a consumer group is
            - how ACLs work
            - Kafka architecture or configuration

        - 'all-others'

        3. **Output only one label** — one of:
            kafka-logs
            schema-registry-logs
            kafka-docs

        Do not include any reasoning or explanation in your response.
    """
    
#     # Build the classification prompt with conversation history if available
#     user_prompt = input_text
#     if conversation_history:
#         user_prompt = f"""Conversation History:
# {conversation_history}

# Current User Message:
# {input_text}

# Based on the conversation history and current message, classify the message type."""
    
    result = classifier_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_text},
    ])
    return result.message_type


def get_sys_prompt():
    return """You are a router assistant.
        Step 1: Use 'classify_message_tool' to determine the message_type.
        Step 2: Based on the message_type, call exactly **one** of the available tools:
        - kafka_docs_tool
        - kafka_logs_tool
        - schema_registry_logs_tool

        After calling the correct tool, **do not call any further tools**.
        Return only the final assistant message in plain text.
        """
