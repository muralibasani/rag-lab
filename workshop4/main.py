from http.client import HTTPException
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.llm import LlmModel
from src.kafka_utils.producer import KafkaProducer
from src.kafka_utils.consumer import KafkaConsumer
from src.kafka_utils.config import TOPICS
import logging
import uuid
from typing import Dict, Optional
import asyncio

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


db_uri = os.getenv("DB_URI")
db = SQLDatabase.from_uri(db_uri)

print("💬 Ask your question (type 'exit' to quit):")

def get_schema(_):
    return db.get_table_info()

def get_prompt():
    template = """
    Based on the table schema below, write only the SQL query (no explanation):
    {schema}

    Question: {question}
    SQL Query:
    """
    return ChatPromptTemplate.from_template(template)


llm = LlmModel.get_llm()

# Extract SQL text only
def extract_sql(text):
    return text.strip().split("SQL Query:")[-1].strip()


def clean_sql(query: str) -> str:
    q = query.strip()
    q = q.replace("```sql", "").replace("```", "").strip()
    return q

sql_chain = (
    RunnablePassthrough.assign(schema=get_schema)
    | get_prompt()
    | llm.bind(stop=["\nSQL Result:"])
    | StrOutputParser()
    | (lambda text: clean_sql(text.strip().split("SQL Query:")[-1].strip()))
)


def get_final_prompt():
    template = """
    Based on the table schema, the question, the SQL query, and the SQL result,
    write a natural-language answer. Use ONLY the SQL result.

    Schema:
    {schema}

    Question: {question}
    SQL Query: {query}
    SQL Result: {response}

    Answer:
    """
    return ChatPromptTemplate.from_template(template)


def run_query(query):
    raw = db.run(query)
    # If it's like [(25,)] → convert to nice Python object
    if isinstance(raw, list) and len(raw) > 0:
        row = raw[0]
        if isinstance(row, (list, tuple)):
            return row if len(row) > 1 else row[0]
    return raw


full_chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
        schema=get_schema,
        response=lambda vars: run_query(vars["query"])
    )
    | get_final_prompt()
    | llm
)


# -------------------------------
#      INTERACTIVE LOOP
# -------------------------------

while True:
    user_question = input("\n🧑‍💻 Your Question: ")

    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye! 👋")
        break

    # ------------------------------------------------------------------------------------
    # produce to kafka topic `events.input.raw`
    producer = KafkaProducer()
    print(f"Producing message to topic 'events.input.raw': {user_question}")

    message_id = str(uuid.uuid4())
    
    logger.info("")
    logger.info("🚀 [GATEWAY] New user message received")
    logger.info(f"   Message ID: {message_id}")
    logger.info(f"   User Message: {user_question}")
    logger.info(f"   Source: web_ui")
    logger.info("")
    
    event = {
        "message": user_question,
        "message_id": message_id,
        "source": "web_ui"
    }

    if not producer.connect():
        logger.error("Failed to connect Kafka producer")
    else:
        logger.info("Kafka producer connected")

    success = producer.produce(
        topic=TOPICS["raw_input"],
        message=event,
        key=message_id
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send message")
    
    producer.flush(timeout=5.0)


    # ------------------------------------------------------------------------------------
    #consume from kafka topic `events.input.raw`
    input_consumer: Optional[KafkaConsumer] = None
    event_loop: Optional[asyncio.AbstractEventLoop] = None

    input_consumer = KafkaConsumer(
        group_id="gateway-consumer-group",
        topics=[TOPICS["raw_input"]]
    )

    if not input_consumer.connect():
        logger.error("Failed to connect output consumer")

    def handle_message(message):
        message_id = message.get("message_id", "unknown")
        logger.info("")
        logger.info("📨 [GATEWAY] Received final response from pipeline")
        logger.info(f"   Message ID: {message_id}")
        logger.info(f"   Message : {message.get("message", "unknown")}")
        logger.info(f"   Intent: {message.get('intent', 'N/A')}")
        logger.info(f"   Action: {message.get('action', 'N/A')}")
        logger.info(f"   Success: {message.get('success', False)}")
        logger.info(f"   Broadcasting to WebSocket clients...")
        logger.info("")
        # Step 1: Generate SQL
        sql_query = sql_chain.invoke({"question": user_question})
        print("\nGenerated SQL Query:")
        print(sql_query)

        # Step 2: Execute SQL and get natural answer
        answer = full_chain.invoke({"question": user_question}).content

        print("\nFinal Answer:")
        print(answer)
        
        # if event_loop and not event_loop.is_closed():
        #     asyncio.run_coroutine_threadsafe(
        #         manager.broadcast_to_all({
        #             "type": "agent_response",
        #             "data": message
        #         }),
        #         event_loop
        #     )

    logger.info("Starting output message consumer...")
    input_consumer.consume(handle_message)

    #      # Step 1: Generate SQL
    sql_query = sql_chain.invoke({"question": user_question})
    print("\nGenerated SQL Query:")
    print(sql_query)

    # Step 2: Execute SQL and get natural answer
    answer = full_chain.invoke({"question": user_question}).content

    print("\nFinal Answer:")
    print(answer)
    if input_consumer:
        input_consumer.stop()
        input_consumer.close()
    logger.info("Gateway shutdown complete")


   
