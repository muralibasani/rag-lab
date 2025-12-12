import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from workshop4.llm import LlmModel
from kafka_utils.producer import KafkaProducer
from kafka_utils.consumer import KafkaConsumer

load_dotenv()

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

    # produce to kafka topic `events.input.raw`
    producer = KafkaProducer(topic="events.input.raw")
    producer.send("events.input.raw", user_question)
    producer.close()

    #consume from kafka topic `events.input.raw`
    consumer = KafkaConsumer(topic="events.input.raw", group_id="workshop4-group")

    for message in consumer.consume():
        user_question = message.value
         # Step 1: Generate SQL
        sql_query = sql_chain.invoke({"question": user_question})
        print("\nGenerated SQL Query:")
        print(sql_query)

        # Step 2: Execute SQL and get natural answer
        answer = full_chain.invoke({"question": user_question}).content

        print("\nFinal Answer:")
        print(answer)
    consumer.close()    


   
