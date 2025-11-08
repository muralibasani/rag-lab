import os
from dotenv import load_dotenv
from src.llm import LlmModel
from langchain.agents import create_agent
from src.models import Context, ResponseFormat
from src.tools import classify_message_tool, kafka_docs_tool, kafka_logs_tool, schema_registry_logs_tool
from langchain_core.messages import HumanMessage, AIMessage


load_dotenv()

def main():
    print("Hello from workshop2!")

    llm = LlmModel.get_llm()
    tools = [classify_message_tool, kafka_docs_tool, kafka_logs_tool, schema_registry_logs_tool]
    
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are a router assistant.
        Step 1: Use 'classify_message_tool' to determine the message_type.
        Step 2: Based on the message_type, call exactly **one** of the available tools:
        - kafka_docs_tool
        - kafka_logs_tool
        - schema_registry_logs_tool

        After calling the correct tool, **do not call any further tools**.
        Return only the final assistant message in plain text.
        """,
        response_format=ResponseFormat,
        context_schema=Context
    )

    config = {"configurable": {"conversation_id": "fdjshgjfdsgfjhsdgj"}}

    while True:
        query = input("You : ")
        if query.lower() in ["exit", "quit"]:
            print(f"🙏 Thank you for using ai assistant! ")
            break
        agent_messages = [HumanMessage(content=query)]  # create new list
        response_obj = agent.invoke({"messages": agent_messages}, config=config)
        response = response_obj['structured_response'].final_response
        print(f"\n🤖 Assistant: {response}")
    

if __name__ == "__main__":
    main()
