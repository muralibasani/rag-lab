import os
from dotenv import load_dotenv
from src.llm import LlmModel
from src.agents import create_agent
from src.models import Context, ResponseFormat
from src.agent import get_order_count_tool, get_order_status_tool, cancel_order_tool, refund_order_tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


def main():
    print("Hello from workshop4!")

    llm = LlmModel.get_llm()

    tools = [
        get_order_count_tool,
        get_order_status_tool,
        cancel_order_tool,
        refund_order_tool
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are an order management assistant.

Step 1: Based on the user's question, call exactly **one** of the available tools:

- get_order_count_tool: When user asks about total number of orders or order counts
- get_order_status_tool: When user asks about order status, whether order exists, or if order can be cancelled/refunded
- cancel_order_tool: When user asks to cancel an order
- refund_order_tool: When user asks to refund an order

After calling the correct tool, **do not call any further tools**.

Return only the final assistant message in plain text.
""",
        response_format=ResponseFormat,
        context_schema=Context
    )

    config = {"configurable": {"conversation_id": "workshop4_order_management"}}

    while True:
        query = input("You : ")
        if query.lower() in ["exit", "quit"]:
            print(f"🙏 Thank you for using ai assistant!")
            break

        agent_messages = [HumanMessage(content=query)]  # create new list
        response_obj = agent.invoke({"messages": agent_messages}, config=config)
        response = response_obj['structured_response'].final_response
        print(f"\n🤖 Assistant: {response}")


if __name__ == "__main__":
    main()
