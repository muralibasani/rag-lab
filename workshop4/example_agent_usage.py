"""
Example usage of the order management agent with tools.
"""
from src.agent import create_agent, all_tools
from src.llm import LlmModel
from src.models import ResponseFormat, Context

# Initialize LLM
llm = LlmModel.get_llm()

# Define tools
tools = [
    # Existing tools
    # classify_message_tool,  # Uncomment if needed
    # kafka_docs_tool,
    # kafka_logs_tool,
    # schema_registry_logs_tool,
    
    # Order management tools
    *all_tools
]

# Create agent
agent = create_agent(
    model=llm,
    tools=all_tools,
    system_prompt="""You are a router assistant.

Step 1: Use 'classify_message_tool' to determine the message_type.

Step 2: Based on the message_type and user question, call exactly **one** of the available tools:

- get_order_count_tool: For questions about total number of orders
- get_order_status_tool: For questions about order status, existence, or cancellation/refund eligibility
- cancel_order_tool: When user wants to cancel an order
- refund_order_tool: When user wants to refund an order
- kafka_docs_tool: For Kafka documentation questions
- kafka_logs_tool: For Kafka log analysis
- schema_registry_logs_tool: For Schema Registry log analysis

After calling the correct tool, **do not call any further tools**.

Return only the final assistant message in plain text.
""",
    response_format=ResponseFormat,
    context_schema=Context
)

# Example usage
if __name__ == "__main__":
    # Example queries
    queries = [
        "How many orders are there?",
        "What is the status of order 20?",
        "Can order 20 be cancelled?",
        "Cancel order 20",
        "Refund order 33"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        response = agent(query)
        print(f"Response: {response}")

