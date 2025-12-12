"""
Agent setup for order management with tools.
"""
from langchain_core.tools import tool
from src.llm import LlmModel
import logging

logger = logging.getLogger(__name__)

llm = LlmModel.get_llm()


# Create structured tools for LangChain
@tool
def get_order_count_tool() -> str:
    """Get the total count of orders and count by status. Use this when user asks about total number of orders or order counts."""
    from src.order_service import get_order_count
    try:
        result = get_order_count()
        return f"Total orders: {result['total_orders']}. Orders by status: {result['by_status']}"
    except Exception as e:
        return f"Error getting order count: {str(e)}"


@tool
def get_order_status_tool(order_id: int) -> str:
    """Get the status and details of a specific order. Use this when user asks about order status, whether order exists, or if order can be cancelled/refunded."""
    from src.order_service import get_order_status
    try:
        result = get_order_status(order_id)
        if not result.get("exists"):
            return f"Order {order_id} does not exist."
        
        status_info = f"Order {order_id} status: {result.get('status')}"
        if result.get("total_amount"):
            status_info += f", Total amount: {result.get('total_amount')}"
        if result.get("has_cancellation"):
            status_info += ", Has cancellation record"
        if result.get("has_refund"):
            status_info += ", Has refund record"
        
        return status_info
    except Exception as e:
        return f"Error getting order status: {str(e)}"


@tool
def cancel_order_tool(order_id: int) -> str:
    """Cancel an order by updating its status and creating a cancellation record. Use this when user asks to cancel an order."""
    from src.order_service import cancel_order
    try:
        result = cancel_order(order_id)
        return result.get("message", "Unknown error")
    except Exception as e:
        return f"Error cancelling order: {str(e)}"


@tool
def refund_order_tool(order_id: int) -> str:
    """Refund an order by updating its status and creating a refund record. Use this when user asks to refund an order."""
    from src.order_service import refund_order
    try:
        result = refund_order(order_id)
        message = result.get("message", "Unknown error")
        if result.get("success") and result.get("refund_amount"):
            message += f" Refund amount: {result.get('refund_amount')}"
        return message
    except Exception as e:
        return f"Error refunding order: {str(e)}"


# Define tools list for the agent
order_tools = [
    get_order_count_tool,
    get_order_status_tool,
    cancel_order_tool,
    refund_order_tool
]

# Combined tools list (order tools only)
all_tools = [
    *order_tools
]


def create_agent(model, tools, system_prompt, response_format=None, context_schema=None):
    """
    Create an agent with tools.
    This is a simplified agent that uses the LLM with tool calling.
    
    Args:
        model: The LLM model to use
        tools: List of tool functions (can be @tool decorated functions or regular functions)
        system_prompt: System prompt for the agent
        response_format: Optional response format schema
        context_schema: Optional context schema
    
    Returns:
        Agent function that processes messages
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Bind tools to the model (if they're LangChain tools)
    langchain_tools = [t for t in tools if hasattr(t, 'name')]
    if langchain_tools:
        model_with_tools = model.bind_tools(langchain_tools)
    else:
        model_with_tools = model
    
    def agent(message: str, context: dict = None):
        """Agent function that processes messages and calls tools."""
        try:
            # Prepare messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            
            # Get initial response
            response = model_with_tools.invoke(messages)
            
            # Check if tool calls are needed
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    
                    # Find and call the tool
                    tool_func = None
                    for tool in tools:
                        if hasattr(tool, 'name') and tool.name == tool_name:
                            tool_func = tool
                            break
                    
                    if tool_func:
                        try:
                            result = tool_func.invoke(tool_args) if hasattr(tool_func, 'invoke') else tool_func(**tool_args)
                            tool_results.append({
                                "tool_call_id": tool_call.get("id"),
                                "content": str(result) if not isinstance(result, dict) else result.get("messages", [{}])[0].get("content", str(result))
                            })
                        except Exception as e:
                            logger.error(f"Error calling tool {tool_name}: {e}")
                            tool_results.append({
                                "tool_call_id": tool_call.get("id"),
                                "content": f"Error: {str(e)}"
                            })
                
                # Get final response with tool results
                from langchain_core.messages import ToolMessage
                messages.append(response)
                for tool_result in tool_results:
                    messages.append(ToolMessage(
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"]
                    ))
                
                final_response = model_with_tools.invoke(messages)
                return final_response.content if hasattr(final_response, 'content') else str(final_response)
            else:
                return response.content if hasattr(response, 'content') else str(response)
                
        except Exception as e:
            logger.error(f"Error in agent: {e}", exc_info=True)
            return f"Error processing request: {str(e)}"
    
    return agent



