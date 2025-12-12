from src.llm import LlmModel
from src.log_prompt import _run_message_classifier
from langchain_core.messages import AIMessage

llm = LlmModel.get_llm()

# -------------------------------------------------------------------------
# 🔹 Message classifier
# -------------------------------------------------------------------------
def classify_message_tool(user_message: str):
    """Classify a message into one of the known types."""
    message_type = _run_message_classifier(user_message, llm)
    print(f"Routed to: {message_type.value}")
    return {"message_type": message_type.value}


# -------------------------------------------------------------------------
# 🔹 Order management tools
# -------------------------------------------------------------------------
def get_order_count_tool(**kwargs):
    """Get the total count of orders and count by status."""
    from src.order_service import get_order_count
    try:
        result = get_order_count()
        return {
            "messages": [AIMessage(content=f"Total orders: {result['total_orders']}. Orders by status: {result['by_status']}")]
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error getting order count: {str(e)}")]
        }


def get_order_status_tool(order_id: int, **kwargs):
    """Get the status and details of a specific order."""
    from src.order_service import get_order_status
    try:
        result = get_order_status(order_id)
        if not result.get("exists"):
            return {
                "messages": [AIMessage(content=f"Order {order_id} does not exist.")]
            }
        
        status_info = f"Order {order_id} status: {result.get('status')}"
        if result.get("total_amount"):
            status_info += f", Total amount: {result.get('total_amount')}"
        if result.get("has_cancellation"):
            status_info += ", Has cancellation record"
        if result.get("has_refund"):
            status_info += ", Has refund record"
        
        return {
            "messages": [AIMessage(content=status_info)]
        }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error getting order status: {str(e)}")]
        }


def cancel_order_tool(order_id: int, **kwargs):
    """Cancel an order by updating its status and creating a cancellation record."""
    from src.order_service import cancel_order
    try:
        result = cancel_order(order_id)
        if result.get("success"):
            return {
                "messages": [AIMessage(content=result.get("message"))]
            }
        else:
            return {
                "messages": [AIMessage(content=result.get("message"))]
            }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error cancelling order: {str(e)}")]
        }


def refund_order_tool(order_id: int, **kwargs):
    """Refund an order by updating its status and creating a refund record."""
    from src.order_service import refund_order
    try:
        result = refund_order(order_id)
        if result.get("success"):
            message = result.get("message")
            if result.get("refund_amount"):
                message += f" Refund amount: {result.get('refund_amount')}"
            return {
                "messages": [AIMessage(content=message)]
            }
        else:
            return {
                "messages": [AIMessage(content=result.get("message"))]
            }
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error refunding order: {str(e)}")]
        }
