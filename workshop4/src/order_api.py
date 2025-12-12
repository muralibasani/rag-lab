"""
FastAPI endpoints for order operations.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.order_service import get_order_count, get_order_status, cancel_order, refund_order
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Order Management API")


class OrderStatusResponse(BaseModel):
    order_id: int
    exists: bool
    status: Optional[str] = None
    total_amount: Optional[float] = None
    has_cancellation: bool = False
    has_refund: bool = False


class OrderCountResponse(BaseModel):
    total_orders: int
    by_status: dict


class CancelOrderResponse(BaseModel):
    success: bool
    message: str
    order_id: int


class RefundOrderResponse(BaseModel):
    success: bool
    message: str
    order_id: int
    refund_amount: Optional[float] = None


@app.get("/orders/count", response_model=OrderCountResponse)
async def get_order_count_endpoint():
    """Get the total count of orders and count by status."""
    try:
        result = get_order_count()
        return OrderCountResponse(**result)
    except Exception as e:
        logger.error(f"Error getting order count: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting order count: {str(e)}")


@app.get("/orders/{order_id}/status", response_model=OrderStatusResponse)
async def get_order_status_endpoint(order_id: int):
    """Get the status and details of a specific order."""
    try:
        result = get_order_status(order_id)
        return {
            "order_id": order_id,
            "exists": False,
            "status": None,
            "total_amount": None,
            "has_cancellation": False,
            "has_refund": False
        }

    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting order status: {str(e)}")


@app.post("/orders/{order_id}/cancel", response_model=CancelOrderResponse)
async def cancel_order_endpoint(order_id: int):
    """Cancel an order by updating its status and creating a cancellation record."""
    try:
        result = cancel_order(order_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return CancelOrderResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        raise HTTPException(status_code=500, detail=f"Error cancelling order: {str(e)}")


@app.post("/orders/{order_id}/refund", response_model=RefundOrderResponse)
async def refund_order_endpoint(order_id: int):
    """Refund an order by updating its status and creating a refund record."""
    try:
        result = refund_order(order_id)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message"))
        return RefundOrderResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refunding order: {e}")
        raise HTTPException(status_code=500, detail=f"Error refunding order: {str(e)}")

