"""
Order service layer - business logic for order operations.
"""
import os
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
import psycopg2
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Parse database URI and create connection
def get_db_connection():
    """Get a PostgreSQL database connection."""
    db_uri = os.getenv("DB_URI")
    if not db_uri:
        raise ValueError("DB_URI environment variable is not set")
    
    # Parse the URI
    parsed = urlparse(db_uri)
    
    # Extract connection parameters
    conn_params = {
        'host': parsed.hostname,
        'port': parsed.port or 5432,
        'database': parsed.path.lstrip('/'),
        'user': parsed.username,
        'password': parsed.password
    }
    
    # Remove None values
    conn_params = {k: v for k, v in conn_params.items() if v is not None}
    
    return psycopg2.connect(**conn_params)


def execute_query(query: str, params: Optional[Tuple] = None, fetch: bool = True) -> List[Tuple]:
    """
    Execute a SQL query and return results.
    
    Args:
        query: SQL query string
        params: Optional tuple of parameters for parameterized queries
        fetch: Whether to fetch results (True for SELECT, False for INSERT/UPDATE/DELETE)
    
    Returns:
        List of tuples representing rows
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                return cursor.fetchall()
            else:
                conn.commit()
                return []
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error executing query: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()



def get_order_count() -> Dict:
    """Get the total count of orders and count by status."""
    try:
        # Get total count
        total_rows = execute_query("SELECT COUNT(*) FROM lg_orders;")
        total_count = int(total_rows[0][0]) if total_rows else 0

        # Get count by status
        status_rows = execute_query("""
            SELECT order_status, COUNT(*)
            FROM lg_orders
            GROUP BY order_status;
        """)

        by_status = {
            str(status): int(count)
            for status, count in status_rows
        }

        return {
            "total_orders": total_count,
            "by_status": by_status
        }

    except Exception as e:
        logger.error("Error getting order count", exc_info=True)
        raise


def get_order_status(order_id: int) -> Dict:
    """Get the status and details of a specific order."""
    try:
        rows = execute_query("""
            SELECT
                o.order_id,
                o.order_status,
                o.total_amount,
                c.cancel_id IS NOT NULL AS has_cancellation,
                r.refund_id IS NOT NULL AS has_refund
            FROM lg_orders o
            LEFT JOIN lg_order_cancellation c ON o.order_id = c.order_id
            LEFT JOIN lg_refunds r ON o.order_id = r.order_id
            WHERE o.order_id = %s;
        """, params=(order_id,))

        if not rows:
            return {
                "order_id": order_id,
                "exists": False
            }

        order_id_val, status, total, has_cancel, has_refund = rows[0]

        return {
            "order_id": int(order_id_val),
            "exists": True,
            "status": str(status),
            "total_amount": float(total) if total is not None else None,
            "has_cancellation": bool(has_cancel),
            "has_refund": bool(has_refund),
        }
    except Exception as e:
        logger.error(f"Error getting order status: {e}", exc_info=True)
        raise


def cancel_order(order_id: int) -> Dict:
    """Cancel an order by updating its status and creating a cancellation record."""
    try:
        # First check if order exists and can be cancelled
        status = get_order_status(order_id)
        
        if not status.get("exists"):
            return {
                "success": False,
                "message": f"Order {order_id} does not exist",
                "order_id": order_id
            }
        
        if status.get("has_cancellation") or status.get("has_refund") or status.get("status") in ['Cancelled', 'Refunded']:
            return {
                "success": False,
                "message": f"Order {order_id} cannot be cancelled. It is already cancelled or refunded.",
                "order_id": order_id
            }
        
        # Execute cancellation
        # 1. Update order status
        execute_query(
            "UPDATE lg_orders SET order_status = 'Cancelled' WHERE order_id = %s;",
            params=(order_id,),
            fetch=False
        )
        
        # 2. Insert cancellation record
        execute_query(
            "INSERT INTO lg_order_cancellation (order_id, reason, cancelled_by) VALUES (%s, %s, %s);",
            params=(order_id, 'Customer request', 'system'),
            fetch=False
        )
        
        return {
            "success": True,
            "message": f"Order {order_id} has been successfully cancelled",
            "order_id": order_id
        }
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        return {
            "success": False,
            "message": f"Error cancelling order: {str(e)}",
            "order_id": order_id
        }


def refund_order(order_id: int) -> Dict:
    """Refund an order by updating its status and creating a refund record."""
    try:
        # First check if order exists and can be refunded
        status = get_order_status(order_id)
        
        if not status.get("exists"):
            return {
                "success": False,
                "message": f"Order {order_id} does not exist",
                "order_id": order_id,
                "refund_amount": None
            }
        
        if status.get("has_cancellation") or status.get("has_refund") or status.get("status") in ['Cancelled', 'Refunded']:
            return {
                "success": False,
                "message": f"Order {order_id} cannot be refunded. It is already cancelled or refunded.",
                "order_id": order_id,
                "refund_amount": None
            }
        
        total_amount = status.get("total_amount")
        
        # Execute refund
        # 1. Update order status
        execute_query(
            "UPDATE lg_orders SET order_status = 'Refunded' WHERE order_id = %s;",
            params=(order_id,),
            fetch=False
        )
        
        # 2. Insert refund record
        if total_amount is not None:
            execute_query(
                "INSERT INTO lg_refunds (order_id, refund_amount, refund_status) VALUES (%s, %s, %s);",
                params=(order_id, total_amount, 'Completed'),
                fetch=False
            )
        else:
            execute_query(
                "INSERT INTO lg_refunds (order_id, refund_amount, refund_status) VALUES (%s, (SELECT total_amount FROM lg_orders WHERE order_id = %s), %s);",
                params=(order_id, order_id, 'Completed'),
                fetch=False
            )
        
        return {
            "success": True,
            "message": f"Order {order_id} has been successfully refunded",
            "order_id": order_id,
            "refund_amount": total_amount
        }
    except Exception as e:
        logger.error(f"Error refunding order: {e}")
        return {
            "success": False,
            "message": f"Error refunding order: {str(e)}",
            "order_id": order_id,
            "refund_amount": None
        }

