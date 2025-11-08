from enum import Enum
from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from dataclasses import dataclass

from pydantic import BaseModel, Field

class EventType(Enum):
    KAFKA = "kafka-logs"
    SCHEMA_REGISTRY = "schema-registry-logs"
    KAFKA_DOCS = "kafka-docs"


class MessageClassifier(BaseModel):
    message_type: EventType = Field(
        ...,
        description="Classify if the message is about kafka logs, schema registry logs, or kafka documentation"
    )

class State(TypedDict):
    messages : Annotated[list, add_messages]
    message_type: str | None
    next : str | None
    
@dataclass
class ResponseFormat:
    """Agent structured output."""
    final_response: str
    reasoning_path: str | None = None

@dataclass
class Context:
    user_id: str
