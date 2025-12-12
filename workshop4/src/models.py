from enum import Enum
from typing import Annotated, TypedDict
from langgraph.graph import add_messages
from dataclasses import dataclass

from pydantic import BaseModel, Field

class EventType(Enum):
    INFO = "get-info"
    ACTION = "take-action"


class MessageClassifier(BaseModel):
    message_type: EventType = Field(
        ...,
        description="Classify if the message is to get-info or take-action",
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
