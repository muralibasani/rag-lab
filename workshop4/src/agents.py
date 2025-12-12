"""
LangChain agents module - provides create_agent function compatible with the expected interface.
"""
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class StructuredResponse:
    """Structured response wrapper."""
    final_response: str
    reasoning_path: str = None


class Agent:
    """Agent class that matches the expected interface."""
    
    def __init__(self, model, tools, system_prompt, response_format=None, context_schema=None):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.context_schema = context_schema
        
        # Bind tools to the model
        langchain_tools = [t for t in tools if hasattr(t, 'name')]
        if langchain_tools:
            self.model_with_tools = model.bind_tools(langchain_tools)
        else:
            self.model_with_tools = model
    
    def invoke(self, input_data: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """
        Invoke the agent with messages.
        
        Args:
            input_data: Dict with "messages" key containing list of messages
            config: Optional configuration dict
        
        Returns:
            Dict with "structured_response" key containing StructuredResponse
        """
        try:
            messages = input_data.get("messages", [])
            
            # Add system prompt if not already in messages
            has_system = any(isinstance(m, SystemMessage) for m in messages)
            if not has_system:
                messages = [SystemMessage(content=self.system_prompt)] + messages
            
            # Convert HumanMessage if needed
            processed_messages = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    processed_messages.append(msg)
                elif isinstance(msg, dict):
                    if msg.get("role") == "user" or msg.get("role") == "human":
                        processed_messages.append(HumanMessage(content=msg.get("content", "")))
                    elif msg.get("role") == "system":
                        processed_messages.append(SystemMessage(content=msg.get("content", "")))
                    else:
                        processed_messages.append(msg)
                else:
                    processed_messages.append(msg)
            
            # Get initial response
            response = self.model_with_tools.invoke(processed_messages)
            
            # Check if tool calls are needed
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    
                    # Find and call the tool
                    tool_func = None
                    for tool in self.tools:
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
                processed_messages.append(response)
                for tool_result in tool_results:
                    processed_messages.append(ToolMessage(
                        content=tool_result["content"],
                        tool_call_id=tool_result["tool_call_id"]
                    ))
                
                final_response = self.model_with_tools.invoke(processed_messages)
                final_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
            else:
                final_text = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "structured_response": StructuredResponse(
                    final_response=final_text,
                    reasoning_path=None
                )
            }
                
        except Exception as e:
            logger.error(f"Error in agent invoke: {e}", exc_info=True)
            return {
                "structured_response": StructuredResponse(
                    final_response=f"Error processing request: {str(e)}",
                    reasoning_path=None
                )
            }


def create_agent(model, tools, system_prompt, response_format=None, context_schema=None):
    """
    Create an agent with tools.
    
    Args:
        model: The LLM model to use
        tools: List of tool functions
        system_prompt: System prompt for the agent
        response_format: Optional response format schema
        context_schema: Optional context schema
    
    Returns:
        Agent instance with invoke method
    """
    return Agent(model, tools, system_prompt, response_format, context_schema)

