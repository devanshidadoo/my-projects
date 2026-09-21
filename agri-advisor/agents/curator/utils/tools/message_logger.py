from typing import Type, Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import asyncio
import datetime
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import os
import requests
import sys

# Add the project root to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from agents.config import Config as config

class MessageHistoryLoggerInput(BaseModel):
    action: str = Field(..., description="Action to perform (store, retrieve, update, delete)")
    conversation_id: str = Field(..., description="Conversation ID to identify the message history")
    agent_type: str = Field(..., description="Type of agent (e.g., curator, planner)")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Messages to store or update")

class MessageHistoryLoggerTool(BaseTool):
    name: str = "message_history_logger_tool"
    description: str = "Tool to log and retrieve message history for each conversation, with separate histories for different agent types"
    args_schema: Type[MessageHistoryLoggerInput] = MessageHistoryLoggerInput
    db_url: str = config.DB_URL

    def _serialize_message(self, message):
        """Convert Message objects to serializable dictionaries"""
        if isinstance(message, dict):
            return message

        if isinstance(message, HumanMessage):
            return {
                "type": "human",
                "content": message.content,
                "name": getattr(message, "name", "user")
            }
        elif isinstance(message, AIMessage):
            result = {
                "type": "ai",
                "content": message.content,
                "name": getattr(message, "name", "assistant")
            }
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = message.tool_calls
            return result
        elif isinstance(message, SystemMessage):
            return {
                "type": "system",
                "content": message.content
            }
        elif isinstance(message, ToolMessage):
            return {
                "type": "tool",
                "content": message.content,
                "tool_call_id": getattr(message, "tool_call_id", ""),
                "name": getattr(message, "name", "")
            }
        else:
            return {
                "type": "unknown",
                "content": str(message)
            }

    def _deserialize_message(self, message_dict):
        """Convert serialized dictionaries back to Message objects"""
        msg_type = message_dict.get("type")
        content = message_dict.get("content", "")

        if msg_type == "human":
            return HumanMessage(
                content=content,
                name=message_dict.get("name", "user")
            )
        elif msg_type == "ai":
            msg = AIMessage(
                content=content,
                name=message_dict.get("name", "assistant")
            )
            if "tool_calls" in message_dict:
                msg.tool_calls = message_dict["tool_calls"]
            return msg
        elif msg_type == "system":
            return SystemMessage(content=content)
        elif msg_type == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=message_dict.get("tool_call_id", ""),
                name=message_dict.get("name", "")
            )
        else:
            return HumanMessage(content=f"Unknown message type: {content}")

    async def _arun(
        self,
        action: str,
        conversation_id: str,
        agent_type: str,
        messages: Optional[List[Any]] = None
    ) -> str:
        """Asynchronous execution of the tool"""
        try:
            timestamp = datetime.datetime.now().isoformat()

            if action == "store":
                if not messages:
                    return f"Error: No messages provided for {agent_type} message history storage"

                serialized_messages = [
                    self._serialize_message(msg) for msg in messages
                ]

                # Retrieve existing message document
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "messages",
                        "_id": conversation_id,
                    }
                )

                if response.status_code == 200:
                    doc = response.json().get('data', {})
                else:
                    doc = {}

                # Update this agent_type's messages
                doc[agent_type] = {
                    "messages": serialized_messages,
                    "updated_at": timestamp
                }

                # Persist the updated document
                response = requests.post(
                    self.db_url,
                    params={
                        "collection_name": "messages",
                        "_id": conversation_id,
                    },
                    json=doc
                )
                
                if response.status_code == 200:
                    return f"{agent_type.capitalize()} message history stored for conversation ID: {conversation_id}"
                else:
                    return f"Error storing messages: {response.text}"

            elif action == "retrieve":
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "messages",
                        "_id": conversation_id,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json().get('data', {})
                    agent_history = data.get(agent_type, {})
                    messages = agent_history.get('messages', [])
                    return [self._deserialize_message(msg) for msg in messages]
                else:
                    return []

            elif action == "delete":
                # Retrieve existing
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "messages",
                        "_id": conversation_id,
                    }
                )

                if response.status_code == 200:
                    doc = response.json().get('data', {})
                    if agent_type in doc:
                        doc.pop(agent_type)

                    # Persist updated doc
                    response = requests.post(
                        self.db_url,
                        params={
                            "collection_name": "messages",
                            "_id": conversation_id,
                        },
                        json=doc
                    )

                    if response.status_code == 200:
                        return f"{agent_type.capitalize()} message history deleted for conversation ID: {conversation_id}"
                    else:
                        return f"Error deleting messages: {response.text}"
                else:
                    return f"Error fetching existing messages: {response.text}"

            else:
                return f"Error: Invalid action '{action}'. Use 'store', 'retrieve', or 'delete'."

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _run(
        self,
        action: str,
        conversation_id: str,
        agent_type: str,
        messages: Optional[List[Any]] = None
    ) -> str:
        """Synchronous execution wrapper"""
        return asyncio.run(self._arun(action=action, conversation_id=conversation_id, agent_type=agent_type, messages=messages))
    
if __name__ == "__main__":
    logger_tool = MessageHistoryLoggerTool()

    messages_to_store = [
        HumanMessage(content="Hello, how can I help you?"),
        AIMessage(content="I need assistance with my order."),
        SystemMessage(content="User initiated a conversation.")
    ]

    response = logger_tool._run(
        action="store",
        conversation_id="68048e6b11964da0866d63de",
        agent_type="curator",
        messages=messages_to_store
    )
    
    print(response)
