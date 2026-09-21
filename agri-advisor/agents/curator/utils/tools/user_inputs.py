from typing import Type, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
import json
import asyncio
import datetime
from langchain.tools import BaseTool
import os
import requests
import sys

# Add the parent directory (project root) to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from agents.config import Config as config

class UserDataLoggerInput(BaseModel):
    action: str = Field(..., description="Action to perform (store, retrieve, update, delete, get_history)")
    key: str = Field(..., description="Key to identify the user_inputs")
    data: Optional[Union[Dict[str, Any], str]] = Field(None, description="Data to store or update (can be dictionary or string)")

class UserDataLoggerTool(BaseTool):
    name: str = "UserDataLoggerTool"
    description: str = "Log and manage user travel inputs, storing preferences, retrieving past data, updating details, and deleting entries."
    args_schema: Type[UserDataLoggerInput] = UserDataLoggerInput
    db_url: str = config.DB_URL

    def update_history(self, action: str, history: list, data: Optional[Dict[str, Any]] = None) -> list:
        """Update history with a new entry"""
        timestamp = datetime.datetime.now().isoformat()
        history_entry = {
            "timestamp": timestamp,
            "action": action,
            "data_summary": str(data)[:100] + "..." if data and len(str(data)) > 100 else str(data)
        }
        history.append(history_entry)
        return history

    async def _arun(self, action: str, key: str, data: Optional[Union[Dict[str, Any], str]] = None) -> str:
        timestamp = datetime.datetime.now().isoformat() 
        try:
            if action == "store":
                if data:
                    if isinstance(data, dict):
                        data["timestamp"] = timestamp

                # Retrieve existing data and history
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "user_inputs",
                        "_id": key
                    }
                )

                print(f"Response: {response}")

                if response.status_code == 200:
                    user_inputs = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("user_inputs", {})
                    history = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("history", [])
                else:
                    user_inputs = data
                    history = []

                history = self.update_history(action, history, data)

                # Store via POST request
                response = requests.post(
                    self.db_url,
                    params={
                        "collection_name": "user_inputs",
                        "_id": key
                    },
                    json={
                        "user_inputs": user_inputs,
                        "history": history
                    }
                )
                
                if response.status_code == 200:
                    return f"User inputs stored for key: {key}"
                else:
                    return f"Error storing data: {response.text}"

            elif action == "retrieve":
                # Retrieve via GET request
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "user_inputs",
                        "_id": key
                    }
                )
                
                if response.status_code == 200:
                    user_inputs = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("user_inputs", {})
                    history = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("history", [])
                    history = self.update_history(action, history)
                    
                    # Update history in the database
                    requests.post(
                        self.db_url,
                        params={
                            "collection_name": "user_inputs",
                            "_id": key
                        },
                        json={
                            "user_inputs": user_inputs,
                            "history": history
                        }
                    )
                    return json.dumps(user_inputs, indent=2)
                else:
                    return "{}"

            elif action == "update":
                if not data:
                    return "Error: Data required for update"

                # Create tasks for each field update
                for field_key, value in data.items():
                    response = requests.put(
                        self.db_url,
                        params={
                            "collection_name": "user_inputs",
                            "_id": key,
                            "key": field_key
                        },
                        json=value
                    )
                    if response.status_code != 200:
                        return f"Error updating {field_key}: {response.text}"
                
                # Update history
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "user_inputs",
                        "_id": key
                    }
                )
                if response.status_code == 200:
                    user_inputs = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("user_inputs", {})
                    user_inputs["updated_at"] = timestamp
                    history = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("history", [])
                    history = self.update_history(action, history, data)
                    
                    # Update history in the database
                    requests.post(
                        self.db_url,
                        params={
                            "collection_name": "user_inputs",
                            "_id": key
                        },
                        json={
                            "user_inputs": user_inputs,
                            "history": history
                        }
                    )
                
                return f"User inputs updated for key: {key}"

            elif action == "delete":
                # Delete via POST request
                response = requests.get(
                    self.db_url,
                    params={
                        "collection_name": "user_inputs",
                        "_id": key
                    }
                )
                if response.status_code == 200:
                    history = json.loads(json.dumps(response.json().get('data', {}), indent=2)).get("history", [])
                    history = self.update_history(action, history)
                    
                    # Update history in the database
                    requests.post(
                        self.db_url,
                        params={
                            "collection_name": "user_inputs",
                            "_id": key
                        },
                        json={
                            "history": history
                        }
                    )

                    return f"User inputs deleted for key: {key}"
                else:
                    return f"Error deleting data: {response.text}"

            else:
                return f"Error: Invalid action '{action}'. Use 'store', 'retrieve', 'update', or 'delete'."

        except Exception as e:
            return f"Error processing request: {str(e)}"

    def _run(self, action: str, key: str, data: Optional[Union[Dict[str, Any], str]] = None) -> str:
        return asyncio.run(self._arun(action=action, key=key, data=data))
    
if __name__ == "__main__":
    # Example usage
    tool = UserDataLoggerTool()
    action = "store"
    key = "68048e6b11964da0866d63ce"
    data = {
        "destination": "Taiwan"
    }
    
    result = tool._run(action=action, key=key, data=data)
    print(result)