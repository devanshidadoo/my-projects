import time
import json
from typing import Dict, Any, List, Optional
import asyncio

import os
import sys

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from utils.tools.message_logger import MessageHistoryLoggerTool
from agents.curator.utils.tools.user_inputs import UserDataLoggerTool
from utils.tools.pest_detection import PestDetectionTool
from utils.tools.policy_fetcher import PolicyFetcherTool
from utils.tools.news_tool import NewsFetcherTool

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
)

from agents.config import Config as config
from utils.prompts import SYSTEM_PROMPT, QUERY_ROUTER_PROMPT

class QueryRouter:
    def __init__(self, model, system_prompt: str, ):
        """
        Initialize the QueryRouter.

        Args:
            model: A language model with tools capability
            system_prompt: The system prompt to use for the curator
        """
        self.model = model
        self.system_prompt = system_prompt
        self.message_logger = MessageHistoryLoggerTool()

    async def process_state(self, state: Dict[str, Any], skip: bool = False, usecase_type: str = None) -> Dict[str, Any]:
        """
        Routes the user query to appropriate actions or responses 

        Args:
            state: Current state of the agent 
            skip: If True, skips normal routing and goes straight to response formatting
            usecase_type: Type of usecase (pest, policy, etc.) for tool execution

        Returns:
            Updated agent state with router analysis
        """
        print("QueryRouter: Starting Query Router")
        start_time = time.time()
        conversation_id = state['message_to_curator']['conversation_id']

        # Extract relevant information from state
        query = state['message_to_curator']['query']
        user_inputs = state.get('user_inputs', {})
        image_url = state.get('image_url')
        policy_details = state.get('policy_details')
        news_url = state.get('news_url', '')

        print(f"QueryRouter: news_url: {news_url}")

        # Retrieve existing message history for this conversation_id
        message_history: List[BaseMessage] = await self.message_logger._arun(
            action="retrieve",
            conversation_id=conversation_id,
            agent_type="curator"
        )

        # Initialize with system prompt if no history
        if not isinstance(message_history, list) or len(message_history) == 0:
            message_history = [SystemMessage(content=self.system_prompt)]

        # Try to fetch existing user inputs using Data Logger
        existing_user_inputs = {}
        user_inputs_result = await UserDataLoggerTool()._arun(
            action="retrieve", 
            key=conversation_id
        )
        
        if user_inputs_result and not isinstance(user_inputs_result, str):
            existing_user_inputs = user_inputs_result
        elif isinstance(user_inputs_result, str):
            try:
                existing_user_inputs = json.loads(user_inputs_result)
            except:
                print(f"Error parsing user inputs: {user_inputs_result}")
        
        # Merge user inputs
        merged_user_inputs = {**existing_user_inputs, **user_inputs}

        language = merged_user_inputs.get("language", "English")

        print(f"QueryRouter: language: {language}")
        
        if skip:
            # If skipping, create a simple message flow for response formatting
            print("QueryRouter: Skipping normal routing, creating simple message flow")

            print(f"QueryRouter: usecase_type: {usecase_type}, image_url: {image_url}, policy_details: {policy_details}")
            print(f"QueryRouter: Debug - image_url type: {type(image_url)}, policy_details type: {type(policy_details)}")
            
            # Format the query router prompt with the actual values (same as normal flow)
            formatted_prompt = QUERY_ROUTER_PROMPT.format(
                query=query,
                conversation_id=conversation_id,
                user_persona=json.dumps(merged_user_inputs, indent=2),
                language=language
            )
            
            # Add formatted query router prompt if not already present
            if not any(isinstance(msg, HumanMessage) and msg.content == formatted_prompt for msg in message_history):
                message_history.append(HumanMessage(content=formatted_prompt, name="user"))
            
            # Execute tools based on usecase type and add results to message history
            if usecase_type == "pest" and image_url and isinstance(image_url, str):
                print("QueryRouter: Executing pest detection tool")
                print("=== PEST DETECTION TOOL INPUTS ===")
                print(f"image: {image_url} (type: {type(image_url)})")
                print("==================================")
                try:
                    pest_tool = PestDetectionTool()
                    tool_result = pest_tool._run(
                        image=image_url,
                    )

                    router_message = {
                        "agent_message": "",
                        "CTAs": [],
                        "tool_calls": [
                            {
                                "tool_name": "PestDetectionTool",
                                "args": {
                                    "image": image_url
                                }
                            }
                        ],
                        "tasks": "",
                        "routing_skipped": True,
                        "note": "Routing stage was skipped for this specialized endpoint. Proceeding directly to response formatting."
                    }

                    message_history.append(AIMessage(content=json.dumps(router_message, indent=2)))
                    
                    # Add tool result as an AI message
                    tool_message = {
                        "tool_name": "PestDetectionTool",
                        "result": tool_result,
                        "note": "Pest detection tool executed successfully"
                    }
                    message_history.append(AIMessage(content=json.dumps(tool_message, indent=2)))
                    
                except Exception as e:
                    print(f"Error executing pest detection tool: {e}")
                    error_message = {
                        "tool_name": "PestDetectionTool",
                        "error": str(e),
                        "note": "Pest detection tool failed to execute"
                    }
                    message_history.append(AIMessage(content=json.dumps(error_message, indent=2)))
            
            elif usecase_type == "policy" and policy_details and isinstance(policy_details, dict):
                print("QueryRouter: Executing policy fetcher tool")
                print("=== POLICY FETCHER TOOL INPUTS ===")
                print(f"policy_details: {policy_details} (type: {type(policy_details)})")
                if isinstance(policy_details, dict):
                    for key, value in policy_details.items():
                        print(f"  {key}: {value} (type: {type(value)})")
                print("==================================")
                try:
                    policy_tool = PolicyFetcherTool()
                    tool_result = policy_tool._run(
                        **policy_details
                    )

                    router_message = {
                        "agent_message": "",
                        "CTAs": [],
                        "tool_calls": [
                            {
                                "tool_name": "PolicyFetcherTool",
                                "args": policy_details
                            }
                        ],
                        "tasks": "",
                        "routing_skipped": True,
                        "note": "Routing stage was skipped for this specialized endpoint. Proceeding directly to response formatting."
                    }

                    message_history.append(AIMessage(content=json.dumps(router_message, indent=2)))

                    # Add tool result as an AI message
                    tool_message = {
                        "tool_name": "PolicyFetcherTool",
                        "result": tool_result,
                        "note": "Policy fetcher tool executed successfully"
                    }
                    message_history.append(AIMessage(content=json.dumps(tool_message, indent=2)))
                    
                except Exception as e:
                    print(f"Error executing policy fetcher tool: {e}")
                    error_message = {
                        "tool_name": "PolicyFetcherTool",
                        "error": str(e),
                        "note": "Policy fetcher tool failed to execute"
                    }
                    message_history.append(AIMessage(content=json.dumps(error_message, indent=2)))
            
            elif usecase_type == "news" and news_url and isinstance(news_url, str):
                print("QueryRouter: Executing news fetcher tool")
                print("=== NEWS FETCHER TOOL INPUTS ===")
                print(f"news_url: {news_url} (type: {type(news_url)})")
                print("==================================")
                
                try:
                    news_tool = NewsFetcherTool()
                    news_snippet = news_tool.scrape(
                        news_url
                    )
                    
                    router_message = {
                        "agent_message": "",
                        "CTAs": [],
                        "tool_calls": [
                            {
                                "tool_name": "NewsFetcherTool",
                                "args": news_url
                            }
                        ],
                        "tasks": "",
                        "routing_skipped": True,
                        "note": "Routing stage was skipped for this specialized endpoint. Proceeding directly to response formatting."
                    }
                    
                    message_history.append(AIMessage(content=json.dumps(router_message, indent=2)))
                    
                    # Add tool result as an AI message
                    tool_message = {
                        "tool_name": "NewsFetcherTool",
                        "result": news_snippet,
                        "note": "News fetcher tool executed successfully"
                    }
                    message_history.append(AIMessage(content=json.dumps(tool_message, indent=2)))
                    
                except Exception as e:
                    print(f"Error executing news fetcher tool: {e}")
                    error_message = {
                        "tool_name": "NewsFetcherTool",
                        "error": str(e),
                        "note": "News fetcher tool failed to execute"
                    }
                    message_history.append(AIMessage(content=json.dumps(error_message, indent=2)))
            
        else:
            # Normal routing flow
            print("QueryRouter: Processing normal routing flow")
            
            # Format the query router prompt with the actual values
            formatted_prompt = QUERY_ROUTER_PROMPT.format(
                query=query,
                conversation_id=conversation_id,
                user_persona=json.dumps(merged_user_inputs, indent=2),
                language=language
            )
            
            # Ask router to generate the final response
            message_history.append(
                HumanMessage(
                    content=formatted_prompt,
                    name="user"
                )
            )

            # Get router response
            final_router_response = await self.model.ainvoke(message_history)

            # Add router response to message history
            message_history.append(final_router_response)

        # Persist updated message history back to DB
        await self.message_logger._arun(
            action="store",
            conversation_id=conversation_id,
            agent_type="curator",
            messages=message_history
        )

        # Update state with new message history
        state["message_history"] = message_history
        
        print(f"QueryRouter: Query Router Completed in {time.time() - start_time:.2f} seconds")
        
        return state

if __name__ == "__main__":
    async def test_query_router():
        """
        Test function to demonstrate the QueryRouter functionality.
        """
        print("Testing QueryRouter...")
        
        # Initialize the model
        model = ChatOpenAI(model="gpt-5-mini", temperature=0.2, api_key=config.OPENAI_API_KEY)
        
        # Define system prompt
        system_prompt = SYSTEM_PROMPT
        
        # Initialize the QueryRouter
        router = QueryRouter(model, system_prompt)
        
        # Test state
        test_state = {
            'message_to_curator': {
                'query': "Hi",
                'conversation_id': "test123"
            },
            'message_history': [],
            'user_inputs': {}
        }
        
        # Process the state
        result = await router.process_state(test_state)
        
        # Print the results
        print("\n=== Test Results ===")
        print(f"Query: {test_state['message_to_curator']['query']}")
        print(f"Conversation ID: {test_state['message_to_curator']['conversation_id']}")

        router_response = result.get('message_history', [])[-1].content.replace("```json", "").replace("```", "")
        router_response = json.loads(router_response)

        print(router_response)
        
        return result

    # Run the test
    asyncio.run(test_query_router())