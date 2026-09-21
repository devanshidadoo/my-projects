import time
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

import os
import sys

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tools.message_logger import MessageHistoryLoggerTool
from agents.curator.utils.tools.user_inputs import UserDataLoggerTool

class ResponseFormatter:
    """
    A class responsible for formatting tool results into agricultural advice and responses.
    It processes raw tool outputs into a user-friendly format and provides
    structured responses with agent_message, CTAs, and tasks.
    """

    def __init__(self, model, tools):
        """
        Initialize the ResponseFormatter with models and tools.

        Args:
            model: The language model for generating responses
            tools: List of tool objects available for use
        """
        self.model = model
        self.tools = tools

    async def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats tool results into agricultural advice and responses.

        Args:
            state: Current state of the agent

        Returns:
            Updated agent state with formatted response containing agent_message, CTAs, and tasks
        """
        print("ResponseFormatter: Starting Response Formatter")
        start_time = time.time()

        # Get message history
        message_history = state.get("message_history", [])

        # Extract URLs from search results for reference
        urls = []
        task_results = state.get('task_results', {})
        if task_results:
            for search_result in task_results.get('GoogleSearchTool', []):
                if isinstance(search_result.get('result'), list):
                    for item in search_result.get('result', []):
                        if isinstance(item, dict) and 'Link' in item:
                            urls.append(item['Link'])

        # Extract conversation_id for data retrieval
        conversation_id = state['message_to_curator']['conversation_id']

        # Fetch user inputs
        user_inputs = await self._fetch_user_inputs(conversation_id)

        # Generate comprehensive agricultural advice in one call
        message_history = await self._generate_complete_response(
            state, message_history, user_inputs
        )

        # Retrieve final user inputs for the response
        user_inputs = await self._fetch_user_inputs(conversation_id)

        # Extract the response from the last AI message
        last_ai_message = message_history[-1]
        try:
            response_content = last_ai_message.content.replace('```json', '').replace('```', '').strip()
            parsed_response = json.loads(response_content)

            # Enforce: If tasks is not empty, CTAs must be an empty list
            tasks_val = parsed_response.get("tasks", "")
            ctas_val = parsed_response.get("CTAs", [])
            if tasks_val and isinstance(tasks_val, str) and tasks_val.strip() != "":
                ctas_val = []

            # Update the state with the complete response format
            state["message_from_curator"] = {
                "user_inputs": user_inputs,
                "agent_message": parsed_response.get("agent_message", ""),
                "CTAs": ctas_val,
                "tasks": tasks_val
            }
        except (json.JSONDecodeError, AttributeError):
            # Fallback if parsing fails
            state["message_from_curator"] = {
                "user_inputs": user_inputs,
                "agent_message": last_ai_message.content,
                "CTAs": [],
                "tasks": ""
            }

        # Update message history in state
        state["message_history"] = message_history

        print(f"ResponseFormatter: Response Formatter Completed in {time.time() - start_time:.2f} seconds")

        return state

    async def _fetch_user_inputs(self, conversation_id: str) -> Dict:
        """
        Fetch user inputs for the current conversation.

        Args:
            conversation_id: Unique identifier for the conversation

        Returns:
            Dictionary containing user input data
        """
        user_inputs = {}

        # Try to fetch existing user inputs using Data Logger
        data_logger_tool = UserDataLoggerTool()

        if data_logger_tool:
            try:
                # Retrieve user inputs for this conversation
                user_inputs_result = await data_logger_tool._arun(
                    action="retrieve",
                    key=conversation_id
                )

                if user_inputs_result and not isinstance(user_inputs_result, str):
                    user_inputs = user_inputs_result
                elif isinstance(user_inputs_result, str):
                    try:
                        user_inputs = json.loads(user_inputs_result)
                    except:
                        print("Failed to parse user inputs as JSON")
            except Exception as e:
                print(f"Error retrieving user inputs: {e}")

        return user_inputs

    async def _generate_complete_response(
        self,
        state: Dict[str, Any],
        message_history: List[BaseMessage],
        user_inputs: Dict
    ) -> List[BaseMessage]:
        """
        Generate a complete agricultural advice response in a single call.

        Args:
            state: Current agent state
            message_history: Conversation history
            user_inputs: User inputs for context

        Returns:
            Updated conversation history with complete response
        """
        # Streamlined single prompt focused on agent message output
        message_history.append(
            HumanMessage(content=f"""
            User Query: {state['message_to_curator']['query']}

            User Context: {json.dumps(user_inputs, indent=2) if user_inputs else "No additional context available"}

            Language: {user_inputs.get("language", "English")}

            The agent message, CTAs and tasks should be in the language of the user, {user_inputs.get("language", "English")}, other than that, the internal workings of the agent should be in English.

            The keys should be in English, but the content in the keys should be in the language of the user, {user_inputs.get("language", "English")}.

            Respond in this exact JSON format:
            {{
                "agent_message": "Your comprehensive agricultural advice here (Markdown Response)",
                "CTAs": ["The next message the user is likely to send, not a question from you. If you are pushing out a task, DO NOT generate CTAs at all."],
                "tasks": "Specific tasks or actions assigned to the farmer based on the current context. Leave empty string if none."
            }}

            Language should always be ENGLISH.

            For the agent_message, provide a curated response based on tool results, including specific recommendations for crops, soil, weather, or pest management only if they are relevant to the user's query.
            Mention local practices and government schemes only when they are contextually appropriate to the query. Use uppercase and lowercase properly, it should be semi-formal!

            Maintain a warm, friendly tone throughout the length of the conversation.
            Try to keep the response length in check, within 70-150 words, would suffice.

            CTAs guidelines:
            - CTAs should always be the next message the user is likely to send, not a question from you (the agent).
            - Never generate CTAs if you are pushing out a task (i.e., if the "tasks" field is not empty, "CTAs" must be an empty list). This is critical.
            - So basically it is a next word prediction task, but you are predicting the user's response to your answer

            For tasks:
            - Only assign when contextually necessary and valuable
            - Use empty string "" when no specific tasks are needed
            - Make tasks specific and actionable when assigned
            - Never keep it more than 5-10 words! Short and simple it should be.
            """)
        )

        # Get complete response in one call
        complete_response = self.model.invoke(message_history)

        # Add response to message history
        message_history.append(complete_response)

        return message_history

