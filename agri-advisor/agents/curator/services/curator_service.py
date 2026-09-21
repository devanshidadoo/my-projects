import json
import time
import asyncio
from typing import TypedDict, List, Dict, Any, Literal, Tuple, Optional
import nest_asyncio
import os
import sys

# Add the parent directory (project root) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

from utils.query_router import QueryRouter
from utils.task_manager import TaskManager
from utils.response_formatter import ResponseFormatter
from utils.tools.search_tool import WebSearchTool
from utils.tools.user_inputs import UserDataLoggerTool
from utils.prompts import SYSTEM_PROMPT
from utils.tools.message_logger import MessageHistoryLoggerTool
from utils.tools.pest_detection import PestDetectionTool
from utils.tools.policy_fetcher import PolicyFetcherTool
from utils.tools.weather_tool import WeatherAnalysisTool
from utils.tools.price_fetcher import PriceFetcherTool
from utils.tools.retrieval_tool import RetrievalTool

nest_asyncio.apply()

# Enhanced state structure for LangGraph
class CuratorState(TypedDict):
    messages: List[BaseMessage]  # LangGraph messages state
    conversation_id: str
    user_inputs: Dict[str, Any]
    task_results: Optional[Dict[str, Any]]
    response: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]  # Additional metadata like lat, long, place, etc.
    query: str  # Current user query passed explicitly
    image_url: Optional[str]
    policy_details: Optional[Dict[str, Any]]
    news_url: Optional[str]

class CuratorNode:
    """
    CuratorNode orchestrates the query routing, task management, and response formatting
    components using LangGraph for better state management and workflow control.
    """
    
    def __init__(self, model, tools, system_prompt=SYSTEM_PROMPT, skip_routing=False):
        """
        Initialize the CuratorNode with models, tools, and system prompt.
        
        Args:
            model: Language model for generating summaries
            tools: List of tool objects available for use
            system_prompt: System prompt for the curator
            skip_routing: If True, skips the routing stage and goes directly to response formatter
        """
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.skip_routing = skip_routing
        
        # Initialize components
        self.query_router = QueryRouter(model, system_prompt)
        self.task_manager = TaskManager(tools)
        self.response_formatter = ResponseFormatter(model, tools)
        self.message_logger = MessageHistoryLoggerTool()
        
        # Create LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """
        Create the LangGraph workflow for the curator.
        
        Returns:
            Configured StateGraph for the curator workflow
        """
        print("CuratorNode: Creating workflow...")
        
        # Create the state graph
        workflow = StateGraph(CuratorState)
        
        # Add nodes
        workflow.add_node("query_router", self._query_router_node)
        workflow.add_node("task_manager", self._task_manager_node)
        workflow.add_node("response_formatter", self._response_formatter_node)
        workflow.add_node("final_response", self._final_response_node)
        
        # Set entry point
        workflow.set_entry_point("query_router")
        
        # Add conditional edges
        if self.skip_routing:
            workflow.add_edge("query_router", "response_formatter")
        else:
            workflow.add_conditional_edges(
                "query_router",
                self._should_execute_tasks,
                {
                    "execute_tasks": "task_manager",
                    "skip_tasks": "final_response"
                }
            )
        
        workflow.add_conditional_edges(
            "task_manager",
            self._should_format_response,
            {
                "format_response": "response_formatter",
                "skip_format": "final_response"
            }
        )
        
        workflow.add_edge("response_formatter", "final_response")
        workflow.add_edge("final_response", END)
        
        print("CuratorNode: Workflow created successfully")
        return workflow.compile()
    
    async def _query_router_node(self, state: CuratorState) -> CuratorState:
        """
        Query router node that analyzes the query and determines next steps.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with router analysis
        """
        print("CuratorNode: Starting Query Router")
        start_time = time.time()
        
        # Use query from state; fallback to extracting from messages
        query = state.get("query") or self._extract_latest_query(state["messages"])
        
        # Create internal state for router
        # Validate and ensure correct data types
        image_url = state.get("image_url")
        policy_details = state.get("policy_details")
        news_url = state.get("news_url")
        
        # Log the actual types for debugging
        print(f"CuratorNode: Debug - image_url type: {type(image_url)}, value: {image_url}")
        print(f"CuratorNode: Debug - policy_details type: {type(policy_details)}, value: {policy_details}")
        print(f"CuratorNode: Debug - news_url type: {type(news_url)}, value: {news_url}")
        
        internal_state = {
            "message_to_curator": {
                "query": query,
                "conversation_id": state["conversation_id"],
                "image_url": image_url,
                "policy_details": policy_details,
                "news_url": news_url
            },
            "message_history": state["messages"],
            "user_inputs": state["user_inputs"],
            "image_url": image_url,
            "policy_details": policy_details,
            "news_url": news_url
        }

        # Process with router - pass the skip_routing flag and determine usecase type
        usecase_type = None
        for tool in self.tools:
            # Check for pest detection tool
            if hasattr(tool, "name") and tool.name == "PestDetectionTool" and state.get("image_url"):
                usecase_type = "pest"
                break
            # Check for policy fetcher tool
            if hasattr(tool, "name") and tool.name == "PolicyFetcherTool" and state.get("policy_details"):
                usecase_type = "policy"
                break
            # Check for news summarizer tool
            if hasattr(tool, "name") and tool.name == "NewsFetcherTool" and state.get("news_url"):
                usecase_type = "news"
                break
        
        updated_state = await self.query_router.process_state(
            internal_state, 
            skip=self.skip_routing, 
            usecase_type=usecase_type
        )
        
        # Update the LangGraph state
        state["messages"] = updated_state.get("message_history", state["messages"])
        
        print(f"CuratorNode: Query Router Completed in {time.time() - start_time:.2f} seconds")
        return state
    
    async def _task_manager_node(self, state: CuratorState) -> CuratorState:
        """
        Task manager node that executes tool calls.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with task results
        """
        print("CuratorNode: Starting Task Manager")
        start_time = time.time()
        
        # Create internal state for task manager
        internal_state = {
            "message_history": state["messages"],
            "task_results": state.get("task_results", {})
        }
        
        # Process with task manager
        updated_state = await self.task_manager.process_state(internal_state)
        
        # Update the LangGraph state
        state["messages"] = updated_state.get("message_history", state["messages"])
        state["task_results"] = updated_state.get("task_results", {})
        
        # Persist message history after task execution
        try:
            await self.message_logger._arun(
                action="store",
                conversation_id=state["conversation_id"],
                agent_type="curator",
                messages=state["messages"],
            )
        except Exception as e:
            print(f"Warning: failed to store messages after task manager: {e}")
        
        print(f"CuratorNode: Task Manager Completed in {time.time() - start_time:.2f} seconds")
        return state
    
    async def _response_formatter_node(self, state: CuratorState) -> CuratorState:
        """
        Response formatter node that formats tool results into agricultural advice.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with formatted agricultural advice
        """
        print("CuratorNode: Starting Response Formatter")
        start_time = time.time()
        
        # Create internal state for response formatter
        internal_state = {
            "message_history": state["messages"],
            "task_results": state["task_results"],
            "message_to_curator": {
                "conversation_id": state["conversation_id"],
                "query": state.get("query", "")
            }
        }
        
        # Process with response formatter
        updated_state = await self.response_formatter.process_state(internal_state)
        
        # Update the LangGraph state
        state["messages"] = updated_state.get("message_history", state["messages"])
        state["response"] = updated_state.get("message_from_curator", {})
        
        # Persist message history after formatting
        try:
            await self.message_logger._arun(
                action="store",
                conversation_id=state["conversation_id"],
                agent_type="curator",
                messages=state["messages"],
            )
        except Exception as e:
            print(f"Warning: failed to store messages after response formatter: {e}")
        
        # Upsert user inputs snapshot each run
        try:
            await UserDataLoggerTool()._arun(
                action="store",
                key=state["conversation_id"],
                data=state.get("user_inputs", {}),
            )
        except Exception as e:
            print(f"Warning: failed to store user inputs: {e}")
        
        print(f"CuratorNode: Response Formatter Completed in {time.time() - start_time:.2f} seconds")
        return state
    
    async def _final_response_node(self, state: CuratorState) -> CuratorState:
        """
        Final response node that prepares the final response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with final response
        """
        print("CuratorNode: Preparing Final Response")
        
        # Extract the last AI message to get the response
        last_ai_message = self._extract_last_ai_message(state["messages"])
        
        if last_ai_message:
            try:
                # Parse the JSON response
                response_content = last_ai_message.content.replace('```json', '').replace('```', '').strip()
                parsed_response = json.loads(response_content)
                
                # Prepare final response
                state["response"] = {
                    "user_inputs": state["user_inputs"],
                    "agent_message": parsed_response.get("agent_message", ""),
                    "CTAs": parsed_response.get("CTAs", []),
                    "tasks": parsed_response.get("tasks", "")
                }
                
                # Execute UserDataLoggerTool if present (for cases where we skip tasks)
                tool_calls = parsed_response.get("tool_calls", [])
                for tool_call in tool_calls:
                    if tool_call.get("name") == "UserDataLoggerTool":
                        try:
                            args = tool_call.get("args", {})
                            await UserDataLoggerTool()._arun(**args)
                            print("UserDataLoggerTool executed successfully in final response")
                        except Exception as e:
                            print(f"Error executing UserDataLoggerTool: {e}")
                
            except (json.JSONDecodeError, AttributeError):
                # Fallback if parsing fails
                state["response"] = {
                    "user_inputs": state["user_inputs"],
                    "agent_message": last_ai_message.content,
                    "CTAs": [],
                    "tasks": ""
                }
        
        return state
    
    def _should_execute_tasks(self, state: CuratorState) -> str:
        """
        Determine if tasks should be executed based on the router response.
        
        Args:
            state: Current state
            
        Returns:
            Decision string for conditional edge
        """
        # If routing was skipped, go directly to response formatter
        if self.skip_routing:
            return "skip_tasks"
        
        last_ai_message = self._extract_last_ai_message(state["messages"])
        
        if not last_ai_message:
            return "skip_tasks"
        
        try:
            response_content = last_ai_message.content.replace('```json', '').replace('```', '').strip()
            parsed_response = json.loads(response_content)
            
            # Check if there are tool calls that need execution
            tool_calls = parsed_response.get("tool_calls", [])
            
            if not tool_calls:
                return "skip_tasks"
            
            # Filter out simple tools that don't need task manager
            simple_tools = {"UserDataLoggerTool", "MessageHistoryLoggerTool"}
            
            needs_task_manager = any(
                tool_call.get("name") not in simple_tools 
                for tool_call in tool_calls
            )
            
            return "execute_tasks" if needs_task_manager else "skip_tasks"
            
        except (json.JSONDecodeError, AttributeError):
            return "skip_tasks"
    
    def _should_format_response(self, state: CuratorState) -> str:
        """
        Determine if response should be formatted based on task results.
        
        Args:
            state: Current state
            
        Returns:
            Decision string for conditional edge
        """
        # Check if we have task results that need formatting
        task_results = state.get("task_results", {})
        
        # If we have search results or other complex results, format them
        if task_results and any(key != "errors" for key in task_results.keys()):
            print("CuratorNode: Task results found, proceeding to formatting")
            return "format_response"
        
        print("CuratorNode: No task results, skipping formatting")
        return "skip_format"
    
    def _extract_latest_query(self, messages: List[BaseMessage]) -> str:
        """
        Extract the latest user query from messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Latest user query
        """
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message.content
        return ""
    
    def _extract_last_ai_message(self, messages: List[BaseMessage]) -> Optional[AIMessage]:
        """
        Extract the last AI message from messages.
        
        Args:
            messages: List of messages
            
        Returns:
            Last AI message or None
        """
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        return None
    
    async def __call__(self, query: str, conversation_id: str, inputs: Dict[str, Any], image_url: Optional[str] = None, policy_details: Optional[Dict[str, Any]] = None, news_url: Optional[str] = None) -> Dict:
        """
        Main execution flow for the CuratorNode using LangGraph.

        Args:
            query: User's query string
            conversation_id: Unique conversation identifier
            inputs: User inputs and metadata

        Returns:
            Dictionary containing user_inputs, agent_message, CTAs, and tasks
        """
        print("CuratorNode: Starting Execution")
        start_time = time.time()

        # Initialize state
        initial_state = CuratorState(
            messages=[
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=query)
            ],
            conversation_id=conversation_id,
            user_inputs=inputs,
            task_results=None,
            response=None,
            metadata={
                "lat": inputs.get("latitude"),
                "long": inputs.get("longitude"),
                "place": inputs.get("place")
            },
            query=query,
            image_url=image_url,
            policy_details=policy_details,
            news_url=news_url
        )

        # Execute the workflow
        try:
            print(f"CuratorNode: Executing workflow with conversation_id: {conversation_id}")
            print(f"CuratorNode: Inputs: {inputs}")
            print(f"CuratorNode: Initial state keys: {list(initial_state.keys())}")
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            print(f"CuratorNode: Final state keys: {list(final_state.keys()) if final_state else 'None'}")
            
            # Extract the final response
            result = final_state.get("response", {}) if final_state else {}
            
            # Ensure we have the required fields
            if not result:
                result = {
                    "user_inputs": final_state.get("user_inputs", {}) if final_state else {},
                    "agent_message": "",
                    "CTAs": [],
                    "tasks": ""
                }
            
            print(f"CuratorNode: Execution Completed in {time.time() - start_time:.2f} seconds")
            return result
            
        except Exception as e:
            print(f"Error in workflow execution: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            # Return a fallback response
            return {
                "user_inputs": inputs,
                "agent_message": f"Sorry, I encountered an error: {str(e)}",
                "CTAs": [],
                "tasks": ""
            }


if __name__ == "__main__":
    async def test_curator():
        """
        Test function to demonstrate the CuratorNode functionality.
        """
        print("Testing CuratorNode...")
        
        # Initialize the model
        model = ChatOpenAI(model="gpt-5-mini", temperature=0.2)
        
        # Define basic tools
        tools = [
            WebSearchTool(), 
            UserDataLoggerTool(),
            RetrievalTool(),
            PriceFetcherTool(),
            WeatherAnalysisTool()
        ]
        
        # Initialize the CuratorNode
        curator = CuratorNode(model, tools)
        
        # Test query and conversation ID
        test_query = "Hi I am a very poor farmer living in Salkia, Howrah. I am planning to grow rice, as I can't grow anything else"
        test_conversation_id = "23048e6b11964da0866d63dd"
        test_inputs = {}

        # Call the curator
        result = await curator(test_query, test_conversation_id, test_inputs)
        
        # Print the results
        print("\n=== Test Results ===")
        print(f"Query: {test_query}")
        print(f"Conversation ID: {test_conversation_id}")
        print(f"Agent Message: {result.get('agent_message', 'No message')}")
        print(f"CTAs: {result.get('CTAs', [])}")
        print(f"Tasks: {result.get('tasks', '')}")
        print(f"Tool Calls: {result.get('tool_calls','')}")
        
        return result

    # Run the test
    asyncio.run(test_curator())