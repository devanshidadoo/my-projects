import asyncio
import json
import time
import uuid
from typing import Dict, List, Tuple, Any, Optional
from langchain_core.messages import ToolMessage, AIMessage

class TaskManager:
    """
    TaskManager handles the execution of tool calls for an agent system.
    It processes tool calls from router responses and manages the execution
    of these tools with proper concurrency control.
    """
    def __init__(self, tools: List[Any]):
        """
        Initialize the TaskManager with a list of available tools.
        
        Args:
            tools: List of tool objects that can be executed
        """
        self.tools = tools
        self.tool_instances = {getattr(tool, 'name', tool.__class__.__name__): tool for tool in tools}
        
    async def process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the agent state to execute any pending tool calls.
        
        Args:
            state: Current state of the agent
            
        Returns:
            Updated agent state with executed tool results
        """
        print("TaskManager: Starting Task Processing")
        start_time = time.time()
        
        # Extract message history from state
        message_history = state.get("message_history", [])
        if not message_history:
            return state
            
        # Find the last AI message
        last_ai_message = next((msg for msg in reversed(message_history)
                            if isinstance(msg, AIMessage)), None)
        
        if not last_ai_message:
            return state
            
        # Parse the message content
        try:
            message_content = last_ai_message.content.replace('```json','').replace('```','').strip()
            last_ai_message_parsed = json.loads(message_content)
        except (json.JSONDecodeError, AttributeError):
            print("TaskManager: Failed to parse last AI message")
            return state
            
        # Check for tool calls
        if not last_ai_message_parsed or not last_ai_message_parsed.get("tool_calls", False):
            return state
            
        # Extract and format action plan from tool calls
        action_plan = last_ai_message_parsed.get("tool_calls", [])
        formatted_action_plan = self._format_action_plan(action_plan)
        
        if not formatted_action_plan:
            print("TaskManager: No valid tool calls found")
            return state
            
        # Execute the tasks
        task_history = state.get("task_history", [])
        updated_task_history, results = await self.execute_tasks(
            formatted_action_plan,
            task_history
        )
        
        # Update state with results
        state["task_history"] = updated_task_history
        state["task_results"] = results
        
        # Add tool message responses to the message history
        state["message_history"] = self._update_message_history(
            message_history, 
            action_plan, 
            updated_task_history
        )
        
        print(f"TaskManager: Task Processing Completed in {time.time() - start_time:.2f} seconds")
        
        return state
        
    def _format_action_plan(self, action_plan: List[Dict]) -> List[Dict]:
        """
        Format the action plan to ensure tool calls match our registry format.
        
        Args:
            action_plan: List of tool calls from the router
            
        Returns:
            Formatted action plan with normalized structure
        """
        formatted_plan = []
        
        for tool_call in action_plan:
            # Extract key information making sure we have proper field names
            tool_name = tool_call.get("name", None)
            if not tool_name and "function" in tool_call:
                # Try to get name from function field if direct name not available
                tool_name = tool_call.get("function", {}).get("name", None)
                
            # Make sure we have a valid tool name that matches our registry
            if tool_name in self.tool_instances:
                # Get args from either direct args or function arguments
                args = tool_call.get("args", {})
                if tool_call.get("id", None) is None:
                    tool_call['id'] = str(uuid.uuid4())[:16]
                if not args and "function" in tool_call:
                    args_str = tool_call.get("function", {}).get("arguments", "{}")
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    
                formatted_plan.append({
                    "id": tool_call.get("id", f"call_{str(uuid.uuid4())[:16]}"),
                    "name": tool_name,
                    "args": args,
                    "type": "tool_call"
                })
                
        return formatted_plan
        
    def _update_message_history(self, 
                               message_history: List, 
                               action_plan: List[Dict], 
                               task_history: List[Dict]) -> List:
        """
        Update message history with tool execution results.
        
        Args:
            message_history: Current message history
            action_plan: Original action plan with tool calls
            task_history: Updated task history with results
            
        Returns:
            Updated message history with tool responses
        """
        updated_history = message_history.copy()
        
        for tool_call in action_plan:
            tool_id = tool_call.get("id")
            if tool_id is None:
                tool_id = str(uuid.uuid4())[:16]
                tool_call["id"] = tool_id
                
            tool_name = tool_call.get("name")
            if not tool_name and "function" in tool_call:
                tool_name = tool_call.get("function", {}).get("name")
                
            # Find the corresponding result for this tool call
            tool_result = None
            for result in task_history:
                if result.get("tool_id") == tool_id:
                    tool_result = result.get("result", result.get("error", "No result found"))
                    break
                    
            if tool_result:
                # Add ToolMessage to message history
                updated_history.append(
                    AIMessage(
                        content=str(tool_result) if not isinstance(tool_result, str) else tool_result,
                        # tool_call_id=tool_id,
                        name=tool_name
                    )
                )
                
        return updated_history
    
    async def _execute_single_tool(self, tool_call: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """
        Execute a single tool call with semaphore control.
        
        Args:
            tool_call: The tool call to execute
            semaphore: Semaphore for controlling concurrency
            
        Returns:
            Dictionary containing the tool execution result or error
        """
        async with semaphore:
            try:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                args = tool_call.get("args", {})
                
                print(f"Executing tool: {tool_name}, ID: {tool_id}, Args: {args}")
                
                if tool_name in self.tool_instances:
                    try:
                        # Get the tool instance
                        tool = self.tool_instances[tool_name]
                        
                        # Execute the tool with proper arguments
                        tool_result = await tool._arun(**args)
                        
                        return {
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "result": tool_result
                        }
                    except Exception as e:
                        print(f"Error executing tool {tool_name}: {e}")
                        return {
                            "tool_id": tool_id,
                            "tool_name": tool_name,
                            "error": str(e)
                        }
                else:
                    print(f"Tool {tool_name} not found in registry")
                    return {
                        "tool_id": tool_id,
                        "tool_name": tool_name,
                        "error": f"Tool '{tool_name}' not found in registry"
                    }
            except Exception as e:
                print(f"General error: {e}")
                return {
                    "tool_id": tool_call.get("id", "unknown"),
                    "error": f"Execution error: {str(e)}"
                }
    
    async def execute_tasks(self, 
                           action_plan: List[Dict], 
                           task_history: Optional[List[Dict]] = None) -> Tuple[List[Dict], Dict]:
        """
        Execute tasks from the action plan with proper parallelization.
        
        Args:
            action_plan: List of tool calls to execute
            task_history: Previous task execution history
            
        Returns:
            Tuple containing:
            - Updated task history
            - Consolidated results
        """
        start = time.time()
        print("TaskManager: Starting Task Execution")
        print(f"TaskManager: Action Plan: {action_plan}")
        
        updated_task_history = task_history.copy() if task_history else []
        consolidated_results = {}
        
        # Set maximum concurrent tasks
        max_concurrent_tasks = 10
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Create tasks with semaphore control
        tool_futures = [self._execute_single_tool(tool_call, semaphore) for tool_call in action_plan]
        
        # Execute all tasks with proper concurrency control
        completed_tools = await asyncio.gather(*tool_futures)
        
        # Add tool results to task history
        for tool_result in completed_tools:
            updated_task_history.append(tool_result)
            
        # Organize results by tool type
        for tool_result in completed_tools:
            tool_name = tool_result.get("tool_name", "unknown")
            
            if tool_name not in consolidated_results:
                consolidated_results[tool_name] = []
                
            consolidated_results[tool_name].append(tool_result)
            
            # Track errors separately
            if "error" in tool_result:
                if "errors" not in consolidated_results:
                    consolidated_results["errors"] = []
                consolidated_results["errors"].append(tool_result)
                
        print(f"TaskManager: Task Execution Completed in {time.time() - start:.2f} seconds")
        
        return updated_task_history, consolidated_results
    

    
