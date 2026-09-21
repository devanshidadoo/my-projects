from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from tavily import TavilyClient

import os
import sys

# Add the parent directory (project root) to the Python path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from agents.config import Config as config
from pydantic import PrivateAttr

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(5, description="Number of search results to return (default: 5)")

class WebSearchTool(BaseTool):
    name: str = "WebSearchTool"
    description: str = (
        "Search the web using Tavily by providing a query and number of search results. "
        "Returns a list of results with title, link, snippet, favicon, and raw content."
    )
    args_schema: Type[WebSearchInput] = WebSearchInput

    _client: TavilyClient = PrivateAttr(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_key = getattr(config, "TAVILY_API_KEY", None)
        self._client = TavilyClient(api_key)

    def _run(self, **kwargs) -> str:
        try:
            query = kwargs.get("query")
            k = kwargs.get("k", 5)
            if not k:
                k = 3

            response = self._tavily_search_sync(
                query=query,
                k=k
            )
            return str(response)
        except Exception as e:
            error_msg = str(e)
            return str([{
                "Title": "Search Error",
                "Link": "",
                "Snippet": f"Error during search: {error_msg}",
                "Success": False,
                "Error": error_msg,
                "Content": f"Error during search: {error_msg}"
            }])

    async def _arun(self, **kwargs) -> str:
        try:
            query = kwargs.get("query")
            k = kwargs.get("k", 5)
            if not k:
                k = 3

            response = await self._tavily_search_async(
                query=query,
                k=k
            )
            return response
        except Exception as e:
            error_msg = str(e)
            return [{
                "Title": "Search Error",
                "Link": "",
                "Snippet": f"Error during search: {error_msg}",
                "Success": False,
                "Error": error_msg,
                "Content": f"Error during search: {error_msg}"
            }]
        
    async def _tavily_search_async(self, query, k):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._tavily_search_sync, query, k
        )

    def _tavily_search_sync(self, query, k):
        response = self._client.search(
            query=query,
            search_depth="basic",
            include_favicon=True,
            include_raw_content="text",
            max_results=k
        )
        results = []
        for item in response.get("results", []):
            results.append({
                "Title": item.get("title", ""),
                "Link": item.get("url", ""),
                "Snippet": item.get("content", ""),
                "Favicon": item.get("favicon", ""),
                "Success": True,
                "Error": "",
                "Content": item.get("raw_content", "") or item.get("content", "")
            })
        return results

if __name__ == "__main__":
    import asyncio
    import json

    async def test_search():
        tool = WebSearchTool()
        # Example query
        query = "latest news in artificial intelligence"
        k = 3
        print(f"Testing WebSearchTool with query: '{query}', k={k}")
        results = await tool._arun(query=query, k=k)
        print("Results:")
        print(json.dumps(results, indent=2, ensure_ascii=False))

    asyncio.run(test_search())