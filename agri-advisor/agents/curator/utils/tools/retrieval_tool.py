import logging
import sys
import os
from typing import Optional, List, Dict, Any
from langchain.tools import BaseTool, tool
from pydantic import BaseModel, Field, ConfigDict

# Ensure config import works regardless of working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
from agents.config import Config

from gradio_client import Client as GradioClient
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .retriever.metadata_selector import LLMMetadataSubsetSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RetrievalInput(BaseModel):
    """Input schema for the retrieval tool."""
    query: str = Field(..., description="The search query to find relevant documents")
    max_metadata: int = Field(5, description="Maximum number of metadata fields to request from the selector")
    limit: int = Field(5, description="Number of results to return")
    use_metadata_filter: bool = Field(True, description="Whether to use metadata filtering (True) or not (False)")
    model_config = ConfigDict(extra="allow")

class RetrievalTool(BaseTool):
    name: str = "search_qdrant_with_metadata"
    description: str = (
        "Search documents in Qdrant using a vector embedding, optionally applying an LLM-selected metadata filter.\n"
        "This tool can search through indexed documents and return relevant results with metadata.\n"
        "Supports both vector similarity search and metadata-based filtering for more precise results."
    )
    args_schema: type[BaseModel] = RetrievalInput
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def _get_embedding_from_hf(self, text: str) -> List[float]:
        """
        Get embedding for the input text using the HuggingFace Gradio API.
        """
        logger.info(f"Getting embedding for text via GradioClient: {text!r}")
        client = GradioClient(getattr(Config, "HF_GRADIO_EMBEDDING_SPACE"))
        result = client.predict(
            texts=[text],
            api_name="/predict"
        )

        if not isinstance(result, dict) or "vectors" not in result:
            raise ValueError(f"Unexpected embedding format: {result}")

        vectors = result["vectors"]
        if not vectors or not isinstance(vectors[0], list):
            raise ValueError(f"Unexpected vectors format: {vectors}")

        embedding = vectors[0]
        if not all(isinstance(x, (float, int)) for x in embedding):
            raise ValueError("Invalid embedding format — expected list of floats.")

        return embedding

    def _build_qdrant_filter_from_metadata(self, metadata_subset: List[Dict[str, Any]]) -> Optional[Filter]:
        """Build Qdrant filter from metadata subset."""
        if not metadata_subset:
            return None

        INDEXED_FIELDS = getattr(Config, "QDRANT_INDEXED_FIELDS", ["year"])
        conditions = []

        for entry in metadata_subset:
            for key, value in entry.items():
                if key not in INDEXED_FIELDS:
                    logger.debug(f"Skipping metadata key '{key}' — not indexed.")
                    continue
                if isinstance(value, (str, int, float, bool)):
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], (str, int, float, bool)):
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value[0])))
                else:
                    logger.debug(f"Skipping key '{key}' — unsupported type for filtering.")

        logger.info(f"Built OR filter with {len(conditions)} conditions")
        return Filter(should=conditions) if conditions else None

    def _run(
        self,
        query: str,
        max_metadata: int = 5,
        limit: int = 5,
        use_metadata_filter: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Search documents in Qdrant using a vector embedding, optionally applying an LLM-selected metadata filter.

        Args:
            query (str): The search query.
            max_metadata (int): Maximum number of metadata fields to request from the selector.
            limit (int): Number of results to return.
            use_metadata_filter (bool): Whether to use metadata filtering (True) or not (False).

        Returns:
            List[Dict[str, Any]]: Search results with 'id', 'score', 'text', and 'metadata'.
        """
        logger.info(f"Starting search for query: {query!r}")

        # Init Qdrant client
        client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)

        # Decide whether to use metadata filtering
        qdrant_filter = None
        if use_metadata_filter:
            selector = LLMMetadataSubsetSelector()
            relevant_subset = selector.select_relevant_metadata(query, max_metadata=max_metadata)
            logger.info(f"Relevant metadata subset: {relevant_subset}")
            qdrant_filter = self._build_qdrant_filter_from_metadata(relevant_subset)
        else:
            logger.info("Skipping metadata filtering — retrieving purely by vector similarity.")

        # Get embedding
        query_vector = self._get_embedding_from_hf(query)
       
        # Perform search
        try:
            results = client.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit,
                using="text_embedding"
            ).points
        except Exception as e:
            logger.error(f"Qdrant query failed: {e}")
            raise

        # Format results as list of dicts
        output = []
        for r in results:
            result_dict = {
                "id": r.id,
                "score": r.score,
                "text": r.payload.get("text"),
                "metadata": r.payload
            }
            logger.info(result_dict)
            output.append(result_dict)

        return output

    async def arun(
        self,
        query: str,
        max_metadata: int = 5,
        limit: int = 5,
        use_metadata_filter: bool = True,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Async version of the run method for the retrieval tool."""
        # For now, just call the synchronous version
        # In the future, this could be made truly async if needed
        return self._run(
            query=query,
            max_metadata=max_metadata,
            limit=limit,
            use_metadata_filter=use_metadata_filter,
            **kwargs
        )

# # Legacy function for backward compatibility
# @tool("search_qdrant_with_metadata", return_direct=False)
# def search_with_metadata_tool(query: str, max_metadata: int = 5, limit: int = 5, use_metadata_filter: bool = True) -> List[dict]:
#     """
#     Legacy function for backward compatibility.
#     Use RetrievalTool class instead.
#     """
#     tool = RetrievalTool()
#     return tool._run(
#         query=query,
#         max_metadata=max_metadata,
#         limit=limit,
#         use_metadata_filter=use_metadata_filter
#     )

if __name__ == "__main__":
    # Test RetrievalInput schema
    print("Testing RetrievalInput schema creation:")
    try:
        ri = RetrievalInput(query="how to become agri scientist")
        print("RetrievalInput instance:", ri)
        ri2 = RetrievalInput(query="agriculture techniques", max_metadata=3, limit=10)
        print("RetrievalInput with custom parameters:", ri2)
    except Exception as e:
        print("Error creating RetrievalInput:", e)
    
    print("\n" + "="*60)
    print("RETRIEVAL TOOL DEMONSTRATION")
    print("="*60)
    
    # Create retrieval tool instance
    retrieval_tool = RetrievalTool()
    
    # Demo queries
    demo_queries = [
        "how to become agri scientist",
        "agriculture techniques",
        "crop management",
        "soil health improvement"
    ]
    
    print("\nRunning demo queries:")
    print("-" * 40)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        try:
            result = retrieval_tool._run(query=query, limit=3)
            print(f"Results found: {len(result)}")
            for j, res in enumerate(result, 1):
                print(f"  {j}. Score: {res['score']:.4f}, ID: {res['id']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 40)
