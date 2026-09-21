import os
import sys
import json
import logging
from typing import List, Optional, Dict, Any

from langchain_openai import ChatOpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from agents.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_metadata_file() -> Dict[str, List[Any]]:
    """
    Loads the metadata file specified in Config and returns its content as a dict of lists.
    The file is expected to be a JSON object with keys like "document_type", "key_entities", etc.
    """
    metadata_path = getattr(Config, "METADATA_FILE", None)
    if not metadata_path or not os.path.exists(metadata_path):
        logger.error(f"Metadata file not found at {metadata_path}")
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.error("Metadata file does not contain a dict at the top level.")
            return {}
        return data
    except Exception as e:
        logger.error(f"Error loading metadata file: {str(e)}")
        return {}

class LLMMetadataSubsetSelector:
    """
    Uses an LLM to select the most relevant subset of metadata values from the metadata file.
    Returns a list of dicts, each dict containing a single key-value pair, e.g. [{"year": 2021}, {"topic": "agriculture"}]
    """

    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.openai_api_key = openai_api_key or getattr(Config, "OPENAI_API_KEY", None)
        self.model = model
        self._llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model=self.model,
            temperature=0.0,
            max_tokens=1000,
        )
        self._all_metadata = load_metadata_file()

    def select_relevant_metadata(
        self,
        user_query: str,
        max_metadata: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Uses the LLM to select the most relevant metadata key-value pairs for the user query.

        Args:
            user_query (str): The user's query or information need.
            max_metadata (int): Maximum number of metadata key-value pairs to return.

        Returns:
            List[Dict[str, Any]]: The LLM-selected relevant metadata as a list of dicts.
        """
        all_metadata = self._all_metadata
        if not all_metadata:
            logger.error("No metadata loaded.")
            return []

        # To avoid sending the entire metadata file, do a simple keyword match to prefilter values
        def filter_candidates(field_values, query):
            query_lower = query.lower()
            if not isinstance(field_values, list):
                return []
            # For years, match if any year is mentioned in the query
            if all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in field_values):
                years_in_query = [int(word) for word in query.split() if word.isdigit()]
                return [v for v in field_values if int(v) in years_in_query] if years_in_query else []
            # For other fields, match if query substring is present
            return [v for v in field_values if query_lower in str(v).lower()]

        candidate_metadata = {}
        for key, values in all_metadata.items():
            filtered = filter_candidates(values, user_query)
            if filtered:
                candidate_metadata[key] = filtered
        # If nothing matched, just take the first N from each field
        if not candidate_metadata:
            for key, values in all_metadata.items():
                if isinstance(values, list):
                    candidate_metadata[key] = values[:min(10, len(values))]
        # Limit the number of candidates sent to the LLM per field
        for key in candidate_metadata:
            candidate_metadata[key] = candidate_metadata[key][:30]

        prompt = f"""
        You are an expert assistant for an agriculture and farming knowledge system.

        Given the following user query and a metadata dictionary (with keys like "document_type", "key_entities", "topics", "year", each mapping to a list of possible values), select the most relevant metadata key-value pairs for answering the query.

        <user_query>
        {user_query}
        </user_query>

        <all_metadata>
        {json.dumps(candidate_metadata, ensure_ascii=False, indent=2)}
        </all_metadata>

        **Instructions:**
        - Carefully read the user query and the metadata values.
        - For each field, select the most relevant values (at most {max_metadata} in total, across all fields).
        - Return the output as a JSON array of objects, where each object contains a single key-value pair, e.g. [{{"year": 2021}}, {{"topic": "agriculture"}}].
        - Do not include any explanation or extra text.
        - If none are relevant, return an empty list.
        - Preserve the original value format.

        Example output:
        [
            {{"year": 2024}},
            {{"topics": "agricultural research"}},
            {{"document_type": "annual report"}}
        ]
        """
        response = self._llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip()
        # Remove code block markers if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        try:
            subset = json.loads(content)
            if not isinstance(subset, list):
                logger.error("LLM did not return a list of metadata key-value dicts.")
                return []
            # Ensure each item is a dict with a single key-value pair
            cleaned = []
            for item in subset:
                if isinstance(item, dict) and len(item) == 1:
                    cleaned.append(item)
                elif isinstance(item, dict) and len(item) > 1:
                    # If LLM returns a dict with multiple keys, split into single-key dicts
                    for k, v in item.items():
                        cleaned.append({k: v})
                else:
                    logger.warning(f"Skipping invalid metadata item: {item}")
            return cleaned
        except Exception as e:
            logger.error(f"Error parsing LLM metadata subset response: {str(e)}")
            logger.error(f"Raw LLM response: {content}")
            return []

if __name__ == "__main__":
    # Example usage
    user_query = "how to become agri scientist"
    selector = LLMMetadataSubsetSelector()
    relevant_subset = selector.select_relevant_metadata(user_query, max_metadata=5)
    print("Relevant metadata subset:", relevant_subset)