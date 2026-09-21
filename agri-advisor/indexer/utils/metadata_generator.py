import os
import sys
import logging
import json
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

from dotenv import load_dotenv

# Use Google Gemini API directly
import google.generativeai as genai

METADATA_SAVE_DIR = getattr(Config, "METADATA_SAVE_DIR", ".cache/metadata_list")
Path(METADATA_SAVE_DIR).mkdir(parents=True, exist_ok=True)
METADATA_LOG_FILE = os.path.join(METADATA_SAVE_DIR, "all_metadata.json")

load_dotenv()

sample_json = {
    "document_type": "farming report",
    "key_entities": ["maize", "fertilizer", "irrigation"],
    "topics": ["crop yield", "pest management", "soil health"],
    "year": 2024,
}

# Updated prompt: only 3-4 broad data, concise, and only 4 keys
prompt_template = """
You are an expert assistant for an agriculture and farming knowledge system.

Analyze the provided document and extract only the 3-4 most broad and important pieces of metadata relevant to agriculture and farming. Focus on general categories, not specifics.

<document_name>
{doc_name}
</document_name>

<document>
{doc_text}
</document>

Return the results as a JSON object with exactly these 4 fields, using lowercase for all string values:

- "document_type" (string): The broad type of document (e.g., "report", "advisory", "bulletin", "policy").
- "key_entities" (list of up to 3 strings): The most important crops, organizations, or people mentioned.
- "topics" (list of up to 3 strings): The main agricultural topics or issues discussed.
- "year" (integer): The year the document was created or is about.

All text values in the JSON output must be in lowercase.

Example output:
{sample_json}

Return only the JSON object in the specified format.
"""

class MetadataGrouper:
    def __init__(self, fuzzy_threshold=90, max_workers=None):
        self.fuzzy_threshold = fuzzy_threshold
        self.max_workers = max_workers

    def _cheap_bucket_key(self, metadata):
        key_entities = metadata.get("key_entities", [])
        first_entity = key_entities[0] if key_entities else ""
        year = metadata.get("year", "")
        return f"{first_entity[:2].lower()}_{len(metadata.get('topics', []))}_{year}"

    def _fuzzy_score(self, meta_a, meta_b):
        str_a = (meta_a.get("document_type", "") + " " +
                 " ".join(meta_a.get("key_entities", []) + meta_a.get("topics", [])))
        str_b = (meta_b.get("document_type", "") + " " +
                 " ".join(meta_b.get("key_entities", []) + meta_b.get("topics", [])))
        return fuzz.token_sort_ratio(str_a, str_b)

    def group(self, metadata_list):
        buckets = defaultdict(list)
        for md in metadata_list:
            buckets[self._cheap_bucket_key(md)].append(md)

        groups = []
        for _, bucket in buckets.items():
            visited = set()
            for i, meta in enumerate(bucket):
                if i in visited:
                    continue
                group = [meta]
                visited.add(i)
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self._fuzzy_score, meta, bucket[j]): j
                        for j in range(i + 1, len(bucket)) if j not in visited
                    }
                    for future in concurrent.futures.as_completed(futures):
                        j = futures[future]
                        if future.result() >= self.fuzzy_threshold:
                            group.append(bucket[j])
                            visited.add(j)
                groups.append(group)
        return groups

class MetadataGenerator:
    """
    A class to generate document metadata using Google Gemini API directly.
    Only generates metadata for the complete document, not for each chunk.
    Instead of caching, appends all unique generated metadata to a JSON file.
    Optionally groups metadata using MetadataGrouper if use_grouping is True.
    """
    def __init__(self, use_grouping=False, grouper_kwargs=None, model_name=None, google_api_key=None):
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY") or getattr(Config, "GOOGLE_API_KEY", None)
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable or config not set.")

        self.model_name = model_name or "gemini-2.5-flash-lite"

        genai.configure(api_key=self.google_api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

        self.seen_metadata = set()
        self.use_grouping = use_grouping
        self.grouper = None
        if self.use_grouping:
            if grouper_kwargs is None:
                grouper_kwargs = {}
            self.grouper = MetadataGrouper(**grouper_kwargs)

    def _format_prompt(self, doc_text, doc_name):
        return prompt_template.format(
            doc_text=doc_text,
            doc_name=doc_name,
            sample_json=json.dumps(sample_json, indent=4),
        )

    def _filter_and_merge_unique(self, existing, new_metadata):
        """
        Merge new_metadata into existing, ensuring only unique elements for each key.
        Only the four keys: document_type, key_entities, topics, year.
        Each value is a list of unique values for that key.
        """
        keys = ["document_type", "key_entities", "topics", "year"]
        # Initialize result dict with empty lists
        result = {k: [] for k in keys}
        # Add existing values
        if isinstance(existing, dict):
            for k in keys:
                if k in existing:
                    if isinstance(existing[k], list):
                        result[k].extend(existing[k])
                    elif existing[k] is not None:
                        result[k].append(existing[k])
        # Add new values
        for k in keys:
            v = new_metadata.get(k)
            if isinstance(v, list):
                result[k].extend(v)
            elif v is not None:
                result[k].append(v)
        # Remove duplicates and sort for consistency
        for k in ["key_entities", "topics"]:
            result[k] = sorted(list(set([str(x).lower() for x in result[k] if x])))
        # For document_type and year, keep unique and sorted
        if result["document_type"]:
            result["document_type"] = sorted(list(set([str(x).lower() for x in result["document_type"] if x])))
        if result["year"]:
            # Only keep unique years as ints
            result["year"] = sorted(list(set([int(x) for x in result["year"] if str(x).isdigit()])))
        return result

    def generate(self, doc_text, doc_name):
        content = self._format_prompt(doc_text, doc_name)
        try:
            response = self.model.generate_content(content)
            enhanced_data = response.text.strip() if hasattr(response, "text") else response.candidates[0].content.parts[0].text.strip()
            enhanced_data = enhanced_data.replace("```json", "").replace("```", "").strip()
            try:
                addn_metadata = json.loads(enhanced_data)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from model output: {enhanced_data}")
                raise e

            # Only keep the four keys, and lists for all
            keys = ["document_type", "key_entities", "topics", "year"]
            filtered_metadata = {}
            for k in keys:
                v = addn_metadata.get(k)
                if k in ["key_entities", "topics"]:
                    filtered_metadata[k] = [str(x).lower() for x in v] if isinstance(v, list) else []
                elif k == "document_type":
                    filtered_metadata[k] = [str(v).lower()] if v else []
                elif k == "year":
                    filtered_metadata[k] = [int(v)] if v is not None and str(v).isdigit() else []

            # Use a stringified version for uniqueness check
            metadata_str = json.dumps(filtered_metadata, sort_keys=True)
            if metadata_str not in self.seen_metadata:
                self.seen_metadata.add(metadata_str)
                # Save as a single JSON object with only unique elements in lists for each key
                if os.path.exists(METADATA_LOG_FILE):
                    try:
                        with open(METADATA_LOG_FILE, "r", encoding="utf-8") as f:
                            existing = json.load(f)
                        if not isinstance(existing, dict):
                            existing = {k: [] for k in keys}
                    except Exception:
                        existing = {k: [] for k in keys}
                else:
                    existing = {k: [] for k in keys}
                merged = self._filter_and_merge_unique(existing, filtered_metadata)
                with open(METADATA_LOG_FILE, "w", encoding="utf-8") as f:
                    json.dump(merged, f, ensure_ascii=False, indent=4)
                self.logger.info(f"Appended new unique metadata for {doc_name} to {METADATA_LOG_FILE}")
            else:
                self.logger.info(f"Metadata for {doc_name} already exists in this run, not appending.")

            # If grouping is enabled, group all metadata in the log file and return the group containing this metadata
            if self.use_grouping:
                try:
                    with open(METADATA_LOG_FILE, "r", encoding="utf-8") as f:
                        all_metadata = json.load(f)
                    # Grouping doesn't make much sense for this format, but return the lists for now
                    return all_metadata
                except Exception as e:
                    self.logger.error(f"Error during grouping: {str(e)}")
                    return filtered_metadata
            else:
                return filtered_metadata

        except Exception as e:
            self.logger.error(f"Error generating metadata: {str(e)}")
            return {
                "document_type": [],
                "key_entities": [],
                "topics": [],
                "year": [],
            }

if __name__ == "__main__":
    import PyPDF2

    pdf_path = r"C:\Users\subar\OneDrive\Desktop\capitalone\indexer\utils\documents\Draft_RRs_JSO_inNCONF.pdf"
    doc_name = "Subarno_Resume (2).pdf"

    # Extract text from the PDF
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        doc_text = ""
        for page in reader.pages:
            doc_text += page.extract_text() or ""

    metadata_generator = MetadataGenerator(use_grouping=True)
    metadata = metadata_generator.generate(doc_text, doc_name)
    print(json.dumps(metadata, indent=4, ensure_ascii=False))
