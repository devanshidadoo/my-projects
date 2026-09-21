import time
from typing import Optional, Dict
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, PointStruct
import os
import sys
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import embedding and metadata generator
from utils.embedding_generator import EmbeddingGenerator
from utils.metadata_generator import MetadataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vector_store")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config as config
from utils.file_parser import PDFParser  

# --- Hardcoded schema for payload fields ---
# All string fields are KEYWORD, year is INTEGER
HARDCODED_PAYLOAD_SCHEMA = {
    "document_type": models.PayloadSchemaType.KEYWORD,
    "key_entities": models.PayloadSchemaType.KEYWORD,
    "topics": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "text": models.PayloadSchemaType.KEYWORD,
    "image": models.PayloadSchemaType.KEYWORD,
    "chunk_index": models.PayloadSchemaType.INTEGER,
    "chunk_count": models.PayloadSchemaType.INTEGER,
    "doc_name": models.PayloadSchemaType.KEYWORD,
}

def detect_schema_type(key, value):
    # Use hardcoded schema if available
    if key in HARDCODED_PAYLOAD_SCHEMA:
        return HARDCODED_PAYLOAD_SCHEMA[key]
    # Fallback: all strings/lists are KEYWORD, year is int
    if key == "year":
        return models.PayloadSchemaType.INTEGER
    if isinstance(value, int):
        return models.PayloadSchemaType.INTEGER
    elif isinstance(value, float):
        return models.PayloadSchemaType.FLOAT
    elif isinstance(value, str):
        return models.PayloadSchemaType.KEYWORD
    elif isinstance(value, list):
        return models.PayloadSchemaType.KEYWORD
    else:
        return None  # unsupported

class VectorStore:
    """
    VectorStore is a high-level interface for managing a Qdrant vector database collection
    that supports both text and image embeddings. It simplifies collection setup, document
    insertion (with text and/or image embeddings and optional metadata), and similarity search.
    """

    def __init__(
        self,
        collection_name: str,
        vector_sizes: Dict[str, int] = {"text_embedding": 1024, "image_embedding": 512},
        distance: Distance = Distance.COSINE
    ):
        logger.info(f"Initializing VectorStore for collection '{collection_name}'")
        self.collection_name = collection_name
        self.client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
        self.vector_sizes = vector_sizes
        self.distance = distance
        self.text_embedding_model = config.TEXT_EMBEDDING_MODEL
        self.image_embedding_model = config.IMAGE_EMBEDDING_MODEL

        self.embedding_generator = EmbeddingGenerator(model_name=self.text_embedding_model)
        self.metadata_generator = MetadataGenerator()

        self._initialize_collection()

    def _initialize_collection(self):
        logger.info(f"Checking if collection '{self.collection_name}' exists in Qdrant...")
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in existing:
            logger.info(f"Collection '{self.collection_name}' already exists.")
            return

        logger.info(f"Creating collection '{self.collection_name}' with vector sizes: {self.vector_sizes}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text_embedding": VectorParams(size=self.vector_sizes["text_embedding"], distance=self.distance),
                "image_embedding": VectorParams(size=self.vector_sizes["image_embedding"], distance=self.distance),
            }
        )
        logger.info(f"Collection '{self.collection_name}' created successfully.")

        # Create payload indexes for all fields in the hardcoded schema
        for field, schema_type in HARDCODED_PAYLOAD_SCHEMA.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type
                )
                logger.info(f"Index for '{field}' ({schema_type}) created.")
            except Exception as e:
                logger.warning(f"Index creation for '{field}' may have failed or already exists: {e}")

    def _ensure_payload_indexes(self, metadata):
        """
        Ensure all fields in the metadata are indexed in Qdrant.
        """
        if not metadata:
            return
        field_types = {}
        for k, v in metadata.items():
            schema_type = detect_schema_type(k, v)
            if schema_type and k not in field_types:
                field_types[k] = schema_type

        for field, schema_type in field_types.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type
                )
                logger.info(f"Index for '{field}' ({schema_type}) created.")
            except Exception as e:
                logger.warning(f"Index creation for '{field}' may have failed or already exists: {e}")

    def insert_document(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
        metadata: Optional[dict] = None,
        id: Optional[int] = None,
        chunk: bool = False,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        doc_name: Optional[str] = None
    ):
        """
        Insert a document into the vector store. If chunk=True and text is provided,
        the text will be split into chunks using LangChain's RecursiveCharacterTextSplitter,
        and each chunk will be inserted as a separate point with its own id.
        Metadata is generated using the MetadataGenerator and all fields are indexed.
        """
        logger.debug(f"Preparing to insert document. Text provided: {bool(text)}, Image provided: {bool(image)}")
        if not text and not image:
            logger.error("Attempted to insert document without text or image.")
            raise ValueError("At least one of `text` or `image` must be provided.")

        # Generate metadata if not provided
        if metadata is None and text:
            logger.info("Generating metadata using MetadataGenerator.")
            metadata = self.metadata_generator.generate(doc_text=text, doc_name=doc_name or "document")
            logger.info(f"Generated metadata: {metadata}")

        # Ensure all metadata fields are indexed
        self._ensure_payload_indexes(metadata)

        # If chunking is enabled and text is provided, split and insert each chunk
        if chunk and text:
            logger.info("Chunking enabled. Splitting text into chunks for insertion.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks.")

            for idx, chunk_text in enumerate(chunks):
                vectors = {}
                logger.debug("Generating text embedding for chunked document.")
                # Use embedding generator for text embedding
                text_emb = self.embedding_generator.encode([chunk_text])[0]
                vectors["text_embedding"] = text_emb
                if image:
                    logger.debug("Generating image embedding for document (applies to all chunks).")
                    # Placeholder: image embedding logic if needed
                    vectors["image_embedding"] = [0.0] * self.vector_sizes["image_embedding"]

                payload_data = metadata.copy() if metadata else {}
                payload_data["text"] = chunk_text
                if image:
                    payload_data["image"] = image
                payload_data["chunk_index"] = idx
                payload_data["chunk_count"] = len(chunks)

                point_id = (id or self._generate_id()) + idx
                logger.info(f"Inserting chunk {idx+1}/{len(chunks)} with id={point_id} into collection '{self.collection_name}'")
                point = PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload_data
                )

                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=[point]
                    )
                    logger.info(f"Chunk {idx+1}/{len(chunks)} with id={point_id} inserted successfully.")
                except Exception as e:
                    logger.error(f"Failed to insert chunk {idx+1}/{len(chunks)} with id={point_id}: {e}")
                    raise
            return 

        vectors = {}
        if text:
            logger.debug("Generating text embedding for document.")
            text_emb = self.embedding_generator.encode([text])[0]
            vectors["text_embedding"] = text_emb
        if image:
            logger.debug("Generating image embedding for document.")
            # Placeholder: image embedding logic if needed
            vectors["image_embedding"] = [0.0] * self.vector_sizes["image_embedding"]

        payload_data = metadata.copy() if metadata else {}
        if text:
            payload_data["text"] = text
        if image:
            payload_data["image"] = image

        point_id = id or self._generate_id()
        logger.info(f"Inserting document with id={point_id} into collection '{self.collection_name}'")
        point = PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload_data
        )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            logger.info(f"Document with id={point_id} inserted successfully.")
        except Exception as e:
            logger.error(f"Failed to insert document with id={point_id}: {e}")
            raise

    def _generate_id(self):
        generated_id = int(time.time() * 1000)
        logger.debug(f"Generated new document id: {generated_id}")
        return generated_id
    
    def query(
        self,
        query_text: str,
        top_k: int = 3,
        using: str = "text_embedding",
    ):
        logger.info(f"Querying collection '{self.collection_name}' for top {top_k} results using '{using}' embedding.")
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=models.Document(text=query_text, model=self.text_embedding_model),
                limit=top_k,
                using=using
            )
            logger.info(f"Query successful. Retrieved {len(results.points)} results.")
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

        return [
            {
                "id": r.id,
                "score": r.score,
                "text": r.payload.get("text"),
                "metadata": r.payload
            }
            for r in results.points
        ]