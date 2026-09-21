import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))

from utils.file_parser import PDFParser
from utils.vector_store import VectorStore
from utils.metadata_generator import MetadataGenerator
from utils.embedding_generator import EmbeddingGenerator
from config import Config
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("indexer")

def index_documents(documents_dir, collection_name=Config.COLLECTION_NAME):
    store = VectorStore(collection_name)
    metadata_gen = MetadataGenerator()
    for filename in os.listdir(documents_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(documents_dir, filename)
            logger.info(f"Processing file: {filename}")
            try:
                parser = PDFParser(pdf_path)
                extracted_text = parser.extract_text()
                logger.debug(f"Extracted text from {filename}: {extracted_text[:100]}..." if extracted_text else "No text extracted.")

                generated_metadata = metadata_gen.generate(extracted_text, filename)
                logger.info(f"Generated metadata for {filename}: {generated_metadata}")

                payload = {"source": filename}
                if isinstance(generated_metadata, dict):
                    payload.update(generated_metadata)
                else:
                    payload["generated_metadata"] = generated_metadata

                store.insert_document(
                    text=extracted_text,
                    metadata=payload,
                    chunk=True,
                    chunk_size=1000,
                    chunk_overlap=200
                )
                logger.info(f"Inserted document from {filename} (with chunking and metadata)")
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                logger.error(f"Please ensure the PDF file '{filename}' exists in the 'documents' folder.")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
    return store  # <-- Ensure the store is returned for later use

if __name__ == "__main__":
    documents_dir = os.path.join(os.path.dirname(__file__), "utils", "documents")
    logger.info(f"Looking for PDF documents in directory: {documents_dir}")
    
    try:
        store = index_documents(documents_dir)
        logger.info("Indexing complete.")

        # Defensive: Check if store is not None before proceeding
        if store is None:
            logger.critical("VectorStore was not initialized properly. Exiting.")
            sys.exit(1)

        # ---------------- Filtered Query ----------------
        logger.info("Running filtered query...")
        filter_conditions = Filter(
            should=[
                FieldCondition(
                    key="year",  
                    range=Range(gte=2020)
                ),
                FieldCondition(
                    key="key_entities", 
                    match=MatchValue(value="government of india")
                )
            ]
        )

        # Defensive: Check if embedding_generator exists
        if not hasattr(store, "embedding_generator") or store.embedding_generator is None:
            logger.critical("VectorStore does not have a valid embedding_generator. Exiting.")
            sys.exit(1)

        query_vector = store.embedding_generator.encode(
            ["how to become a agri scientist"]
        )[0]

        results = store.client.query_points(
            collection_name=store.collection_name,
            query=query_vector,
            query_filter=filter_conditions,
            limit=3,
            using="text_embedding"
        ).points

        output = [
            {
                "id": r.id,
                "score": r.score,
                "text": r.payload.get("text"),
                "metadata": r.payload
            }
            for r in results
        ]

        logger.info(f"Filtered query results: {output}")
        print(output)

    except Exception as e:
        logger.critical(f"Fatal error: {e}")