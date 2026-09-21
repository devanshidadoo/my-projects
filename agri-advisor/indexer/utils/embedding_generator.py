
import logging
from sentence_transformers import SentenceTransformer

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("embedding_generator")


class EmbeddingGenerator:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def encode(self, texts, prompt_name=None):
        """
        Generate embeddings for a list of texts.
        Args:
            texts (list of str): The texts to embed.
            prompt_name (str, optional): If using a model that supports prompt_name (e.g., "query"), pass it here.
        Returns:
            np.ndarray: Embeddings for the input texts.
        """
        logger.info(f"Encoding {len(texts)} texts...")
        if prompt_name is not None:
            embeddings = self.model.encode(texts, prompt_name=prompt_name)
        else:
            embeddings = self.model.encode(texts)
        logger.info("Encoding complete.")
        return embeddings

# Example usage:
if __name__ == "__main__":
    generator = EmbeddingGenerator()
    queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    query_embeddings = generator.encode(queries, prompt_name="query")
    print(query_embeddings)
