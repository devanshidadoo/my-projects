import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

class Config:
    # Database settings
    DB_HOST = os.getenv("DB_HOST", "")
    DB_ROUTE = os.getenv("DB_ROUTE", "")
    DB_URL = DB_HOST + "/" + DB_ROUTE

    # # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8001))

    # # Logging settings
    # LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    # LOG_FILE = os.getenv("LOG_FILE", "app.log")

    # # Other settings
    # SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    # ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")

    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # MongoDB settings
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    
    # Qdrant Config
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")
    IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL")
    QDRANT_INDEXED_FIELDS = ["document_type" ,"key_entities","topics", "year"]
    
    METADATA_FILE = os.path.join(os.path.dirname(__file__), "curator", "utils", "tools", "retriever", "all_metadata.json")
    COLLECTION_NAME = "new3"
    HF_GRADIO_EMBEDDING_SPACE="subarnoM/Qwen3-Embedding-0.6B"
    
if __name__=="__main__":
    config = Config()
    print("Database URL:", config.DB_URL)
    print("API Host:", config.API_HOST)
    print("API Port:", config.API_PORT)
    print("Google API Key:", config.GOOGLE_API_KEY)
    print("OpenAI API Key:", config.OPENAI_API_KEY)
    print("MongoDB URI:", config.MONGO_URI)
    print("MongoDB Name:", config.MONGO_DB_NAME)
