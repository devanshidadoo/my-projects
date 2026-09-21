import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # AWS S3 Config
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION')
    AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')

    # Qdrant Config
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

    # Model Config
    TEXT_EMBEDDING_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")
    IMAGE_EMBEDDING_MODEL = os.getenv("IMAGE_EMBEDDING_MODEL")

    COLLECTION_NAME = "new3"
    
if __name__ == "__main__":
    print(Config.AWS_ACCESS_KEY_ID)
    print(Config.AWS_SECRET_ACCESS_KEY)
    print(Config.AWS_REGION)
    print(Config.AWS_BUCKET_NAME)

    print(Config.QDRANT_API_KEY)
    print(Config.QDRANT_URL)