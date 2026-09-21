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

    # MongoDB settings
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")

if __name__=="__main__":
    config = Config()
    print("Database URL:", config.DB_URL)
    print("API Host:", config.API_HOST)
    print("API Port:", config.API_PORT)
    print("Google API Key:", config.GOOGLE_API_KEY)
    print("OpenAI API Key:", config.OPENAI_API_KEY)
    print("MongoDB URI:", config.MONGO_URI)
    print("MongoDB Name:", config.MONGO_DB_NAME)