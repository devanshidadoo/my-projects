from pymongo import MongoClient
from datetime import datetime
import uvicorn
import sys
from typing import Any
import os
from dotenv import load_dotenv, find_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.core.config import *
from data.utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Set up environment variables
app_env = APP_ENV
if app_env == "development":
    dotenv_path = find_dotenv("./config/environments/dev.env")

if dotenv_path is not None:
    load_dotenv(dotenv_path=dotenv_path)

from data.routes import *

if __name__ == "__main__":
    logger.info(f"Starting server in {app_env} environment")
    uvicorn.run(
        "routes:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,
    )
