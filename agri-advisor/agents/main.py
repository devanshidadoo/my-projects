from pymongo import MongoClient
from datetime import datetime
import nest_asyncio
import threading
import uvicorn
import requests
import json
import signal
import sys
import atexit
from typing import Any
import os

from routes import *

if __name__ == "__main__":
    uvicorn.run(
        "routes:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False,
    )