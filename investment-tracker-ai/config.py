"""
Configuration for API keys and defaults.
"""
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG"]