"""
Fetch historical and live price data for tickers and manage holdings storage.
"""
import yfinance as yf
import json
from pathlib import Path

HOLDINGS_PATH = Path("holdings.json")

def fetch_prices(tickers, period="1y"):
    """
    Download closing prices for given tickers over the specified period.
    """
    data = yf.download(tickers, period=period, auto_adjust=True)
    return data["Close"]


def load_holdings():
    """
    Load holdings from a local JSON file.
    """
    if not HOLDINGS_PATH.exists():
        return {}
    return json.loads(HOLDINGS_PATH.read_text())


def save_holdings(holdings):
    """
    Save the holdings dict to JSON.
    """
    HOLDINGS_PATH.write_text(json.dumps(holdings, indent=2))