# test_metrics.py

from data_ingestion import fetch_prices, load_holdings
from metrics import total_return, allocation, volatility_contribution

# Load your sample holdings
holdings = load_holdings()

# Fetch 6 months of data
prices = fetch_prices(list(holdings.keys()), period="6mo")

# Compute and print metrics
print("Total Return:", total_return(prices, holdings))
print("\nAllocation:\n", allocation(prices, holdings))
print("\nVolatility Contribution:\n", volatility_contribution(prices, holdings))
