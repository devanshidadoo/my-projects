# test_visualization.py

from data_ingestion import fetch_prices, load_holdings
from metrics import allocation
from visualization import plot_portfolio_value, plot_allocation

# Load and fetch data
holdings = load_holdings()
prices = fetch_prices(list(holdings.keys()), period="6mo")

# Compute time‐series of portfolio value
#  — prices is a DataFrame: index=dates, columns=tickers
#  — holdings.values() is the list of quantities in the same order
port_vals = (prices * list(holdings.values())).sum(axis=1)

# Plot portfolio value over time
plot_portfolio_value(port_vals)

# Plot current allocation pie chart
weights = allocation(prices, holdings)
plot_allocation(weights)
