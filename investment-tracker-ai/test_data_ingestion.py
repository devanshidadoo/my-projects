from data_ingestion import fetch_prices, load_holdings

tickers = list(load_holdings().keys())
print(fetch_prices(tickers).tail())
