"""
Compute portfolio metrics: total returns, allocation, volatility contributions.
"""
import pandas as pd


def total_return(prices: pd.DataFrame, holdings: dict) -> float:
    """
    Calculate cumulative total return of the portfolio.
    """
    df = prices.copy()
    for ticker, qty in holdings.items():
        df[ticker] *= qty
    portfolio = df.sum(axis=1)
    return portfolio.pct_change().add(1).cumprod().iloc[-1] - 1


def allocation(prices: pd.DataFrame, holdings: dict) -> pd.Series:
    """
    Compute current weight of each asset in portfolio.
    """
    latest = prices.iloc[-1]
    values = {t: latest[t] * qty for t, qty in holdings.items()}
    total = sum(values.values())
    return pd.Series({t: v / total for t, v in values.items()})


def volatility_contribution(prices: pd.DataFrame, holdings: dict, window: int = 30) -> pd.Series:
    """
    Rolling volatility contribution of each asset.
    """
    returns = prices.pct_change().dropna()
    w = allocation(prices, holdings)
    vol = returns.rolling(window).std().iloc[-1]
    contr = w * vol
    return contr / contr.sum()
