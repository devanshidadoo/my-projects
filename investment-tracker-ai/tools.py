# tools.py

import numpy as np
import pandas as pd
from typing import Dict

def get_total_return(prices: pd.DataFrame, holdings: Dict[str, float]) -> float:
    df = prices.copy()
    for t, q in holdings.items():
        df[t] *= q
    port = df.sum(axis=1)
    return port.pct_change().add(1).cumprod().iloc[-1] - 1

def get_irr(cash_flows: list) -> float:
    """
    cash_flows: list of floats, where negative = investment, positive = returns
    """
    return np.irr(cash_flows)

def get_exposure(prices: pd.DataFrame, holdings: Dict[str, float]) -> Dict[str, float]:
    latest = prices.iloc[-1]
    vals = {t: latest[t] * q for t, q in holdings.items()}
    total = sum(vals.values())
    return {t: v/total for t, v in vals.items()}
