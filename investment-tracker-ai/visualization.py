"""
Plot portfolio metrics and dashboards using matplotlib.
"""
import matplotlib.pyplot as plt


def plot_portfolio_value(portfolio_series):
    plt.figure()
    plt.plot(portfolio_series.index, portfolio_series.values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


def plot_allocation(weights):
    plt.figure()
    weights.plot(kind="pie", autopct="%.1f%%")
    plt.title("Asset Allocation")
    plt.ylabel("")  # hide y-label
    plt.tight_layout()
    plt.show()