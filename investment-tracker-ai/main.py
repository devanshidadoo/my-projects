"""
Entry point: CLI for the Personal Investment Tracker AI with chat.
"""
import argparse
from data_ingestion import fetch_prices, load_holdings, save_holdings
from metrics import total_return, allocation, volatility_contribution
from visualization import plot_portfolio_value, plot_allocation
from agent import agent, DummyAgent


def main():
    parser = argparse.ArgumentParser(description="Personal Investment Tracker AI")
    sub = parser.add_subparsers(dest="cmd")

    # update holdings
    upd = sub.add_parser("update", help="Add/update ticker holdings")
    upd.add_argument("ticker")
    upd.add_argument("qty", type=float)

    # show metrics
    sub.add_parser("metrics", help="Show portfolio metrics")

    # visualize portfolio
    sub.add_parser("visualize", help="Plot portfolio charts")

    # chat with AI agent
    chat = sub.add_parser("chat", help="Ask the AI about your portfolio")
    chat.add_argument("query", type=str)

    args = parser.parse_args()
    holdings = load_holdings()

    if args.cmd == "update":
        holdings[args.ticker] = args.qty
        save_holdings(holdings)
        print(f"Updated {args.ticker}: {args.qty}")
    elif args.cmd == "metrics":
        prices = fetch_prices(list(holdings.keys()))
        print("Total Return:", total_return(prices, holdings))
        print("Allocation:", allocation(prices, holdings))
        print("Volatility Contribution:", volatility_contribution(prices, holdings))
    elif args.cmd == "visualize":
        prices = fetch_prices(list(holdings.keys()))
        port_val = (prices * list(holdings.values())).sum(axis=1)
        plot_portfolio_value(port_val)
        plot_allocation(allocation(prices, holdings))
    elif args.cmd == "chat":
    # Intercept basic IRR queries without relying on the LLM agent
     import re
     if "IRR" in args.query.upper():
        nums = re.findall(r'-?\d+\.?\d*', args.query)
        if len(nums) >= 2:
            from agent import irr_tool
            print(irr_tool(','.join(nums)))
            return

    # Otherwise try the LLM agent with a simple fallback
        try:
         response = agent.run(args.query)
        except Exception:
            response = (
            "Chat agent unavailable or failed during execution. "
            "Please set a valid API key or try a different query."
        )
        print(response)


if __name__ == "__main__":
    main()