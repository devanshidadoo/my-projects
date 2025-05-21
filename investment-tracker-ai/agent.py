"""
LLM agent setup with financial tools for interactive queries.
Fallbacks: OpenAI or HuggingFaceHub, with error-handling; DummyAgent if both fail.
"""
import os
from langchain.tools import tool
from data_ingestion import fetch_prices, load_holdings
from metrics import total_return, allocation, volatility_contribution
import numpy as np
import numpy_financial as npf
from langchain_community.vectorstores import FAISS

from config import OPENAI_API_KEY

# Dummy fallback agent
class DummyAgent:
    """Fallback agent that notifies missing or invalid API key."""
    def run(self, query: str) -> str:
        return (
            "Chat agent unavailable or failed to initialize. "
            "Ensure OPENAI_API_KEY or HUGGINGFACEHUB_API_TOKEN is set and valid."
        )

# Financial tool functions
@tool
def total_return_tool(period: str = "1y") -> str:
     """Calculate total portfolio return over the specified period."""
     holdings = load_holdings()
     prices = fetch_prices(list(holdings.keys()), period=period)
     ret = total_return(prices, holdings)
     return f"Total return over {period}: {ret:.2%}"

@tool
def irr_tool(cash_flows: str) -> str:
    """Compute the internal rate of return from a comma-separated cash flow string."""
    flows = [float(x) for x in cash_flows.split(",")]
    # Use numpy_financial for IRR calculation
    rate = npf.irr(flows)
    return f"IRR: {rate:.2%}"

@tool
def exposure_tool(period: str = "1d") -> str:
     """Calculate current portfolio exposure (asset weightings) over the specified period."""
     holdings = load_holdings()
     prices = fetch_prices(list(holdings.keys()), period=period)
     latest = prices.iloc[-1]
     vals = {t: latest[t] * q for t, q in holdings.items()}
     total = sum(vals.values())
     exp = {t: v / total for t, v in vals.items()}
     return "Exposure: " + ", ".join(f"{t}: {w:.1%}" for t, w in exp.items())

# Initialize agent
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if OPENAI_API_KEY:
    # OpenAI branch
    try:
        from langchain_community.llms import OpenAI
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain.agents import initialize_agent, AgentType

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vector_store = FAISS.from_texts([], embeddings)
        agent = initialize_agent(
            [total_return_tool, irr_tool, exposure_tool],
            OpenAI(openai_api_key=OPENAI_API_KEY),
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    except Exception as e:
        print(f"Warning: OpenAI agent failed ({e}), falling back to DummyAgent.")
        agent = DummyAgent()
elif HF_TOKEN:
    # HuggingFaceEndpoint branch
    try:
        from langchain_huggingface import HuggingFaceEndpoint
        from langchain.agents import initialize_agent, AgentType

        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-small",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.7,
        )
        # Initialize agent without vector store (no static docs)
        agent = initialize_agent(
            [total_return_tool, irr_tool, exposure_tool],
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
    except Exception as e:
        print(f"Warning: HF agent failed ({e}), falling back to DummyAgent.")
        agent = DummyAgent()
else:
    # No valid API keys
    agent = DummyAgent()