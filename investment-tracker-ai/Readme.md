# Personal Investment Tracker AI

> An interactive, AI-powered portfolio manager for logging, analyzing, and visualizing your investments.

---

## 🚀 Features

- **Automated Data Ingestion**  
  Fetch historical and live closing prices for your holdings via [yfinance](https://github.com/ranaroussi/yfinance).

- **Robust Metrics**  
  - **Total Return**: Cumulative portfolio gain/loss.  
  - **Asset Allocation**: Current weight of each holding.  
  - **Volatility Contribution**: Relative risk contribution per asset.

- **Beautiful Visualizations**  
  - Line chart of portfolio value over time.  
  - Pie chart of current asset allocation.

- **AI-Driven Q&A**  
  Chat naturally to get answers like “What’s my IRR?” or “Show my exposure today.”  
  Falls back to local IRR calculation if no API key is provided.

---

## 📦 Prerequisites

- **Python 3.8+**  
- (Optional) **OpenAI API Key** or **Hugging Face Hub Token** for full AI chat functionality  
- GitHub account (for version control)

---

## 🔧 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/YourUsername/my-projects.git
   cd my-projects/investment-tracker-ai
