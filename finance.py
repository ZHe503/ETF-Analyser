import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Utility Functions ---

def fetch_data(ticker: str, period: str = "1y"):
    data = yf.download(ticker, period=period, auto_adjust=False)

    # Prefer "Adj Close", but fallback to "Close" if missing
    if "Adj Close" in data.columns:
        return data["Adj Close"].dropna()
    elif "Close" in data.columns:
        return data["Close"].dropna()
    else:
        raise KeyError(f"No 'Adj Close' or 'Close' data available for {ticker}")


def calculate_returns(prices: pd.Series):
    return prices.pct_change().dropna()

def annualized_return(returns: pd.Series):
    return np.mean(returns) * 252

def annualized_volatility(returns: pd.Series):
    return np.std(returns) * np.sqrt(252)

def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.01):
    if returns.empty:
        return 0.0  # no data, no sharpe ratio

    excess = float(annualized_return(returns) - risk_free_rate)
    vol = float(annualized_volatility(returns))

    return excess / vol if vol != 0 else 0.0


# --- Streamlit UI ---

st.title("📈 ETF Risk/Return Analyzer")

tickers = st.multiselect(
    "Select ETFs/Stocks",
    ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    ["SPY", "QQQ"]
)

if tickers:
    results = []
    for t in tickers:
        prices = fetch_data(t)
        returns = calculate_returns(prices)
        ret = annualized_return(returns)
        vol = annualized_volatility(returns)
        sr = sharpe_ratio(returns)
        results.append((t, ret, vol, sr))

    st.subheader("Risk vs Return Scatter")
    fig, ax = plt.subplots()
    for t, ret, vol, sr in results:
        ax.scatter(vol, ret, label=f"{t} (SR={sr:.2f})")
        ax.annotate(t, (vol, ret))
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Annualized Return")
    ax.legend()
    st.pyplot(fig)

    # Optional: Show Data Table
    st.subheader("Summary Table")
    summary_df = pd.DataFrame(
        results, columns=["Ticker", "Annualized Return", "Volatility", "Sharpe Ratio"]
    ).set_index("Ticker")
    st.dataframe(summary_df.round(3))

# --- Basic Unit Tests ---

def test_calculate_returns():
    prices = pd.Series([100, 102, 104])
    returns = calculate_returns(prices)
    assert len(returns) == 2
    assert round(returns.iloc[0], 2) == 0.02

def test_annualized_metrics():
    prices = pd.Series([100, 101, 102, 103])
    returns = calculate_returns(prices)
    assert annualized_return(returns) != 0
    assert annualized_volatility(returns) >= 0

if __name__ == "__main__":
    # Run basic tests if launched as a script
    test_calculate_returns()
    test_annualized_metrics()
    print("All tests passed.")
