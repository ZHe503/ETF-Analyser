import pandas as pd
import numpy as np

def test_calculate_returns():
    prices = pd.Series([100, 102, 104])
    returns = prices.pct_change().dropna()
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], 0.02)

def test_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.015])
    mean_return = np.mean(returns) * 252
    vol = np.std(returns) * np.sqrt(252)
    sharpe = (mean_return - 0.01) / vol
    assert sharpe != 0
