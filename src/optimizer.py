import logging

import numpy as np
import pandas as pd
import scipy.optimize as sco

from config import MAX_WEIGHT_PER_STOCK, N_PORTFOLIOS, RISK_FREE_RATE, TRADING_DAYS

logger = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _annualized_performance(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
) -> tuple:
    """Return (annualized_volatility, annualized_return) for a weight vector."""
    ret = float(np.sum(mean_returns * weights) * TRADING_DAYS)
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights))) * np.sqrt(TRADING_DAYS))
    return vol, ret


def _neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    vol, ret = _annualized_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol


def _neg_sortino(weights, daily_returns_matrix, mean_returns, cov_matrix, risk_free_rate):
    """Negative Sortino ratio using realized downside deviation of daily returns."""
    _, ret = _annualized_performance(weights, mean_returns, cov_matrix)
    portfolio_daily = daily_returns_matrix @ weights
    downside = portfolio_daily[portfolio_daily < 0]
    if len(downside) == 0:
        return 0.0
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS))
    if downside_dev == 0:
        return 0.0
    return -(ret - risk_free_rate) / downside_dev


def _volatility_only(weights, mean_returns, cov_matrix):
    return _annualized_performance(weights, mean_returns, cov_matrix)[0]


def _constraints_and_bounds(n: int) -> tuple:
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    bounds = tuple((0.0, MAX_WEIGHT_PER_STOCK) for _ in range(n))
    return constraints, bounds


def _to_allocation(weights: np.ndarray, tickers: pd.Index) -> pd.Series:
    """Convert raw weight array to a cleaned Series (drop near-zero positions)."""
    allocation = pd.Series(weights, index=tickers).round(6)
    return allocation[allocation > 0.001]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_portfolios(
    prices: pd.DataFrame,
    n_portfolios: int = N_PORTFOLIOS,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple:
    """Simulate n_portfolios random weight allocations.

    Returns:
        sim_df  — DataFrame with columns [volatility, annual_return, sharpe]
        weights — ndarray of shape (n_portfolios, n_assets)
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)

    all_weights = np.zeros((n_portfolios, n))
    vols = np.zeros(n_portfolios)
    rets = np.zeros(n_portfolios)

    for i in range(n_portfolios):
        w = np.random.random(n)
        w /= w.sum()
        all_weights[i] = w
        vol, ret = _annualized_performance(w, mean_returns, cov_matrix)
        vols[i] = vol
        rets[i] = ret

    sharpes = (rets - risk_free_rate) / np.where(vols > 0, vols, np.nan)
    sim_df = pd.DataFrame({"volatility": vols, "annual_return": rets, "sharpe": sharpes})

    logger.info("Generated %d random portfolios.", n_portfolios)
    return sim_df, all_weights


def optimize_max_sharpe(
    prices: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Find the maximum-Sharpe-ratio portfolio via SLSQP constrained optimization."""
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    constraints, bounds = _constraints_and_bounds(n)

    result = sco.minimize(
        _neg_sharpe,
        x0=np.full(n, 1.0 / n),
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    vol, ret = _annualized_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol

    logger.info(
        "Max Sharpe  →  return=%.2f%%  vol=%.2f%%  sharpe=%.3f",
        ret * 100, vol * 100, sharpe,
    )
    return {
        "label": "Max Sharpe",
        "weights": _to_allocation(weights, prices.columns),
        "annual_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def optimize_min_volatility(prices: pd.DataFrame) -> dict:
    """Find the minimum-variance portfolio via SLSQP constrained optimization."""
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    constraints, bounds = _constraints_and_bounds(n)

    result = sco.minimize(
        _volatility_only,
        x0=np.full(n, 1.0 / n),
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    vol, ret = _annualized_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - RISK_FREE_RATE) / vol

    logger.info(
        "Min Vol     →  return=%.2f%%  vol=%.2f%%  sharpe=%.3f",
        ret * 100, vol * 100, sharpe,
    )
    return {
        "label": "Min Volatility",
        "weights": _to_allocation(weights, prices.columns),
        "annual_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


def optimize_max_sortino(
    prices: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
) -> dict:
    """Find the maximum-Sortino-ratio portfolio via SLSQP constrained optimization.

    The Sortino ratio penalizes only downside volatility, making it preferable
    to Sharpe in asymmetric return distributions common in equity portfolios.
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    constraints, bounds = _constraints_and_bounds(n)

    result = sco.minimize(
        _neg_sortino,
        x0=np.full(n, 1.0 / n),
        args=(returns.values, mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x
    vol, ret = _annualized_performance(weights, mean_returns, cov_matrix)
    portfolio_daily = returns.values @ weights
    downside = portfolio_daily[portfolio_daily < 0]
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS)) if len(downside) > 0 else 0.0
    sortino = (ret - risk_free_rate) / downside_dev if downside_dev > 0 else 0.0

    logger.info(
        "Max Sortino →  return=%.2f%%  vol=%.2f%%  sortino=%.3f",
        ret * 100, vol * 100, sortino,
    )
    return {
        "label": "Max Sortino",
        "weights": _to_allocation(weights, prices.columns),
        "annual_return": ret,
        "volatility": vol,
        "sharpe": (ret - risk_free_rate) / vol,
        "sortino": sortino,
    }
