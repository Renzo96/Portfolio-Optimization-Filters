import logging

import numpy as np
import pandas as pd
import yfinance as yf

from config import RISK_FREE_RATE, TRADING_DAYS

logger = logging.getLogger(__name__)


# ── Building blocks ───────────────────────────────────────────────────────────

def compute_portfolio_returns(weights: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Compute the daily return series of a portfolio given its weights.

    Args:
        weights: Series indexed by ticker symbols.
        prices:  DataFrame of adjusted close prices (rows = dates, cols = tickers).

    Returns:
        Daily portfolio return series aligned to the price history.
    """
    aligned = prices[weights.index]
    daily_returns = aligned.pct_change().dropna()
    return daily_returns.dot(weights.values)


def _nav(daily_returns: pd.Series) -> pd.Series:
    """Compute cumulative NAV starting at 1.0."""
    return (1 + daily_returns).cumprod()


# ── Individual metrics ────────────────────────────────────────────────────────

def annualized_return(daily_returns: pd.Series) -> float:
    """CAGR over the full history of daily_returns."""
    total = float((1 + daily_returns).prod())
    n_years = len(daily_returns) / TRADING_DAYS
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    nav = _nav(daily_returns)
    peak = nav.cummax()
    return float(((nav - peak) / peak).min())


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualized Sortino ratio using downside deviation as the risk measure."""
    ann_ret = annualized_return(daily_returns)
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return float("nan")
    downside_dev = float(np.sqrt((downside ** 2).mean()) * np.sqrt(TRADING_DAYS))
    return (ann_ret - risk_free_rate) / downside_dev if downside_dev > 0 else float("nan")


def var_historical(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk at the given confidence level.

    Returns the loss threshold (negative number) not exceeded with 'confidence'
    probability over a single trading day.
    """
    return float(np.percentile(daily_returns, (1 - confidence) * 100))


def cvar_historical(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall (CVaR) — mean return in the tail beyond VaR.

    More conservative than VaR; accounts for the magnitude of tail losses.
    """
    var = var_historical(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float("nan")


def beta_vs_benchmark(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """OLS beta of the portfolio relative to a benchmark."""
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        logger.warning("Only %d overlapping observations for beta — result may be unreliable.", len(aligned))
    cov_mat = np.cov(aligned.iloc[:, 0].values, aligned.iloc[:, 1].values)
    bench_var = cov_mat[1, 1]
    return float(cov_mat[0, 1] / bench_var) if bench_var != 0 else float("nan")


def calmar_ratio(daily_returns: pd.Series) -> float:
    """Calmar ratio: annualized return divided by absolute maximum drawdown."""
    ann_ret = annualized_return(daily_returns)
    mdd = max_drawdown(daily_returns)
    return ann_ret / abs(mdd) if mdd != 0 else float("nan")


# ── Aggregate report ──────────────────────────────────────────────────────────

def full_risk_report(
    portfolio: dict,
    prices: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_prices: pd.Series = None,
) -> dict:
    """Compute all risk metrics for a portfolio in a single call.

    Args:
        portfolio:        Output dict from any optimizer function (must have 'weights').
        prices:           Adjusted close prices for the screened universe.
        benchmark_ticker: Ticker to use as benchmark (default: SPY).
        risk_free_rate:   Annualized risk-free rate for ratio computations.
        benchmark_prices: Pre-downloaded benchmark prices (optional, avoids re-download).

    Returns:
        Dict with all metrics plus '_daily_returns' and '_benchmark_returns'
        series for downstream chart generation.
    """
    weights = portfolio["weights"]
    returns = compute_portfolio_returns(weights, prices)

    if benchmark_prices is None:
        logger.info("Downloading benchmark (%s) for risk calculations...", benchmark_ticker)
        raw = yf.download(
            benchmark_ticker,
            start=prices.index[0],
            end=prices.index[-1],
            auto_adjust=True,
            progress=False,
        )["Close"]
        benchmark_prices = raw.squeeze()

    bench_returns = benchmark_prices.pct_change().dropna()

    ann_ret = annualized_return(returns)
    ann_vol = annualized_volatility(returns)
    sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else float("nan")

    return {
        "label": portfolio.get("label", "Portfolio"),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "var_95": var_historical(returns, 0.95),
        "cvar_95": cvar_historical(returns, 0.95),
        "beta": beta_vs_benchmark(returns, bench_returns),
        "calmar_ratio": calmar_ratio(returns),
        # Internal series — used by charts, not exported to CSV
        "_daily_returns": returns,
        "_benchmark_returns": bench_returns,
    }
