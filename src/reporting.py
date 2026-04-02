import logging
import os

import pandas as pd

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_portfolios(portfolios: list) -> str:
    """Export a side-by-side weight comparison of all portfolios.

    Output: output/portfolios.csv
    Columns: one per portfolio, rows = ticker symbols.
    """
    series = [p["weights"].rename(p["label"]) for p in portfolios]
    df = pd.concat(series, axis=1).fillna(0.0).sort_index()
    df.index.name = "Ticker"
    path = os.path.join(OUTPUT_DIR, "portfolios.csv")
    df.to_csv(path)
    logger.info("Exported portfolio weights → %s", path)
    return path


def export_risk_metrics(risk_reports: list) -> str:
    """Export a comparative risk metrics table for all portfolios.

    Output: output/risk_metrics.csv
    """
    rows = []
    for r in risk_reports:
        rows.append({
            "Portfolio":              r["label"],
            "Annual Return (%)":      round(r["annual_return"] * 100, 2),
            "Annual Volatility (%)":  round(r["annual_volatility"] * 100, 2),
            "Sharpe Ratio":           round(r["sharpe_ratio"], 3),
            "Sortino Ratio":          round(r["sortino_ratio"], 3),
            "Max Drawdown (%)":       round(r["max_drawdown"] * 100, 2),
            "VaR 95% (daily %)":      round(r["var_95"] * 100, 2),
            "CVaR 95% (daily %)":     round(r["cvar_95"] * 100, 2),
            "Beta vs SPY":            round(r["beta"], 3),
            "Calmar Ratio":           round(r["calmar_ratio"], 3),
        })
    df = pd.DataFrame(rows).set_index("Portfolio")
    path = os.path.join(OUTPUT_DIR, "risk_metrics.csv")
    df.to_csv(path)
    logger.info("Exported risk metrics → %s", path)
    return path


def export_screened_stocks(summary_df: pd.DataFrame) -> str:
    """Export the list of stocks that passed all screening stages.

    Output: output/screened_stocks.csv
    Columns: symbol, sector, score (number of filters passed).
    """
    path = os.path.join(OUTPUT_DIR, "screened_stocks.csv")
    summary_df.to_csv(path, index=False)
    logger.info("Exported screened stocks → %s", path)
    return path
