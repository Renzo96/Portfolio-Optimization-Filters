"""
main.py — Portfolio Optimization Pipeline Entry Point

Run:
    python main.py

Steps:
    1. Load S&P 500 universe (cached)
    2. Download price history (cached)
    3. Price-based screening (nulls, Sharpe)
    4. Fundamental screening (12 metrics, sector-relative)
    5. Portfolio optimization (Max Sharpe / Min Vol / Max Sortino)
    6. Risk analysis (Sharpe, Sortino, VaR, CVaR, Beta, Calmar, MaxDD)
    7. Chart generation (5 charts → output/)
    8. CSV export (3 files → output/)
"""

import logging
import os
import sys

import pandas as pd
import yfinance as yf

from config import DATE_END, DATE_START, LOG_LEVEL, N_PORTFOLIOS, OUTPUT_DIR, RISK_FREE_RATE
from src.charts import (
    plot_correlation_heatmap,
    plot_drawdown,
    plot_efficient_frontier,
    plot_nav_vs_benchmark,
    plot_weights,
)
from src.optimizer import (
    generate_portfolios,
    optimize_max_sharpe,
    optimize_max_sortino,
    optimize_min_volatility,
)
from src.reporting import export_portfolios, export_risk_metrics, export_screened_stocks
from src.risk import full_risk_report
from src.screening import FundamentalScreener, StockFilter, fetch_all_fundamentals
from src.universe import download_prices, load_sp500_components

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "run.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run() -> None:
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║        PORTFOLIO OPTIMIZATION — PIPELINE START           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    # ── 1. Universe ───────────────────────────────────────────────────────────
    logger.info("▶ STEP 1/8 — Loading S&P 500 universe")
    components = load_sp500_components()
    tickers = components["symbol"].tolist()
    sector_map = dict(zip(components["symbol"], components["sector"]))
    logger.info("Universe: %d tickers  |  period: %s → %s", len(tickers), DATE_START.date(), DATE_END.date())

    # ── 2. Price history ──────────────────────────────────────────────────────
    logger.info("▶ STEP 2/8 — Downloading price history")
    prices_raw = download_prices(tickers, start=DATE_START, end=DATE_END)

    # ── 3. Price-based screening ──────────────────────────────────────────────
    logger.info("▶ STEP 3/8 — Price-based screening")
    sf = StockFilter()
    prices = sf.filtro_nulo(prices_raw)
    prices, sharpe_ratios = sf.filtro_sharpe(prices)

    # ── 4. Fundamental screening ──────────────────────────────────────────────
    logger.info("▶ STEP 4/8 — Fundamental screening (%d stocks)", len(prices.columns))
    fundamentals = fetch_all_fundamentals(prices.columns.tolist())
    screener = FundamentalScreener()
    prices, score_counts, fundamentals_df = screener.screen(prices, sharpe_ratios, fundamentals)
    prices, stock_summary = screener.top_sector(prices, score_counts, fundamentals_df)

    if len(prices.columns) < 3:
        logger.error(
            "Only %d stocks survived screening — cannot build a diversified portfolio. "
            "Consider lowering MIN_FILTER_PASSES in config.py.",
            len(prices.columns),
        )
        sys.exit(1)

    logger.info(
        "Final universe: %d stocks — %s",
        len(prices.columns),
        ", ".join(prices.columns.tolist()),
    )

    # ── 5. Download benchmark once (reused by all risk reports) ───────────────
    logger.info("▶ STEP 5/8 — Downloading benchmark (SPY)")
    spy_raw = yf.download("SPY", start=prices.index[0], end=prices.index[-1], auto_adjust=True, progress=False)["Close"]
    spy_prices = spy_raw.squeeze()

    # ── 6. Portfolio optimization ─────────────────────────────────────────────
    logger.info("▶ STEP 6/8 — Optimizing portfolios (%d simulations)", N_PORTFOLIOS)
    sim_df, _ = generate_portfolios(prices, N_PORTFOLIOS, RISK_FREE_RATE)
    p_max_sharpe   = optimize_max_sharpe(prices, RISK_FREE_RATE)
    p_min_vol      = optimize_min_volatility(prices)
    p_max_sortino  = optimize_max_sortino(prices, RISK_FREE_RATE)
    portfolios = [p_max_sharpe, p_min_vol, p_max_sortino]

    # ── 7. Risk analysis ──────────────────────────────────────────────────────
    logger.info("▶ STEP 7/8 — Computing risk metrics")
    risk_reports = [
        full_risk_report(p, prices, benchmark_prices=spy_prices, risk_free_rate=RISK_FREE_RATE)
        for p in portfolios
    ]

    # ── 8. Charts ─────────────────────────────────────────────────────────────
    logger.info("▶ STEP 8/8 — Generating charts and exporting results")
    plot_efficient_frontier(sim_df, portfolios)
    plot_correlation_heatmap(prices)
    for p in portfolios:
        plot_weights(p, sector_map)
    for r in risk_reports:
        plot_nav_vs_benchmark(r["_daily_returns"], r["_benchmark_returns"], label=r["label"])
        plot_drawdown(r["_daily_returns"], label=r["label"])

    # ── Export ────────────────────────────────────────────────────────────────
    export_portfolios(portfolios)
    export_risk_metrics(risk_reports)
    export_screened_stocks(stock_summary)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print(f"{'PORTFOLIO SUMMARY':^68}")
    print("=" * 68)
    summary_rows = [
        {
            "Portfolio":  r["label"],
            "Return":     f"{r['annual_return']:>7.1%}",
            "Vol":        f"{r['annual_volatility']:>6.1%}",
            "Sharpe":     f"{r['sharpe_ratio']:>6.2f}",
            "Sortino":    f"{r['sortino_ratio']:>7.2f}",
            "MaxDD":      f"{r['max_drawdown']:>7.1%}",
            "Beta":       f"{r['beta']:>5.2f}",
            "Calmar":     f"{r['calmar_ratio']:>6.2f}",
        }
        for r in risk_reports
    ]
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 68)
    print(f"\nOutputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print()

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    run()
