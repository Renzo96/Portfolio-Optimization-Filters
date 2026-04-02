
# Portfolio Optimization

A quantitative equity portfolio construction system built on **Markowitz Mean-Variance Optimization** with multi-factor fundamental screening. Designed to replicate the rigor of institutional asset management workflows.

---

## What it does

The system operates as a sequential pipeline:

1. **Universe** — Loads the full S&P 500 constituent list from Wikipedia (cached locally).
2. **Price download** — Fetches adjusted close prices via yfinance from 2015 to present (cached to disk, TTL configurable).
3. **Structural screening** — Eliminates tickers with missing price history and those with a negative historical Sharpe ratio.
4. **Fundamental screening** — Scores each stock across 12 financial metrics (P/E, Debt/Equity, EBITDA margins, ROA, ROE, revenue growth, earnings growth, gross margins, beta, current ratio, EV/EBITDA, Sharpe). Scoring is **sector-relative**: stocks compete against peers in the same GICS sector. Only those passing a configurable minimum threshold advance.
5. **Sector diversification** — From the qualifying pool, retains only stocks above the 80th percentile of their sector (configurable), ensuring no single sector dominates.
6. **Portfolio optimization** — Runs 30,000 Monte Carlo simulations to map the feasible risk-return space, then uses SLSQP constrained optimization to find three distinct optimal portfolios:
   - **Max Sharpe** — best risk-adjusted return
   - **Min Volatility** — lowest annualized standard deviation
   - **Max Sortino** — best downside-risk-adjusted return
7. **Risk analysis** — Computes a full institutional risk report per portfolio: Sortino ratio, Maximum Drawdown, VaR 95%, CVaR/Expected Shortfall, Beta vs SPY, and Calmar ratio.
8. **Visualization** — Generates 5 chart types: Efficient Frontier, portfolio weights by sector, correlation heatmap, cumulative NAV vs SPY benchmark, and drawdown chart.
9. **Export** — Writes results to `output/`: portfolio weights CSV, risk metrics comparison CSV, and screened stocks CSV.

---

## Project structure

```
Portfolio-Optimization-Filters/
├── config.py          ← all tunable parameters (risk-free rate, dates, thresholds)
├── main.py            ← pipeline entry point: python main.py
├── notebook.ipynb     ← interactive walkthrough with inline charts
├── src/
│   ├── universe.py    ← S&P 500 constituent fetch + disk cache
│   ├── screening.py   ← structural and fundamental filters
│   ├── optimizer.py   ← Markowitz optimization (3 portfolios)
│   ├── risk.py        ← risk metrics computation
│   ├── charts.py      ← all visualizations
│   └── reporting.py   ← CSV export
├── data/cache/        ← auto-generated cache files
└── output/            ← generated charts (.png) and results (.csv)
```

---

## Quickstart

```bash
# Install dependencies
pip install numpy pandas scipy yfinance requests matplotlib seaborn

# Run the full pipeline
python main.py

# Or explore interactively
jupyter notebook notebook.ipynb
```

---

## Configuration

All parameters are centralized in `config.py`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RISK_FREE_RATE` | `0.0425` | Annual risk-free rate |
| `N_PORTFOLIOS` | `30,000` | Monte Carlo simulations |
| `MIN_FILTER_PASSES` | `6` | Minimum fundamental filters to pass (out of 12) |
| `QUANTILE_THRESHOLD` | `0.80` | Sector percentile cutoff |
| `MAX_WEIGHT_PER_STOCK` | `1.0` | Per-asset weight cap (set `0.20` for a 20% limit) |
| `DATE_START` | `2015-01-01` | Start of price history |

---

## Authors

- [@RenzoSosa](https://www.github.com/Renzo96)
- [@DuamCastro](https://www.github.com/duamc)

