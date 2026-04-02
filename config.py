import os
from datetime import datetime, timedelta

# ── Universe ────────────────────────────────────────────────────────────────
DATE_START = datetime(2015, 1, 1)
DATE_END = datetime.now() - timedelta(days=3)

# ── Screening ────────────────────────────────────────────────────────────────
MIN_FILTER_PASSES = 6       # minimum filters a stock must pass to qualify
QUANTILE_THRESHOLD = 0.80   # percentile cutoff for sector-relative selection

# ── Optimization ─────────────────────────────────────────────────────────────
N_PORTFOLIOS = 30_000       # number of random portfolios to simulate
TRADING_DAYS = 252          # annualization factor
RISK_FREE_RATE = 0.0425     # annualized risk-free rate (update as needed)
MAX_WEIGHT_PER_STOCK = 1.0  # max allocation per asset (1.0 = unconstrained)
                             # set to e.g. 0.20 to cap any single position at 20%

# ── Cache ────────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join("data", "cache")
CACHE_TTL_HOURS = 24        # hours before cached data is considered stale

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
