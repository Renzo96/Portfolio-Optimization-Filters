import datetime as dt
import logging
import os
import pickle

import pandas as pd
import requests
import yfinance as yf

from config import CACHE_DIR, CACHE_TTL_HOURS, DATE_END, DATE_START

logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.pkl")


def _is_cache_valid(path: str) -> bool:
    if not os.path.exists(path):
        return False
    age_hours = (dt.datetime.now().timestamp() - os.path.getmtime(path)) / 3600
    return age_hours < CACHE_TTL_HOURS


def _load_cache(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(path: str, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ── Public API ────────────────────────────────────────────────────────────────

def load_sp500_components() -> pd.DataFrame:
    """Return S&P 500 components as a DataFrame (symbol, name, sector, sub_industry).

    Result is cached to disk for CACHE_TTL_HOURS hours to avoid repeated
    Wikipedia requests.
    """
    path = _cache_path("sp500_components")
    if _is_cache_valid(path):
        logger.info("S&P 500 components loaded from cache.")
        return _load_cache(path)

    logger.info("Fetching S&P 500 components from Wikipedia...")
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        timeout=15,
    )
    response.raise_for_status()
    components = pd.read_html(response.content)[0]
    components = components[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    components.columns = ["symbol", "name", "sector", "sub_industry"]
    # Wikipedia uses "." as separator but yfinance expects "-"
    components["symbol"] = components["symbol"].str.replace(".", "-", regex=False)

    _save_cache(path, components)
    logger.info("S&P 500 components cached (%d tickers).", len(components))
    return components


def download_prices(
    tickers: list,
    start: dt.datetime = DATE_START,
    end: dt.datetime = DATE_END,
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Uses a file cache keyed by the number of tickers and date range.
    The cache key intentionally omits specific ticker names so that minor
    universe changes do not invalidate the entire cache — clear data/cache/
    manually to force a full re-download.
    """
    cache_key = f"prices_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}_{len(tickers)}"
    path = _cache_path(cache_key)
    if _is_cache_valid(path):
        logger.info("Price data loaded from cache (%d tickers).", len(tickers))
        return _load_cache(path)

    logger.info("Downloading prices for %d tickers via yfinance...", len(tickers))
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
    )["Close"]

    _save_cache(path, data)
    logger.info("Price data cached: %d tickers × %d trading days.", data.shape[1], data.shape[0])
    return data
