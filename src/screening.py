import collections
import logging
import time

import numpy as np
import pandas as pd
import yfinance as yf

from config import MIN_FILTER_PASSES, QUANTILE_THRESHOLD

logger = logging.getLogger(__name__)

# Fields fetched once per ticker and reused across all filters
_FUNDAMENTAL_FIELDS = [
    "sector",
    "currentRatio",
    "trailingPE",
    "forwardPE",
    "debtToEquity",
    "ebitdaMargins",
    "enterpriseToEbitda",
    "returnOnAssets",
    "returnOnEquity",
    "revenueGrowth",
    "earningsGrowth",
    "beta",
    "grossMargins",
]


# ── Single-stock fetch with retry ─────────────────────────────────────────────

def fetch_fundamentals(symbol: str, retries: int = 3, backoff: float = 2.0) -> dict:
    """Fetch fundamental data for one ticker with exponential-backoff retries.

    Returns a dict with all _FUNDAMENTAL_FIELDS keys; missing values are None.
    """
    for attempt in range(retries):
        try:
            info = yf.Ticker(symbol).info
            result = {field: info.get(field) for field in _FUNDAMENTAL_FIELDS}
            # Prefer trailingPE; fall back to forwardPE if invalid
            pe = result["trailingPE"]
            if pe is not None:
                try:
                    result["trailingPE"] = float(pe)
                except (TypeError, ValueError):
                    result["trailingPE"] = result.get("forwardPE")
            return result
        except Exception as exc:
            if attempt < retries - 1:
                wait = backoff ** attempt
                logger.warning(
                    "Fetch failed for %s (attempt %d/%d): %s — retrying in %.1fs.",
                    symbol, attempt + 1, retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error("All retries exhausted for %s: %s.", symbol, exc)
                return {field: None for field in _FUNDAMENTAL_FIELDS}


def fetch_all_fundamentals(symbols: list) -> dict:
    """Fetch fundamentals for every symbol in one pass (single API call per ticker).

    This consolidates data that was previously fetched redundantly by each
    individual filter, reducing total yfinance calls from O(n × filters) → O(n).
    """
    logger.info("Fetching fundamentals for %d stocks (one call each)...", len(symbols))
    result = {}
    for i, symbol in enumerate(symbols, 1):
        logger.debug("  [%d/%d] %s", i, len(symbols), symbol)
        result[symbol] = fetch_fundamentals(symbol)
    logger.info("Fundamentals fetch complete.")
    return result


# ── Price-based filters ───────────────────────────────────────────────────────

class StockFilter:
    """Structural and price-based filters applied before fundamental screening."""

    def filtro_nulo(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drop any column that contains at least one NaN."""
        before = len(data.columns)
        data = data.dropna(axis=1)
        logger.info("filtro_nulo: %d → %d stocks (dropped %d)", before, len(data.columns), before - len(data.columns))
        return data

    def filtro_sharpe(self, data: pd.DataFrame) -> tuple:
        """Drop stocks with a negative annualized Sharpe ratio (rf = 0).

        Returns (filtered_data, sharpe_series).
        """
        returns = np.log1p(data.pct_change())
        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        to_drop = sharpe[sharpe < 0].index.tolist()
        data = data.drop(columns=to_drop)
        logger.info(
            "filtro_sharpe: dropped %d stocks with Sharpe < 0, %d remain",
            len(to_drop), len(data.columns),
        )
        return data, sharpe

    def filtro_tiempo(self, data: pd.DataFrame, n_years: float) -> pd.DataFrame:
        """Drop stocks with more than n_years * 252 missing trading days."""
        min_missing = int(252 * n_years)
        n_total = len(data)
        to_drop = [col for col in data.columns if (n_total - data[col].count()) >= min_missing]
        data = data.drop(columns=to_drop)
        logger.info("filtro_tiempo: dropped %d stocks with insufficient history", len(to_drop))
        return data

    def filtro_precio(self, data: pd.DataFrame, precio_min: float, precio_max: float) -> pd.DataFrame:
        """Drop stocks whose last observed price falls outside [precio_min, precio_max]."""
        to_drop = [
            col for col in data.columns
            if not (precio_min <= data[col].iloc[-1] <= precio_max)
        ]
        data = data.drop(columns=to_drop)
        logger.info(
            "filtro_precio: dropped %d stocks outside [%.0f, %.0f]",
            len(to_drop), precio_min, precio_max,
        )
        return data


# ── Fundamental screener ──────────────────────────────────────────────────────

class FundamentalScreener:
    """Sector-relative multi-factor screening using pre-fetched fundamentals.

    Each metric is scored relative to sector peers. A stock advances to the
    next stage only if it passes MIN_FILTER_PASSES or more criteria.
    """

    def screen(
        self,
        data: pd.DataFrame,
        sharpe_ratios: pd.Series,
        fundamentals: dict,
    ) -> tuple:
        """Score every stock against its sector peers.

        Returns:
            (filtered_prices, score_counts, fundamentals_df)

        fundamentals_df is kept for downstream use so that sector info does
        NOT need to be re-fetched (fixing the original redundant-call bug).
        """
        records = []
        for symbol in data.columns:
            f = fundamentals.get(symbol, {}) or {}
            records.append({
                "symbol": symbol,
                "sector": f.get("sector"),
                "CurrentRatio": f.get("currentRatio"),
                "PE": f.get("trailingPE"),
                "DebtToEquity": f.get("debtToEquity"),
                "EbitdaMargins": f.get("ebitdaMargins"),
                "EnterpriseToEbitda": f.get("enterpriseToEbitda"),
                "ROA": f.get("returnOnAssets"),
                "ROE": f.get("returnOnEquity"),
                "RevenueGrowth": f.get("revenueGrowth"),
                "EarningsGrowth": f.get("earningsGrowth"),
                "Beta": f.get("beta"),
                # BUG FIX: was `data_dict['GrossMargins'] = gross_Margin`
                # (overwrote the entire key with a scalar instead of per-symbol)
                "GrossMargins": f.get("grossMargins"),
                "SharpeRatio": sharpe_ratios.get(symbol),
            })

        df = pd.DataFrame(records).set_index("symbol")
        score_counter = collections.Counter()

        for sector in df["sector"].dropna().unique():
            peers = df[df["sector"] == sector]

            def _above(col):
                m = peers[col].mean()
                return peers.index[peers[col] > m].tolist()

            def _below(col):
                m = peers[col].mean()
                return peers.index[peers[col] < m].tolist()

            def _within_one_std(col):
                m, s = peers[col].mean(), peers[col].std()
                mask = (peers[col] > m - s) & (peers[col] < m + s)
                return peers.index[mask].tolist()

            def _below_mean_plus_std(col):
                m, s = peers[col].mean(), peers[col].std()
                return peers.index[peers[col] < m + s].tolist()

            buckets = [
                _above("SharpeRatio"),
                _above("CurrentRatio"),
                _within_one_std("PE"),           # P/E within ±1σ of sector mean
                _above("EbitdaMargins"),
                _below("DebtToEquity"),           # lower debt is better
                _below("EnterpriseToEbitda"),     # lower EV/EBITDA is better
                _above("ROA"),
                _above("ROE"),
                _above("RevenueGrowth"),
                _above("EarningsGrowth"),
                _below_mean_plus_std("Beta"),     # not an outlier on the upside
                _above("GrossMargins"),
            ]

            for bucket in buckets:
                for symbol in bucket:
                    score_counter[symbol] += 1

        qualified = {s for s, count in score_counter.items() if count >= MIN_FILTER_PASSES}
        data = data[[c for c in data.columns if c in qualified]]
        counts = pd.Series(score_counter, name="score")

        logger.info(
            "Fundamental screening: %d stocks qualify (min_passes=%d)",
            len(data.columns), MIN_FILTER_PASSES,
        )
        return data, counts, df

    def top_sector(
        self,
        data: pd.DataFrame,
        score_counts: pd.Series,
        fundamentals_df: pd.DataFrame,
    ) -> tuple:
        """Keep only stocks above the QUANTILE_THRESHOLD within each sector.

        Sector information is taken from fundamentals_df (already fetched),
        eliminating the redundant per-ticker yfinance calls of the original code.
        """
        df = score_counts.reset_index()
        df.columns = ["symbol", "score"]

        # Reuse sector data — no re-fetch needed
        df["sector"] = df["symbol"].map(fundamentals_df["sector"])
        df = df.dropna(subset=["sector"])

        sector_q = df.groupby("sector")["score"].quantile(QUANTILE_THRESHOLD)
        df = df.merge(sector_q.rename("quantile"), on="sector")

        qualified = df[df["score"] > df["quantile"]]["symbol"].tolist()
        data = data[[c for c in data.columns if c in qualified]]
        summary = (
            df[df["symbol"].isin(qualified)][["symbol", "sector", "score"]]
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(
            "top_sector (q=%.0f%%): %d stocks selected",
            QUANTILE_THRESHOLD * 100, len(data.columns),
        )
        return data, summary
