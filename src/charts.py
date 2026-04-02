import logging
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR

logger = logging.getLogger(__name__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4a",
    "axes.labelcolor": "#c8ccd8",
    "axes.titlecolor": "#e8eaf0",
    "xtick.color": "#9095a3",
    "ytick.color": "#9095a3",
    "text.color": "#c8ccd8",
    "grid.color": "#2a2d3a",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4a",
    "font.family": "sans-serif",
})

_PORTFOLIO_COLORS = {
    "Max Sharpe":    "#2ecc71",
    "Min Volatility": "#3498db",
    "Max Sortino":   "#e67e22",
}
_PORTFOLIO_MARKERS = {
    "Max Sharpe":    "*",
    "Min Volatility": "D",
    "Max Sortino":   "^",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("Chart saved → %s", path)


# ── Charts ────────────────────────────────────────────────────────────────────

def plot_efficient_frontier(
    sim_df: pd.DataFrame,
    portfolios: list,
) -> plt.Figure:
    """Scatter of simulated portfolios colored by Sharpe ratio with optimal portfolios highlighted.

    Args:
        sim_df:     DataFrame from generate_portfolios() with [volatility, annual_return, sharpe].
        portfolios: List of portfolio dicts from the optimizer functions.
    """
    fig, ax = plt.subplots(figsize=(13, 7))

    sc = ax.scatter(
        sim_df["volatility"] * 100,
        sim_df["annual_return"] * 100,
        c=sim_df["sharpe"],
        cmap="plasma",
        alpha=0.35,
        s=6,
        rasterized=True,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Sharpe Ratio", color="#c8ccd8")
    cbar.ax.yaxis.set_tick_params(color="#c8ccd8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#c8ccd8")

    for p in portfolios:
        color = _PORTFOLIO_COLORS.get(p["label"], "#ffffff")
        marker = _PORTFOLIO_MARKERS.get(p["label"], "o")
        ax.scatter(
            p["volatility"] * 100,
            p["annual_return"] * 100,
            marker=marker,
            s=280,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
            label=f"{p['label']}  (Sharpe {p['sharpe']:.2f})",
        )

    ax.set_xlabel("Annualized Volatility (%)", fontsize=11)
    ax.set_ylabel("Annualized Return (%)", fontsize=11)
    ax.set_title("Efficient Frontier — Portfolio Optimization", fontsize=14, pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    _save(fig, "efficient_frontier.png")
    return fig


def plot_weights(portfolio: dict, sector_map: dict = None) -> plt.Figure:
    """Horizontal bar chart of portfolio weights, optionally colored by sector.

    Args:
        portfolio:  Portfolio dict with 'weights' (pd.Series) and 'label'.
        sector_map: Optional {ticker: sector} mapping for color coding.
    """
    weights = portfolio["weights"].sort_values(ascending=True)
    label = portfolio.get("label", "Portfolio")
    color = _PORTFOLIO_COLORS.get(label, "#7f8c8d")

    fig, ax = plt.subplots(figsize=(11, max(4, len(weights) * 0.45)))

    if sector_map:
        sectors = [sector_map.get(t, "Unknown") for t in weights.index]
        unique_sectors = sorted(set(sectors))
        palette = plt.cm.Set2(np.linspace(0, 1, len(unique_sectors)))
        sector_colors = dict(zip(unique_sectors, palette))
        bar_colors = [sector_colors[s] for s in sectors]
        ax.barh(weights.index, weights.values * 100, color=bar_colors)
        handles = [mpatches.Patch(color=sector_colors[s], label=s) for s in unique_sectors]
        ax.legend(handles=handles, title="Sector", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    else:
        ax.barh(weights.index, weights.values * 100, color=color)

    ax.set_xlabel("Allocation (%)", fontsize=10)
    ax.set_title(f"{label} — Portfolio Weights", fontsize=13)
    ax.axvline(0, color="#555", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    safe = label.lower().replace(" ", "_")
    _save(fig, f"weights_{safe}.png")
    return fig


def plot_correlation_heatmap(prices: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap of annualized daily returns for the selected universe."""
    returns = prices.pct_change().dropna()
    corr = returns.corr()
    n = len(corr)
    size = max(7, n * 0.55)

    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(
        corr,
        annot=n <= 25,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        linecolor="#0f1117",
        annot_kws={"size": 7},
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Return Correlation Matrix — Screened Universe", fontsize=13, pad=12)
    fig.tight_layout()
    _save(fig, "correlation_heatmap.png")
    return fig


def plot_nav_vs_benchmark(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    label: str = "Portfolio",
) -> plt.Figure:
    """Cumulative NAV (base = 100) of portfolio vs benchmark.

    Includes shaded drawdown regions for visual clarity.
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    aligned.columns = [label, "SPY"]
    nav = (1 + aligned).cumprod() * 100
    color = _PORTFOLIO_COLORS.get(label, "#2ecc71")

    fig, ax = plt.subplots(figsize=(14, 6))
    nav[label].plot(ax=ax, color=color, linewidth=2, label=label)
    nav["SPY"].plot(ax=ax, color="#e74c3c", linewidth=1.5, linestyle="--", label="Benchmark (SPY)", alpha=0.85)

    # Shade periods where portfolio underperforms
    diff = nav[label] - nav["SPY"]
    ax.fill_between(nav.index, nav[label], nav["SPY"], where=(diff < 0), alpha=0.15, color="#e74c3c", label="Underperformance")
    ax.fill_between(nav.index, nav[label], nav["SPY"], where=(diff >= 0), alpha=0.1, color=color)

    ax.set_ylabel("NAV (Base = 100)", fontsize=10)
    ax.set_title(f"Cumulative Performance: {label} vs SPY", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    fig.tight_layout()
    safe = label.lower().replace(" ", "_")
    _save(fig, f"nav_{safe}.png")
    return fig


def plot_drawdown(daily_returns: pd.Series, label: str = "Portfolio") -> plt.Figure:
    """Area chart of portfolio drawdown over time."""
    nav = (1 + daily_returns).cumprod()
    peak = nav.cummax()
    drawdown = (nav - peak) / peak * 100
    color = _PORTFOLIO_COLORS.get(label, "#e74c3c")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(drawdown.index, drawdown.values, 0, color=color, alpha=0.45)
    ax.plot(drawdown.index, drawdown.values, color=color, linewidth=1)
    ax.axhline(0, color="#555", linewidth=0.8)

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.annotate(
        f"Max DD: {max_dd:.1f}%",
        xy=(max_dd_date, max_dd),
        xytext=(max_dd_date, max_dd - 2),
        fontsize=8,
        color="white",
        ha="center",
    )

    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.set_title(f"{label} — Drawdown", fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    safe = label.lower().replace(" ", "_")
    _save(fig, f"drawdown_{safe}.png")
    return fig
