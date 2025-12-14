from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from functools import lru_cache
import plotly.express as px

import matplotlib.pyplot as plt
from io import BytesIO

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (
        SimpleDocTemplate,
        Table,
        TableStyle,
        Paragraph,
        Spacer,
        Image,
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ============================================================================
# APP CONFIG
# ============================================================================

st.set_page_config(page_title="Portfolio Analyzer 2.19", layout="wide")

# ============================================================================
# CSV NORMALIZATION & REBASING (v2.9) + YAHOO NAMES (v2.11)
# ============================================================================


def normalize_and_rebase_to_pounds(
    df_tx: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Normalize CSV prices to match Yahoo, then rebase EVERYTHING to pounds.

    v2.9: Use Yahoo's currency metadata (GBp/GBX) instead of ratio detection.
    v2.11: Also extract Yahoo longName/shortName for each ticker.

    Returns:
    - df_tx with prices in pounds
    - conversion_factors dict {ticker: factor} to apply to Yahoo prices
    - yahoo_names dict {ticker: display_name}
    """
    st.info("🔄 Normalizing and rebasing all prices to pounds (£)...")

    tickers = df_tx["ticker"].unique()
    conversion_log: list[str] = []
    conversion_factors: dict[str, float] = {}
    yahoo_names: dict[str, str] = {}

    for ticker in tickers:
        try:
            # Get Yahoo Finance metadata
            ticker_info = yf.Ticker(ticker)

            # Try name
            long_name = None
            try:
                info_dict = ticker_info.info
                long_name = info_dict.get(
                    "longName") or info_dict.get("shortName")
            except Exception:
                long_name = None
            yahoo_names[ticker] = str(long_name) if long_name else ticker

            # Try to get currency from fast_info, fallback to info
            currency = None
            try:
                fast_info = ticker_info.fast_info
                currency = getattr(fast_info, "currency", None)
            except Exception:
                try:
                    info_dict = ticker_info.info
                    currency = info_dict.get("currency", None)
                except Exception:
                    currency = None

            # Get most recent CSV price for logging
            ticker_txs = df_tx[df_tx["ticker"] == ticker]
            csv_price_original = float(ticker_txs["price"].iloc[-1])

            # Determine conversion based on Yahoo's currency
            if currency in ("GBp", "GBX"):
                # Yahoo quotes in pence, need to convert to pounds
                conversion_log.append(
                    f"✅ {ticker}: Yahoo in pence (currency={currency})"
                    f" → Yahoo prices will be ÷100 to £"
                )
                conversion_factors[ticker] = 100.0  # Divide Yahoo by 100

                # Check if CSV needs conversion
                if csv_price_original > 1000:
                    conversion_log.append(
                        f" → CSV={csv_price_original:.0f}p converted to £{csv_price_original/100:.2f}"
                    )
                    mask = df_tx["ticker"] == ticker
                    df_tx.loc[mask, "price"] = df_tx.loc[mask, "price"] / 100
                    # df_tx.loc[mask, "total_cost"] = df_tx.loc[mask, "total_cost"] / 100 ; DO NOT SCALE
                else:
                    conversion_log.append(
                        f" → CSV=£{csv_price_original:.2f} already in £"
                    )

            elif currency in ("GBP", "USD", "EUR"):
                # Yahoo already in major currency units
                conversion_log.append(
                    f"✅ {ticker}: Yahoo in £/$ (currency={currency})"
                )
                conversion_factors[ticker] = 1.0  # No conversion needed

                # Check if CSV is in pence
                if csv_price_original > 1000:
                    conversion_log.append(
                        f" → CSV={csv_price_original:.0f}p converted to £{csv_price_original/100:.2f}"
                    )
                    mask = df_tx["ticker"] == ticker
                    df_tx.loc[mask, "price"] = df_tx.loc[mask, "price"] / 100
                    # df_tx.loc[mask, "total_cost"] = df_tx.loc[mask, "total_cost"] / 100 ; DO NOT SCALE
                else:
                    conversion_log.append(
                        f" → CSV=£{csv_price_original:.2f} already in £"
                    )

            else:
                # Currency unknown, use fallback ratio detection
                conversion_log.append(
                    f"⚠️ {ticker}: Unknown currency ({currency}), using ratio detection"
                )
                csv_date = ticker_txs["date"].iloc[-1]
                yahoo_data = yf.download(
                    ticker,
                    start=csv_date - timedelta(days=10),
                    end=datetime.now(),
                    progress=False,
                )
                if not yahoo_data.empty:
                    if "Close" in yahoo_data.columns:
                        yahoo_price = float(yahoo_data["Close"].iloc[-1])
                    else:
                        yahoo_price = csv_price_original
                    ratio = yahoo_price / csv_price_original
                    if ratio > 50:
                        conversion_log.append(
                            f" → Ratio={ratio:.1f}x suggests Yahoo in pence"
                        )
                        conversion_factors[ticker] = 100.0
                    else:
                        conversion_log.append(
                            f" → Ratio={ratio:.2f}x suggests both in £"
                        )
                        conversion_factors[ticker] = 1.0
                else:
                    conversion_log.append(
                        f" → Could not download price, assuming £"
                    )
                    conversion_factors[ticker] = 1.0

        except Exception as e:
            import traceback

            conversion_log.append(f"❌ {ticker}: Error - {str(e)}")
            conversion_log.append(
                f" Details: {traceback.format_exc()[:200]}"
            )
            conversion_factors[ticker] = 1.0
            if ticker not in yahoo_names:
                yahoo_names[ticker] = ticker

    # Display conversion log
    # Decide whether to auto-open if there are warnings/errors
    has_issues = any(
        entry.startswith(("⚠️", "❌")) for entry in conversion_log
    )

    with st.expander(
        "📋 Price Normalization, Rebasing & Name Lookup Log",
        expanded=has_issues,  # collapsed if everything looks clean
    ):
        for log_entry in conversion_log:
            st.text(log_entry)

    st.success("✅ All prices rebased to pounds (£) for accurate calculations")

    return df_tx, conversion_factors, yahoo_names

# ============================================================================
# TRANSACTION LOADING
# ============================================================================


def load_transactions(file) -> tuple[pd.DataFrame, dict, dict]:
    """Load and normalize transactions, return with conversion factors and Yahoo names."""
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "ticker", "shares", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(
        df["date"].astype(str).str.strip(),
        format="mixed",
        dayfirst=True,
        errors="raise",
    )

    df["shares"] = pd.to_numeric(df["shares"], errors="raise")
    df["price"] = pd.to_numeric(df["price"], errors="raise")

    if "total_cost" in df.columns:
        df["total_cost"] = pd.to_numeric(df["total_cost"], errors="coerce")
        mask_nan = df["total_cost"].isna()
        if mask_nan.any():
            df.loc[mask_nan, "total_cost"] = (
                df.loc[mask_nan, "shares"] * df.loc[mask_nan, "price"]
            )
    else:
        df["total_cost"] = df["shares"] * df["price"]

    df = df.sort_values("date")

    # Normalize and rebase to pounds (v2.9) + names (v2.11)
    df, conversion_factors, yahoo_names = normalize_and_rebase_to_pounds(df)
    return df, conversion_factors, yahoo_names

# ============================================================================
# YAHOO DATA PROVIDER (v2.11 - DATA QUALITY LOGGING)
# ============================================================================


def fetch_yahoo_prices(
    symbols,
    start_date,
    end_date,
    auto_adjust=True,
    progress=False,
):
    """
    Wrapper around yf.download that:
    - downloads prices
    - returns (close_df, issues)
    issues is a list of {'symbol': str, 'problem': str}.
    """
    data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=progress,
    )

    issues: list[dict[str, str]] = []

    if data.empty:
        issues.append(
            {
                "symbol": ",".join(symbols if isinstance(symbols, list) else [symbols]),
                "problem": "No data returned for any symbol",
            }
        )
        return data, issues

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            issues.append(
                {
                    "symbol": ",".join(
                        symbols if isinstance(symbols, list) else [symbols]
                    ),
                    "problem": "Expected 'Close' in yfinance columns but not found",
                }
            )
            raise KeyError("Expected 'Close' in yfinance download columns.")
        close = data["Close"]
    else:
        if "Close" in data.columns:
            if isinstance(symbols, list) and len(symbols) == 1:
                close = data[["Close"]].rename(columns={"Close": symbols[0]})
            else:
                close = data[["Close"]]
        elif "Adj Close" in data.columns:
            close = data["Adj Close"]
        else:
            issues.append(
                {
                    "symbol": ",".join(
                        symbols if isinstance(symbols, list) else [symbols]
                    ),
                    "problem": "Neither 'Close' nor 'Adj Close' found in yfinance output",
                }
            )
            raise KeyError(
                "Neither 'Close' nor 'Adj Close' found in yfinance output."
            )

    requested = symbols if isinstance(symbols, list) else [symbols]
    for s in requested:
        if s not in close.columns:
            issues.append(
                {"symbol": s, "problem": "Symbol missing from Yahoo download"}
            )
        else:
            series = close[s]
            if series.isna().all():
                issues.append(
                    {
                        "symbol": s,
                        "problem": "All prices are NaN (Yahoo returned no usable data)",
                    }
                )

    return close, issues

# ============================================================================
# PRICE HISTORY (WITH REBASING + LOGGING v2.11 + MISSING RANGES v2.12)
# ===========================================================================


def get_price_history(
    tickers, benchmark, start_date, end_date, conversion_factors
) -> pd.DataFrame:
    """Fetch prices and apply conversion factors to rebase to pounds, with Yahoo issues logging (v2.11)."""
    all_symbols = list(set(tickers) | {benchmark})
    close, issues = fetch_yahoo_prices(
        all_symbols,
        start_date,
        end_date,
        auto_adjust=True,
        progress=False,
    )

    if close.empty:
        raise RuntimeError(
            "Yahoo Finance returned no price data for any symbol.")

    for s in all_symbols:
        if s not in close.columns:
            close[s] = np.nan
            issues.append(
                {
                    "symbol": s,
                    "problem": "Column added as all-NaN because Yahoo omitted symbol",
                }
            )

    close = close[all_symbols]

    # Rebase all prices to pounds
    for ticker in tickers:
        if ticker in close.columns and ticker in conversion_factors:
            factor = conversion_factors[ticker]
            if factor > 1:
                close[ticker] = close[ticker] / factor

    if benchmark in conversion_factors:
        factor = conversion_factors[benchmark]
        if factor > 1:
            close[benchmark] = close[benchmark] / factor

    # v2.10: handle price gaps
    close = close.dropna(how="all")
    close = close.sort_index()

    # v2.12: detect leading periods with missing prices per symbol
    missing_ranges: list[dict[str, str]] = []
    close_raw = close.copy()  # pre-forward-fill view
    first_index = close_raw.index.min() if not close_raw.empty else None
    last_index = close_raw.index.max() if not close_raw.empty else None

    if first_index is not None and last_index is not None:
        for s in all_symbols:
            if s not in close_raw.columns:
                continue
            series = close_raw[s]

            # Case 1: Yahoo returned no usable data for this symbol at all
            if series.isna().all():
                missing_ranges.append(
                    {
                        "symbol": s,
                        "type": "no_data",
                        "start": first_index.strftime("%d/%m/%Y"),
                        "end": last_index.strftime("%d/%m/%Y"),
                    }
                )
            else:
                # Case 2: there is data, but the start of the period is NaN
                first_valid = series.first_valid_index()
                if first_valid is not None and first_valid > first_index:
                    used_from = first_valid.strftime("%d/%m/%Y")
                    missing_ranges.append(
                        {
                            "symbol": s,
                            "type": "leading_nan",
                            "start": first_index.strftime("%d/%m/%Y"),
                            "end": (first_valid - pd.Timedelta(days=1)).strftime(
                                "%d/%m/%Y"
                            ),
                            "used_price_from": used_from,
                        }
                    )

                    # v2.19: Backfill missing leading period using the first available price
                    # This avoids a stepped jump in TWR when the first valid Yahoo price appears.
                    try:
                        fill_value = close.at[first_valid, s]
                        mask_leading = (close.index >= first_index) & (
                            close.index < first_valid)
                        close.loc[mask_leading, s] = fill_value
                    except Exception:
                        pass

    # Forward-fill to remove internal gaps for calculations
    close[all_symbols] = close[all_symbols].ffill()

    # Attach metadata for later display in the app
    close._yahoo_issues = issues  # attach
    close._missing_ranges = missing_ranges

    return close

# ============================================================================
# RISK-FREE RATE
# ============================================================================


def load_risk_free_series(rf_file, target_index: pd.DatetimeIndex) -> pd.Series | None:
    """Load risk-free rate CSV: columns date, rate (in percent)."""
    if rf_file is None:
        return None

    rf = pd.read_csv(rf_file, parse_dates=["date"])
    rf = rf.sort_values("date")
    rf["rate"] = pd.to_numeric(rf["rate"], errors="coerce") / 100.0
    rf = rf.set_index("date")["rate"]
    rf_aligned = rf.reindex(target_index).ffill()
    rf_daily = rf_aligned / 252.0
    return rf_daily

# ============================================================================
# HOLDINGS & PORTFOLIO CALCULATIONS
# ============================================================================


def build_holdings(df_tx: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative holdings (shares) for each ticker over time (v2.12).

    - Uses BEGINNING-of-day holdings (before transactions on that day)
    - Ensures transactions on non-trading days (weekends / holidays) are
      reflected from the next available trading day
    - Keeps full fractional share precision for all calculations
    """
    # Trading days from Yahoo price history
    days = price_df.index

    # Pivot raw transactions by transaction date and ticker
    tx_pivot = df_tx.pivot_table(
        index="date",
        columns="ticker",
        values="shares",
        aggfunc="sum",
    )

    # Union of all transaction dates and all price dates
    union_index = tx_pivot.index.union(days).sort_values()

    # Reindex to the union, fill missing with 0 shares
    tx_pivot_union = tx_pivot.reindex(union_index).fillna(0.0)

    # Cumulative shares over time, then shift to get BEGINNING-of-day holdings
    holdings_union = tx_pivot_union.cumsum().shift(1, fill_value=0.0)

    # Restrict to trading days and forward-fill between events
    holdings = holdings_union.reindex(days).ffill().fillna(0.0)

    # Ensure we have all asset columns present (including any symbols
    # Yahoo added as empty columns that still appear in price_df)
    holdings = holdings.reindex(columns=price_df.columns, fill_value=0.0)

    return holdings


def format_shares(x):
    """Format share quantities:
    - Up to 4 decimal places of precision
    - No trailing zeros
    - No decimal point for whole numbers (e.g. 61, 58.3852)
    """
    import pandas as _pd
    if isinstance(x, str):
        return x
    if _pd.isna(x):
        return ""
    s = f"{x:.4f}"
    s = s.rstrip("0")
    s = s.rstrip(".")
    return s


def compute_drawdown(cum_returns: pd.Series):
    """Return max drawdown and drawdown series from cumulative returns."""
    wealth = 1.0 + cum_returns
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    max_dd = dd.min() if not dd.empty and not dd.isna().all() else np.nan
    return max_dd, dd


def compute_twr_from_prices(
    asset_prices: pd.DataFrame, holdings_all: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    """
    Compute Time-Weighted Return using lagged holdings weights.
    All prices are in pounds, so calculations are accurate.
    """
    values_all = holdings_all * asset_prices
    port_value_all = values_all.sum(axis=1)
    asset_rets = asset_prices.pct_change()
    weights_lag = values_all.shift(1).div(port_value_all.shift(1), axis=0)
    twr_daily = (weights_lag * asset_rets).sum(axis=1)

    valid_mask = port_value_all.shift(1) > 0
    twr_daily = twr_daily.where(valid_mask, 0.0)
    twr_daily = twr_daily.fillna(0.0)
    twr_cum = (1.0 + twr_daily).cumprod() - 1.0
    return twr_daily, twr_cum


def compute_risk_metrics(twr_daily: pd.Series, rf_daily):
    """Compute portfolio risk metrics from TWR daily returns."""
    if twr_daily.empty:
        return {
            "total_return": np.nan,
            "vol_annual": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "downside_dev": np.nan,
            "max_drawdown": np.nan,
        }

    if isinstance(rf_daily, pd.Series):
        rf_aligned = rf_daily.reindex(twr_daily.index).ffill().fillna(0.0)
    else:
        rf_aligned = pd.Series(float(rf_daily), index=twr_daily.index)

    twr_cum = (1.0 + twr_daily).cumprod() - 1.0
    total_return = twr_cum.iloc[-1] if not twr_cum.empty else np.nan

    vol_daily = twr_daily.std(ddof=1)
    vol_annual = vol_daily * np.sqrt(252) if np.isfinite(vol_daily) else np.nan

    excess = twr_daily - rf_aligned
    mean_excess_daily = excess.mean()
    sharpe_daily = (
        mean_excess_daily / vol_daily
        if vol_daily and np.isfinite(vol_daily) and vol_daily > 0
        else np.nan
    )
    sharpe = sharpe_daily * \
        np.sqrt(252) if np.isfinite(sharpe_daily) else np.nan

    downside = twr_daily[twr_daily < 0]
    downside_dev_daily = downside.std(ddof=1) if len(downside) > 0 else np.nan
    downside_dev_annual = (
        downside_dev_daily *
        np.sqrt(252) if np.isfinite(downside_dev_daily) else np.nan
    )

    mean_return_annual = twr_daily.mean() * 252
    rf_mean_annual = rf_aligned.mean() * 252
    sortino = (
        (mean_return_annual - rf_mean_annual) / downside_dev_annual
        if downside_dev_annual
        and np.isfinite(downside_dev_annual)
        and downside_dev_annual > 0
        else np.nan
    )

    max_dd, _ = compute_drawdown(twr_cum)

    return {
        "total_return": total_return,
        "vol_annual": vol_annual,
        "sharpe": sharpe,
        "sortino": sortino,
        "downside_dev": downside_dev_annual,
        "max_drawdown": max_dd,
    }


def build_holdings_overview_full(
    pdata,
    benchmark: str,
    bench_prices: pd.Series,
    yahoo_names_all: dict[str, str],
    start_effective: date,
    end_effective: date,
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    """
    v2.17: Build Holdings Overview with period return columns.
    Returns (hold_table, total_val, port_ytd, bench_ytd, port_period, bench_period).
    """
    latest_hold = pdata["holdings"].iloc[-1]
    latest_values = pdata["values"].iloc[-1]
    total_val = float(latest_values.sum())

    weights_now = (
        latest_values / total_val if total_val > 0 else latest_values * 0
    )

    # YTD dates and portfolio YTD
    year_start_date = datetime(date.today().year, 1, 1).date()
    twr_ytd_mask = pdata["twr_daily_all"].index.date >= year_start_date
    if twr_ytd_mask.any():
        port_ytd_ret = pdata["twr_daily_all"][twr_ytd_mask]
        port_ytd = (1.0 + port_ytd_ret).prod() - 1.0
    else:
        port_ytd = 0.0

    # Benchmark YTD
    bench_ytd_mask = bench_prices.index.date >= year_start_date
    if bench_ytd_mask.any():
        bench_ytd_prices = bench_prices[bench_ytd_mask]
        if len(bench_ytd_prices) > 1:
            bench_ytd = (
                bench_ytd_prices.iloc[-1] / bench_ytd_prices.iloc[0]
            ) - 1.0
        else:
            bench_ytd = 0.0
    else:
        bench_ytd = 0.0

    # Asset-level YTD
    ytd_prices = pdata["asset_prices"][
        pdata["asset_prices"].index.date >= year_start_date
    ]
    if not ytd_prices.empty and len(ytd_prices) > 1:
        ytd_asset = (ytd_prices.iloc[-1] / ytd_prices.iloc[0]) - 1.0
    else:
        ytd_asset = pd.Series(0.0, index=latest_hold.index)

    # v2.17: Period return calculations based on selected date range
    twr_period_mask = (pdata["twr_daily_all"].index.date >= start_effective) & (
        pdata["twr_daily_all"].index.date <= end_effective
    )
    if twr_period_mask.any():
        port_period_ret = pdata["twr_daily_all"][twr_period_mask]
        port_period = (1.0 + port_period_ret).prod() - 1.0
    else:
        port_period = 0.0

    bench_period_mask = (bench_prices.index.date >= start_effective) & (
        bench_prices.index.date <= end_effective
    )
    if bench_period_mask.any():
        bench_period_prices = bench_prices[bench_period_mask]
        if len(bench_period_prices) > 1:
            bench_period = (
                bench_period_prices.iloc[-1] / bench_period_prices.iloc[0]
            ) - 1.0
        else:
            bench_period = 0.0
    else:
        bench_period = 0.0

    period_prices = pdata["asset_prices"][
        (pdata["asset_prices"].index.date >= start_effective)
        & (pdata["asset_prices"].index.date <= end_effective)
    ]
    if not period_prices.empty and len(period_prices) > 1:
        period_asset = (period_prices.iloc[-1] / period_prices.iloc[0]) - 1.0
    else:
        period_asset = pd.Series(0.0, index=latest_hold.index)

    tickers_idx = latest_hold.index
    names_series = pd.Index(tickers_idx).map(
        lambda t: yahoo_names_all.get(t, t)
    )

    # Cumulative cost per ticker from CSV
    tx_all = pdata["tx"]
    cost_by_ticker = tx_all.groupby("ticker")["total_cost"].sum()
    cost_series = cost_by_ticker.reindex(tickers_idx).fillna(0.0)

    hold_table = pd.DataFrame(
        {
            "Ticker": tickers_idx,
            "Name": names_series,
            "Shares": latest_hold.values,
            "Cost (£)": cost_series.values,
            "Value (£)": latest_values.values,
            "Weight": weights_now.values,
            "YTD Return": ytd_asset.reindex(tickers_idx).values,
            "Vs Portfolio YTD": (
                ytd_asset.reindex(tickers_idx) - port_ytd
            ).values,
            "Vs Benchmark YTD": (
                ytd_asset.reindex(tickers_idx) - bench_ytd
            ).values,
            "Period Return": period_asset.reindex(tickers_idx).values,
            "Vs Portfolio Period": (
                period_asset.reindex(tickers_idx) - port_period
            ).values,
            "Vs Benchmark Period": (
                period_asset.reindex(tickers_idx) - bench_period
            ).values,
        }
    )

    # Remove zero positions
    hold_table = hold_table[
        (hold_table["Shares"] != 0) | (hold_table["Value (£)"] != 0)
    ]

    hold_table = hold_table.sort_values(
        "Weight", ascending=False
    ).reset_index(drop=True)

    return (
        hold_table,
        total_val,
        float(port_ytd),
        float(bench_ytd),
        float(port_period),
        float(bench_period),
    )


def generate_pdf_report(
    pname: str,
    pdata: dict,
    benchmark: str,
    bench_prices: pd.Series,
    bench_metrics: dict,
    yahoo_names_all: dict[str, str],
    bench_cum: pd.Series,
    start_effective: date,
    end_effective: date,
) -> bytes:
    """
    v2.17: Generate PDF with period columns in holdings.
    Returns PDF bytes.
    """
    from reportlab.lib.pagesizes import landscape
    from reportlab.lib.units import inch

    buffer = BytesIO()

    # **CRITICAL: Define doc and elements FIRST**
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        topMargin=36,
        bottomMargin=36,
        leftMargin=36,
        rightMargin=36,
    )

    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(
        Paragraph(f"Portfolio Report: {pname}", styles["Heading1"])
    )
    elements.append(
        Paragraph(
            f"Period: {start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 12))

    # 1) Cumulative Return vs Benchmark chart
    base_cum = pdata["twr_cum"]
    common_idx = base_cum.index.intersection(bench_cum.index)

    if not common_idx.empty:
        base_series = base_cum.loc[common_idx]
        bench_series = bench_cum.loc[common_idx]

        base_series = base_series - base_series.iloc[0]
        bench_series = bench_series - bench_series.iloc[0]

        base_pct = base_series * 100.0
        bench_pct = bench_series * 100.0

        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(common_idx, base_pct, label=pname)
        ax.plot(
            common_idx,
            bench_pct,
            label=f"Benchmark ({benchmark})",
        )
        chart_title = (
            f"{pname} - Cumulative returns from "
            f"{start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}"
        )
        ax.set_title(chart_title)
        ax.set_ylabel("Cumulative Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        # Create the in-memory PNG
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        img_buf.seek(0)

        # Constrain image so it always fits the landscape page
        max_width = doc.width * 0.9      # 90% of frame width
        max_height = 3.5 * inch          # ~3.5 inches high
        img = Image(img_buf, width=max_width, height=max_height)
        elements.append(img)

        elements.append(Spacer(1, 12))

    # 2) Key Metrics table (portfolio + benchmark)
    elements.append(Paragraph("Key Metrics", styles["Heading2"]))

    m = pdata["metrics"]
    metrics_rows = [
        {
            "Series": pname,
            "Total Return": m["total_return"],
            "Volatility (annual)": m["vol_annual"],
            "Sharpe": m["sharpe"],
            "Sortino": m["sortino"],
            "Downside Dev": m["downside_dev"],
            "Max Drawdown": m["max_drawdown"],
        },
        {
            "Series": f"Benchmark ({benchmark})",
            "Total Return": bench_metrics["total_return"],
            "Volatility (annual)": bench_metrics["vol_annual"],
            "Sharpe": bench_metrics["sharpe"],
            "Sortino": bench_metrics["sortino"],
            "Downside Dev": bench_metrics["downside_dev"],
            "Max Drawdown": bench_metrics["max_drawdown"],
        },
    ]

    metrics_df = pd.DataFrame(metrics_rows)

    # Format as strings for PDF
    fmt_df = metrics_df.copy()
    fmt_df["Total Return"] = fmt_df["Total Return"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    fmt_df["Volatility (annual)"] = fmt_df["Volatility (annual)"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    fmt_df["Sharpe"] = fmt_df["Sharpe"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )
    fmt_df["Sortino"] = fmt_df["Sortino"].map(
        lambda x: f"{x:.2f}" if pd.notna(x) else ""
    )
    fmt_df["Downside Dev"] = fmt_df["Downside Dev"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    fmt_df["Max Drawdown"] = fmt_df["Max Drawdown"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )

    metrics_table_data = [fmt_df.columns.tolist()] + fmt_df.values.tolist()
    metrics_table = Table(metrics_table_data, hAlign="LEFT")
    metrics_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ])
    )
    elements.append(metrics_table)
    elements.append(Spacer(1, 6))

    metrics_note = (
        f"Metrics are calculated for the period from "
        f"{start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}. "
        f"Volatility, Sharpe, and Sortino ratios are annualized, i.e. multiplied by √252 trading days."
    )
    elements.append(Paragraph(metrics_note, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # 3) Holdings Overview (full values)
    elements.append(Paragraph("Holdings Overview", styles["Heading2"]))

    hold_table, total_val, port_ytd, bench_ytd, port_period, bench_period = build_holdings_overview_full(
        pdata,
        benchmark,
        bench_prices,
        yahoo_names_all,
        start_effective,
        end_effective,
    )

    # Format holdings for PDF
    ht = hold_table.copy()
    ht["Shares"] = ht["Shares"].map(format_shares)
    ht["Cost (£)"] = ht["Cost (£)"].map(
        lambda x: f"£{x:,.2f}" if pd.notna(x) else ""
    )
    ht["Value (£)"] = ht["Value (£)"].map(
        lambda x: f"£{x:,.2f}" if pd.notna(x) else ""
    )
    ht["Weight"] = ht["Weight"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    ht["YTD Return"] = ht["YTD Return"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    ht["Vs Portfolio YTD"] = ht["Vs Portfolio YTD"].map(
        lambda x: f"{x:+.2%}" if pd.notna(x) else ""
    )
    ht["Vs Benchmark YTD"] = ht["Vs Benchmark YTD"].map(
        lambda x: f"{x:+.2%}" if pd.notna(x) else ""
    )
    ht["Period Return"] = ht["Period Return"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) else ""
    )
    ht["Vs Portfolio Period"] = ht["Vs Portfolio Period"].map(
        lambda x: f"{x:+.2%}" if pd.notna(x) else ""
    )
    ht["Vs Benchmark Period"] = ht["Vs Benchmark Period"].map(
        lambda x: f"{x:+.2%}" if pd.notna(x) else ""
    )

    hold_data = [ht.columns.tolist()] + ht.values.tolist()
    hold_table_pdf = Table(
        hold_data,
        hAlign="LEFT",
        repeatRows=1,
    )
    hold_table_pdf.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ])
    )
    elements.append(hold_table_pdf)
    elements.append(Spacer(1, 6))

    elements.append(
        Paragraph(
            f"Total Portfolio Value: £{total_val:,.2f}", styles["Normal"]
        )
    )
    elements.append(
        Paragraph(
            f"Portfolio YTD: {port_ytd:.2%} | {benchmark} YTD: {bench_ytd:.2%} | "
            f"Portfolio Period: {port_period:.2%} | {benchmark} Period: {bench_period:.2%}",
            styles["Normal"],
        )
    )

    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================================
# SCENARIO & MONTE CARLO
# ============================================================================


def run_weight_scenario(asset_prices_window: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Build hypothetical constant-weight portfolio."""
    asset_rets = asset_prices_window.pct_change().dropna()
    w = weights.reindex(asset_rets.columns).fillna(0.0)
    if w.sum() == 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    scen_ret = (asset_rets * w).sum(axis=1)
    return scen_ret


def run_monte_carlo(
    asset_prices: pd.DataFrame,
    weight_series: pd.Series,
    years: int = 10,
    paths: int = 500,
    seed: int | None = 123,
    starting_value: float = 100_000.0,
) -> pd.DataFrame:
    """Historical Monte Carlo simulation (original method)."""
    cols = [c for c in weight_series.index if c in asset_prices.columns]
    if not cols:
        return pd.DataFrame()

    asset_rets = asset_prices[cols].pct_change().dropna()
    if asset_rets.empty:
        return pd.DataFrame()

    w_raw = weight_series.reindex(cols).fillna(0.0)
    if w_raw.sum() == 0:
        return pd.DataFrame()
    w = (w_raw / w_raw.sum()).values

    mu_daily = asset_rets.mean().values
    cov_daily = asset_rets.cov().values
    if cov_daily.size == 0:
        return pd.DataFrame()

    n_steps = int(252 * years)
    rng = np.random.default_rng(seed)

    paths_arr = np.zeros((n_steps + 1, paths), dtype=float)
    paths_arr[0, :] = starting_value

    for t in range(1, n_steps + 1):
        rets = rng.multivariate_normal(mu_daily, cov_daily, size=paths)
        port_rets = rets @ w
        paths_arr[t] = paths_arr[t - 1] * (1.0 + port_rets)

    idx = pd.date_range(date.today() + timedelta(days=1),
                        periods=n_steps + 1, freq="B")
    df_paths = pd.DataFrame(paths_arr, index=idx)

    percentiles = df_paths.quantile([0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    percentiles.columns = ["P10", "P25", "P50", "P75", "P90"]
    return percentiles


def run_monte_carlo_mean_reverting(
    asset_prices: pd.DataFrame,
    weight_series: pd.Series,
    years: int = 10,
    paths: int = 500,
    seed: int | None = 123,
    starting_value: float = 100_000.0,
) -> pd.DataFrame:
    """
    Mean-reverting GBM Monte Carlo (experimental).

    Steps:
    - Compute historical portfolio returns from current weights.
    - Convert to log-returns and fit AR(1): r_t = a + phi * r_{t-1} + eps_t.
    - Rewrite as mean-reverting form with long-run mean mu and parameter phi.
    - Simulate future log-returns as Ornstein–Uhlenbeck in log-space and
      compound to portfolio values.
    """
    cols = [c for c in weight_series.index if c in asset_prices.columns]
    if not cols:
        return pd.DataFrame()

    asset_rets = asset_prices[cols].pct_change().dropna()
    if asset_rets.empty:
        return pd.DataFrame()

    w_raw = weight_series.reindex(cols).fillna(0.0)
    if w_raw.sum() == 0:
        return pd.DataFrame()
    w = (w_raw / w_raw.sum())

    # Historical portfolio return series
    port_rets = (asset_rets[w.index] * w.values).sum(axis=1)
    if port_rets.empty:
        return pd.DataFrame()

    log_rets = np.log1p(port_rets.dropna())
    if len(log_rets) < 5:
        # Not enough data to fit AR(1) reliably; fall back to simple GBM-like
        mu = log_rets.mean()
        sigma = log_rets.std(ddof=1)
        if not np.isfinite(sigma) or sigma <= 0:
            return pd.DataFrame()
        phi = 0.0
        sigma_eps = sigma
    else:
        x = log_rets.values
        x_lag = x[:-1]
        x_t = x[1:]
        x_lag_mean = x_lag.mean()
        denom = ((x_lag - x_lag_mean) ** 2).sum()
        if denom == 0:
            phi = 0.0
        else:
            phi = ((x_lag - x_lag_mean) * (x_t - x_t.mean())).sum() / denom
        # Clamp phi to a reasonable range to avoid instability
        phi = float(np.clip(phi, -0.99, 0.99))
        # Intercept a and implied long-run mean mu
        a = x_t.mean() - phi * x_lag_mean
        if abs(1.0 - phi) < 1e-6:
            mu = log_rets.mean()
        else:
            mu = a / (1.0 - phi)
        # Residual std dev
        eps = x_t - (a + phi * x_lag)
        sigma_eps = eps.std(ddof=1)
        if not np.isfinite(sigma_eps) or sigma_eps <= 0:
            sigma_eps = log_rets.std(ddof=1)
        if not np.isfinite(sigma_eps) or sigma_eps <= 0:
            return pd.DataFrame()

    n_steps = int(252 * years)
    rng = np.random.default_rng(seed)

    paths_arr = np.zeros((n_steps + 1, paths), dtype=float)
    paths_arr[0, :] = starting_value

    # Start each path from last observed log-return (or mu if missing)
    last_r = float(log_rets.iloc[-1]) if len(log_rets) > 0 else float(mu)
    cur_r = np.full(paths, last_r, dtype=float)

    for t in range(1, n_steps + 1):
        eps = rng.normal(0.0, sigma_eps, size=paths)
        # Mean-reverting step: r_{t+1} = mu + phi * (r_t - mu) + eps
        cur_r = mu + phi * (cur_r - mu) + eps
        paths_arr[t] = paths_arr[t - 1] * np.exp(cur_r)

    idx = pd.date_range(date.today() + timedelta(days=1),
                        periods=n_steps + 1, freq="B")
    df_paths = pd.DataFrame(paths_arr, index=idx)

    percentiles = df_paths.quantile([0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    percentiles.columns = ["P10", "P25", "P50", "P75", "P90"]
    return percentiles

# ===========================================================================
# MAIN APP
# ============================================================================


def main():
    st.title("📊 Portfolio Analyzer 2.19 - based on Yahoo Data")
    st.warning(
        "⚠️ **IMPORTANT**: Please ensure all prices in your CSV are in **major currency units** "
        "(£ or $), **NOT** in pence/cents (p or ¢).\n\n"
        "✅ Correct: £147.92, $98.50\n"
        "❌ Wrong: 14792p, 9850¢\n\n"
        "The app will automatically detect and convert all prices to pounds (£) for accurate calculations."
    )

    # ---- SIDEBAR: LOAD PORTFOLIOS ----
    st.sidebar.header("📁 Portfolio Management")

    uploaded_files = st.sidebar.file_uploader(
        "Upload portfolio CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload CSV files with columns: date, ticker, shares, price, total_cost",
    )

    if not uploaded_files:
        st.info("👈 Please upload at least one portfolio CSV file to get started.")
        return

    portfolios: dict[str, pd.DataFrame] = {}
    portfolio_names: list[str] = []
    conversion_factors_all: dict[str, dict] = {}
    yahoo_names_all: dict[str, str] = {}

    for file in uploaded_files:
        portfolio_name = file.name.replace(".csv", "")
        try:
            df, conversion_factors, yahoo_names = load_transactions(file)
            portfolios[portfolio_name] = df
            conversion_factors_all[portfolio_name] = conversion_factors
            yahoo_names_all.update(yahoo_names)
            portfolio_names.append(portfolio_name)
        except Exception as e:
            st.sidebar.error(f"Error loading {file.name}: {e}")

    if not portfolios:
        st.error("No valid portfolios loaded.")
        return

    # ---- SIDEBAR: SELECT PORTFOLIOS & SETTINGS ----
    st.sidebar.subheader("Select Portfolios")
    selected_portfolios = st.sidebar.multiselect(
        "Choose portfolios to analyze:",
        portfolio_names,
        default=portfolio_names,
        key="portfolio_selector",
    )

    if not selected_portfolios:
        st.warning("Please select at least one portfolio.")
        return

    st.sidebar.subheader("⚙️ Settings")

    benchmark = st.sidebar.selectbox(
        "Benchmark Index",
        ["^GSPC", "^FTSE", "^STOXX50E"],
        index=0,
        help="Market index to compare against",
    )

    today = datetime.today().date()
    default_start = today - timedelta(days=365)

    # New: date-range presets
    date_preset = st.sidebar.selectbox(
        "Date range",
        ["Custom", "Last 3 months", "Last 6 months",
            "YTD", "Last 12 months", "Last 24 months"],
        index=4,
        help="Choose a preset or use Custom with the Start Date below",
    )

    start_date_input = st.sidebar.date_input("Start Date", default_start)

    # v2.17: End date input (only for Custom)
    if date_preset == "Custom":
        end_date_input = st.sidebar.date_input("End Date", today)
    else:
        end_date_input = today

    # Resolve effective requested start_date and end_date before applying earliest-tx clamp
    if date_preset == "Custom":
        start_date = start_date_input
        end_date = end_date_input
        if end_date <= start_date:
            st.error("❌ End date must be later than start date")
            return
    else:
        end_date = today
        if date_preset == "Last 3 months":
            start_date = today - timedelta(days=90)
        elif date_preset == "Last 6 months":
            start_date = today - timedelta(days=180)
        elif date_preset == "YTD":
            start_date = date(today.year, 1, 1)
        elif date_preset == "Last 12 months":
            start_date = today - timedelta(days=365)
        elif date_preset == "Last 24 months":
            start_date = today - timedelta(days=730)
        else:
            start_date = start_date_input  # Fallback

    rf_annual = st.sidebar.number_input(
        "Risk-Free Rate (annual, %)", value=4.0, step=0.25
    ) / 100.0

    rf_upload = st.sidebar.file_uploader(
        "Risk-free curve CSV (optional)\nColumns: date, rate[%]",
        type=["csv"],
    )

    # =========================================================================
    # PROCESS ALL SELECTED PORTFOLIOS
    # =========================================================================
    portfolio_data: dict[str, dict] = {}
    all_tickers: set[str] = set()
    min_start_date: date | None = None
    merged_conversion_factors: dict[str, float] = {}

    for pname in selected_portfolios:
        tickers = portfolios[pname]["ticker"].unique()
        all_tickers.update(tickers)
        merged_conversion_factors.update(conversion_factors_all[pname])

        min_date = portfolios[pname]["date"].min().date()
        if min_start_date is None or min_date < min_start_date:
            min_start_date = min_date

    start_all = min_start_date
    start_effective = max(start_date, start_all)
    end_effective = end_date

    with st.spinner("Downloading and rebasing price data..."):
        try:
            price_df = get_price_history(
                list(all_tickers),
                benchmark,
                start_all,
                end_effective,
                merged_conversion_factors,
            )
            if price_df.empty:
                st.error("No price data returned.")
                return

            issues = getattr(price_df, "_yahoo_issues", []) or []
            missing_ranges = getattr(price_df, "_missing_ranges", []) or []
            warn_lines = []

            # Existing issues from fetch_yahoo_prices
            for item in issues:
                sym = item.get("symbol", "?")
                prob = item.get("problem", "Unknown issue")
                warn_lines.append(f"- {sym}: {prob}")

            # Explicit missing-date ranges per symbol (v2.12)
            for m in missing_ranges:
                sym = m.get("symbol", "?")
                start = m.get("start", "?")
                end_str = m.get("end", "?")
                if m.get("type") == "leading_nan":
                    used_from = m.get("used_price_from")
                    if used_from:
                        warn_lines.append(
                            f"❌ {sym}: missing prices from {start} to {end_str}; used price from {used_from} to backfill missing period"
                        )
                    else:
                        warn_lines.append(
                            f"❌ {sym}: missing prices from {start} to {end_str}"
                        )
                elif m.get("type") == "no_data":
                    warn_lines.append(
                        f"❌ {sym}: no usable Yahoo price history between {start} and {end_str}"
                    )

            if warn_lines:
                st.warning(
                    "⚠️ Yahoo Finance did not return full historical data for some symbols.\n\n"
                    "This may understate their contribution to portfolio returns:\n\n"
                    + "\n".join(warn_lines)
                )

            if benchmark not in price_df.columns:
                st.error(f"Benchmark {benchmark} not found.")
                return

        except Exception as e:
            st.error(f"Error downloading prices: {e}")
            return

    bench_prices = price_df[benchmark]
    asset_prices = price_df.drop(columns=[benchmark], errors="ignore")

    mask_bench = (bench_prices.index.date >= start_effective) & (
        bench_prices.index.date <= end_effective
    )
    bench_prices_window = bench_prices[mask_bench]
    bench_ret = bench_prices_window.pct_change().dropna()

    # Align benchmark cumulative to window start (baseline 0 at start_effective)
    bench_cum_all = (1 + bench_prices.pct_change().dropna()).cumprod() - 1
    bench_cum = bench_cum_all[bench_ret.index]

    rf_daily_series_all = load_risk_free_series(rf_upload, bench_ret.index)
    if rf_daily_series_all is not None:
        bench_rf_daily_used = (
            rf_daily_series_all.reindex(bench_ret.index)
            .ffill()
            .fillna(rf_annual / 252.0)
        )
    else:
        bench_rf_daily_used = pd.Series(
            rf_annual / 252.0, index=bench_ret.index
        )

    bench_metrics = compute_risk_metrics(bench_ret, bench_rf_daily_used)

    for pname in selected_portfolios:
        tx_all = portfolios[pname][portfolios[pname]
                                   ["date"].dt.date <= end_effective].copy()
        holdings_all = build_holdings(tx_all, asset_prices)
        twr_daily_all, twr_cum_all = compute_twr_from_prices(
            asset_prices, holdings_all
        )

        mask = (twr_daily_all.index.date >= start_effective) & (
            twr_daily_all.index.date <= end_effective
        )
        if not mask.any():
            st.error(
                f"No data for portfolio {pname} on or after {start_effective}."
            )
            continue

        twr_daily = twr_daily_all[mask]
        twr_cum = twr_cum_all[mask]  # v2.10: preserve cumulative baseline

        if rf_daily_series_all is not None:
            rf_daily_used = (
                rf_daily_series_all.reindex(twr_daily.index)
                .ffill()
                .fillna(rf_annual / 252.0)
            )
        else:
            rf_daily_used = pd.Series(
                rf_annual / 252.0, index=twr_daily.index
            )

        metrics = compute_risk_metrics(twr_daily, rf_daily_used)

        portfolio_data[pname] = {
            "holdings": holdings_all,
            "values": holdings_all * asset_prices,
            "asset_prices": asset_prices,
            "tickers": tx_all["ticker"].unique(),
            "twr_daily_all": twr_daily_all,
            "twr_cum_all": twr_cum_all,
            "twr_daily": twr_daily,
            "twr_cum": twr_cum,
            "metrics": metrics,
            "rf_daily_used": rf_daily_used,
            "tx": tx_all,
        }

    if not portfolio_data:
        st.error("Could not process any portfolios.")
        return

    # =========================================================================
    # PDF REPORT (SIDEBAR)
    # =========================================================================
    st.sidebar.subheader("📄 PDF Report")
    if not REPORTLAB_AVAILABLE:
        st.sidebar.warning(
            "PDF export is disabled because 'reportlab' is not installed. "
            "Install it in your environment to enable PDF reports."
        )
    else:
        # Add "All" option to the dropdown
        report_options = ["All"] + selected_portfolios
        report_portfolio = st.sidebar.selectbox(
            "Portfolio for report:",
            report_options,
            key="report_portfolio_selector",
        )

        if report_portfolio == "All":
            # Generate combined PDF for all portfolios
            try:
                from reportlab.platypus import PageBreak
                from reportlab.lib.pagesizes import landscape
                from reportlab.lib.units import inch
                from io import BytesIO

                buffer = BytesIO()
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=landscape(A4),
                    topMargin=36,
                    bottomMargin=36,
                    leftMargin=36,
                    rightMargin=36,
                )
                all_elements = []

                for idx, pname in enumerate(selected_portfolios):
                    if pname in portfolio_data:
                        # Generate content for this portfolio
                        styles = getSampleStyleSheet()

                        # Title for this portfolio
                        all_elements.append(
                            Paragraph(
                                f"Portfolio Report: {pname}", styles["Heading1"]
                            )
                        )
                        all_elements.append(
                            Paragraph(
                                f"Period: {start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}",
                                styles["Normal"],
                            )
                        )
                        all_elements.append(Spacer(1, 12))

                        # Get portfolio data
                        pdata = portfolio_data[pname]

                        # 1) Cumulative Return Chart
                        base_cum = pdata["twr_cum"]
                        common_idx = base_cum.index.intersection(
                            bench_cum.index)
                        if not common_idx.empty:
                            base_series = base_cum.loc[common_idx]
                            bench_series = bench_cum.loc[common_idx]
                            base_series = base_series - base_series.iloc[0]
                            bench_series = bench_series - bench_series.iloc[0]
                            base_pct = base_series * 100.0
                            bench_pct = bench_series * 100.0

                            fig, ax = plt.subplots(figsize=(7.5, 4))
                            ax.plot(common_idx, base_pct, label=pname)
                            ax.plot(common_idx, bench_pct,
                                    label=f"Benchmark ({benchmark})")
                            chart_title = (
                                f"{pname} - Cumulative returns from "
                                f"{start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}"
                            )
                            ax.set_title(chart_title)
                            ax.set_ylabel("Cumulative Return (%)")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            fig.autofmt_xdate()

                            img_buf = BytesIO()
                            fig.savefig(img_buf, format="png",
                                        dpi=150, bbox_inches="tight")
                            plt.close(fig)
                            img_buf.seek(0)

                            max_width = doc.width * 0.9
                            max_height = 3.5 * inch
                            img = Image(img_buf, width=max_width,
                                        height=max_height)
                            all_elements.append(img)
                            all_elements.append(Spacer(1, 12))

                        # 2) Key Metrics Table
                        all_elements.append(
                            Paragraph("Key Metrics", styles["Heading2"]))
                        m = pdata["metrics"]
                        metrics_rows = [
                            {
                                "Series": pname,
                                "Total Return": m["total_return"],
                                "Volatility (annual)": m["vol_annual"],
                                "Sharpe": m["sharpe"],
                                "Sortino": m["sortino"],
                                "Downside Dev": m["downside_dev"],
                                "Max Drawdown": m["max_drawdown"],
                            },
                            {
                                "Series": f"Benchmark ({benchmark})",
                                "Total Return": bench_metrics["total_return"],
                                "Volatility (annual)": bench_metrics["vol_annual"],
                                "Sharpe": bench_metrics["sharpe"],
                                "Sortino": bench_metrics["sortino"],
                                "Downside Dev": bench_metrics["downside_dev"],
                                "Max Drawdown": bench_metrics["max_drawdown"],
                            },
                        ]

                        metrics_df = pd.DataFrame(metrics_rows)
                        fmt_df = metrics_df.copy()
                        fmt_df["Total Return"] = fmt_df["Total Return"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                        fmt_df["Volatility (annual)"] = fmt_df["Volatility (annual)"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                        fmt_df["Sharpe"] = fmt_df["Sharpe"].map(
                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                        )
                        fmt_df["Sortino"] = fmt_df["Sortino"].map(
                            lambda x: f"{x:.2f}" if pd.notna(x) else ""
                        )
                        fmt_df["Downside Dev"] = fmt_df["Downside Dev"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                        fmt_df["Max Drawdown"] = fmt_df["Max Drawdown"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )

                        metrics_table_data = [
                            fmt_df.columns.tolist()] + fmt_df.values.tolist()
                        metrics_table = Table(
                            metrics_table_data, hAlign="LEFT")
                        metrics_table.setStyle(
                            TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 8),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                            ])
                        )
                        all_elements.append(metrics_table)
                        all_elements.append(Spacer(1, 6))

                        metrics_note = (
                            f"Metrics are calculated for the period from "
                            f"{start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}. "
                            f"Volatility, Sharpe, and Sortino ratios are annualized, i.e. multiplied by √252 trading days."
                        )
                        all_elements.append(
                            Paragraph(metrics_note, styles["Normal"]))
                        all_elements.append(Spacer(1, 12))

                        # 3) Holdings Overview
                        all_elements.append(
                            Paragraph("Holdings Overview", styles["Heading2"]))
                        hold_table, total_val, port_ytd, bench_ytd, port_period, bench_period = build_holdings_overview_full(
                            pdata, benchmark, bench_prices, yahoo_names_all,
                            start_effective, end_effective
                        )

                        ht = hold_table.copy()
                        ht["Shares"] = ht["Shares"].map(format_shares)
                        ht["Cost (£)"] = ht["Cost (£)"].map(
                            lambda x: f"£{x:,.2f}" if pd.notna(x) else ""
                        )
                        ht["Value (£)"] = ht["Value (£)"].map(
                            lambda x: f"£{x:,.2f}" if pd.notna(x) else ""
                        )
                        ht["Weight"] = ht["Weight"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                        ht["YTD Return"] = ht["YTD Return"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                        ht["Vs Portfolio YTD"] = ht["Vs Portfolio YTD"].map(
                            lambda x: f"{x:+.2%}" if pd.notna(x) else ""
                        )
                        ht["Vs Benchmark YTD"] = ht["Vs Benchmark YTD"].map(
                            lambda x: f"{x:+.2%}" if pd.notna(x) else ""
                        )
                        ht["Period Return"] = ht["Period Return"].map(
                            lambda x: f"{x:.2%}" if pd.notna(x) else ""
                        )
                        ht["Vs Portfolio Period"] = ht["Vs Portfolio Period"].map(
                            lambda x: f"{x:+.2%}" if pd.notna(x) else ""
                        )
                        ht["Vs Benchmark Period"] = ht["Vs Benchmark Period"].map(
                            lambda x: f"{x:+.2%}" if pd.notna(x) else ""
                        )

                        hold_data = [ht.columns.tolist()] + ht.values.tolist()
                        hold_table_pdf = Table(
                            hold_data, hAlign="LEFT", repeatRows=1)
                        hold_table_pdf.setStyle(
                            TableStyle([
                                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, -1), 6),
                                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                            ])
                        )
                        all_elements.append(hold_table_pdf)
                        all_elements.append(Spacer(1, 6))
                        all_elements.append(
                            Paragraph(
                                f"Total Portfolio Value: £{total_val:,.2f}", styles["Normal"])
                        )
                        all_elements.append(
                            Paragraph(
                                f"Portfolio YTD: {port_ytd:.2%} | {benchmark} YTD: {bench_ytd:.2%}",
                                styles["Normal"],
                            )
                        )

                        # Add page break between portfolios (except after the last one)
                        if idx < len(selected_portfolios) - 1:
                            all_elements.append(PageBreak())

                # Build the combined PDF
                doc.build(all_elements)
                buffer.seek(0)
                pdf_bytes = buffer.getvalue()

                st.sidebar.download_button(
                    "⬇️ Download PDF report (All Portfolios)",
                    data=pdf_bytes,
                    file_name=f"All_Portfolios_report_v2.19.pdf",
                    mime="application/pdf",
                    key="download_pdf_report",
                )
            except Exception as e:
                st.sidebar.error(f"Error generating combined PDF report: {e}")

        elif report_portfolio and report_portfolio in portfolio_data:
            # Generate single portfolio PDF (existing code)
            try:
                pdf_bytes = generate_pdf_report(
                    report_portfolio,
                    portfolio_data[report_portfolio],
                    benchmark,
                    bench_prices,
                    bench_metrics,
                    yahoo_names_all,
                    bench_cum,
                )
                st.sidebar.download_button(
                    "⬇️ Download PDF report",
                    data=pdf_bytes,
                    file_name=f"{report_portfolio}_report_v2.19.pdf",
                    mime="application/pdf",
                    key="download_pdf_report",
                )
            except Exception as e:
                st.sidebar.error(f"Error generating PDF report: {e}")

    # =========================================================================
    # CSV EXPORT (SIDEBAR)
    # =========================================================================
    st.sidebar.subheader("📊 Export CSV tables")
    try:
        # Build Key Metrics table
        metrics_rows = []
        for pname in selected_portfolios:
            if pname not in portfolio_data:
                continue
            m = portfolio_data[pname]["metrics"]
            metrics_rows.append({
                "Portfolio": pname,
                "Total Return": m["total_return"],
                "Volatility (annual)": m["vol_annual"],
                "Sharpe": m["sharpe"],
                "Sortino": m["sortino"],
                "Downside Dev": m["downside_dev"],
                "Max Drawdown": m["max_drawdown"],
            })

        # Add benchmark metrics
        metrics_rows.append({
            "Portfolio": f"Benchmark ({benchmark})",
            "Total Return": bench_metrics["total_return"],
            "Volatility (annual)": bench_metrics["vol_annual"],
            "Sharpe": bench_metrics["sharpe"],
            "Sortino": bench_metrics["sortino"],
            "Downside Dev": bench_metrics["downside_dev"],
            "Max Drawdown": bench_metrics["max_drawdown"],
        })

        metrics_df = pd.DataFrame(metrics_rows)

        # Build complete export CSV
        export_lines = []

        # Add KEY METRICS section
        export_lines.append(",KEY METRICS,,,,,,,,")
        metrics_csv = metrics_df.to_csv(index=True, lineterminator='\r\n')
        export_lines.append(metrics_csv.strip())
        export_lines.append(",,,,,,,,,")

        # Add Holdings Overview for each portfolio
        for pname in selected_portfolios:
            if pname not in portfolio_data:
                continue

            pdata = portfolio_data[pname]

            # Get holdings overview (raw values, not formatted)
            hold_table, total_val, port_ytd, bench_ytd, port_period, bench_period = build_holdings_overview_full(
                pdata,
                benchmark,
                bench_prices,
                yahoo_names_all,
                start_effective,
                end_effective,
            )

            # Add section header
            export_lines.append(f",Holdings Overview - {pname},,,,,,,,")

            # Add holdings table
            holdings_csv = hold_table.to_csv(index=True, lineterminator='\r\n')
            export_lines.append(holdings_csv.strip())
            export_lines.append(",,,,,,,,,")

        # Combine all lines
        full_csv = '\r\n'.join(export_lines)

        st.sidebar.download_button(
            "⬇️ Download CSV Export",
            data=full_csv,
            file_name="export_tables_2.19.csv",
            mime="text/csv",
            key="download_csv_export",
        )
    except Exception as e:
        st.sidebar.error(f"Error generating CSV export: {e}")

    # =========================================================================
    # EXPORT ALL CHARTS (SIDEBAR)  (v2.19)
    # =========================================================================
    st.sidebar.subheader("Export all Charts")
    try:
        import zipfile

        zip_buf = BytesIO()

        with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pname in selected_portfolios:
                if pname not in portfolio_data:
                    continue

                pdata = portfolio_data[pname]
                base_cum = pdata["twr_cum"]
                common_idx = base_cum.index.intersection(bench_cum.index)

                if common_idx.empty:
                    continue

                base_series = base_cum.loc[common_idx]
                bench_series = bench_cum.loc[common_idx]

                # Keep identical baseline treatment to the PDF chart
                base_series = base_series - base_series.iloc[0]
                bench_series = bench_series - bench_series.iloc[0]

                base_pct = base_series * 100.0
                bench_pct = bench_series * 100.0

                fig, ax = plt.subplots(figsize=(7.5, 4))
                ax.plot(common_idx, base_pct, label=pname)
                ax.plot(common_idx, bench_pct,
                        label=f"Benchmark ({benchmark})")

                chart_title = (
                    f"{pname} - Cumulative returns from "
                    f"{start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}"
                )
                ax.set_title(chart_title)

                ax.set_ylabel("Cumulative Return (%)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.autofmt_xdate()

                img_buf = BytesIO()
                fig.savefig(img_buf, format="png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
                img_buf.seek(0)

                # “CISA_LW.csv -> CISA_LW.png”
                zf.writestr(f"{pname}.png", img_buf.read())

        zip_buf.seek(0)

        if zip_buf.getbuffer().nbytes > 0:
            st.sidebar.download_button(
                "⬇️ Download Charts (ZIP)",
                data=zip_buf.getvalue(),
                file_name="portfolio_charts.zip",
                mime="application/zip",
                key="download_all_charts_zip",
            )
        else:
            st.sidebar.info(
                "No charts available to export for the current selection.")
    except Exception as e:
        st.sidebar.error(f"Error generating charts export: {e}")

    # =========================================================================
    # TABS LAYOUT
    # =========================================================================
    tabs = st.tabs(
        [
            "📈 Cumulative Return",
            "📊 Risk Analysis",
            "💰 Key Metrics",
            "🎯 Allocations",
            "📋 Holdings",
            "🔮 Scenarios",
            "🎲 Monte Carlo",
        ]
    )

    # ---- TAB 1: CUMULATIVE RETURN ----
    with tabs[0]:
        st.subheader(f"Cumulative Return vs Benchmark ({benchmark})")
        selected_for_cum = st.multiselect(
            "Select portfolios to display:",
            selected_portfolios,
            default=selected_portfolios,
            key="cum_selector",
        )

        if selected_for_cum:
            cum_data = {}
            show_bench = st.checkbox(
                f"Show Benchmark Index ({benchmark})", True, key="main_bench"
            )

            if show_bench:
                bench_series = bench_cum.copy()
                if not bench_series.empty:
                    bench_series = bench_series - bench_series.iloc[0]
                cum_data[f"Benchmark ({benchmark})"] = bench_series

            for pname in selected_for_cum:
                if pname in portfolio_data:
                    twr_series = portfolio_data[pname]["twr_cum"]
                    if not twr_series.empty:
                        twr_series = twr_series - twr_series.iloc[0]
                        cum_data[f"{pname} (TWR)"] = twr_series

            if cum_data:
                st.line_chart(pd.DataFrame(cum_data))
            else:
                st.info("Select at least one series to display.")
        else:
            st.info("Select at least one portfolio to display.")

    # ---- TAB 2: RISK ----
    with tabs[1]:
        st.subheader("Risk (Annualized Volatility)")
        selected_for_risk = st.multiselect(
            "Select portfolios to display:",
            selected_portfolios,
            default=selected_portfolios,
            key="risk_selector",
        )

        if selected_for_risk:
            risk_data = {
                "Series": selected_for_risk + [f"Benchmark ({benchmark})"],
                "Volatility": [
                    portfolio_data[pname]["metrics"]["vol_annual"]
                    for pname in selected_for_risk
                ]
                + [bench_metrics["vol_annual"]],
            }
            risk_df = pd.DataFrame(risk_data)
            st.plotly_chart(
                px.bar(risk_df, x="Series", y="Volatility", text_auto=".2%"),
                use_container_width=True,
            )
        else:
            st.info("Select at least one portfolio to display.")

    # ---- TAB 3: KEY METRICS ----
    with tabs[2]:
        st.subheader(f"Key Risk Metrics (vs {benchmark} Benchmark)")

        metrics_rows = []
        for pname in selected_portfolios:
            if pname not in portfolio_data:
                continue
            m = portfolio_data[pname]["metrics"]
            metrics_rows.append(
                {
                    "Portfolio": pname,
                    "Total Return": m["total_return"],
                    "Volatility (annual)": m["vol_annual"],
                    "Sharpe": m["sharpe"],
                    "Sortino": m["sortino"],
                    "Downside Dev": m["downside_dev"],
                    "Max Drawdown": m["max_drawdown"],
                }
            )

        metrics_rows.append(
            {
                "Portfolio": f"Benchmark ({benchmark})",
                "Total Return": bench_metrics["total_return"],
                "Volatility (annual)": bench_metrics["vol_annual"],
                "Sharpe": bench_metrics["sharpe"],
                "Sortino": bench_metrics["sortino"],
                "Downside Dev": bench_metrics["downside_dev"],
                "Max Drawdown": bench_metrics["max_drawdown"],
            }
        )

        metrics_display_df = pd.DataFrame(metrics_rows)
        styled_metrics = metrics_display_df.style.format(
            {
                "Total Return": "{:.2%}",
                "Volatility (annual)": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Downside Dev": "{:.2%}",
                "Max Drawdown": "{:.2%}",
            }
        )
        st.dataframe(styled_metrics, use_container_width=True)

        st.caption(
            f"Metrics for period: {start_effective.strftime('%d/%m/%Y')} to {end_effective.strftime('%d/%m/%Y')}. "
            f"Volatility, Sharpe, and Sortino annualized using √252 trading days."
        )

        csv = metrics_display_df.to_csv(index=False)
        st.download_button(
            "📥 Download Metrics CSV",
            csv,
            "portfolio_metrics.csv",
            "text/csv",
            key="download_metrics",
        )

    # ---- TAB 4: ALLOCATIONS ----
    with tabs[3]:
        st.subheader("Current Allocation (by Market Value)")
        selected_for_alloc = st.multiselect(
            "Select portfolios to display:",
            selected_portfolios,
            default=selected_portfolios,
            key="alloc_selector",
        )

        if selected_for_alloc:
            alloc_tabs = st.tabs(selected_for_alloc)
            for idx, pname in enumerate(selected_for_alloc):
                with alloc_tabs[idx]:
                    if pname not in portfolio_data:
                        st.write("No data for this portfolio.")
                        continue

                    latest_values = portfolio_data[pname]["values"].iloc[-1]
                    pie_df = latest_values[latest_values > 0].reset_index()
                    pie_df.columns = ["Ticker", "Value"]

                    # Build legend/label as: TICKER (Name from Yahoo)
                    pie_df["Label"] = pie_df["Ticker"].map(
                        lambda t: f"{t} ({yahoo_names_all.get(t, t)})"
                    )

                    if not pie_df.empty:
                        st.plotly_chart(
                            px.pie(
                                pie_df,
                                names="Label",   # use combined label in legend
                                values="Value",
                                title=pname,
                            ),
                            use_container_width=True,
                        )

                    else:
                        st.write("No current holdings to display.")
        else:
            st.info("Select at least one portfolio to display.")

    # ---- TAB 5: HOLDINGS ----
    with tabs[4]:
        st.subheader("Holdings Overview")
        selected_holding = st.selectbox(
            "Select portfolio:",
            selected_portfolios,
            key="holdings_selector",
        )

        if selected_holding:
            pdata = portfolio_data[selected_holding]
            latest_hold = pdata["holdings"].iloc[-1]
            latest_values = pdata["values"].iloc[-1]
            total_val = latest_values.sum()
            weights_now = (
                latest_values / total_val if total_val > 0 else latest_values * 0
            )

            year_start_date = datetime(date.today().year, 1, 1).date()
            twr_ytd_mask = pdata["twr_daily_all"].index.date >= year_start_date
            if twr_ytd_mask.any():
                port_ytd_ret = pdata["twr_daily_all"][twr_ytd_mask]
                port_ytd = (1.0 + port_ytd_ret).prod() - 1.0
            else:
                port_ytd = 0.0

            bench_ytd_mask = bench_prices.index.date >= year_start_date
            if bench_ytd_mask.any():
                bench_ytd_prices = bench_prices[bench_ytd_mask]
                if len(bench_ytd_prices) > 1:
                    bench_ytd = (
                        bench_ytd_prices.iloc[-1] / bench_ytd_prices.iloc[0]
                    ) - 1.0
                else:
                    bench_ytd = 0.0
            else:
                bench_ytd = 0.0

            ytd_prices = pdata["asset_prices"][
                pdata["asset_prices"].index.date >= year_start_date
            ]
            if not ytd_prices.empty and len(ytd_prices) > 1:
                ytd_asset = (ytd_prices.iloc[-1] / ytd_prices.iloc[0]) - 1.0
            else:
                ytd_asset = pd.Series(0.0, index=latest_hold.index)

            # v2.17: Period return calculations
            twr_period_mask = (pdata["twr_daily_all"].index.date >= start_effective) & (
                pdata["twr_daily_all"].index.date <= end_effective
            )
            if twr_period_mask.any():
                port_period_ret = pdata["twr_daily_all"][twr_period_mask]
                port_period = (1.0 + port_period_ret).prod() - 1.0
            else:
                port_period = 0.0

            bench_period_mask = (bench_prices.index.date >= start_effective) & (
                bench_prices.index.date <= end_effective
            )
            if bench_period_mask.any():
                bench_period_prices = bench_prices[bench_period_mask]
                if len(bench_period_prices) > 1:
                    bench_period = (
                        bench_period_prices.iloc[-1] / bench_period_prices.iloc[0]) - 1.0
                else:
                    bench_period = 0.0
            else:
                bench_period = 0.0

            period_prices = pdata["asset_prices"][
                (pdata["asset_prices"].index.date >= start_effective) &
                (pdata["asset_prices"].index.date <= end_effective)
            ]
            if not period_prices.empty and len(period_prices) > 1:
                period_asset = (
                    period_prices.iloc[-1] / period_prices.iloc[0]) - 1.0
            else:
                period_asset = pd.Series(0.0, index=latest_hold.index)

            tickers_idx = latest_hold.index
            names_series = pd.Index(tickers_idx).map(
                lambda t: yahoo_names_all.get(t, t)
            )

            # Cumulative cost per ticker from CSV (informational only)
            tx_all = pdata["tx"]  # original transactions for this portfolio
            cost_by_ticker = tx_all.groupby("ticker")["total_cost"].sum()
            cost_series = cost_by_ticker.reindex(tickers_idx).fillna(0.0)

            hold_table = pd.DataFrame(
                {
                    "Ticker": tickers_idx,
                    "Name": names_series,
                    "Shares": latest_hold.values,
                    "Cost (£)": cost_series.values,
                    "Value (£)": latest_values.values,
                    "Weight": weights_now.values,
                    "YTD Return": ytd_asset.reindex(tickers_idx).values,
                    "Vs Portfolio YTD": (
                        ytd_asset.reindex(tickers_idx) - port_ytd
                    ).values,
                    "Vs Benchmark YTD": (
                        ytd_asset.reindex(tickers_idx) - bench_ytd
                    ).values,
                    "Period Return": period_asset.reindex(tickers_idx).values,
                    "Vs Portfolio Period": (
                        period_asset.reindex(tickers_idx) - port_period
                    ).values,
                    "Vs Benchmark Period": (
                        period_asset.reindex(tickers_idx) - bench_period
                    ).values,
                }
            )

            hold_table = hold_table[
                (hold_table["Shares"] != 0) | (hold_table["Value (£)"] != 0)
            ]

            hold_table = hold_table.sort_values(
                "Weight", ascending=False
            ).reset_index(drop=True)

            # Privacy masking checkbox (does not change underlying data)
            show_values = st.checkbox(
                "Display Full Portfolio",
                value=False,
                key="show_portfolio_values",
                help="Show actual shares, cost, and value amounts"
            )

            if show_values:
                # Normal numeric formatting
                styled_hold = hold_table.style.format(
                    {
                        "Shares": format_shares,
                        "Cost (£)": "£{:,.2f}",
                        "Value (£)": "£{:,.2f}",
                        "Weight": "{:.2%}",
                        "YTD Return": "{:.2%}",
                        "Vs Portfolio YTD": "{:+.2%}",
                        "Vs Benchmark YTD": "{:+.2%}",
                        "Period Return": "{:.2%}",
                        "Vs Portfolio Period": "{:+.2%}",
                        "Vs Benchmark Period": "{:+.2%}",
                    }
                )
            else:
                # Mask sensitive numeric columns at render-time
                styled_hold = hold_table.style.format(
                    {
                        "Shares": lambda x: "*******" if not pd.isna(x) else "",
                        "Cost (£)": lambda x: "*******" if not pd.isna(x) else "",
                        "Value (£)": lambda x: "*******" if not pd.isna(x) else "",
                        "Weight": "{:.2%}",
                        "YTD Return": "{:.2%}",
                        "Vs Portfolio YTD": "{:+.2%}",
                        "Vs Benchmark YTD": "{:+.2%}",
                        "Period Return": "{:.2%}",
                        "Vs Portfolio Period": "{:+.2%}",
                        "Vs Benchmark Period": "{:+.2%}",
                    }
                )

            st.dataframe(styled_hold, use_container_width=True)

            if show_values:
                total_text = f"**Total Portfolio Value: £{total_val:,.2f}**"
            else:
                total_text = "**Total Portfolio Value: **********"

            st.caption(
                total_text + "\n\n"
                f"**Vs Portfolio YTD**: Holding YTD - Portfolio YTD ({port_ytd:.2%})\n"
                f"**Vs Benchmark YTD**: Holding YTD - {benchmark} YTD ({bench_ytd:.2%})"
            )

    # ---- TAB 6: SCENARIOS ----
    with tabs[5]:
        st.subheader("Scenario Analysis")
        st.info("Test hypothetical portfolio allocations")

        selected_scen = st.selectbox(
            "Select portfolio:",
            selected_portfolios,
            key="scen_selector",
        )

        if selected_scen:
            pdata = portfolio_data[selected_scen]
            latest_values = pdata["values"].iloc[-1]
            total_val = latest_values.sum()
            weights_now = (
                latest_values / total_val if total_val > 0 else latest_values * 0
            )

            hold_table = pd.DataFrame({"Weight": weights_now}).sort_values(
                "Weight", ascending=False
            )
            hold_table = hold_table[hold_table["Weight"] > 0]

            if not hold_table.empty:
                with st.expander("Open scenario sandbox"):
                    default_rows = pd.DataFrame(
                        {
                            "Investment": hold_table.index,
                            "Weight (%)": (hold_table["Weight"] * 100).round(2),
                        }
                    )
                    scen_table = st.data_editor(
                        default_rows, num_rows="dynamic"
                    )

                    scen_table = scen_table[
                        scen_table["Investment"].astype(str).str.strip() != ""
                    ]
                    scen_table = scen_table[scen_table["Weight (%)"] > 0]

                    if not scen_table.empty:
                        scen_w = (
                            scen_table.set_index("Investment")["Weight (%)"]
                            / 100.0
                        )

                        total_w = scen_w.sum()

                        # Hard cap: do not allow allocations to exceed 100%
                        if total_w > 1.001:  # allow tiny FP wiggle
                            st.warning(
                                f"Scenario weights sum to {total_w*100:.2f}%. "
                                "Please reduce them so total allocation does not exceed 100%."
                            )
                        else:
                            # Optional: warn if significantly below 100% (implied cash)
                            if total_w < 0.99:
                                st.info(
                                    f"Scenario weights sum to {total_w*100:.2f}%. "
                                    "Remaining weight is treated as uninvested cash."
                                )

                            scen_ret = run_weight_scenario(
                                pdata["asset_prices"].loc[pdata["twr_daily"].index],
                                scen_w,
                            )

                            if not scen_ret.empty:
                                scen_cum = (1 + scen_ret).cumprod() - 1
                                base_cum = pdata["twr_cum"]

                                # Align on common dates (your fixed version)
                                common_idx = base_cum.index.intersection(
                                    scen_cum.index)
                                base_cum_aligned = base_cum.loc[common_idx]
                                scen_cum_aligned = scen_cum.loc[common_idx]

                                if not common_idx.empty:
                                    base_cum_rebased = base_cum_aligned - \
                                        base_cum_aligned.iloc[0]
                                    scen_cum_rebased = scen_cum_aligned - \
                                        scen_cum_aligned.iloc[0]

                                    chart_df = pd.DataFrame(
                                        {
                                            f"{selected_scen}": base_cum_rebased,
                                            "Scenario": scen_cum_rebased,
                                        }
                                    )
                                    st.line_chart(chart_df)
                                else:
                                    st.warning(
                                        "Scenario could not be plotted (no overlapping dates "
                                        "between baseline and scenario)."
                                    )

    # ---- TAB 7: MONTE CARLO ----
    with tabs[6]:
        st.subheader("Monte Carlo Projection")
        st.info("Project future portfolio value")

        selected_mc = st.selectbox(
            "Select portfolio:",
            selected_portfolios,
            key="mc_selector",
        )

        if selected_mc:
            pdata = portfolio_data[selected_mc]
            with st.expander("Run Monte Carlo"):
                years = st.number_input(
                    "Years", min_value=1, max_value=40, value=10
                )
                n_paths = st.number_input(
                    "Paths", min_value=100, max_value=2000, value=500, step=100
                )

                mc_method = st.selectbox(
                    "Method",
                    [
                        "Historical (current)",
                        "Mean-reverting GBM (experimental)",
                    ],
                    index=0,
                    help="Historical = multivariate normal on asset returns; "
                         "Mean-reverting GBM = AR(1) on log portfolio returns.",
                )

                run_mc = st.checkbox("Run simulation")
                if run_mc:
                    total_val0 = pdata["values"].iloc[-1].sum()
                    weights_now = pdata["values"].iloc[-1] / total_val0

                    if mc_method.startswith("Historical"):
                        mc_baseline = run_monte_carlo(
                            pdata["asset_prices"],
                            weights_now,
                            years=int(years),
                            paths=int(n_paths),
                            starting_value=float(total_val0),
                        )
                    else:
                        mc_baseline = run_monte_carlo_mean_reverting(
                            pdata["asset_prices"],
                            weights_now,
                            years=int(years),
                            paths=int(n_paths),
                            starting_value=float(total_val0),
                        )

                    if not mc_baseline.empty:
                        df_lines = []
                        for col in ["P10", "P50", "P90"]:
                            df_lines.append(
                                pd.DataFrame(
                                    {
                                        "Date": mc_baseline.index,
                                        "Value": mc_baseline[col].values,
                                        "Series": f"{col}",
                                    }
                                )
                            )
                        plot_df = pd.concat(df_lines, ignore_index=True)
                        fig = px.line(
                            plot_df, x="Date", y="Value", color="Series"
                        )
                        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
