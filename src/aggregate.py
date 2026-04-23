"""
aggregate.py
─────────────────────────────────────────────────────────────────────────────
Temporal aggregation of daily parquet files to monthly and yearly resolution.

Reads each instrument's daily parquet from storage, applies the canonical
OHLCV aggregation rules, and writes monthly and yearly parquet files.

Aggregation rules — formally defined:
  PX_OPEN   → first()  : The first open of the period IS the period open.
                          Taking any other value is mathematically incorrect.
  PX_HIGH   → max()    : The true period high is the maximum of all daily highs.
                          This is the only value consistent with the OHLC contract.
  PX_LOW    → min()    : The true period low is the minimum of all daily lows.
  PX_LAST   → last()   : The last close of the period IS the period close.
                          For index levels (SPXT, NDX), last() is the period-end
                          index level — the canonical reference value.
  PX_VOLUME → sum()    : Volume is additive. Period volume = sum of daily volumes.
                          This is the only financially correct aggregation.
  date      → max()    : The period anchor date is the LAST TRADING DAY of the
                          period, not the calendar month/year end.
                          January 2026 ends on 2026-01-30, not 2026-01-31.
                          Using truncated calendar dates would produce anchor
                          dates that never existed as trading sessions.

Null handling:
  Polars first(), last(), max(), min(), sum() all skip nulls by default.
  This is correct behaviour: a null in one daily bar (e.g. early VIX OHLC
  gaps) does not propagate to null the entire monthly/yearly aggregate.
  A monthly open, high, or low is computed from the available non-null values.

Writes are atomic: .tmp file → rename.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import shutil
from pathlib import Path

import polars as pl

import config

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / 'aggregate.log', mode='a', encoding='utf-8'),
    ],
)
log = logging.getLogger('aggregate')


def _atomic_write(df: pl.DataFrame, output_path: Path) -> None:
    """
    Write a Polars DataFrame to parquet atomically.

    Writes to a temporary file in the same directory, then performs an
    atomic rename to the final path. A crash mid-write leaves the previous
    version of the output file intact.
    """
    tmp_path = output_path.with_suffix('.tmp.parquet')
    try:
        df.write_parquet(tmp_path)
        shutil.move(str(tmp_path), str(output_path))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _build_aggregation_expressions(
    prefix:      str,
    has_ohlc:    bool,
    has_volume:  bool,
) -> list:
    """
    Construct the list of Polars aggregation expressions for one instrument.

    The date expression always comes first and returns the last trading day
    of the period — the canonical period anchor date.

    Parameters
    ----------
    prefix     : Instrument column prefix (e.g. 'SPX', 'SPXT', 'VIX')
    has_ohlc   : True if the instrument has PX_OPEN/HIGH/LOW in addition to
                 PX_LAST. False for single-field instruments (SPXT).
    has_volume : True if the instrument has PX_VOLUME (traded ETFs).
                 False for synthetic indices.

    Returns
    -------
    List of Polars expression objects ready for use in .agg().
    """
    # The date anchor is always the last trading day of the period.
    exprs = [pl.col("date").last().alias("date")]

    if has_ohlc:
        # Full OHLC aggregation
        exprs.append(pl.col(f"{prefix}_PX_OPEN").first().alias(f"{prefix}_PX_OPEN"))
        exprs.append(pl.col(f"{prefix}_PX_HIGH").max().alias(f"{prefix}_PX_HIGH"))
        exprs.append(pl.col(f"{prefix}_PX_LOW").min().alias(f"{prefix}_PX_LOW"))
        exprs.append(pl.col(f"{prefix}_PX_LAST").last().alias(f"{prefix}_PX_LAST"))
    else:
        # Single price level (SPXT total return index)
        exprs.append(pl.col(f"{prefix}_PX_LAST").last().alias(f"{prefix}_PX_LAST"))

    if has_volume:
        exprs.append(pl.col(f"{prefix}_PX_VOLUME").sum().alias(f"{prefix}_PX_VOLUME"))

    return exprs


def _resample_to_period(
    df:        pl.DataFrame,
    period:    str,
    agg_exprs: list,
) -> pl.DataFrame:
    """
    Resample a sorted daily DataFrame to monthly or yearly resolution.

    The grouping key is a synthetic internal column ('_period') that is
    dropped after aggregation. The output 'date' column is the last trading
    day of each period, computed inside the aggregation expressions.

    Parameters
    ----------
    df        : Sorted daily Polars DataFrame with a 'date' column of type pl.Date
    period    : 'monthly' or 'yearly'
    agg_exprs : List of Polars aggregation expressions from
                _build_aggregation_expressions()

    Returns
    -------
    Polars DataFrame with one row per period, sorted chronologically.
    """
    if period == "monthly":
        # Truncate to the first calendar day of the month — used only as a
        # grouping key. The actual output date is the last trading day of the
        # month, computed via pl.col("date").last() inside the agg expressions.
        group_key_expr = pl.col("date").dt.truncate("1mo").alias("_period")
    elif period == "yearly":
        # Extract the calendar year as an integer grouping key.
        group_key_expr = pl.col("date").dt.year().alias("_period")
    else:
        raise ValueError(
            f"Unsupported period '{period}'. Valid values: 'monthly', 'yearly'."
        )

    df_resampled = (
        df
        .sort("date")                      # Guarantee chronological order within groups
        .with_columns(group_key_expr)      # Add the grouping key column
        .group_by("_period")              # Group by period
        .agg(agg_exprs)                    # Apply aggregation rules
        .drop("_period")                   # Remove the internal grouping key
        .sort("date")                      # Sort output chronologically
    )

    return df_resampled


def aggregate_entry(entry: dict) -> None:
    """
    Aggregate one instrument's daily data to monthly and yearly resolution.

    Reads the daily parquet file, applies canonical OHLCV aggregation rules,
    and writes monthly and yearly parquet files atomically.

    Parameters
    ----------
    entry : A single instrument dict from config.REGISTRY.
    """
    ticker      = entry["ticker"]
    prefix      = entry["prefix"]
    source_path = config.daily_path(prefix)

    # Determine field structure from the registry entry
    has_ohlc   = len(entry["daily_fields"]) > 1   # True if more than just PX_LAST
    has_volume = entry["volume"]

    log.info(f"Aggregating {ticker} | prefix={prefix} | has_ohlc={has_ohlc} | has_volume={has_volume}")

    # ── Source file guard ─────────────────────────────────────────────────────
    if not source_path.exists():
        log.error(
            f"Daily source file not found: {source_path.name}. "
            f"Run ingest.py before aggregate.py."
        )
        return

    # ── Load daily data ───────────────────────────────────────────────────────
    df_daily = pl.read_parquet(source_path)
    log.info(f"Loaded {df_daily.height:,} daily rows from {source_path.name}")

    # ── Build aggregation expressions ─────────────────────────────────────────
    agg_exprs = _build_aggregation_expressions(prefix, has_ohlc, has_volume)

    # ── Monthly aggregation ───────────────────────────────────────────────────
    df_monthly = _resample_to_period(df_daily, "monthly", agg_exprs)
    monthly_out = config.monthly_path(prefix)
    _atomic_write(df_monthly, monthly_out)

    # Post-write verification
    written_monthly = pl.read_parquet(monthly_out)
    if written_monthly.height != df_monthly.height:
        raise RuntimeError(
            f"Post-write integrity check failed for {monthly_out.name}. "
            f"Expected {df_monthly.height} rows, found {written_monthly.height}."
        )
    log.info(
        f"Monthly: {df_monthly.height:,} periods "
        f"| {df_monthly['date'].min()} → {df_monthly['date'].max()} "
        f"| → {monthly_out.name}"
    )

    # ── Yearly aggregation ────────────────────────────────────────────────────
    df_yearly = _resample_to_period(df_daily, "yearly", agg_exprs)
    yearly_out = config.yearly_path(prefix)
    _atomic_write(df_yearly, yearly_out)

    # Post-write verification
    written_yearly = pl.read_parquet(yearly_out)
    if written_yearly.height != df_yearly.height:
        raise RuntimeError(
            f"Post-write integrity check failed for {yearly_out.name}. "
            f"Expected {df_yearly.height} rows, found {written_yearly.height}."
        )
    log.info(
        f"Yearly : {df_yearly.height:,} periods  "
        f"| {df_yearly['date'].min()} → {df_yearly['date'].max()} "
        f"| → {yearly_out.name}"
    )


def main() -> None:
    """
    Entry point. Iterates config.REGISTRY and aggregates every instrument.
    All instruments are aggregated — the daily source file must already exist.
    """
    log.info("=" * 72)
    log.info("Aggregation pipeline started.")
    log.info(f"Instruments: {[e['ticker'] for e in config.REGISTRY]}")
    log.info("=" * 72)

    failed = []
    for entry in config.REGISTRY:
        try:
            aggregate_entry(entry)
        except Exception as exc:
            log.error(f"FAILED: {entry['ticker']} — {exc}", exc_info=True)
            failed.append(entry["ticker"])

    log.info("=" * 72)
    if failed:
        log.error(f"Aggregation completed with errors. Failed tickers: {failed}")
    else:
        log.info("Aggregation complete. All instruments serialized successfully.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()