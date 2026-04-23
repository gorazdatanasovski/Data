"""
ingest_tick.py
─────────────────────────────────────────────────────────────────────────────
Sub-minute raw tick ingestion from Bloomberg via blp.bdtick().

This module is architecturally distinct from ingest_intraday.py:

  Bar data (bdib):
    - Fixed-interval OHLCV aggregates
    - Schema: (datetime, open, high, low, close, volume, numEvents)
    - Minimum resolution: 1 minute
    - Use case: technical analysis, strategy backtesting, regime detection

  Tick data (bdtick):
    - Raw event stream: individual trade prints, bid updates, ask updates
    - Schema: (datetime, type, value, size)
    - Resolution: millisecond timestamp per event
    - Use case: microstructure analysis, spread analysis, order flow

Tick type contracts by instrument:
  Traded ETFs (SPY, QQQ):
    ['TRADE', 'BID', 'ASK'] — full tape: prints + quote updates
  Synthetic indices (SPX, VIX, NDX):
    ['BID', 'ASK'] only — no TRADE events exist for non-traded instruments
  SPXT:
    Excluded entirely — total return index, no intraday computation

Bloomberg window: same 139-day hard constraint as bar data.

Output schema:
  datetime  : pl.Datetime (microsecond precision, event timestamp)
  type      : pl.String   (event type: 'TRADE', 'BID', or 'ASK')
  value     : pl.Float64  (price of the event)
  size      : pl.Float64  (trade size for TRADE events; 0 or null for BID/ASK)

Files are large. A single instrument over 139 days at full tick resolution
can produce millions of rows. These files are never aggregated — they are
the terminal resolution of the data lake.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from xbbg import blp

import config

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / 'ingest_tick.log', mode='a', encoding='utf-8'),
    ],
)
log = logging.getLogger('ingest_tick')


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


def pull_tick_data(
    ticker:     str,
    prefix:     str,
    tick_types: list[str],
) -> None:
    """
    Pull raw tick data from Bloomberg for a single instrument.

    Parameters
    ----------
    ticker     : Bloomberg ticker string
    prefix     : Column prefix (e.g. 'SPX', 'SPY')
    tick_types : List of Bloomberg tick event types to request.
                 Valid values: 'TRADE', 'BID', 'ASK'
    """
    end_dt      = datetime.now()
    start_dt    = end_dt - timedelta(days=config.BLOOMBERG_INTRADAY_WINDOW_DAYS)
    output_path = config.tick_path(prefix)

    log.info(
        f"{ticker} "
        f"| types={tick_types} "
        f"| {start_dt.date()} → {end_dt.date()}"
    )

    # ── Bloomberg pull ────────────────────────────────────────────────────────
    df_raw = blp.bdtick(
        ticker         = ticker,
        start_datetime = start_dt.strftime('%Y-%m-%d'),
        end_datetime   = end_dt.strftime('%Y-%m-%d'),
        types          = tick_types,
    )

    # ── Convert from Narwhals/PyArrow to native Polars ────────────────────────
    df = pl.from_arrow(df_raw.to_arrow())

    # ── Empty guard ───────────────────────────────────────────────────────────
    if df.is_empty():
        log.warning(
            f"No tick data returned for {ticker} with types={tick_types}. "
            f"Output file not written."
        )
        return

    log.info(f"Received {df.height:,} raw tick rows. Columns: {df.columns}")

    # ── Normalise the datetime column name ────────────────────────────────────
    # xbbg bdtick may return the timestamp column as 'time' or 'datetime'
    # depending on version. We normalise to 'datetime' in all cases.
    time_col_candidates = [c for c in df.columns if c.lower() in ('time', 'datetime')]
    if not time_col_candidates:
        raise RuntimeError(
            f"No timestamp column found in bdtick response for {ticker}. "
            f"Columns: {df.columns}"
        )
    time_col = time_col_candidates[0]
    if time_col != 'datetime':
        df = df.rename({time_col: 'datetime'})

    # ── Type enforcement ──────────────────────────────────────────────────────
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime(time_unit='us')))

    # ── Sort chronologically ──────────────────────────────────────────────────
    df = df.sort('datetime')

    # ── Validate event type coverage ──────────────────────────────────────────
    if 'type' in df.columns:
        observed_types = df['type'].unique().to_list()
        log.info(f"Tick event types observed: {sorted(observed_types)}")
        unexpected = [t for t in observed_types if t not in tick_types]
        if unexpected:
            log.warning(
                f"Unexpected tick event types returned by Bloomberg "
                f"for {ticker}: {unexpected}. These rows are retained."
            )

    # ── Atomic write ──────────────────────────────────────────────────────────
    _atomic_write(df, output_path)

    # ── Post-write verification ───────────────────────────────────────────────
    written = pl.read_parquet(output_path)
    if written.height != df.height:
        raise RuntimeError(
            f"Post-write integrity check failed for {output_path.name}. "
            f"In-memory rows: {df.height}, On-disk rows: {written.height}."
        )

    log.info(
        f"Serialized {df.height:,} ticks "
        f"| {df['datetime'].min()} → {df['datetime'].max()} "
        f"| → {output_path.name}"
    )


def main() -> None:
    """
    Entry point. Iterates config.REGISTRY and pulls tick data for every
    instrument where tick_types is not None (i.e. SPXT is excluded).
    """
    log.info("=" * 72)
    log.info("Tick ingestion pipeline started.")
    log.info(f"Bloomberg window: last {config.BLOOMBERG_INTRADAY_WINDOW_DAYS} calendar days.")

    tick_entries = [e for e in config.REGISTRY if e.get("tick_types") is not None]
    log.info(f"Instruments: {[e['ticker'] for e in tick_entries]}")
    log.info("=" * 72)

    failed = []
    for entry in tick_entries:
        try:
            pull_tick_data(
                ticker     = entry["ticker"],
                prefix     = entry["prefix"],
                tick_types = entry["tick_types"],
            )
        except Exception as exc:
            log.error(f"FAILED: {entry['ticker']} — {exc}", exc_info=True)
            failed.append(entry["ticker"])

    log.info("=" * 72)
    if failed:
        log.error(f"Tick ingestion completed with errors. Failed tickers: {failed}")
    else:
        log.info("Tick ingestion complete. All instruments serialized successfully.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()