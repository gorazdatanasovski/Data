"""
ingest_intraday.py
─────────────────────────────────────────────────────────────────────────────
Intraday bar ingestion from Bloomberg via blp.bdib().

Pulls bar data at 10 standard intervals (1min through 4h) for every
instrument in config.REGISTRY that has intraday=True. SPXT is excluded
because Bloomberg does not compute IntradayBarRequest for total return
indices.

Bloomberg constraints:
  - Hard window: 140 calendar days maximum. We use 139.
  - Minimum interval: 1 minute. Sub-minute requires bdtick (see ingest_tick.py)
  - Maximum interval: Any integer number of minutes up to 1440.

Output schema per file:
  datetime         : pl.Datetime (timestamp of bar open)
  {PREFIX}_PX_OPEN : Float64
  {PREFIX}_PX_HIGH : Float64
  {PREFIX}_PX_LOW  : Float64
  {PREFIX}_PX_LAST : Float64
  {PREFIX}_PX_VOLUME  : Float64   (traded instruments only: SPY, QQQ)
  {PREFIX}_NUM_EVENTS : Int32     (traded instruments only: SPY, QQQ)

Volume and numEvents are explicitly excluded for synthetic indices (SPX,
VIX, NDX) because Bloomberg either returns zeros or nulls — these values
carry no information for non-traded instruments.

Writes are atomic: .tmp file → rename.
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
        logging.FileHandler(config.LOG_DIR / 'ingest_intraday.log', mode='a', encoding='utf-8'),
    ],
)
log = logging.getLogger('ingest_intraday')


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


def pull_intraday_bars(
    ticker:       str,
    prefix:       str,
    interval_min: int,
    output_path:  Path,
    volume_cols:  bool,
) -> None:
    """
    Pull intraday bars from Bloomberg for a single instrument and interval.

    Parameters
    ----------
    ticker       : Bloomberg ticker string (e.g. 'SPX Index', 'SPY US Equity')
    prefix       : Column prefix (e.g. 'SPX', 'SPY')
    interval_min : Bar interval in minutes. Must be a key in config.INTRADAY_INTERVALS.
    output_path  : Destination parquet path.
    volume_cols  : True  → retain volume and numEvents columns (traded ETFs)
                   False → drop volume and numEvents (synthetic indices)
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=config.BLOOMBERG_INTRADAY_WINDOW_DAYS)
    label    = config.INTRADAY_INTERVALS[interval_min]

    log.info(
        f"[{label}] {ticker} "
        f"| {start_dt.date()} → {end_dt.date()} "
        f"| volume_cols={volume_cols}"
    )

    # ── Bloomberg pull ────────────────────────────────────────────────────────
    df_raw = blp.bdib(
        ticker         = ticker,
        start_datetime = start_dt.strftime('%Y-%m-%d'),
        end_datetime   = end_dt.strftime('%Y-%m-%d'),
        interval       = interval_min,
    )

    # ── Convert from Narwhals/PyArrow to native Polars ────────────────────────
    df = pl.from_arrow(df_raw.to_arrow())

    # ── Empty guard ───────────────────────────────────────────────────────────
    if df.is_empty():
        log.warning(
            f"[{label}] No data returned for {ticker}. "
            f"Output file not written."
        )
        return

    log.info(f"[{label}] Received {df.height:,} raw rows. Columns: {df.columns}")

    # ── Column selection ──────────────────────────────────────────────────────
    # bdib return schema (xbbg): ticker, time, open, high, low, close, volume, numEvents
    # We always retain: time, open, high, low, close
    # We conditionally retain: volume, numEvents (only for traded instruments)

    base_cols   = ['time', 'open', 'high', 'low', 'close']
    volume_cols_names = ['volume', 'numEvents']

    existing_cols = set(df.columns)

    # Verify base columns are present
    missing_base = [c for c in base_cols if c not in existing_cols]
    if missing_base:
        raise RuntimeError(
            f"[{label}] {ticker}: Expected base columns {missing_base} "
            f"not found in bdib response. Actual columns: {df.columns}"
        )

    cols_to_select = [c for c in base_cols if c in existing_cols]
    if volume_cols:
        cols_to_select += [c for c in volume_cols_names if c in existing_cols]

    df = df.select(cols_to_select)

    # ── Column renaming ───────────────────────────────────────────────────────
    rename_map: dict[str, str] = {
        'time':  'datetime',
        'open':  f'{prefix}_PX_OPEN',
        'high':  f'{prefix}_PX_HIGH',
        'low':   f'{prefix}_PX_LOW',
        'close': f'{prefix}_PX_LAST',
    }
    if volume_cols:
        rename_map['volume']    = f'{prefix}_PX_VOLUME'
        rename_map['numEvents'] = f'{prefix}_NUM_EVENTS'

    # Apply only renames for columns that exist in the selected frame
    active_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(active_rename)

    # ── Type enforcement ──────────────────────────────────────────────────────
    # Guarantee datetime column is pl.Datetime with microsecond precision.
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime(time_unit='us')))

    # ── Sort chronologically ──────────────────────────────────────────────────
    df = df.sort('datetime')

    # ── Duplicate timestamp guard ─────────────────────────────────────────────
    n_dupes = df.height - df['datetime'].n_unique()
    if n_dupes > 0:
        log.warning(
            f"[{label}] {ticker}: {n_dupes} duplicate timestamps detected. "
            f"Deduplicating by keeping last occurrence."
        )
        df = df.unique(subset=['datetime'], keep='last').sort('datetime')

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
        f"[{label}] Serialized {df.height:,} bars "
        f"| {df['datetime'].min()} → {df['datetime'].max()} "
        f"| → {output_path.name}"
    )


def main() -> None:
    """
    Entry point. Iterates all intraday-eligible instruments in config.REGISTRY
    and pulls every interval defined in config.INTRADAY_INTERVALS.

    SPXT is skipped automatically via the entry["intraday"] flag.
    """
    log.info("=" * 72)
    log.info("Intraday bar ingestion pipeline started.")
    log.info(f"Bloomberg window: last {config.BLOOMBERG_INTRADAY_WINDOW_DAYS} calendar days.")
    log.info(f"Intervals: {list(config.INTRADAY_INTERVALS.values())}")

    intraday_entries = [e for e in config.REGISTRY if e["intraday"]]
    log.info(f"Instruments: {[e['ticker'] for e in intraday_entries]}")
    log.info("=" * 72)

    failed = []
    for entry in intraday_entries:
        ticker   = entry["ticker"]
        prefix   = entry["prefix"]
        vol_cols = entry["volume"]

        log.info(f"── {ticker} ──")
        for interval_min, _label in config.INTRADAY_INTERVALS.items():
            output_path = config.intraday_path(prefix, interval_min)
            try:
                pull_intraday_bars(
                    ticker       = ticker,
                    prefix       = prefix,
                    interval_min = interval_min,
                    output_path  = output_path,
                    volume_cols  = vol_cols,
                )
            except Exception as exc:
                key = f"{ticker} [{_label}]"
                log.error(f"FAILED: {key} — {exc}", exc_info=True)
                failed.append(key)

    log.info("=" * 72)
    if failed:
        log.error(f"Intraday ingestion completed with errors. Failed: {failed}")
    else:
        log.info("Intraday bar ingestion complete. All intervals serialized successfully.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()