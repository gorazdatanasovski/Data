"""
ingest.py
─────────────────────────────────────────────────────────────────────────────
Daily OHLCV ingestion from Bloomberg via blp.bdh().

Pulls daily history from START_DATE (1900-01-01) to today for every
instrument defined in config.REGISTRY. Serializes each instrument to a
wide-format parquet file with the naming convention:
    {prefix_lower}_raw_inception_daily.parquet

Architectural notes:
  - blp.bdh() returns a Narwhals DataFrame backed by PyArrow in recent
    versions of xbbg. We convert to native Polars immediately via
    pl.from_arrow(df_raw.to_arrow()) before any manipulation.
  - The raw return schema is long-format: (ticker, date, field, value).
    We pivot to wide format on the 'field' column.
  - All output columns are prefixed with the instrument prefix (e.g. SPX_).
  - Writes are atomic: data is written to a .tmp file first, then renamed
    to the final path. A crash mid-write cannot corrupt the existing file.
  - All operations are logged to both stdout and logs/pipeline.log.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import shutil
from datetime import datetime
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
        logging.FileHandler(config.LOG_DIR / 'ingest_daily.log', mode='a', encoding='utf-8'),
    ],
)
log = logging.getLogger('ingest_daily')


def _atomic_write(df: pl.DataFrame, output_path: Path) -> None:
    """
    Write a Polars DataFrame to parquet atomically.

    Writes first to a temporary file in the same directory, then performs
    an atomic rename to the final path. This guarantees that the output
    file is never in a partially-written state — a crash mid-write leaves
    the previous version of the file intact.

    Parameters
    ----------
    df          : Polars DataFrame to serialize
    output_path : Final destination path
    """
    tmp_path = output_path.with_suffix('.tmp.parquet')
    try:
        df.write_parquet(tmp_path)
        shutil.move(str(tmp_path), str(output_path))
    except Exception:
        # Clean up the temp file if anything fails before the rename
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def pull_and_save_daily(entry: dict) -> None:
    """
    Pull full daily history for a single instrument and serialize to parquet.

    Parameters
    ----------
    entry : A single instrument dict from config.REGISTRY. Required keys:
              ticker        Bloomberg ticker string
              prefix        Column prefix (e.g. 'SPX')
              daily_fields  List of Bloomberg field mnemonics
    """
    ticker      = entry["ticker"]
    prefix      = entry["prefix"]
    fields      = entry["daily_fields"]
    output_path = config.daily_path(prefix)
    end_date    = datetime.today().strftime('%Y-%m-%d')

    log.info(f"Extracting daily series | ticker={ticker} | fields={fields}")
    log.info(f"Date range: {config.START_DATE} → {end_date}")

    # ── Bloomberg pull ────────────────────────────────────────────────────────
    df_raw = blp.bdh(
        tickers    = [ticker],
        flds       = fields,
        start_date = config.START_DATE,
        end_date   = end_date,
    )

    # ── Convert from Narwhals/PyArrow to native Polars ────────────────────────
    # blp.bdh() returns a Narwhals DataFrame backed by PyArrow in xbbg >= 0.8.
    # Narwhals does not implement pivot() for PyArrow backends. Converting to
    # native Polars via the Arrow IPC layer is zero-copy and instantaneous.
    df = pl.from_arrow(df_raw.to_arrow())

    # ── Empty guard ───────────────────────────────────────────────────────────
    if df.is_empty():
        log.warning(f"No data returned for {ticker}. Output file not written.")
        return

    log.info(f"Received {df.height:,} raw long-format rows for {ticker}.")

    # ── Schema validation ─────────────────────────────────────────────────────
    # Verify that the expected long-format columns are present before pivoting.
    expected_cols = {'ticker', 'date', 'field', 'value'}
    actual_cols   = set(df.columns)
    if not expected_cols.issubset(actual_cols):
        raise RuntimeError(
            f"Unexpected schema from blp.bdh() for {ticker}. "
            f"Expected columns {expected_cols}, got {actual_cols}."
        )

    # ── Filter to this ticker only ────────────────────────────────────────────
    # blp.bdh() is called with a single ticker but we filter defensively to
    # guard against any case where Bloomberg appends metadata rows.
    df = df.filter(pl.col("ticker") == ticker)

    # ── Pivot long → wide ─────────────────────────────────────────────────────
    # Input:  (ticker, date, field, value)  — one row per (date, field)
    # Output: (date, PX_OPEN, PX_HIGH, ...)  — one row per date
    df_wide = df.pivot(index="date", on="field", values="value")

    # ── Column renaming ───────────────────────────────────────────────────────
    # Prefix every field column with the instrument prefix.
    # 'date' column is left unchanged.
    # Result: (date, SPX_PX_OPEN, SPX_PX_HIGH, SPX_PX_LOW, SPX_PX_LAST, SPX_PX_VOLUME)
    rename_map = {
        col: f"{prefix}_{col}"
        for col in df_wide.columns
        if col != "date"
    }
    df_wide = df_wide.rename(rename_map)

    # ── Type enforcement ──────────────────────────────────────────────────────
    # Guarantee the date column is pl.Date (not Datetime or String).
    # Polars pivot preserves the source type — the from_arrow conversion
    # already sets it to Date from the Arrow date32 type, but we enforce
    # explicitly to be safe.
    df_wide = df_wide.with_columns(pl.col("date").cast(pl.Date))

    # ── Sort chronologically ──────────────────────────────────────────────────
    df_wide = df_wide.sort("date")

    # ── Verify field coverage ─────────────────────────────────────────────────
    # Confirm every requested field appears as a column in the output.
    expected_output_cols = {f"{prefix}_{f}" for f in fields}
    actual_output_cols   = set(df_wide.columns) - {"date"}
    missing_fields       = expected_output_cols - actual_output_cols
    if missing_fields:
        log.warning(
            f"Bloomberg did not return data for fields: {missing_fields} "
            f"on ticker {ticker}. These columns will be absent from the output."
        )

    # ── Atomic write ──────────────────────────────────────────────────────────
    _atomic_write(df_wide, output_path)

    # ── Post-write verification ───────────────────────────────────────────────
    # Read back the written file and verify row count matches in-memory frame.
    written = pl.read_parquet(output_path)
    if written.height != df_wide.height:
        raise RuntimeError(
            f"Post-write integrity check failed for {output_path.name}. "
            f"In-memory rows: {df_wide.height}, On-disk rows: {written.height}."
        )

    log.info(
        f"Serialized {df_wide.height:,} trading days "
        f"| {df_wide['date'].min()} → {df_wide['date'].max()} "
        f"| → {output_path.name}"
    )


def main() -> None:
    """
    Entry point. Iterates config.REGISTRY and pulls daily history for every
    instrument. SPXT and all others are included — no instrument is excluded
    from the daily pull.
    """
    log.info("=" * 72)
    log.info("Daily ingestion pipeline started.")
    log.info(f"Instruments: {[e['ticker'] for e in config.REGISTRY]}")
    log.info("=" * 72)

    failed = []
    for entry in config.REGISTRY:
        try:
            pull_and_save_daily(entry)
        except Exception as exc:
            log.error(f"FAILED: {entry['ticker']} — {exc}", exc_info=True)
            failed.append(entry["ticker"])

    log.info("=" * 72)
    if failed:
        log.error(f"Daily ingestion completed with errors. Failed tickers: {failed}")
    else:
        log.info("Daily ingestion complete. All instruments serialized successfully.")
    log.info("=" * 72)


if __name__ == "__main__":
    main()