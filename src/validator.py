"""
validator.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive structural and statistical validation of all parquet files
in the data lake.

Validation checks performed per file:
  1. Existence       — file is present on disk
  2. Timeline        — min and max timestamp
  3. Row count       — total periods captured
  4. Schema          — column names and data types
  5. Null audit      — null count and percentage per column
  6. Temporal uniqueness — duplicate timestamp detection
  7. Monotonicity    — date/datetime column is strictly increasing
  8. Statistical summary — mean, std, min, p25, p50, p75, max per
                           numeric column (Float64, Int32, Int64)
                           This allows immediate detection of price
                           level anomalies, zero values, or outliers
                           that a simple null count would not catch.

The matrix registry is built programmatically from config.REGISTRY.
Adding a new instrument to config.REGISTRY automatically adds it to
validation without any changes to this file.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
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
        logging.FileHandler(config.LOG_DIR / 'validator.log', mode='a', encoding='utf-8'),
    ],
)
log = logging.getLogger('validator')


def _build_matrix_registry() -> list[tuple[Path, str, str]]:
    """
    Programmatically construct the full list of (path, label, dt_col) tuples
    from config.REGISTRY.

    dt_col is 'date' for daily/monthly/yearly files (pl.Date column)
    and 'datetime' for intraday bar and tick files (pl.Datetime column).

    Returns
    -------
    List of (file_path, human_label, datetime_column_name) tuples covering
    every parquet file the pipeline is expected to produce.
    """
    matrices: list[tuple[Path, str, str]] = []

    for entry in config.REGISTRY:
        prefix = entry["prefix"]

        # ── Daily ─────────────────────────────────────────────────────────────
        matrices.append((
            config.daily_path(prefix),
            f"{prefix} Daily",
            "date",
        ))

        # ── Monthly ───────────────────────────────────────────────────────────
        matrices.append((
            config.monthly_path(prefix),
            f"{prefix} Monthly",
            "date",
        ))

        # ── Yearly ────────────────────────────────────────────────────────────
        matrices.append((
            config.yearly_path(prefix),
            f"{prefix} Yearly",
            "date",
        ))

        # ── Intraday bars ─────────────────────────────────────────────────────
        if entry["intraday"]:
            for interval_min, label in config.INTRADAY_INTERVALS.items():
                matrices.append((
                    config.intraday_path(prefix, interval_min),
                    f"{prefix} {label}",
                    "datetime",
                ))

        # ── Tick data ─────────────────────────────────────────────────────────
        if entry.get("tick_types") is not None:
            tick_file = config.tick_path(prefix)
            matrices.append((
                tick_file,
                f"{prefix} Tick",
                "datetime",
            ))

    return matrices


def _validate_matrix(
    file_path: Path,
    label:     str,
    dt_col:    str,
) -> bool:
    """
    Perform full structural and statistical validation on a single parquet file.

    Parameters
    ----------
    file_path : Path to the parquet file
    label     : Human-readable label for display (e.g. 'SPX Daily')
    dt_col    : Name of the timestamp column ('date' or 'datetime')

    Returns
    -------
    True if the file passes all checks, False if any check fails.
    """
    separator = "─" * 68
    print(f"\n{'═' * 68}")
    print(f"  {label}")
    print(f"{'═' * 68}")

    # ── Check 1: File existence ───────────────────────────────────────────────
    if not file_path.exists():
        print(f"  ✗  NOT FOUND: {file_path.name}")
        print(f"     Run the appropriate ingestion script to generate this file.")
        return False

    # ── Load file ─────────────────────────────────────────────────────────────
    df = pl.read_parquet(file_path)

    # ── Check 2: File metadata ────────────────────────────────────────────────
    file_size_mb = file_path.stat().st_size / (1024 ** 2)
    print(f"\n  File             : {file_path.name}")
    print(f"  Size             : {file_size_mb:.2f} MB")

    # ── Check 3: Timeline ─────────────────────────────────────────────────────
    dt_min = df[dt_col].min()
    dt_max = df[dt_col].max()
    print(f"  Timeline         : {dt_min} → {dt_max}")
    print(f"  Periods Captured : {df.height:,}")

    # ── Check 4: Schema ───────────────────────────────────────────────────────
    print(f"\n  Schema:")
    for col_name, dtype in df.schema.items():
        print(f"    {col_name}: {dtype}")

    # ── Check 5: Null audit ───────────────────────────────────────────────────
    print(f"\n  Null Audit:")
    null_counts = df.null_count()
    has_nulls   = False
    for col in df.columns:
        n = null_counts[col][0]
        if n > 0:
            pct = n / df.height * 100
            print(f"    ⚠  {col}: {n:,} nulls ({pct:.2f}%) — expected historical gap")
            has_nulls = True
    if not has_nulls:
        print(f"    ✓  Zero nulls across all columns.")

    # ── Check 6: Temporal uniqueness ──────────────────────────────────────────
    print(f"\n  Temporal Integrity:")
    n_unique = df[dt_col].n_unique()
    n_dupes  = df.height - n_unique
    if n_dupes == 0:
        print(f"    ✓  No duplicate timestamps. ({n_unique:,} unique)")
    else:
        print(f"    ✗  FAIL: {n_dupes:,} duplicate timestamps detected.")

    # ── Check 7: Monotonicity ─────────────────────────────────────────────────
    # Verify the timestamp column is strictly non-decreasing.
    # We sort on ingest, but this confirms the file on disk is ordered.
    is_sorted = df[dt_col].is_sorted()
    print(f"\n  Monotonicity:")
    if is_sorted:
        print(f"    ✓  Timestamps are strictly non-decreasing.")
    else:
        print(f"    ✗  FAIL: Timestamp column is not sorted. File may be corrupted.")

    # ── Check 8: Statistical summary (numeric columns only) ───────────────────
    numeric_dtypes = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16)
    numeric_cols   = [
        col for col, dtype in df.schema.items()
        if isinstance(dtype, numeric_dtypes) and col != dt_col
    ]

    if numeric_cols:
        print(f"\n  Statistical Summary:")
        print(
            f"    {'Column':<32} "
            f"{'Mean':>12} "
            f"{'Std':>12} "
            f"{'Min':>12} "
            f"{'p25':>12} "
            f"{'p50':>12} "
            f"{'p75':>12} "
            f"{'Max':>12}"
        )
        print(f"    {separator}")

        for col in numeric_cols:
            series = df[col].drop_nulls()
            if series.is_empty():
                print(f"    {col:<32}  [all nulls — no statistics computable]")
                continue

            mean_val = series.mean()
            std_val  = series.std()
            min_val  = series.min()
            max_val  = series.max()

            # Polars quantile returns a scalar float
            p25 = series.quantile(0.25)
            p50 = series.quantile(0.50)
            p75 = series.quantile(0.75)

            print(
                f"    {col:<32} "
                f"{mean_val:>12.4f} "
                f"{std_val:>12.4f} "
                f"{min_val:>12.4f} "
                f"{p25:>12.4f} "
                f"{p50:>12.4f} "
                f"{p75:>12.4f} "
                f"{max_val:>12.4f}"
            )

        # ── Anomaly flags ─────────────────────────────────────────────────────
        # Flag columns where the minimum value is zero or negative.
        # Price series should never be zero or negative.
        print(f"\n  Anomaly Flags:")
        anomalies_found = False
        for col in numeric_cols:
            if 'PX_' not in col:
                continue  # Only check price columns
            series     = df[col].drop_nulls()
            if series.is_empty():
                continue
            min_val    = series.min()
            zero_count = (series == 0.0).sum()
            neg_count  = (series < 0.0).sum()
            if min_val is not None and min_val <= 0:
                anomalies_found = True
                print(
                    f"    ⚠  {col}: min={min_val:.6f} | "
                    f"zero_count={zero_count:,} | neg_count={neg_count:,}"
                )
        if not anomalies_found:
            print(f"    ✓  No zero or negative price values detected.")

    # ── Summary result ────────────────────────────────────────────────────────
    passed = (n_dupes == 0) and is_sorted
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n  Overall Status   : {status}")

    return passed


def main() -> None:
    """
    Entry point. Builds the full matrix registry from config.REGISTRY and
    validates every expected parquet file.
    """
    print("=" * 68)
    print("  Bloomberg Data Lake — Full Validation Report")
    print(f"  Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)

    matrices      = _build_matrix_registry()
    total         = len(matrices)
    passed_count  = 0
    failed        = []
    not_found     = []

    for file_path, label, dt_col in matrices:
        result = _validate_matrix(file_path, label, dt_col)
        if result:
            passed_count += 1
        elif not file_path.exists():
            not_found.append(label)
        else:
            failed.append(label)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print(f"  Validation Summary")
    print(f"{'=' * 68}")
    print(f"  Total matrices expected : {total}")
    print(f"  Passed                  : {passed_count}")
    print(f"  Not yet ingested        : {len(not_found)}")
    print(f"  Failed                  : {len(failed)}")

    if not_found:
        print(f"\n  Not yet ingested:")
        for label in not_found:
            print(f"    - {label}")

    if failed:
        print(f"\n  Failed validation:")
        for label in failed:
            print(f"    - {label}")

    if not failed and not not_found:
        print(f"\n  ✓ ALL {total} MATRICES VALIDATED SUCCESSFULLY.")
    elif not failed:
        print(f"\n  ✓ All present files validated successfully. {len(not_found)} not yet ingested.")
    else:
        print(f"\n  ✗ {len(failed)} files failed validation. Investigate immediately.")

    print(f"{'=' * 68}\n")


if __name__ == "__main__":
    main()