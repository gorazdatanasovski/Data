import polars as pl
from pathlib import Path
import config

# ── Matrix registry ─────────────────────────────────────────────────────────
#
# Each entry: (path, series_name, frequency, datetime_col)
# datetime_col distinguishes daily ('date') from intraday ('datetime')

MATRICES = [
    # Daily
    (config.SPX_DAILY,    "SPX",  "Daily",   "date"),
    (config.SPX_MONTHLY,  "SPX",  "Monthly", "date"),
    (config.SPX_YEARLY,   "SPX",  "Yearly",  "date"),
    (config.SPXT_DAILY,   "SPXT", "Daily",   "date"),
    (config.SPXT_MONTHLY, "SPXT", "Monthly", "date"),
    (config.SPXT_YEARLY,  "SPXT", "Yearly",  "date"),
    (config.VIX_DAILY,    "VIX",  "Daily",   "date"),
    (config.VIX_MONTHLY,  "VIX",  "Monthly", "date"),
    (config.VIX_YEARLY,   "VIX",  "Yearly",  "date"),
    # Intraday
    (config.SPX_1MIN,     "SPX",  "1-Minute",  "datetime"),
    (config.SPX_5MIN,     "SPX",  "5-Minute",  "datetime"),
    (config.SPX_1H,       "SPX",  "1-Hour",    "datetime"),
    (config.VIX_1MIN,     "VIX",  "1-Minute",  "datetime"),
    (config.VIX_5MIN,     "VIX",  "5-Minute",  "datetime"),
    (config.VIX_1H,       "VIX",  "1-Hour",    "datetime"),
]


def validate_matrix(file_path: Path, name: str, frequency: str, dt_col: str) -> None:
    print(f"\n--- {name} {frequency} ---")

    if not file_path.exists():
        print(f"  Error: {file_path.name} not found.")
        return

    df = pl.read_parquet(file_path)

    print(f"  File             : {file_path.name}")
    print(f"  Timeline         : {df[dt_col].min()} → {df[dt_col].max()}")
    print(f"  Periods Captured : {df.height:,}")

    print("\n  Schema:")
    for col_name, dtype in df.schema.items():
        print(f"    {col_name}: {dtype}")

    print("\n  Null Audit:")
    null_counts = df.null_count()
    has_nulls = False
    for col in df.columns:
        n = null_counts[col][0]
        if n > 0:
            pct = n / df.height * 100
            print(f"    ⚠  {col}: {n} ({pct:.1f}%) — expected historical gap")
            has_nulls = True
    if not has_nulls:
        print("    ✓  Zero nulls.")

    dupes = df.height - df[dt_col].n_unique()
    if dupes == 0:
        print(f"\n  Temporal Integrity: ✓ Pass")
    else:
        print(f"\n  Temporal Integrity: ✗ Fail ({dupes} duplicate timestamps)")


if __name__ == "__main__":
    for path, name, freq, dt_col in MATRICES:
        validate_matrix(path, name, freq, dt_col)