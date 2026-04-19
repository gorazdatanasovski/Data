import polars as pl
from pathlib import Path
import config

# ── Aggregation specification per series ───────────────────────────────────
#
# Each entry defines:
#   daily_path   : source daily parquet
#   monthly_path : monthly output parquet
#   yearly_path  : yearly output parquet
#   rules        : dict mapping column name → aggregation rule string
#
# Rules:
#   'first' → first chronological value of the period  (OPEN)
#   'max'   → maximum value of the period              (HIGH)
#   'min'   → minimum value of the period              (LOW)
#   'last'  → last chronological value of the period   (CLOSE / index level)
#   'sum'   → arithmetic sum of all period values      (VOLUME)

SPECS = [
    {
        "name":         "SPX",
        "daily_path":   config.SPX_DAILY,
        "monthly_path": config.SPX_MONTHLY,
        "yearly_path":  config.SPX_YEARLY,
        "rules": {
            "SPX_PX_OPEN":   "first",
            "SPX_PX_HIGH":   "max",
            "SPX_PX_LOW":    "min",
            "SPX_PX_LAST":   "last",
            "SPX_PX_VOLUME": "sum",
        },
    },
    {
        "name":         "SPXT",
        "daily_path":   config.SPXT_DAILY,
        "monthly_path": config.SPXT_MONTHLY,
        "yearly_path":  config.SPXT_YEARLY,
        "rules": {
            "SPXT_PX_LAST": "last",
        },
    },
    {
        "name":         "VIX",
        "daily_path":   config.VIX_DAILY,
        "monthly_path": config.VIX_MONTHLY,
        "yearly_path":  config.VIX_YEARLY,
        "rules": {
            "VIX_PX_OPEN": "first",
            "VIX_PX_HIGH": "max",
            "VIX_PX_LOW":  "min",
            "VIX_PX_LAST": "last",
        },
    },
]


def build_agg_expressions(rules: dict) -> list:
    """
    Translate the rules dict into a list of Polars aggregation expressions.
    All expressions correctly handle nulls — Polars ignores nulls in
    min/max/sum by default, and first/last skip nulls unless skip_nulls=False.
    We preserve skip_nulls=True (default) so early sparse data doesn't
    propagate nulls across the entire period.
    """
    dispatch = {
        "first": lambda col: pl.col(col).first(),
        "last":  lambda col: pl.col(col).last(),
        "max":   lambda col: pl.col(col).max(),
        "min":   lambda col: pl.col(col).min(),
        "sum":   lambda col: pl.col(col).sum(),
    }
    return [dispatch[rule](col) for col, rule in rules.items()]


def resample(df: pl.DataFrame, period: str, rules: dict) -> pl.DataFrame:
    """
    Resample a sorted daily DataFrame to the target period.

    period: 'monthly' | 'yearly'

    The period grouping key is derived purely for grouping — it is then
    dropped. The output date column is always the last trading day of the
    period, preserving financial calendar truth.
    """
    if period == "monthly":
        group_key = pl.col("date").dt.truncate("1mo").alias("_period")
    elif period == "yearly":
        group_key = pl.col("date").dt.year().alias("_period")
    else:
        raise ValueError(f"Unsupported period: {period}")

    agg_exprs = [
        pl.col("date").last().alias("date"),   # last trading day of the period
        *build_agg_expressions(rules),
    ]

    return (
        df
        .sort("date")                          # guarantee chronological order within groups
        .with_columns(group_key)
        .group_by("_period")
        .agg(agg_exprs)
        .drop("_period")
        .sort("date")
    )


def aggregate_series(spec: dict) -> None:
    name = spec["name"]
    print(f"\nAggregating {name}...")

    if not spec["daily_path"].exists():
        print(f"  Error: daily source not found at {spec['daily_path'].name}. Run ingest.py first.")
        return

    df_daily = pl.read_parquet(spec["daily_path"])

    # Monthly
    df_monthly = resample(df_daily, "monthly", spec["rules"])
    df_monthly.write_parquet(spec["monthly_path"])
    print(f"  Monthly: {df_monthly.height} periods → {spec['monthly_path'].name}")

    # Yearly
    df_yearly = resample(df_daily, "yearly", spec["rules"])
    df_yearly.write_parquet(spec["yearly_path"])
    print(f"  Yearly : {df_yearly.height} periods  → {spec['yearly_path'].name}")


def main() -> None:
    for spec in SPECS:
        aggregate_series(spec)
    print("\nAggregation complete.")


if __name__ == "__main__":
    main()