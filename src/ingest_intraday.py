import polars as pl
from xbbg import blp
from datetime import datetime, timedelta
from pathlib import Path
import config

INTRADAY_SPECS = [
    {
        "ticker":      config.SPX_TICKER,
        "prefix":      "SPX",
        "volume_cols": True,
        "intervals": [
            (1,  config.SPX_1MIN),
            (5,  config.SPX_5MIN),
            (60, config.SPX_1H),
        ],
    },
    {
        "ticker":      config.VIX_TICKER,
        "prefix":      "VIX",
        "volume_cols": False,
        "intervals": [
            (1,  config.VIX_1MIN),
            (5,  config.VIX_5MIN),
            (60, config.VIX_1H),
        ],
    },
]

BLOOMBERG_INTRADAY_WINDOW_DAYS = 139


def pull_intraday_bars(
    ticker:       str,
    prefix:       str,
    interval_min: int,
    output_path:  Path,
    volume_cols:  bool,
) -> None:
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=BLOOMBERG_INTRADAY_WINDOW_DAYS)

    label = f"{interval_min}min" if interval_min < 60 else "1h"
    print(f"  [{label}] {ticker} | {start_dt.date()} → {end_dt.date()}")

    df_raw = blp.bdib(
        ticker         = ticker,
        start_datetime = start_dt.strftime('%Y-%m-%d'),
        end_datetime   = end_dt.strftime('%Y-%m-%d'),
        interval       = interval_min,
    )

    df = pl.from_arrow(df_raw.to_arrow())

    if df.is_empty():
        print(f"    Warning: No data returned. Skipping.")
        return

    base_rename = {
        'time':  'datetime',
        'open':  f'{prefix}_PX_OPEN',
        'high':  f'{prefix}_PX_HIGH',
        'low':   f'{prefix}_PX_LOW',
        'close': f'{prefix}_PX_LAST',
    }

    volume_rename = {
        'volume':    f'{prefix}_PX_VOLUME',
        'numEvents': f'{prefix}_NUM_EVENTS',
    }

    existing      = set(df.columns)
    cols_to_keep  = [c for c in ['time', 'open', 'high', 'low', 'close'] if c in existing]
    if volume_cols:
        cols_to_keep += [c for c in ['volume', 'numEvents'] if c in existing]

    df = df.select(cols_to_keep)

    rename_map = {k: v for k, v in base_rename.items() if k in df.columns}
    if volume_cols:
        rename_map.update({k: v for k, v in volume_rename.items() if k in df.columns})

    df = (
        df
        .rename(rename_map)
        .with_columns(pl.col('datetime').cast(pl.Datetime))
        .sort('datetime')
    )

    df.write_parquet(output_path)
    print(f"    Serialized {df.height:,} bars → {output_path.name}")


def main() -> None:
    print(f"Intraday ingestion started.")
    print(f"Bloomberg window: last {BLOOMBERG_INTRADAY_WINDOW_DAYS} calendar days.\n")

    for spec in INTRADAY_SPECS:
        print(f"── {spec['ticker']} ──")
        for interval_min, output_path in spec["intervals"]:
            pull_intraday_bars(
                ticker       = spec["ticker"],
                prefix       = spec["prefix"],
                interval_min = interval_min,
                output_path  = output_path,
                volume_cols  = spec["volume_cols"],
            )
        print()

    print("Intraday data lake complete.")


if __name__ == "__main__":
    main()