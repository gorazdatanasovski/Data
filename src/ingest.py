import polars as pl
from xbbg import blp
from datetime import datetime
import config


def pull_and_save_raw_ticker(ticker: str, fields: list[str], output_path) -> None:
    end_date = datetime.today().strftime('%Y-%m-%d')
    print(f"Extracting raw series for {ticker}...")

    df_raw = blp.bdh(
        tickers=[ticker],
        flds=fields,
        start_date=config.START_DATE,
        end_date=end_date,
    )

    df = pl.from_arrow(df_raw.to_arrow())

    if df.is_empty():
        print(f"  Warning: No data returned for {ticker}.")
        return

    prefix  = ticker.split()[0]
    df_wide = df.pivot(index="date", on="field", values="value")

    rename_map = {col: f"{prefix}_{col}" for col in df_wide.columns if col != "date"}
    df_wide = df_wide.rename(rename_map).sort("date")

    df_wide.write_parquet(output_path)
    print(f"  Serialized {df_wide.height} trading days → {output_path.name}")


def main() -> None:
    pull_and_save_raw_ticker(config.SPX_TICKER,  config.SPX_FIELDS,  config.SPX_DAILY)
    pull_and_save_raw_ticker(config.SPXT_TICKER, config.SPXT_FIELDS, config.SPXT_DAILY)
    pull_and_save_raw_ticker(config.VIX_TICKER,  config.VIX_FIELDS,  config.VIX_DAILY)
    print("Daily data lake complete.")


if __name__ == "__main__":
    main()