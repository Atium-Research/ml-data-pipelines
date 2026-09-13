"""EOD stock prices for the universe, one year at a time.

The free stock tier refuses anything before 2023-06-01, so the default
window is the whole of what it serves. Each year pulls that year's members
and overwrites that year's partition; years already on disk are skipped.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.canonical import canonicalize_stock
from pipelines.utils.sessions import year_window, years_in
from pipelines.utils.symbols import universe_symbols
from pipelines.utils.theta import fetch_many, is_missing_data, make_client

START = dt.date(2023, 6, 1)
END = dt.date(2025, 12, 31)
WORKERS = 2


def fetch_stock(symbol: str, start: dt.date, end: dt.date) -> pl.DataFrame | None:
    client = make_client()
    try:
        return client.stock_history_eod(symbol, start, end).with_columns(
            pl.lit(symbol).alias("symbol")
        )
    except Exception as error:
        if is_missing_data(error):
            return None
        raise


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    for year in years_in(start, end):
        if store.partition_exists("underlying", year):
            print(f"underlying {year}: on disk, skipped")
            continue
        year_start, year_end = year_window(year, start, end)
        symbols = universe_symbols("stock", year_start, year_end)
        print(f"underlying {year}: {len(symbols)} symbols, {year_start} .. {year_end}")
        frames = fetch_many(
            symbols, lambda s, a=year_start, b=year_end: fetch_stock(s, a, b), WORKERS
        )
        frames = [f for f in frames if f is not None and f.height]
        if frames:
            prices_df = canonicalize_stock(pl.concat(frames, how="vertical_relaxed"))
            store.write(db, "underlying", prices_df)
            print(f"  {prices_df.height:,} rows")
