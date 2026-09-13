"""EOD open interest per symbol-year.

Cheap: the endpoint accepts a whole date range, so a symbol-year is one
request, stitched across the 365-day cap. The server occasionally faults
(`INTERNAL: ArrayIndexOutOfBounds`) on a window containing one bad session;
the window is bisected until that session is alone and skipped.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.canonical import canonicalize_open_interest
from pipelines.utils.sessions import date_chunks, year_window, years_in
from pipelines.utils.symbols import universe_symbols
from pipelines.utils.theta import fetch_many, is_missing_data, make_client

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
WORKERS = 4


def is_server_fault(error: Exception) -> bool:
    return "INTERNAL" in str(error) or "ArrayIndexOutOfBounds" in str(error)


def fetch_window(client, symbol: str, start: dt.date, end: dt.date) -> list[pl.DataFrame]:
    try:
        chunk_df = client.option_history_open_interest(symbol, "*", start_date=start, end_date=end)
        return [chunk_df] if chunk_df.height else []
    except Exception as error:
        if is_missing_data(error):
            return []
        if not is_server_fault(error):
            raise
        if start >= end:
            print(f"\n  {symbol}: server fault on {start}, skipping that session")
            return []
        midpoint = start + (end - start) / 2
        return fetch_window(client, symbol, start, midpoint) + fetch_window(
            client, symbol, midpoint + dt.timedelta(days=1), end
        )


def fetch_open_interest(symbol: str, start: dt.date, end: dt.date) -> pl.DataFrame | None:
    client = make_client()
    chunks = []
    for chunk_start, chunk_end in date_chunks(start, end):
        chunks.extend(fetch_window(client, symbol, chunk_start, chunk_end))
    return pl.concat(chunks, how="vertical_relaxed") if chunks else None


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    store.ensure(db, "open_interest")
    for year in years_in(start, end):
        year_start, year_end = year_window(year, start, end)
        symbols = universe_symbols("option", year_start, year_end)
        pending = [s for s in symbols if not store.partition_exists("open_interest", year, s)]
        print(f"open_interest {year}: {len(symbols)} symbols ({len(pending)} pending)")

        def fetch_and_store(
            symbol: str, year: int = year, start: dt.date = year_start, end: dt.date = year_end
        ) -> int:
            raw_df = fetch_open_interest(symbol, start, end)
            if raw_df is None:
                return 0
            oi_df = canonicalize_open_interest(raw_df).with_columns(
                pl.lit(year, dtype=pl.Int32).alias("year")
            )
            store.write(db, "open_interest", oi_df)
            return oi_df.height

        rows = fetch_many(pending, fetch_and_store, WORKERS)
        print(f"  {sum(rows):,} contract-days written")
