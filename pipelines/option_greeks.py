"""EOD option chains with greeks, IV and the underlying price, per symbol-year.

The server requires `expiration=*` a day at a time, so a symbol-year is
~250 requests at ~0.3 s each: a 500-name year is ~3 hours at four workers.
Resumable at symbol-year granularity: a partition is written once its whole
year is fetched, and existing partitions are skipped.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.canonical import canonicalize_chain
from pipelines.utils.sessions import trading_sessions, year_window, years_in
from pipelines.utils.symbols import universe_symbols
from pipelines.utils.theta import fetch_many, is_missing_data, make_client

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
WORKERS = 4


def fetch_chain(symbol: str, sessions: list[dt.date]) -> pl.DataFrame | None:
    """Every contract-day for one symbol, one request per session."""
    client = make_client()
    days = []
    for session in sessions:
        try:
            day_df = client.option_history_greeks_eod(symbol, "*", session, session)
        except Exception as error:
            if is_missing_data(error):
                continue
            raise
        if day_df.height:
            days.append(day_df)
    return pl.concat(days, how="vertical_relaxed") if days else None


def pull_year(db, table: str, symbols: list[str], year: int, start: dt.date, end: dt.date) -> None:
    store.ensure(db, table)
    pending = [s for s in symbols if not store.partition_exists(table, year, s)]
    sessions = trading_sessions(*year_window(year, start, end))
    print(
        f"{table} {year}: {len(symbols)} symbols ({len(pending)} pending), {len(sessions)} sessions"
    )

    def fetch_and_store(symbol: str) -> int:
        raw_df = fetch_chain(symbol, sessions)
        if raw_df is None:
            return 0
        chain_df = canonicalize_chain(raw_df).with_columns(
            pl.lit(year, dtype=pl.Int32).alias("year")
        )
        store.write(db, table, chain_df)
        return chain_df.height

    rows = fetch_many(pending, fetch_and_store, WORKERS)
    print(f"  {sum(rows):,} rows written")


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    for year in years_in(start, end):
        symbols = universe_symbols("option", *year_window(year, start, end))
        pull_year(db, "option_greeks", symbols, year, start, end)
