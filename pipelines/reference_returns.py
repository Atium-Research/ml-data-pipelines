"""The reference unit's daily per-vega P&L for every name, and for SPX.

One straddle path per symbol over the whole window, written one partition
per (year, symbol). Resumable: a symbol with any partition in the window is
skipped. About 0.25 s per symbol-year.
"""

import datetime as dt
import time

import polars as pl

from pipelines import store
from pipelines.straddle import StraddleConfig, compute_reference_path
from pipelines.utils.sessions import trading_sessions, years_in
from pipelines.utils.symbols import universe_symbols

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
MARKET_ROOT = "SPX"


def load_chain(table: str, symbol: str, years: list[int], config: StraddleConfig) -> pl.DataFrame:
    return (
        store.scan(table, years, [symbol])
        .filter(
            (pl.col("expiration") - pl.col("date")).dt.total_days() <= config.chain_max_dte,
            (pl.col("strike") / pl.col("underlying") - 1.0).abs() <= config.chain_max_moneyness,
        )
        .collect()
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    store.ensure(db, "reference_returns")
    config = StraddleConfig()
    years = years_in(start, end)
    sessions = trading_sessions(start, end)
    symbols = universe_symbols("option", start, end)
    jobs = [("option_greeks", s) for s in symbols] + [("index_greeks", MARKET_ROOT)]
    started = time.time()
    written = skipped = 0
    for count, (table, symbol) in enumerate(jobs, start=1):
        if any(store.partition_exists("reference_returns", year, symbol) for year in years):
            skipped += 1
            continue
        try:
            chain_df = load_chain(table, symbol, years, config)
        except FileNotFoundError:
            continue
        path_df = compute_reference_path(chain_df, sessions, config)
        store.write(db, "reference_returns", path_df)
        written += 1
        if count % 25 == 0 or count == len(jobs):
            print(
                f"  reference {count}/{len(jobs)}: {written} written, {skipped} skipped,"
                f" {time.time() - started:.0f}s",
                flush=True,
            )
