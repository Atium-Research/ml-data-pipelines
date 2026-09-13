"""Overnight SOFR as a decimal rate, from ThetaData. Tier-capped at 2024 like the index levels."""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.utils.sessions import date_chunks
from pipelines.utils.theta import make_client

RATES = ["SOFR"]

START = dt.date(2024, 1, 1)
END = dt.date(2025, 12, 31)


def fetch_rates(start: dt.date, end: dt.date) -> pl.DataFrame:
    client = make_client()
    frames = []
    for symbol in RATES:
        for chunk_start, chunk_end in date_chunks(start, end):
            frames.append(
                client.interest_rate_history_eod(symbol, chunk_start, chunk_end).select(
                    pl.col("created").str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
                    pl.lit(symbol).alias("symbol"),
                    (pl.col("rate") / 100).alias("rate"),
                )
            )
    return (
        pl.concat(frames, how="vertical_relaxed").unique(["date", "symbol"]).sort("date", "symbol")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    rates_df = fetch_rates(start, end)
    store.write(store.connect(), "rates", rates_df)
    print(f"rates: {rates_df.height} rows")
