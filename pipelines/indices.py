"""EOD index levels: SPX, RUT, OEX, XSP and the VIX term structure.

CBOE-family indices only; ThetaData does not redistribute NDX, DJI or MOVE.
The free index tier refuses history before 2024-01-01.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.utils.sessions import date_chunks
from pipelines.utils.theta import make_client

INDICES = ["SPX", "RUT", "OEX", "XSP"]
VOL_INDICES = ["VIX1D", "VIX9D", "VIX", "VIX3M", "VIX1Y", "VVIX", "SKEW"]

START = dt.date(2024, 1, 1)
END = dt.date(2025, 12, 31)


def fetch_indices(symbols: list[str], start: dt.date, end: dt.date) -> pl.DataFrame:
    client = make_client()
    frames = []
    for symbol in symbols:
        for chunk_start, chunk_end in date_chunks(start, end):
            try:
                frames.append(
                    client.index_history_eod(symbol, chunk_start, chunk_end).select(
                        pl.col("created").dt.date().alias("date"),
                        pl.lit(symbol).alias("symbol"),
                        "open",
                        "high",
                        "low",
                        "close",
                    )
                )
            except Exception as error:
                print(f"  skipped {symbol} {chunk_start}..{chunk_end}: {str(error)[:90]}")
    if not frames:
        raise RuntimeError(f"no index data returned; the free tier starts around {START}")
    return (
        pl.concat(frames, how="vertical_relaxed").unique(["date", "symbol"]).sort("date", "symbol")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    indices_df = fetch_indices(INDICES + VOL_INDICES, start, end)
    store.write(store.connect(), "indices", indices_df)
    print(f"indices: {indices_df.height} rows, {indices_df['symbol'].n_unique()} symbols")
