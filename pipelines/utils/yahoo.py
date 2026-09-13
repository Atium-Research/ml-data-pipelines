"""Yahoo Finance downloads shared by more than one step."""

import datetime as dt
import time

import polars as pl
import yfinance as yf


def fetch_yahoo(yahoo_symbols: list[str], start: dt.date, chunk_size: int = 50) -> pl.DataFrame:
    """Close, dividends and splits for every symbol, in batched downloads.

    Runs from `start` to today on purpose: Yahoo back-adjusts its closes for
    every split up to the present, so stopping at the end of a window
    silently omits later splits and the whole name then disagrees.
    """
    frames = []
    chunks = [yahoo_symbols[i : i + chunk_size] for i in range(0, len(yahoo_symbols), chunk_size)]
    for number, chunk in enumerate(chunks, start=1):
        raw = yf.download(
            chunk,
            start=start,
            end=dt.date.today(),
            actions=True,
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            print(f"\r  chunk {number}/{len(chunks)}: empty", end="", flush=True)
            continue
        tidy = (
            raw.loc[:, ["Close", "Dividends", "Stock Splits"]]
            .stack(level="Ticker", future_stack=True)
            .reset_index()
            .rename(
                columns={
                    "Date": "date",
                    "Ticker": "yahoo_symbol",
                    "Close": "yahoo_close",
                    "Dividends": "dividend",
                    "Stock Splits": "split",
                }
            )
        )
        tidy["date"] = tidy["date"].dt.date
        frames.append(
            pl.from_pandas(tidy).select(
                "yahoo_symbol",
                pl.col("date").cast(pl.Date),
                pl.col("yahoo_close").cast(pl.Float64),
                pl.col("dividend").cast(pl.Float64),
                pl.col("split").cast(pl.Float64),
            )
        )
        print(f"\r  chunk {number}/{len(chunks)} done", end="", flush=True)
        time.sleep(0.5)
    print()
    if not frames:
        raise RuntimeError("Yahoo returned nothing for the whole universe")
    return pl.concat(frames, how="vertical_relaxed").drop_nulls("yahoo_close")
