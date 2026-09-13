"""Earnings announcement dates from Yahoo, for the whole universe.

Yahoo stamps each announcement with its scheduled time; 16:00 or later is
after the close (`amc`, moves the *next* close-to-close return), before
09:30 is before the open (`bmo`). About 100 announcements per name reach
back to roughly 2002.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
import yfinance as yf

from pipelines import store
from pipelines.utils.symbols import symbol_map

LIMIT = 100
WORKERS = 4
MARKET_OPEN_HOUR = 9.5
MARKET_CLOSE_HOUR = 16.0


def fetch_earnings(yahoo_symbol: str, attempts: int = 3) -> pl.DataFrame | None:
    for attempt in range(attempts):
        try:
            raw = yf.Ticker(yahoo_symbol).get_earnings_dates(limit=LIMIT)
            break
        except Exception:
            if attempt == attempts - 1:
                return None
            time.sleep(2**attempt)
    if raw is None or raw.empty:
        return None
    frame = raw.reset_index()
    frame.columns = ["announced_at", "eps_estimate", "reported_eps", "surprise_pct"][
        : len(frame.columns)
    ]
    return pl.from_pandas(frame)


def classify_session(frame_df: pl.DataFrame) -> pl.DataFrame:
    hour = pl.col("announced_at").dt.hour() + pl.col("announced_at").dt.minute() / 60
    return frame_df.with_columns(
        pl.col("announced_at").dt.date().alias("date"),
        pl.when(hour >= MARKET_CLOSE_HOUR)
        .then(pl.lit("amc"))
        .when(hour < MARKET_OPEN_HOUR)
        .then(pl.lit("bmo"))
        .otherwise(pl.lit("unknown"))
        .alias("session"),
    )


def run() -> None:
    names_df = symbol_map()
    pairs = list(zip(names_df["option_symbol"], names_df["yahoo_symbol"], strict=True))
    frames, missing = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(fetch_earnings, yahoo): symbol for symbol, yahoo in pairs}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            frame_df = future.result()
            if frame_df is None:
                missing.append(symbol)
            else:
                frames.append(frame_df.with_columns(pl.lit(symbol).alias("symbol")))
            print(f"\r  {done}/{len(pairs)} | {len(missing)} without data", end="", flush=True)
    print()
    earnings_df = (
        classify_session(pl.concat(frames, how="vertical_relaxed"))
        .select(
            "symbol",
            "date",
            "session",
            pl.col("announced_at").dt.convert_time_zone("America/New_York"),
            pl.col("eps_estimate").cast(pl.Float64),
            pl.col("reported_eps").cast(pl.Float64),
            pl.col("surprise_pct").cast(pl.Float64),
        )
        .unique(subset=["symbol", "date"])
        .sort("symbol", "date")
    )
    store.write(store.connect(), "earnings", earnings_df)
    print(
        f"earnings: {earnings_df.height:,} announcements,"
        f" {earnings_df['symbol'].n_unique()} symbols"
    )
    if missing:
        print(f"  no data for {len(missing)}: {sorted(missing)[:20]}")
