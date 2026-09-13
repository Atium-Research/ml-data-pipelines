"""The CBOE treasury yield curve (13w, 5y, 10y, 30y) as decimal yields, from Yahoo.

Yahoo rather than ThetaData because the free index tier refuses anything
before 2024, which would leave every option year to 2023 without a discount
rate; the two sources agree to six decimals where both answer.
"""

import datetime as dt

import polars as pl
import yfinance as yf

from pipelines import store

YAHOO_YIELD_INDICES = {"^IRX": "13w", "^FVX": "5y", "^TNX": "10y", "^TYX": "30y"}

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)


def fetch_yields(start: dt.date, end: dt.date) -> pl.DataFrame:
    frames = []
    for ticker, tenor in YAHOO_YIELD_INDICES.items():
        history = yf.Ticker(ticker).history(
            start=start, end=end + dt.timedelta(days=1), auto_adjust=False
        )["Close"]
        frames.append(
            pl.DataFrame(
                {
                    "date": [stamp.date() for stamp in history.index],
                    "tenor": tenor,
                    "yield": history.values / 100,  # Yahoo quotes percent
                }
            )
        )
    return (
        pl.concat(frames, how="vertical_relaxed")
        .filter(pl.col("yield").is_not_nan() & pl.col("yield").is_not_null())
        .unique(["date", "tenor"])
        .sort("date", "tenor")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    yields_df = fetch_yields(start, end)
    store.write(store.connect(), "yields", yields_df)
    print(
        f"yields: {yields_df.height} rows, {yields_df['date'].min()} .. {yields_df['date'].max()}"
    )
