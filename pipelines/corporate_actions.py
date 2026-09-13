"""Splits and dividends from Yahoo, one row per (symbol, date, action).

ThetaData serves raw prices and no splits, so a return off `close` books a
-93% day on ORLY's 15:1. Split values are ex-date ratios; spinoff
adjustment factors arrive as non-integer "splits", which is what a return
wants too. Always pulled to today: Yahoo back-adjusts for every split up to
the present, not the end of a window.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.utils.symbols import symbol_map
from pipelines.utils.yahoo import fetch_yahoo

START = dt.date(2017, 1, 1)


def build_actions(quotes_df: pl.DataFrame, names_df: pl.DataFrame) -> pl.DataFrame:
    named_df = quotes_df.join(
        names_df.select(pl.col("option_symbol").alias("symbol"), "yahoo_symbol"),
        on="yahoo_symbol",
        how="inner",
    )
    pieces = []
    for action, column in (("split", "split"), ("dividend", "dividend")):
        pieces.append(
            named_df.filter(pl.col(column) > 0).select(
                "symbol", "date", pl.lit(action).alias("action"), pl.col(column).alias("value")
            )
        )
    return pl.concat(pieces).sort("symbol", "date", "action")


def run(start: dt.date = START) -> None:
    names_df = symbol_map()
    quotes_df = fetch_yahoo(names_df["yahoo_symbol"].to_list(), start)
    actions_df = build_actions(quotes_df, names_df)
    store.write(store.connect(), "corporate_actions", actions_df)
    splits = actions_df.filter(pl.col("action") == "split").height
    print(f"corporate actions: {splits} splits, {actions_df.height - splits:,} dividends")
