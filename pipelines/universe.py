"""Point-in-time S&P 500 membership from Wikipedia.

Walks today's constituent list backwards through the "selected changes"
table, undoing each change as it passes. Tickers are Wikipedia's; `symbol`
is the option-root spelling every other table uses. The walk starts from
*today's* list, so a rebuild months later can differ slightly.
"""

import datetime as dt
import io
import os

import pandas as pd
import polars as pl
import requests

from pipelines import store
from pipelines.utils.sessions import trading_sessions
from pipelines.utils.symbols import normalize_ticker

CURRENT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"

START = dt.date(2016, 1, 1)
END = dt.date(2025, 12, 31)


def fetch_first_table(url: str) -> pd.DataFrame:
    user_agent = os.getenv("WIKIPEDIA_USER_AGENT", "ml-data-pipelines/0.1 (data pipeline)")
    response = requests.get(url, headers={"User-Agent": user_agent}, timeout=60)
    response.raise_for_status()
    return pd.read_html(io.StringIO(response.text))[0]


def clean_current_constituents(current_df: pd.DataFrame) -> pl.DataFrame:
    return (
        pl.from_pandas(current_df)
        .select(pl.col("Symbol").alias("ticker"))
        .drop_nulls("ticker")
        .sort("ticker")
    )


def clean_constituent_changes(changes_df: pd.DataFrame) -> pl.DataFrame:
    pieces = []
    for action in ("Added", "Removed"):
        piece_df = changes_df[["Effective Date", action, "Reason"]].copy()
        piece_df.columns = piece_df.columns.droplevel(0)
        piece_df.columns = ["effective_date", "ticker", "security", "reason"]
        piece_df["action"] = action
        pieces.append(piece_df)
    return (
        pl.from_pandas(pd.concat(pieces, ignore_index=True))
        .with_columns(
            pl.col("effective_date").str.strptime(pl.Date, "%B %d, %Y", strict=False),
            # a few cells carry trailing wiki markup ("ITT |"); keep the symbol
            pl.col("ticker").cast(pl.String).str.extract(r"^\s*([A-Za-z0-9.\-]+)"),
        )
        .drop_nulls(["ticker", "effective_date"])
    )


def build_universe(
    current_df: pl.DataFrame, changes_df: pl.DataFrame, sessions: list[dt.date]
) -> pl.DataFrame:
    """Walk backwards from today's members, undoing each change as we pass it."""
    constituents = set(current_df["ticker"].to_list())
    changes_by_date = {
        row["effective_date"]: list(zip(row["ticker"], row["action"], strict=True))
        for row in changes_df.group_by("effective_date")
        .agg(pl.col("ticker"), pl.col("action"))
        .iter_rows(named=True)
    }

    def undo(effective_date: dt.date) -> None:
        for ticker, action in changes_by_date.get(effective_date, []):
            if action == "Added":
                constituents.discard(ticker)
            else:
                constituents.add(ticker)

    for effective_date in sorted(changes_by_date, reverse=True):
        if effective_date <= max(sessions):
            break
        undo(effective_date)

    snapshots = []
    for session in sorted(sessions, reverse=True):
        snapshots.append({"date": session, "ticker": sorted(constituents)})
        undo(session)

    return (
        pl.DataFrame(snapshots)
        .explode("ticker")
        .with_columns(
            pl.col("ticker")
            .map_elements(lambda t: normalize_ticker(t, "option"), return_dtype=pl.String)
            .alias("symbol")
        )
        .sort("date", "ticker")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    current_df = clean_current_constituents(fetch_first_table(CURRENT_URL))
    changes_df = clean_constituent_changes(fetch_first_table(CHANGES_URL))
    universe_df = build_universe(current_df, changes_df, trading_sessions(start, end))
    store.write(store.connect(), "universe", universe_df)
    print(
        f"universe: {universe_df['date'].n_unique()} sessions,"
        f" {universe_df['ticker'].n_unique()} tickers over the window"
    )
