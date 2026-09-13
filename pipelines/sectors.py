"""GICS sector for each current S&P 500 constituent, from Wikipedia.

A static snapshot: Wikipedia carries the classification of today's members
only, so a name that left the index is missing and a name that changed
sector carries its current one.
"""

import polars as pl

from pipelines import store
from pipelines.universe import CURRENT_URL, fetch_first_table
from pipelines.utils.symbols import normalize_ticker


def fetch_sectors() -> pl.DataFrame:
    return (
        pl.from_pandas(fetch_first_table(CURRENT_URL))
        .select(
            pl.col("Symbol").alias("ticker"),
            pl.col("GICS Sector").alias("sector"),
            pl.col("GICS Sub-Industry").alias("sub_industry"),
        )
        .drop_nulls("ticker")
        .with_columns(
            pl.col("ticker")
            .map_elements(lambda t: normalize_ticker(t, "option"), return_dtype=pl.String)
            .alias("symbol")
        )
        .sort("ticker")
    )


def run() -> None:
    sectors_df = fetch_sectors()
    store.write(store.connect(), "sectors", sectors_df)
    print(f"sectors: {sectors_df.height} names, {sectors_df['sector'].n_unique()} sectors")
