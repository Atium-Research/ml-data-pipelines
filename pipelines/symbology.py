"""Does a stored option root actually belong to the name the universe asked for?

ThetaData answers an unknown symbol with *some* instrument rather than an
error, and the universe carries today's ticker at every historical date, so
a backfill asks for META in 2016 or FI in 2019 and can be handed another
company's chain. The check compares the chain's `underlying` against Yahoo's
close for the modern ticker, on log returns rather than levels: a split is
one outlier day and a constant price ratio differences away, so the same
company scores a median absolute return difference of ~0 and a different
one ~0.005-0.011. Costs no ThetaData requests.

Statuses: `ok`; `thin_overlap` (fewer than 20 overlapping days, unverified,
not wrong: it falls on names Yahoo no longer serves); `suspect`;
`wrong_instrument`. `action_gap_days` counts days the vendors disagree by
more than 5%, usually an unadjusted spinoff: the symbol is right, that one
return is not.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.utils.sessions import years_in
from pipelines.utils.symbols import symbol_map
from pipelines.utils.yahoo import fetch_yahoo

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)

MAX_MEDIAN_RETURN_DIFFERENCE = 0.001
CLEARLY_WRONG_RETURN_DIFFERENCE = 0.005
MIN_OVERLAP_DAYS = 20
CORPORATE_ACTION_RETURN_GAP = 0.05
MAX_ABSOLUTE_LOG_RETURN = 0.3


def read_stored_spots(year: int) -> pl.DataFrame:
    """One (symbol, date, spot) per stored symbol-day, from the chains' `underlying`."""
    return (
        store.scan("option_greeks", [year])
        .filter(pl.col("underlying") > 0)
        .group_by("symbol", "date")
        .agg(pl.col("underlying").median().alias("theta_spot"))
        .collect()
    )


def check_year(year: int) -> pl.DataFrame:
    spots_df = read_stored_spots(year)
    mapping_df = symbol_map().select("option_symbol", "yahoo_symbol").unique("option_symbol")
    named_df = spots_df.join(mapping_df, left_on="symbol", right_on="option_symbol", how="inner")
    wanted = sorted(named_df["yahoo_symbol"].unique().to_list())
    print(f"  {year}: {len(wanted)} symbols, {named_df.height:,} symbol-days; asking Yahoo")
    yahoo_df = (
        fetch_yahoo(wanted, dt.date(year, 1, 1))
        .join(mapping_df, on="yahoo_symbol", how="inner")
        .select(pl.col("option_symbol").alias("symbol"), "date", "yahoo_close")
        .filter(pl.col("yahoo_close") > 0, pl.col("date") <= dt.date(year, 12, 31))
    )
    returns_df = (
        named_df.join(yahoo_df, on=["symbol", "date"], how="inner")
        .sort("symbol", "date")
        .with_columns(
            pl.col("theta_spot").log().diff().over("symbol").alias("theta_return"),
            pl.col("yahoo_close").log().diff().over("symbol").alias("yahoo_return"),
        )
        .drop_nulls(["theta_return", "yahoo_return"])
        .filter(
            pl.col("theta_return").abs() < MAX_ABSOLUTE_LOG_RETURN,
            pl.col("yahoo_return").abs() < MAX_ABSOLUTE_LOG_RETURN,
        )
        .with_columns((pl.col("theta_return") - pl.col("yahoo_return")).abs().alias("return_gap"))
    )
    compared_df = returns_df.group_by("symbol").agg(
        pl.len().alias("overlap_days"),
        pl.corr("theta_return", "yahoo_return").alias("return_correlation"),
        pl.col("return_gap").median().alias("median_return_difference"),
        (pl.col("return_gap") > CORPORATE_ACTION_RETURN_GAP).sum().alias("action_gap_days"),
    )
    return (
        named_df.group_by("symbol")
        .agg(pl.len().alias("theta_days"))
        .join(compared_df, on="symbol", how="left")
        .with_columns(pl.lit(year).alias("year"), pl.col("overlap_days").fill_null(0))
        .with_columns(
            pl.when(pl.col("overlap_days") < MIN_OVERLAP_DAYS)
            .then(pl.lit("thin_overlap"))
            .when(pl.col("median_return_difference") <= MAX_MEDIAN_RETURN_DIFFERENCE)
            .then(pl.lit("ok"))
            .when(pl.col("median_return_difference") >= CLEARLY_WRONG_RETURN_DIFFERENCE)
            .then(pl.lit("wrong_instrument"))
            .otherwise(pl.lit("suspect"))
            .alias("status")
        )
        .sort("status", "symbol")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    years = years_in(start, end)
    checks_df = pl.concat([check_year(year) for year in years], how="vertical_relaxed")
    db = store.connect()
    # accumulate: the table is the record of every year checked so far
    try:
        kept_df = store.scan("symbology_check").filter(~pl.col("year").is_in(years)).collect()
        checks_df = pl.concat([kept_df, store.conform("symbology_check", checks_df)])
    except FileNotFoundError:
        pass
    store.write(db, "symbology_check", checks_df.sort("year", "status", "symbol"))
    print(checks_df.group_by("year", "status").len().sort("year", "status"))
    wrong_df = checks_df.filter(pl.col("status") == "wrong_instrument")
    if wrong_df.height:
        print("\nthese symbol-years are a different company and must be dropped:")
        print(wrong_df)
