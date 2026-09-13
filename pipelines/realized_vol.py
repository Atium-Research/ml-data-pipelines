"""Close-to-close realized vol from the spot stamped on every chain row.

The stock table starts mid-2023; the chains carry the spot every greek was
struck against back to 2017, so this is the estimator that reaches the whole
history. Realized variance over `n` sessions is the mean squared
split-adjusted log return, annualised; `rv_fwd` on date d is the vol over
sessions d+1 .. d+60, null until that window has closed. SPX is included.
"""

import datetime as dt
from dataclasses import dataclass

import polars as pl

from pipelines import store
from pipelines.reference_returns import MARKET_ROOT
from pipelines.utils.sessions import years_in
from pipelines.utils.symbols import universe_symbols

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class RealizedVolConfig:
    windows: tuple[int, ...] = (1, 5, 22)
    forward_window: int = 60
    min_price: float = 1.0


def scan_spot(table: str, symbols: list[str], years: list[int]) -> pl.LazyFrame:
    """`(date, symbol, close, split_ratio)` off the chains' `underlying`."""
    spot = (
        store.scan(table, years, symbols)
        .filter(pl.col("underlying") > 0)
        .select("date", "symbol", pl.col("underlying").alias("close"))
        .unique(subset=["date", "symbol"])
    )
    try:
        splits = (
            store.scan("corporate_actions")
            .filter(pl.col("action") == "split")
            .select("date", "symbol", pl.col("value").alias("split_ratio"))
            .unique(subset=["date", "symbol"])
        )
        spot = spot.join(splits, on=["date", "symbol"], how="left")
    except FileNotFoundError:
        spot = spot.with_columns(pl.lit(None, dtype=pl.Float64).alias("split_ratio"))
    return spot.with_columns(pl.col("split_ratio").fill_null(1.0)).sort("symbol", "date")


def compute_close_to_close_panel(
    closes: pl.LazyFrame, config: RealizedVolConfig = RealizedVolConfig()
) -> pl.LazyFrame:
    """`(date, symbol, rv_1, rv_5, rv_22, rv_fwd)` as annualised vols."""
    log_return = (
        pl.col("close") * pl.col("split_ratio") / pl.col("close").shift(1).over("symbol")
    ).log()
    frame = (
        closes.filter(pl.col("close") >= config.min_price)
        .sort("symbol", "date")
        .with_columns((log_return**2).alias("squared"))
    )

    def variance(n: int) -> pl.Expr:
        daily = pl.col("squared") if n == 1 else pl.col("squared").rolling_mean(n)
        return daily * SESSIONS_PER_YEAR

    n = config.forward_window
    return (
        frame.with_columns(
            [variance(w).over("symbol").sqrt().alias(f"rv_{w}") for w in config.windows]
        )
        .with_columns(variance(n).shift(-n).over("symbol").sqrt().alias("rv_fwd"))
        .select("date", "symbol", *[f"rv_{w}" for w in config.windows], "rv_fwd")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    years = years_in(start, end)
    pieces = []
    symbols = universe_symbols("option", start, end)
    for count, symbol in enumerate(symbols, start=1):
        try:
            pieces.append(
                compute_close_to_close_panel(scan_spot("option_greeks", [symbol], years)).collect()
            )
        except FileNotFoundError:
            continue
        if count % 100 == 0:
            print(f"  realized vol {count}/{len(symbols)}", flush=True)
    pieces.append(
        compute_close_to_close_panel(scan_spot("index_greeks", [MARKET_ROOT], years)).collect()
    )
    rv_df = store.in_window(pl.concat(pieces).lazy(), start, end).collect().sort("date", "symbol")
    store.write(db, "realized_vol", rv_df)
    print(f"realized vol: {rv_df.height:,} rows")
