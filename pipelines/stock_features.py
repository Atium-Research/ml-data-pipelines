"""Per-name features from the spot series, for the signal's controls.

    ret        split-adjusted daily log return
    beta       250-session OLS beta on the SPX log return
    idio_vol   60-session residual vol of that regression, annualised
    gap_freq   share of the trailing 250 sessions with |return| > 3 sigma,
               sigma the trailing 60-session return std

Spot comes off the chains' `underlying`, so there is no volume and no size
proxy; the market return is the SPX chain's underlying.
"""

import datetime as dt
from dataclasses import dataclass

import polars as pl

from pipelines import store
from pipelines.realized_vol import scan_spot
from pipelines.reference_returns import MARKET_ROOT
from pipelines.utils.sessions import years_in
from pipelines.utils.symbols import universe_symbols

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class StockFeatureConfig:
    beta_window: int = 250
    idio_window: int = 60
    gap_window: int = 250
    gap_sigma_window: int = 60
    gap_threshold: float = 3.0


def compute_stock_features(
    closes: pl.LazyFrame, market_df: pl.DataFrame, config: StockFeatureConfig = StockFeatureConfig()
) -> pl.LazyFrame:
    by = "symbol"
    log_return = (pl.col("close") * pl.col("split_ratio") / pl.col("close").shift(1).over(by)).log()
    frame = (
        closes.sort("symbol", "date")
        .with_columns(log_return.alias("ret"))
        .join(market_df.lazy(), on="date", how="inner")
        .sort("symbol", "date")
    )
    beta = (
        pl.rolling_cov("ret", "market_return", window_size=config.beta_window)
        / pl.col("market_return").rolling_var(config.beta_window)
    ).over(by)
    frame = frame.with_columns(beta.alias("beta"))
    frame = frame.with_columns(
        (pl.col("ret") - pl.col("beta") * pl.col("market_return")).alias("residual")
    ).with_columns(
        (
            pl.col("residual").rolling_std(config.idio_window).over(by) * SESSIONS_PER_YEAR**0.5
        ).alias("idio_vol"),
        pl.col("ret").rolling_std(config.gap_sigma_window).over(by).alias("sigma"),
    )
    gap = (pl.col("ret").abs() > config.gap_threshold * pl.col("sigma").shift(1).over(by)).cast(
        pl.Float64
    )
    return frame.with_columns(
        gap.rolling_mean(config.gap_window).over(by).alias("gap_freq")
    ).select("date", "symbol", "ret", "beta", "idio_vol", "gap_freq")


def load_market_return(years: list[int]) -> pl.DataFrame:
    """`(date, market_return)`: the SPX chain's underlying, log-differenced."""
    return (
        scan_spot("index_greeks", [MARKET_ROOT], years)
        .sort("date")
        .select("date", (pl.col("close") / pl.col("close").shift(1)).log().alias("market_return"))
        .collect()
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    years = years_in(start, end)
    market_df = load_market_return(years)
    pieces = []
    symbols = universe_symbols("option", start, end)
    for count, symbol in enumerate(symbols, start=1):
        try:
            closes = scan_spot("option_greeks", [symbol], years)
        except FileNotFoundError:
            continue
        pieces.append(compute_stock_features(closes, market_df).collect())
        if count % 100 == 0:
            print(f"  stock features {count}/{len(symbols)}", flush=True)
    features_df = (
        store.in_window(pl.concat(pieces).lazy(), start, end).collect().sort("date", "symbol")
    )
    store.write(db, "stock_features", features_df)
    print(f"stock features: {features_df.height:,} rows")
