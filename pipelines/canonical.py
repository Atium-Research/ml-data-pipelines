"""ThetaData rows to the canonical schemas the store holds.

This is the only module that knows how ThetaData spells a column. Unit
conventions fixed here:

* `iv` is a decimal, null where the vendor's inversion failed
  (`|iv_error| > 1`) or the vendor reports 0.
* `vega` is per share per **vol point** (a 0.01 change in `iv`). ThetaData
  quotes vega per 1.00 of IV; `detect_vega_scale` confirms that by repricing
  a sample with Black-Scholes and the loader rescales accordingly.
* `theta` is per share per calendar day; `delta` and `gamma` per share.
* `right` is `"C"` / `"P"`; `date` comes from `underlying_timestamp`, the
  stamp on the spot print the greeks were struck against.
* `symbol` is the option-root spelling (BRK.B -> BRKB) in every table.

Every other column the vendor delivers (higher-order greeks, trade OHLC,
quote sizes and conditions, the raw `implied_vol` and `iv_error`, the
timestamps) is carried through unchanged after the canonical columns, so
the store is a complete copy and the vendor files can be deleted.
"""

import numpy as np
import polars as pl

from pipelines.utils import bs

IV_ERROR_TOLERANCE = 1.0

VENDOR_EXTRAS = [
    "implied_vol",
    "iv_error",
    "rho",
    "epsilon",
    "lambda",
    "vanna",
    "charm",
    "vomma",
    "veta",
    "vera",
    "speed",
    "zomma",
    "color",
    "ultima",
    "d1",
    "d2",
    "dual_delta",
    "dual_gamma",
    "open",
    "high",
    "low",
    "close",
    "count",
    "bid_size",
    "bid_exchange",
    "bid_condition",
    "ask_size",
    "ask_exchange",
    "ask_condition",
    "timestamp",
    "underlying_timestamp",
]
STOCK_EXTRAS = [
    "count",
    "bid",
    "ask",
    "bid_size",
    "bid_exchange",
    "bid_condition",
    "ask_size",
    "ask_exchange",
    "ask_condition",
    "created",
    "last_trade",
]


def detect_vega_scale(raw_df: pl.DataFrame, rate: float = 0.04, rows: int = 400) -> float:
    """0.01 when the vendor quotes vega per 1.00 of IV, 1.0 when per vol point.

    Prices each sampled contract at IV and IV + 0.01 and asks which scaling
    of the vendor's vega matches the price change. Needs a two-thirds
    majority; raises otherwise rather than guessing.
    """
    sample_df = (
        raw_df.with_columns(
            (
                pl.col("expiration").str.to_date("%Y-%m-%d")
                - pl.col("underlying_timestamp").dt.date()
            )
            .dt.total_days()
            .alias("dte")
        )
        .filter(
            pl.col("bid") > 0,
            pl.col("iv_error").abs() <= 0.01,
            pl.col("implied_vol").is_between(0.05, 2.0),
            pl.col("dte").is_between(10, 365),
            pl.col("delta").abs().is_between(0.2, 0.8),
            pl.col("vega") > 0,
        )
        .head(rows)
    )
    if sample_df.height < 20:
        raise ValueError("too few clean rows to check the vega convention")
    spot = sample_df["underlying_price"].to_numpy()
    strike = sample_df["strike"].to_numpy()
    tau = sample_df["dte"].to_numpy() / bs.DAYS_PER_YEAR
    sigma = sample_df["implied_vol"].to_numpy()
    is_call = (sample_df["right"] == "CALL").to_numpy()
    vendor_vega = sample_df["vega"].to_numpy()
    price_change = bs.price(spot, strike, tau, rate, sigma + 0.01, is_call) - bs.price(
        spot, strike, tau, rate, sigma, is_call
    )
    per_unit_wins = np.abs(price_change - vendor_vega * 0.01) < np.abs(price_change - vendor_vega)
    share_per_unit = float(per_unit_wins.mean())
    if share_per_unit >= 2 / 3:
        return 0.01
    if share_per_unit <= 1 / 3:
        return 1.0
    raise ValueError(f"vega convention ambiguous: per-unit share {share_per_unit:.2f}")


def canonicalize_chain(raw_df: pl.DataFrame, vega_scale: float | None = None) -> pl.DataFrame:
    """Vendor greeks rows to `CANONICAL_OPTION_SCHEMA` (without `year`)."""
    if vega_scale is None:
        vega_scale = detect_vega_scale(raw_df)
    extras = [pl.col(column) for column in VENDOR_EXTRAS if column in raw_df.columns]
    return raw_df.select(
        pl.col("underlying_timestamp").dt.date().alias("date"),
        pl.col("symbol").cast(pl.String),
        pl.col("expiration").str.to_date("%Y-%m-%d"),
        pl.col("strike").cast(pl.Float64),
        pl.when(pl.col("right") == "CALL").then(pl.lit("C")).otherwise(pl.lit("P")).alias("right"),
        pl.col("bid").cast(pl.Float64),
        pl.col("ask").cast(pl.Float64),
        ((pl.col("bid") + pl.col("ask")) / 2.0).alias("mid"),
        pl.when((pl.col("iv_error").abs() <= IV_ERROR_TOLERANCE) & (pl.col("implied_vol") > 0))
        .then(pl.col("implied_vol"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("iv"),
        pl.col("delta").cast(pl.Float64),
        pl.col("gamma").cast(pl.Float64),
        pl.col("theta").cast(pl.Float64),
        (pl.col("vega") * vega_scale).cast(pl.Float64).alias("vega"),
        pl.col("underlying_price").cast(pl.Float64).alias("underlying"),
        pl.col("volume").cast(pl.Int64),
        *extras,
    ).sort("date", "expiration", "strike", "right")


def canonicalize_open_interest(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Vendor open-interest rows to `(date, symbol, expiration, strike, right, oi)`.

    The vendor stamps OI pre-open with the position after the previous
    close, which is what a trader at today's close knows, so `date` joins
    onto the same session's chain without a shift.
    """
    return (
        raw_df.select(
            pl.col("timestamp").dt.date().alias("date"),
            pl.col("symbol").cast(pl.String),
            pl.col("expiration").str.to_date("%Y-%m-%d"),
            pl.col("strike").cast(pl.Float64),
            pl.when(pl.col("right") == "CALL")
            .then(pl.lit("C"))
            .otherwise(pl.lit("P"))
            .alias("right"),
            pl.col("open_interest").cast(pl.Int64).alias("oi"),
        )
        .unique(subset=["date", "symbol", "expiration", "strike", "right"], keep="last")
        .sort("date", "expiration", "strike", "right")
    )


def canonicalize_stock(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Vendor stock EOD rows with a `date` and the option-root `symbol`, every column kept.

    The vendor emits a final row for a delisted name with a zero close; it is
    kept here so the store is complete, and a return should filter
    `close > 0` before it books a -100% day.
    """
    extras = [pl.col(column) for column in STOCK_EXTRAS if column in raw_df.columns]
    return raw_df.select(
        pl.col("created").dt.date().alias("date"),
        pl.col("symbol").cast(pl.String).str.replace_all(r"\.", ""),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Int64),
        *extras,
    ).sort("date", "symbol")
