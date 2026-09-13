"""Synthetic chains, canonical and vendor-style, from Black-Scholes."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from pipelines.utils import bs


def business_days(start: dt.date, n: int) -> list[dt.date]:
    days, day = [], start
    while len(days) < n:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


def third_friday(year: int, month: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def make_chain(
    dates: list[dt.date],
    spots: dict[dt.date, float],
    ivs: dict[dt.date, float],
    symbol: str = "XYZ",
    expirations: list[dt.date] | None = None,
    strikes: np.ndarray | None = None,
    spread: float = 0.10,
) -> pl.DataFrame:
    """A canonical chain (vega per vol point) priced with Black-Scholes."""
    expirations = expirations or [third_friday(2024, m) for m in range(2, 8)]
    strikes = strikes if strikes is not None else np.arange(80.0, 121.0, 5.0)
    rows = []
    for date_ in dates:
        spot, iv = spots[date_], ivs[date_]
        for expiration in expirations:
            dte = (expiration - date_).days
            if dte <= 0:
                continue
            tau = np.array([dte / 365.0])
            for strike in strikes:
                for right in ("C", "P"):
                    is_call = np.array([right == "C"])
                    args = (
                        np.array([spot]),
                        np.array([strike]),
                        tau,
                        np.array([0.0]),
                        np.array([iv]),
                    )
                    mid = float(bs.price(*args, is_call)[0])
                    rows.append(
                        {
                            "date": date_,
                            "symbol": symbol,
                            "expiration": expiration,
                            "strike": float(strike),
                            "right": right,
                            "bid": max(mid - spread / 2, 0.01),
                            "ask": mid + spread / 2,
                            "mid": mid,
                            "iv": iv,
                            "delta": float(bs.delta(*args, is_call)[0]),
                            "gamma": float(bs.gamma(*args)[0]),
                            "theta": float(bs.theta(*args, is_call)[0]),
                            "vega": float(bs.vega(*args)[0]) / 100.0,
                            "underlying": spot,
                            "volume": 100,
                        }
                    )
    return pl.DataFrame(rows)


def to_vendor_chain(chain_df: pl.DataFrame) -> pl.DataFrame:
    """The same rows spelled the way ThetaData spells them (vega per 1.00 of IV)."""
    return chain_df.select(
        "symbol",
        pl.col("expiration").dt.strftime("%Y-%m-%d"),
        "strike",
        pl.when(pl.col("right") == "C")
        .then(pl.lit("CALL"))
        .otherwise(pl.lit("PUT"))
        .alias("right"),
        "bid",
        "ask",
        "volume",
        "delta",
        "gamma",
        "theta",
        (pl.col("vega") * 100.0).alias("vega"),
        pl.col("iv").alias("implied_vol"),
        pl.lit(0.0).alias("iv_error"),
        pl.col("date")
        .cast(pl.Datetime("ms"))
        .dt.replace_time_zone("America/New_York")
        .alias("underlying_timestamp"),
        pl.col("underlying").alias("underlying_price"),
        pl.lit(0.1).alias("rho"),
        pl.lit(0).cast(pl.Int64).alias("bid_size"),
        pl.col("date")
        .cast(pl.Datetime("ms"))
        .dt.replace_time_zone("America/New_York")
        .alias("timestamp"),
    )


def random_chain(seed: int, n_days: int, symbol: str = "XYZ") -> tuple[list[dt.date], pl.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = business_days(dt.date(2024, 1, 2), n_days)
    spot, spots, ivs = 100.0, {}, {}
    for date_ in dates:
        spot *= float(np.exp(rng.normal(0, 0.02)))
        spots[date_] = spot
        ivs[date_] = float(0.30 + rng.normal(0, 0.02))
    chain_df = make_chain(
        dates,
        spots,
        ivs,
        symbol=symbol,
        expirations=[third_friday(2024, m) for m in range(2, 8)],
        strikes=np.arange(60.0, 141.0, 2.5),
    )
    return dates, chain_df


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_DATA_STORE", str(tmp_path / "store"))
    return tmp_path / "store"
