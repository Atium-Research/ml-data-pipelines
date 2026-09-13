"""The volatility factor model on per-vega reference returns, point in time.

    Sigma = B F B' + D^2

Factors: `market`, the SPX reference straddle's per-vega P&L, and one
factor per GICS sector, the vega-weighted mean of its members' per-vega
P&L orthogonalised to the market with a trailing beta. Per name and
session, a trailing `window` regression on the market and the name's own
sector gives the loadings (B) and the residual std (D); the factor
covariance (F) is the Ledoit-Wolf-shrunk sample covariance over the same
window. Four tables: `factor_returns`, `factor_loadings`,
`factor_covariances`, `idio_vol`.
"""

import datetime as dt
import time
from dataclasses import dataclass

import numpy as np
import polars as pl

from pipelines import store
from pipelines.reference_returns import MARKET_ROOT

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
MARKET = "market"


@dataclass(frozen=True)
class FactorSpec:
    window: int = 250
    min_observations: int = 120
    shrinkage: bool = True


def ledoit_wolf(sample: np.ndarray, n_obs: int) -> np.ndarray:
    """Ledoit-Wolf (2004) shrinkage toward the scaled identity."""
    k = sample.shape[0]
    target = np.trace(sample) / k * np.eye(k)
    delta = np.sum((sample - target) ** 2)
    if delta <= 0 or n_obs <= 1:
        return sample.copy()
    beta = min(np.sum(sample**2) / n_obs, delta)
    intensity = beta / delta
    return intensity * target + (1.0 - intensity) * sample


def build_factor_returns(
    returns_df: pl.DataFrame, market_df: pl.DataFrame, sectors_df: pl.DataFrame, spec: FactorSpec
) -> pl.DataFrame:
    """`(date, factor, ret)`: the market, then each sector net of a trailing market beta."""
    clean_df = returns_df.filter(pl.col("pnl_per_vega").is_not_null(), pl.col("dollar_vega") > 0)
    market = (
        market_df.filter(pl.col("pnl_per_vega").is_not_null())
        .select("date", pl.col("pnl_per_vega").alias("ret"))
        .with_columns(pl.lit(MARKET).alias("factor"))
        .select("date", "factor", "ret")
    )
    weighted = (pl.col("pnl_per_vega") * pl.col("dollar_vega")).sum() / pl.col("dollar_vega").sum()
    rolling = dict(window_size=spec.window, min_samples=spec.min_observations)
    sector = (
        clean_df.join(sectors_df, on="symbol", how="inner")
        .group_by("date", "sector")
        .agg(weighted.alias("raw"))
        .join(market.select("date", pl.col("ret").alias("market")), on="date", how="inner")
        .sort("sector", "date")
        .with_columns(
            (pl.rolling_cov("raw", "market", **rolling) / pl.col("market").rolling_var(**rolling))
            .over("sector")
            .alias("beta"),
            pl.col("raw").rolling_mean(**rolling).over("sector").alias("raw_mean"),
            pl.col("market").rolling_mean(**rolling).over("sector").alias("market_mean"),
        )
        .with_columns(
            (
                pl.col("raw")
                - pl.col("raw_mean")
                - pl.col("beta") * (pl.col("market") - pl.col("market_mean"))
            ).alias("ret")
        )
        .select("date", pl.col("sector").alias("factor"), "ret")
        .drop_nulls("ret")
    )
    return pl.concat([market, sector]).sort("date", "factor")


def regress(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS with intercept: slope coefficients and residual std."""
    design = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return beta[1:], float(np.std(residual, ddof=design.shape[1]))


def estimate_rolling_loadings(
    returns_df: pl.DataFrame,
    factor_returns_df: pl.DataFrame,
    sectors_df: pl.DataFrame,
    spec: FactorSpec,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`(date, symbol, factor, loading)` and `(date, symbol, idio_vol)` at every session.

    A name gets an estimate once it has `min_observations` rows; the window
    expands until it reaches `window`.
    """
    factors_df = factor_returns_df.pivot(index="date", on="factor", values="ret").sort("date")
    panel_df = (
        returns_df.filter(pl.col("pnl_per_vega").is_not_null())
        .join(sectors_df, on="symbol", how="left")
        .join(factors_df, on="date", how="inner")
        .sort("symbol", "date")
    )
    loading_rows: list[tuple] = []
    idio_rows: list[tuple] = []
    for (symbol,), group_df in panel_df.group_by("symbol", maintain_order=True):
        sector = group_df["sector"][0]
        columns = [MARKET]
        if sector is not None and sector in factors_df.columns:
            columns.append(sector)
        clean_df = group_df.drop_nulls(columns)
        if clean_df.height < spec.min_observations:
            continue
        dates = clean_df["date"].to_list()
        y = clean_df["pnl_per_vega"].to_numpy()
        x = clean_df.select(columns).to_numpy()
        for end in range(spec.min_observations, clean_df.height + 1):
            begin = max(0, end - spec.window)
            beta, idio = regress(y[begin:end], x[begin:end])
            day = dates[end - 1]
            loading_rows.extend(
                (day, symbol, name, float(b)) for name, b in zip(columns, beta, strict=True)
            )
            idio_rows.append((day, symbol, idio))
    loadings_df = pl.DataFrame(
        loading_rows,
        schema={"date": pl.Date, "symbol": pl.String, "factor": pl.String, "loading": pl.Float64},
        orient="row",
    )
    idio_df = pl.DataFrame(
        idio_rows,
        schema={"date": pl.Date, "symbol": pl.String, "idio_vol": pl.Float64},
        orient="row",
    )
    return loadings_df.sort("date", "symbol", "factor"), idio_df.sort("date", "symbol")


def estimate_rolling_covariances(factor_returns_df: pl.DataFrame, spec: FactorSpec) -> pl.DataFrame:
    """`(date, factor_1, factor_2, covariance)` at every session, shrunk."""
    factors_df = factor_returns_df.pivot(index="date", on="factor", values="ret").sort("date")
    names = [MARKET] + sorted(c for c in factors_df.columns if c not in ("date", MARKET))
    matrix = factors_df.select(names).fill_null(0.0).to_numpy()
    dates = factors_df["date"].to_list()
    rows: list[tuple] = []
    for end in range(spec.min_observations, len(dates) + 1):
        window = matrix[max(0, end - spec.window) : end]
        sample = np.atleast_2d(np.cov(window, rowvar=False, ddof=1))
        covariance = ledoit_wolf(sample, window.shape[0]) if spec.shrinkage else sample
        day = dates[end - 1]
        rows.extend(
            (day, a, b, float(covariance[i, j]))
            for i, a in enumerate(names)
            for j, b in enumerate(names)
        )
    return pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "factor_1": pl.String,
            "factor_2": pl.String,
            "covariance": pl.Float64,
        },
        orient="row",
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    spec = FactorSpec()
    started = time.time()
    all_df = (
        store.in_window(store.scan("reference_returns"), start, end)
        .select("date", "symbol", "pnl_per_vega", "dollar_vega")
        .collect()
    )
    market_df = all_df.filter(pl.col("symbol") == MARKET_ROOT).select("date", "pnl_per_vega")
    returns_df = all_df.filter(pl.col("symbol") != MARKET_ROOT)
    sectors_df = store.scan("sectors").select("symbol", "sector").collect()

    factor_returns_df = build_factor_returns(returns_df, market_df, sectors_df, spec)
    store.write(db, "factor_returns", factor_returns_df)
    print(
        f"  factor returns: {factor_returns_df['factor'].n_unique()} factors"
        f" ({time.time() - started:.0f}s)"
    )

    covariances_df = estimate_rolling_covariances(factor_returns_df, spec)
    store.write(db, "factor_covariances", covariances_df)
    print(
        f"  factor covariances: {covariances_df['date'].n_unique()} sessions"
        f" ({time.time() - started:.0f}s)"
    )

    loadings_df, idio_df = estimate_rolling_loadings(
        returns_df, factor_returns_df, sectors_df, spec
    )
    store.write(db, "factor_loadings", loadings_df)
    store.write(db, "idio_vol", idio_df)
    print(
        f"  loadings: {loadings_df.height:,} rows, idio vol: {idio_df.height:,} rows"
        f" ({time.time() - started:.0f}s)"
    )
