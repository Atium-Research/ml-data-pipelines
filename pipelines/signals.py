"""Cross-sectional scores, one table with a `signal` name per row.

Every score is a richness z-score: positive means implied vol looks rich for
that name, so a book that sells rich vol shorts it.

`vrp`         ln(iv60) - ln(rv_fcst), regressed cross-sectionally each day on
              beta × SPX premium, the sector median premium and idio vol;
              the residual is averaged over 5 sessions and z-scored.
`iv_zscore`   each name's log iv60 against its own trailing 250-session mean
              and std; no forecast, no regression.
`momentum`    the reference unit's per-vega P&L summed over the 126 sessions
              ending 21 sessions ago, z-scored, sign flipped so a high past
              return reads as *cheap* (Heston, Jones, Khorram, Li and Mo 2023).
"""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from pipelines import store
from pipelines.reference_returns import MARKET_ROOT

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)


@dataclass(frozen=True)
class SignalConfig:
    tenor: int = 60
    smoothing_days: int = 5
    controls: tuple[str, ...] = ("idio_vol",)
    winsor: float = 3.0
    min_names: int = 50
    zscore_window: int = 250
    zscore_min_periods: int = 120
    momentum_window: int = 126
    momentum_skip: int = 21


def compute_vrp(surface_df: pl.DataFrame, forecast_df: pl.DataFrame, tenor: int) -> pl.DataFrame:
    iv_col = f"iv{tenor}"
    return (
        surface_df.select("date", "symbol", iv_col)
        .join(forecast_df.select("date", "symbol", "rv_fcst"), on=["date", "symbol"], how="inner")
        .filter(pl.col(iv_col) > 0, pl.col("rv_fcst") > 0)
        .with_columns((pl.col(iv_col).log() - pl.col("rv_fcst").log()).alias("vrp"))
        .select("date", "symbol", "vrp")
    )


def residualize_daily(
    frame_df: pl.DataFrame, regressors: list[str], min_names: int
) -> pl.DataFrame:
    """OLS of `vrp` on `regressors` with intercept, per date; the residuals."""
    pieces = []
    for (_day,), group_df in frame_df.sort("date").group_by("date", maintain_order=True):
        clean_df = group_df.drop_nulls(["vrp", *regressors])
        if clean_df.height < min_names:
            continue
        x = np.column_stack([np.ones(clean_df.height), clean_df.select(regressors).to_numpy()])
        y = clean_df["vrp"].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        pieces.append(
            clean_df.select("date", "symbol").with_columns(pl.Series("residual", y - x @ beta))
        )
    if not pieces:
        return pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "residual": pl.Float64})
    return pl.concat(pieces)


def compute_vrp_signal(
    surface_df: pl.DataFrame,
    forecast_df: pl.DataFrame,
    features_df: pl.DataFrame,
    sectors_df: pl.DataFrame,
    config: SignalConfig = SignalConfig(),
) -> pl.DataFrame:
    """`(date, symbol, score)`: the residualised, smoothed, z-scored variance premium."""
    vrp_df = compute_vrp(surface_df, forecast_df, config.tenor)
    spx_df = vrp_df.filter(pl.col("symbol") == MARKET_ROOT).select(
        "date", pl.col("vrp").alias("market_vrp")
    )
    frame_df = (
        vrp_df.filter(pl.col("symbol") != MARKET_ROOT)
        .join(
            features_df.select("date", "symbol", "beta", *config.controls),
            on=["date", "symbol"],
            how="left",
        )
        .join(sectors_df.select("symbol", "sector"), on="symbol", how="left")
        .join(spx_df, on="date", how="left")
        .with_columns(
            pl.col("market_vrp").fill_null(pl.col("vrp").mean().over("date")),
        )
        .with_columns(
            (pl.col("beta") * pl.col("market_vrp")).alias("market"),
            pl.col("vrp").median().over("date", "sector").alias("sector_median"),
        )
    )
    regressors = ["market", "sector_median", *config.controls]
    residuals_df = residualize_daily(frame_df, regressors, config.min_names)
    z = ((pl.col("smooth") - pl.col("smooth").mean()) / pl.col("smooth").std()).over("date")
    return (
        residuals_df.sort("symbol", "date")
        .with_columns(
            pl.col("residual")
            .rolling_mean(config.smoothing_days, min_samples=1)
            .over("symbol")
            .alias("smooth")
        )
        .with_columns(z.clip(-config.winsor, config.winsor).alias("score"))
        .select("date", "symbol", "score")
    )


def compute_iv_zscore_signal(
    surface_df: pl.DataFrame, config: SignalConfig = SignalConfig()
) -> pl.DataFrame:
    iv_col = f"iv{config.tenor}"
    rolling = dict(window_size=config.zscore_window, min_samples=config.zscore_min_periods)
    return (
        surface_df.filter(pl.col("symbol") != MARKET_ROOT, pl.col(iv_col) > 0)
        .sort("symbol", "date")
        .with_columns(pl.col(iv_col).log().alias("level"))
        .with_columns(
            pl.col("level").rolling_mean(**rolling).over("symbol").alias("mean"),
            pl.col("level").rolling_std(**rolling).over("symbol").alias("std"),
        )
        .filter(pl.col("std") > 0)
        .with_columns(
            ((pl.col("level") - pl.col("mean")) / pl.col("std"))
            .clip(-config.winsor, config.winsor)
            .alias("score")
        )
        .select("date", "symbol", "score")
    )


def compute_momentum_signal(
    reference_df: pl.DataFrame, config: SignalConfig = SignalConfig()
) -> pl.DataFrame:
    past = (
        pl.col("pnl_per_vega")
        .rolling_sum(config.momentum_window, min_samples=config.momentum_window // 2)
        .shift(config.momentum_skip)
        .over("symbol")
    )
    z = (pl.col("past") - pl.col("past").mean().over("date")) / pl.col("past").std().over("date")
    return (
        reference_df.filter(pl.col("symbol") != MARKET_ROOT)
        .select("date", "symbol", "pnl_per_vega")
        .sort("symbol", "date")
        .with_columns(past.alias("past"))
        .drop_nulls("past")
        .filter(pl.len().over("date") >= config.min_names)
        .with_columns((-z).clip(-config.winsor, config.winsor).alias("score"))
        .select("date", "symbol", "score")
    )


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    config = SignalConfig()
    surface_df = store.in_window(store.scan("surface"), start, end).collect()
    forecast_df = store.in_window(store.scan("forecast"), start, end).collect()
    features_df = store.in_window(store.scan("stock_features"), start, end).collect()
    reference_df = store.in_window(store.scan("reference_returns"), start, end).collect()
    sectors_df = store.scan("sectors").select("symbol", "sector").collect()
    pieces = {
        "vrp": compute_vrp_signal(surface_df, forecast_df, features_df, sectors_df, config),
        "iv_zscore": compute_iv_zscore_signal(surface_df, config),
        "momentum": compute_momentum_signal(reference_df, config),
    }
    signals_df = pl.concat(
        [frame_df.with_columns(pl.lit(name).alias("signal")) for name, frame_df in pieces.items()]
    ).sort("date", "signal", "symbol")
    store.write(db, "signals", signals_df)
    print(
        signals_df.group_by("signal").agg(
            pl.len().alias("rows"), pl.col("date").min().alias("first"), pl.col("date").max().alias("last")
        )
    )
