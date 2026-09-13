"""A 60-session realized-vol forecast: pooled log-HAR (Corsi 2009), point in time.

On the first session of each month the regression of log forward variance
on the three trailing log RVs is refit on every single-name row whose
forward window closed before that session, expanding. Between refits the
last coefficients score each day's trailing RVs. SPX is scored with the
pooled coefficients but never enters the fit. Forecasts are vols, with the
lognormal correction exp(sigma_e^2 / 2) on the variance.
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
class HARConfig:
    windows: tuple[int, ...] = (1, 5, 22)
    forward_window: int = 60
    min_fit_rows: int = 1000


def select_refit_dates(sessions: list[dt.date]) -> list[dt.date]:
    refits, last = [], None
    for day in sessions:
        if (day.year, day.month) != last:
            refits.append(day)
            last = (day.year, day.month)
    return refits


def forecast(
    panel_df: pl.DataFrame, apply_df: pl.DataFrame | None = None, config: HARConfig = HARConfig()
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """`(date, symbol, rv_fcst)` for every row, plus the coefficient history."""
    features = [f"rv_{w}" for w in config.windows]
    panel_df = panel_df.with_columns(pl.lit(True).alias("in_fit"))
    if apply_df is not None:
        panel_df = pl.concat(
            [
                panel_df,
                apply_df.select(panel_df.columns[:-1]).with_columns(pl.lit(False).alias("in_fit")),
            ]
        )
    clean_df = (
        panel_df.filter(
            pl.all_horizontal([pl.col(f) > 0 for f in features]),
            pl.all_horizontal([pl.col(f).is_finite() for f in features]),
        )
        .with_columns([pl.col(f).log().alias(f"x_{f}") for f in features])
        .with_columns(
            pl.when(pl.col("rv_fwd") > 0)
            .then((pl.col("rv_fwd") ** 2).log())
            .otherwise(None)
            .alias("y")
        )
        .sort("date", "symbol")
    )
    sessions = clean_df["date"].unique().sort().to_list()
    position = {day: i for i, day in enumerate(sessions)}
    refit_dates = select_refit_dates(sessions)
    x_columns = [f"x_{f}" for f in features]
    beta: np.ndarray | None = None
    residual_var = 0.0
    coefficient_rows: list[dict] = []
    pieces: list[pl.DataFrame] = []
    next_refit = 0
    for day in sessions:
        while next_refit < len(refit_dates) and refit_dates[next_refit] <= day:
            fit_day = refit_dates[next_refit]
            next_refit += 1
            cutoff_index = position[fit_day] - config.forward_window
            if cutoff_index < 0:
                continue
            train_df = clean_df.filter(
                pl.col("date") <= sessions[cutoff_index],
                pl.col("y").is_not_null(),
                pl.col("in_fit"),
            )
            if train_df.height < config.min_fit_rows:
                continue
            design = np.column_stack(
                [np.ones(train_df.height), train_df.select(x_columns).to_numpy()]
            )
            target = train_df["y"].to_numpy()
            beta, *_ = np.linalg.lstsq(design, target, rcond=None)
            residual_var = float(np.var(target - design @ beta))
            coefficient_rows.append(
                {
                    "date": fit_day,
                    "rows": train_df.height,
                    "const": beta[0],
                    "residual_var": residual_var,
                }
                | {f"b_{f}": b for f, b in zip(features, beta[1:], strict=True)}
            )
        if beta is None:
            continue
        today_df = clean_df.filter(pl.col("date") == day)
        design = np.column_stack([np.ones(today_df.height), today_df.select(x_columns).to_numpy()])
        variance = np.exp(design @ beta + residual_var / 2.0)
        pieces.append(
            today_df.select("date", "symbol").with_columns(pl.Series("rv_fcst", np.sqrt(variance)))
        )
    forecast_df = (
        pl.concat(pieces)
        if pieces
        else pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "rv_fcst": pl.Float64})
    )
    return forecast_df, pl.DataFrame(coefficient_rows)


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    rv_df = store.in_window(store.scan("realized_vol"), start, end).collect()
    names_df = rv_df.filter(pl.col("symbol") != MARKET_ROOT)
    spx_df = rv_df.filter(pl.col("symbol") == MARKET_ROOT)
    forecast_df, coefficients_df = forecast(names_df, spx_df)
    store.write(db, "forecast", forecast_df.sort("date", "symbol"))
    print(f"forecast: {forecast_df.height:,} rows; last fit {coefficients_df.tail(1).to_dicts()}")
