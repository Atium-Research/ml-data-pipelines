import datetime as dt

import numpy as np
import polars as pl
from conftest import business_days, make_chain

from pipelines.forecast import forecast
from pipelines.realized_vol import RealizedVolConfig, compute_close_to_close_panel
from pipelines.signals import SignalConfig, compute_vrp_signal
from pipelines.surface import build_surface_panel
from pipelines.utils.interpolation import interpolate_total_variance


def test_flat_surface_interpolates_to_the_flat_vol():
    dates = business_days(dt.date(2024, 1, 2), 2)
    chain_df = make_chain(dates, dict.fromkeys(dates, 100.0), dict.fromkeys(dates, 0.25))
    surface_df = build_surface_panel(chain_df.lazy()).collect()
    assert surface_df.height == 2
    for column in ("iv60", "iv90"):
        assert (surface_df[column] - 0.25).abs().max() < 1e-9
    # the first listed expiration is 45 days out, so 30 days is outside the pillars and null
    assert surface_df["iv30"].is_null().all()
    assert interpolate_total_variance(np.array([30, 90]), np.array([0.2, 0.4]), 60) > 0.2
    assert np.isnan(interpolate_total_variance(np.array([30, 90]), np.array([0.2, 0.4]), 120))


def test_close_to_close_realized_vol_matches_a_direct_calculation():
    rng = np.random.default_rng(0)
    dates = business_days(dt.date(2024, 1, 2), 100)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    closes_df = pl.DataFrame(
        {"date": dates, "symbol": ["X"] * len(dates), "close": closes, "split_ratio": 1.0}
    )
    panel_df = compute_close_to_close_panel(
        closes_df.lazy(), RealizedVolConfig(forward_window=10)
    ).collect()
    log_returns = np.diff(np.log(closes))
    expected_rv22 = np.sqrt(np.mean(log_returns[-22:] ** 2) * 252)
    assert abs(panel_df["rv_22"][-1] - expected_rv22) < 1e-9
    # forward vol at t is the vol over t+1 .. t+10
    expected_fwd = np.sqrt(np.mean(log_returns[50:60] ** 2) * 252)
    assert abs(panel_df["rv_fwd"][50] - expected_fwd) < 1e-9
    assert panel_df["rv_fwd"][-1] is None


def test_har_forecast_is_point_in_time():
    rng = np.random.default_rng(1)
    dates = business_days(dt.date(2023, 1, 2), 400)
    rows = []
    for symbol in [f"S{i}" for i in range(10)]:
        level = rng.uniform(0.15, 0.5)
        for day in dates:
            rv = level * float(np.exp(rng.normal(0, 0.2)))
            rows.append(
                {
                    "date": day,
                    "symbol": symbol,
                    "rv_1": rv,
                    "rv_5": rv,
                    "rv_22": rv,
                    "rv_fwd": level,
                }
            )
    panel_df = pl.DataFrame(rows)
    forecast_df, coefficients_df = forecast(panel_df)
    assert forecast_df.height > 0 and coefficients_df.height > 0
    cutoff = dates[250]
    scrambled_df = panel_df.with_columns(
        pl.when(pl.col("date") > cutoff)
        .then(pl.col("rv_fwd") * 5)
        .otherwise(pl.col("rv_fwd"))
        .alias("rv_fwd")
    )
    scrambled_forecast_df, _ = forecast(scrambled_df)
    before = forecast_df.filter(pl.col("date") <= cutoff).sort("date", "symbol")
    before_scrambled = scrambled_forecast_df.filter(pl.col("date") <= cutoff).sort("date", "symbol")
    assert (before["rv_fcst"] - before_scrambled["rv_fcst"]).abs().max() < 1e-12


def test_vrp_residuals_are_uncorrelated_with_each_regressor():
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(200)]
    sectors = [f"sec{i % 5}" for i in range(200)]
    dates = business_days(dt.date(2024, 1, 2), 8)
    surface_rows, forecast_rows, feature_rows = [], [], []
    for day in dates:
        surface_rows.append({"date": day, "symbol": "SPX", "iv60": 0.18})
        forecast_rows.append({"date": day, "symbol": "SPX", "rv_fcst": 0.15})
        for i, symbol in enumerate(names):
            beta = 0.5 + i / 200
            idio = rng.uniform(0.15, 0.5)
            vrp = 0.3 * beta * 0.18 + 0.1 * (i % 5) - 0.2 * idio + rng.normal(0, 0.1)
            surface_rows.append({"date": day, "symbol": symbol, "iv60": 0.25 * np.exp(vrp)})
            forecast_rows.append({"date": day, "symbol": symbol, "rv_fcst": 0.25})
            feature_rows.append({"date": day, "symbol": symbol, "beta": beta, "idio_vol": idio})
    scores_df = compute_vrp_signal(
        pl.DataFrame(surface_rows),
        pl.DataFrame(forecast_rows),
        pl.DataFrame(feature_rows),
        pl.DataFrame({"symbol": names, "sector": sectors}),
        SignalConfig(min_names=20, smoothing_days=1),
    )
    last_df = scores_df.filter(pl.col("date") == dates[-1]).join(
        pl.DataFrame(feature_rows), on=["date", "symbol"]
    )
    assert last_df.height == 200
    assert abs(last_df["score"].mean()) < 1e-6
    assert abs(last_df.select(pl.corr("score", "idio_vol")).item()) < 1e-6
    assert abs(last_df.select(pl.corr("score", "beta")).item()) < 1e-6
