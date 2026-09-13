import datetime as dt

import numpy as np
import polars as pl

from pipelines.factor_model import (
    FactorSpec,
    build_factor_returns,
    estimate_rolling_covariances,
    estimate_rolling_loadings,
    ledoit_wolf,
)


def simulate_panel(n_days: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    names = [f"T{i}" for i in range(15)] + [f"E{i}" for i in range(15)]
    sectors = {n: ("Tech" if n.startswith("T") else "Energy") for n in names}
    true_beta = {n: float(rng.uniform(0.5, 1.5)) for n in names}
    true_sector = {n: float(rng.uniform(0.5, 1.5)) for n in names}
    idio = {n: float(rng.uniform(0.2, 0.6)) for n in names}
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
    market = rng.normal(0, 1.0, n_days)
    tech, energy = rng.normal(0, 0.5, n_days), rng.normal(0, 0.7, n_days)
    rows = []
    for i, day in enumerate(dates):
        for symbol, sector in sectors.items():
            factor = tech[i] if sector == "Tech" else energy[i]
            ret = (
                true_beta[symbol] * market[i]
                + true_sector[symbol] * factor
                + rng.normal(0, idio[symbol])
            )
            rows.append({"date": day, "symbol": symbol, "pnl_per_vega": ret, "dollar_vega": 50.0})
    returns_df = pl.DataFrame(rows)
    market_df = pl.DataFrame({"date": dates, "pnl_per_vega": market})
    sectors_df = pl.DataFrame({"symbol": list(sectors), "sector": list(sectors.values())})
    return returns_df, market_df, sectors_df, true_beta, idio


def test_rolling_loadings_and_idio_vol_are_recovered():
    returns_df, market_df, sectors_df, true_beta, idio = simulate_panel()
    spec = FactorSpec(window=400, min_observations=100)
    factors_df = build_factor_returns(returns_df, market_df, sectors_df, spec)
    assert set(factors_df["factor"]) == {"market", "Tech", "Energy"}
    loadings_df, idio_df = estimate_rolling_loadings(returns_df, factors_df, sectors_df, spec)
    last = loadings_df["date"].max()
    market_loadings = loadings_df.filter(pl.col("date") == last, pl.col("factor") == "market")
    for symbol, loading in zip(market_loadings["symbol"], market_loadings["loading"], strict=True):
        assert abs(loading - true_beta[symbol]) < 0.15
    last_idio = idio_df.filter(pl.col("date") == last)
    for symbol, value in zip(last_idio["symbol"], last_idio["idio_vol"], strict=True):
        assert abs(value - idio[symbol]) < 0.1
    # sector factors need min_observations sessions of trailing beta, loadings need
    # min_observations of factor history on top, so estimates start after 2 × 100
    assert idio_df["date"].n_unique() == 400 - 2 * 100 + 2


def test_covariances_are_shrunk_psd_and_market_first():
    returns_df, market_df, sectors_df, *_ = simulate_panel(n_days=200)
    spec = FactorSpec(window=100, min_observations=50)
    factors_df = build_factor_returns(returns_df, market_df, sectors_df, spec)
    covariances_df = estimate_rolling_covariances(factors_df, spec)
    last = covariances_df.filter(pl.col("date") == covariances_df["date"].max())
    names = last["factor_1"].unique(maintain_order=True).to_list()
    assert names[0] == "market"
    matrix = (
        last.pivot(index="factor_1", on="factor_2", values="covariance").select(names).to_numpy()
    )
    assert np.allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0


def test_ledoit_wolf_shrinks_toward_identity_and_keeps_trace():
    rng = np.random.default_rng(2)
    sample = np.cov(rng.normal(size=(30, 8)), rowvar=False)
    shrunk = ledoit_wolf(sample, 30)
    off_diagonal = lambda m: np.sum(np.abs(m - np.diag(np.diag(m))))  # noqa: E731
    assert off_diagonal(shrunk) < off_diagonal(sample)
    assert abs(np.trace(shrunk) - np.trace(sample)) < 1e-9
    assert np.linalg.eigvalsh(shrunk).min() > 0
