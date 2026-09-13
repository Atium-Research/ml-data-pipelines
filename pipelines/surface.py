"""Constant-maturity ATM implied vol per (date, symbol): iv30, iv60, iv90.

For each (date, symbol, expiration): the call and the put nearest |delta| =
0.50 with a usable quote, their IVs averaged, is the expiration's ATM
pillar. Pillars are interpolated in total variance to the tenors. Built one
symbol at a time; SPX is included from the index chains.
"""

import datetime as dt
import time
from dataclasses import dataclass

import polars as pl

from pipelines import store
from pipelines.reference_returns import MARKET_ROOT
from pipelines.utils.interpolation import build_tenor_columns
from pipelines.utils.sessions import years_in
from pipelines.utils.symbols import universe_symbols

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)


@dataclass(frozen=True)
class SurfaceConfig:
    tenors: tuple[int, ...] = (30, 60, 90)
    min_dte: int = 7
    max_dte: int = 400
    delta_band: tuple[float, float] = (0.25, 0.75)


def compute_atm_pillars(options: pl.LazyFrame, config: SurfaceConfig) -> pl.LazyFrame:
    """`(date, symbol, expiration, dte, atm_iv)` from canonical rows, both rights required."""
    candidates = (
        options.filter(
            pl.col("iv").is_not_null(),
            pl.col("bid") > 0,
            pl.col("ask") > pl.col("bid"),
            pl.col("delta").abs().is_between(*config.delta_band),
        )
        .with_columns((pl.col("expiration") - pl.col("date")).dt.total_days().alias("dte"))
        .filter(pl.col("dte").is_between(config.min_dte, config.max_dte))
        .with_columns((pl.col("delta").abs() - 0.5).abs().alias("distance"))
    )
    best_per_right = (
        candidates.sort("distance", "strike")  # strike breaks a tie deterministically
        .group_by("date", "symbol", "expiration", "right", maintain_order=True)
        .agg(pl.col("iv").first(), pl.col("dte").first())
    )
    return (
        best_per_right.group_by("date", "symbol", "expiration")
        .agg(
            pl.col("iv").mean().alias("atm_iv"),
            pl.col("dte").first(),
            pl.col("right").n_unique().alias("rights"),
        )
        .filter(pl.col("rights") == 2)
        .drop("rights")
    )


def build_surface_panel(
    options: pl.LazyFrame, config: SurfaceConfig = SurfaceConfig()
) -> pl.LazyFrame:
    pillars = compute_atm_pillars(options, config)
    return build_tenor_columns(
        pillars.select("date", "symbol", "dte", "atm_iv"),
        group_by=["date", "symbol"],
        dte_col="dte",
        iv_col="atm_iv",
        target_dtes=config.tenors,
    ).sort("date", "symbol")


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    years = years_in(start, end)
    jobs = [("option_greeks", s) for s in universe_symbols("option", start, end)]
    jobs.append(("index_greeks", MARKET_ROOT))
    started = time.time()
    pieces = []
    for count, (table, symbol) in enumerate(jobs, start=1):
        try:
            options = store.in_window(store.scan(table, years, [symbol]), start, end)
        except FileNotFoundError:
            continue
        pieces.append(build_surface_panel(options).collect())
        if count % 50 == 0 or count == len(jobs):
            print(f"  surface {count}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    surface_df = pl.concat(pieces).sort("date", "symbol")
    store.write(db, "surface", surface_df)
    print(f"surface: {surface_df.height:,} rows, {surface_df['symbol'].n_unique()} symbols")
