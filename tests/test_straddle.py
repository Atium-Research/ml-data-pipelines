import datetime as dt

import polars as pl
from conftest import business_days, make_chain, random_chain

from pipelines.straddle import (
    CONTRACT_MULTIPLIER,
    StraddleConfig,
    compute_reference_path,
    select_unit,
)


def test_selection_is_the_monthly_nearest_sixty_days_at_the_money():
    date_ = dt.date(2024, 1, 2)
    chain_df = make_chain([date_], {date_: 101.0}, {date_: 0.3})
    expiration, strike = select_unit(
        chain_df.filter(pl.col("date") == date_), date_, StraddleConfig()
    )
    assert expiration == dt.date(2024, 3, 15)  # 73 days; February's 45 is further from 60
    assert strike == 100.0


def test_reference_path_opens_rolls_and_scales_by_inception_vega():
    dates, chain_df = random_chain(0, 45)
    path_df = compute_reference_path(chain_df, dates)
    assert path_df.height == len(dates)
    assert path_df["event"][0] == "open"
    assert path_df["pnl_per_vega"][0] is None and path_df["entry_vega"][0] is None
    assert (path_df["event"] == "roll").sum() >= 1
    # entry vega is the inception vega of the unit held into the day
    first_unit_vega = path_df["dollar_vega"][0]
    assert path_df["entry_vega"][1] == first_unit_vega
    # exit cost per vega is the half-spread over the current unit's inception vega
    day0 = path_df.row(0, named=True)
    half_spread = 0.10 / 2 * 2 * CONTRACT_MULTIPLIER  # two legs, 0.10 spread each
    assert abs(day0["exit_cost_per_vega"] - half_spread / first_unit_vega) < 1e-9
    # a roll day pays the closing and the opening half-spread, scaled by the old unit
    roll = path_df.filter(pl.col("event") == "roll").row(0, named=True)
    assert roll["cost_per_vega"] > 2 * half_spread / roll["entry_vega"] - 1e-9


def test_theta_only_day_loses_exactly_theta():
    dates = business_days(dt.date(2024, 1, 2), 2)
    day1_df = make_chain(dates[:1], {dates[0]: 100.0}, {dates[0]: 0.30})
    day2_df = day1_df.with_columns(
        pl.lit(dates[1]).alias("date"), (pl.col("mid") + pl.col("theta")).alias("mid")
    ).with_columns((pl.col("mid") - 0.05).alias("bid"), (pl.col("mid") + 0.05).alias("ask"))
    path_df = compute_reference_path(
        pl.concat([day1_df, day2_df]), dates, StraddleConfig(hedge_cost_per_share=0.0)
    )
    expiration, strike = select_unit(day1_df, dates[0], StraddleConfig())
    theta = day1_df.filter(pl.col("expiration") == expiration, pl.col("strike") == strike)[
        "theta"
    ].sum()
    day2 = path_df.row(1, named=True)
    assert abs(day2["pnl_per_vega"] * day2["entry_vega"] - theta * CONTRACT_MULTIPLIER) < 1e-9
    assert day2["cost_per_vega"] == 0.0


def test_unit_with_no_quotes_is_force_closed_and_reopened():
    dates, chain_df = random_chain(1, 20)
    gap = dates[5:12]  # seven sessions without a chain
    path_df = compute_reference_path(chain_df.filter(~pl.col("date").is_in(gap)), dates)
    events = dict(zip(path_df["date"], path_df["event"], strict=True))
    assert events[dates[0]] == "open"
    assert events[dates[10]] == "forced_close"  # stale_days exceeds 5 on the sixth missing session
    assert events[dates[12]] == "open"
    assert dates[11] not in events  # no unit, no chain, no row
