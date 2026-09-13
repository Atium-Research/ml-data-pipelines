import datetime as dt

import polars as pl
from conftest import business_days, make_chain, to_vendor_chain

from pipelines.canonical import canonicalize_chain, detect_vega_scale
from pipelines.store import CANONICAL_OPTION_SCHEMA


def test_vendor_vega_per_unit_iv_is_detected_and_rescaled():
    dates = business_days(dt.date(2024, 1, 2), 3)
    chain_df = make_chain(dates, dict.fromkeys(dates, 100.0), dict.fromkeys(dates, 0.30))
    vendor_df = to_vendor_chain(chain_df)
    assert detect_vega_scale(vendor_df) == 0.01
    canonical_df = canonicalize_chain(vendor_df)
    canonical = [c for c in CANONICAL_OPTION_SCHEMA if c != "year"]
    assert canonical_df.columns[:15] == canonical[:15]
    assert {"rho", "bid_size", "timestamp", "implied_vol", "iv_error"} <= set(canonical_df.columns)
    assert all(c in canonical for c in canonical_df.columns)
    joined_df = canonical_df.join(
        chain_df, on=["date", "expiration", "strike", "right"], suffix="_expected"
    )
    assert (joined_df["vega"] - joined_df["vega_expected"]).abs().max() < 1e-12
    assert (joined_df["iv"] - joined_df["iv_expected"]).abs().max() < 1e-12
    assert set(canonical_df["right"]) == {"C", "P"}
    assert canonical_df.schema["expiration"] == pl.Date


def test_failed_inversions_become_null_iv():
    dates = business_days(dt.date(2024, 1, 2), 2)
    vendor_df = to_vendor_chain(
        make_chain(dates, dict.fromkeys(dates, 100.0), dict.fromkeys(dates, 0.30))
    ).with_columns(
        pl.when(pl.col("strike") == 80.0)
        .then(100.0)
        .otherwise(pl.col("iv_error"))
        .alias("iv_error")
    )
    canonical_df = canonicalize_chain(vendor_df, vega_scale=0.01)
    assert canonical_df.filter(pl.col("strike") == 80.0)["iv"].is_null().all()
    assert canonical_df.filter(pl.col("strike") != 80.0)["iv"].is_not_null().all()
