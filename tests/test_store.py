import datetime as dt

import polars as pl
import pytest
from conftest import business_days, make_chain, third_friday

from pipelines import store


def test_partitioned_write_scan_and_resumability(tmp_store):
    db = store.connect()
    dates = business_days(dt.date(2024, 12, 30), 4)  # straddles two years
    for symbol in ("AAA", "BBB"):
        chain_df = make_chain(
            dates,
            dict.fromkeys(dates, 100.0),
            dict.fromkeys(dates, 0.3),
            symbol=symbol,
            expirations=[third_friday(2025, m) for m in range(2, 8)],
        )
        store.write(db, "option_greeks", chain_df)
    assert store.available_years("option_greeks") == [2024, 2025]
    assert store.available_symbols("option_greeks", 2025) == ["AAA", "BBB"]
    assert store.partition_exists("option_greeks", 2024, "AAA")
    assert not store.partition_exists("option_greeks", 2023, "AAA")
    assert (tmp_store / "option_greeks" / "2025" / "AAA.parquet").exists()

    one_df = store.scan("option_greeks", [2025], ["AAA"]).collect()
    assert set(one_df["symbol"]) == {"AAA"} and set(one_df["year"]) == {2025}
    assert one_df.schema["year"] == pl.Int32
    all_df = store.scan("option_greeks").collect()
    assert all_df["symbol"].n_unique() == 2 and all_df["date"].n_unique() == 4

    # overwriting one partition leaves the others alone
    replacement_df = make_chain(
        dates[2:],
        dict.fromkeys(dates[2:], 200.0),
        dict.fromkeys(dates[2:], 0.3),
        symbol="AAA",
        expirations=[third_friday(2025, m) for m in range(2, 8)],
    )
    store.write(db, "option_greeks", replacement_df)
    assert store.scan("option_greeks", [2025], ["AAA"]).collect()[
        "underlying"
    ].unique().to_list() == [200.0]
    assert store.scan("option_greeks", [2024], ["AAA"]).collect()[
        "underlying"
    ].unique().to_list() == [100.0]

    with pytest.raises(FileNotFoundError):
        store.scan("open_interest")


def test_unpartitioned_table_round_trips(tmp_store):
    db = store.connect()
    store.write(db, "calendar", pl.DataFrame({"date": business_days(dt.date(2024, 1, 1), 5)}))
    assert store.scan("calendar").collect().height == 5
    store.write(db, "calendar", pl.DataFrame({"date": business_days(dt.date(2024, 1, 1), 3)}))
    assert store.scan("calendar").collect().height == 3
