"""Repair index sessions whose EOD quotes came back empty.

The EOD greeks endpoint returns zero bids and asks for the index roots on a
block of sessions, mostly 2020-2021, and a fresh pull gives the same. The
15:59 first-order-greeks bar has the quotes. For each affected session this
pulls that bar for every listed expiration and replaces bid, ask, mid, iv,
delta, theta, vega and the underlying; gamma is nulled, since the vendor
computed it from empty quotes.

One request per (session, expiration) at ~2 s; SPXW lists 40-odd
expirations a day, so a full repair is hours. Bars are checkpointed under
`<store>/.cache/index_repair/` so an interrupted run loses minutes.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.canonical import IV_ERROR_TOLERANCE, detect_vega_scale
from pipelines.index_greeks import INDEX_ROOTS
from pipelines.utils.sessions import years_in
from pipelines.utils.theta import fetch_many, is_missing_data, make_client

START = dt.date(2017, 1, 1)
END = dt.date(2025, 12, 31)
WORKERS = 4
THRESHOLD = 0.5  # repair a session when fewer than this share of near-ATM contracts carry an ask
CHECKPOINT_EVERY = 200
BAR_WINDOWS = (("15:59:00", "16:00:00"), ("12:59:00", "13:00:00"))  # regular close, half day


def find_unquoted_sessions(chain_df: pl.DataFrame) -> list[dt.date]:
    daily_df = (
        chain_df.with_columns((pl.col("expiration") - pl.col("date")).dt.total_days().alias("dte"))
        .filter(
            pl.col("underlying") > 0,
            (pl.col("strike") / pl.col("underlying") - 1).abs() <= 0.02,
            pl.col("dte").is_between(7, 60),
        )
        .group_by("date")
        .agg((pl.col("ask") > 0).mean().alias("quoted"))
    )
    return daily_df.filter(pl.col("quoted") < THRESHOLD).sort("date")["date"].to_list()


def fetch_bar(root: str, session: dt.date, expiration: dt.date) -> pl.DataFrame:
    client = make_client()
    bar_df = pl.DataFrame()
    for start_time, end_time in BAR_WINDOWS:
        try:
            bar_df = client.option_history_greeks_first_order(
                root,
                expiration,
                interval="1h",
                date=session,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as error:
            if not is_missing_data(error):
                raise
            bar_df = pl.DataFrame()
        if not bar_df.is_empty():
            break
    if bar_df.is_empty():
        return bar_df
    return (
        bar_df.sort("timestamp")
        .group_by("expiration", "strike", "right", maintain_order=True)
        .last()
    )


def fetch_bars(root: str, tasks: list[tuple[dt.date, dt.date]], cache_path) -> pl.DataFrame | None:
    cached_df = pl.read_parquet(cache_path) if cache_path.exists() else None
    if cached_df is not None and not cached_df.is_empty():
        done = set(
            zip(
                cached_df["timestamp"].dt.date().to_list(),
                cached_df["expiration"].str.to_date("%Y-%m-%d").to_list(),
                strict=True,
            )
        )
        tasks = [task for task in tasks if task not in done]
        print(f"  resuming: {len(done)} bars cached, {len(tasks)} to go")
    pieces = [cached_df] if cached_df is not None else []
    for offset in range(0, len(tasks), CHECKPOINT_EVERY):
        chunk = tasks[offset : offset + CHECKPOINT_EVERY]
        fetched = [
            bar_df
            for bar_df in fetch_many(chunk, lambda task: fetch_bar(root, *task), WORKERS)
            if not bar_df.is_empty()
        ]
        if fetched:
            pieces.append(pl.concat(fetched, how="vertical_relaxed"))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pl.concat(pieces, how="vertical_relaxed").write_parquet(cache_path)
    pieces = [piece for piece in pieces if piece is not None and not piece.is_empty()]
    return pl.concat(pieces, how="vertical_relaxed") if pieces else None


def build_patch(bars_df: pl.DataFrame) -> pl.DataFrame:
    """The bar rows in canonical columns, keyed by contract-day."""
    try:
        vega_scale = detect_vega_scale(bars_df)
    except ValueError:
        vega_scale = 0.01
    return bars_df.filter(pl.col("ask") > 0).select(
        pl.col("timestamp").dt.date().alias("date"),
        pl.col("expiration").str.to_date("%Y-%m-%d"),
        pl.col("strike").cast(pl.Float64),
        pl.when(pl.col("right") == "CALL").then(pl.lit("C")).otherwise(pl.lit("P")).alias("right"),
        pl.col("bid").cast(pl.Float64).alias("bid_new"),
        pl.col("ask").cast(pl.Float64).alias("ask_new"),
        pl.when((pl.col("iv_error").abs() <= IV_ERROR_TOLERANCE) & (pl.col("implied_vol") > 0))
        .then(pl.col("implied_vol"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("iv_new"),
        pl.col("delta").cast(pl.Float64).alias("delta_new"),
        pl.col("theta").cast(pl.Float64).alias("theta_new"),
        (pl.col("vega") * vega_scale).cast(pl.Float64).alias("vega_new"),
        pl.col("underlying_price").cast(pl.Float64).alias("underlying_new"),
    )


def repair_partition(db, root: str, year: int) -> None:
    chain_df = store.scan("index_greeks", [year], [root]).collect()
    sessions = find_unquoted_sessions(chain_df)
    tasks_df = (
        chain_df.filter(pl.col("date").is_in(sessions))
        .select("date", "expiration")
        .unique()
        .sort("date", "expiration")
    )
    print(f"{root} {year}: {len(sessions)} unquoted sessions, {tasks_df.height} requests")
    if tasks_df.is_empty():
        return
    tasks = [(row["date"], row["expiration"]) for row in tasks_df.iter_rows(named=True)]
    cache_path = store.store_path() / ".cache" / "index_repair" / f"{root}_{year}.parquet"
    bars_df = fetch_bars(root, tasks, cache_path)
    if bars_df is None:
        print(f"  {root} {year}: nothing returned")
        return
    patch_df = build_patch(bars_df)
    patched = pl.col("ask_new").is_not_null()
    replaced = ["bid", "ask", "iv", "delta", "theta", "vega", "underlying"]
    repaired_df = (
        chain_df.join(patch_df, on=["date", "expiration", "strike", "right"], how="left")
        .with_columns(
            [
                pl.when(patched)
                .then(pl.col(f"{column}_new"))
                .otherwise(pl.col(column))
                .alias(column)
                for column in replaced
            ]
        )
        .with_columns(
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"),
            pl.when(patched).then(None).otherwise(pl.col("gamma")).alias("gamma"),
        )
        .select(chain_df.columns)
    )
    store.write(db, "index_greeks", repaired_df)
    cache_path.unlink(missing_ok=True)
    print(f"  {root} {year}: {patch_df.height:,} contract-days repaired")


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    for year in years_in(start, end):
        for root in INDEX_ROOTS:
            if not store.partition_exists("index_greeks", year, root):
                continue
            repair_partition(db, root, year)
