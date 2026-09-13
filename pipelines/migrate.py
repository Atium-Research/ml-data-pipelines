"""Move the legacy `quant-research/data_store/` layout into the bear-lake store.

Reads each legacy parquet, converts it to the canonical schema and inserts
it, partition by partition. Resumable: a chain partition already in the
store is skipped. The legacy directory is never modified.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from pipelines import store
from pipelines.canonical import canonicalize_chain, canonicalize_open_interest, canonicalize_stock
from pipelines.utils.symbols import normalize_ticker

SAMPLE_YEAR = 2025
WORKERS = 4


def legacy_chain_files(source: Path, prefix: str) -> list[tuple[int, Path]]:
    """`(year, path)` for every per-symbol file; the unstamped directory is the sample year."""
    files = []
    for directory in source.glob(f"{prefix}*"):
        if not directory.is_dir():
            continue
        suffix = directory.name[len(prefix) :]
        if suffix == "" and prefix in ("option_greeks", "open_interest"):
            year = SAMPLE_YEAR
        elif suffix.startswith("_") and suffix[1:].isdigit():
            year = int(suffix[1:])
        else:
            continue
        files.extend((year, path) for path in directory.glob("*.parquet"))
    return sorted(files, key=lambda item: (item[0], item[1].name))


def migrate_chains(db, source: Path, prefix: str, table: str) -> None:
    files = legacy_chain_files(source, prefix)
    pending = [
        (year, path) for year, path in files if not store.partition_exists(table, year, path.stem)
    ]
    print(f"{table}: {len(files)} legacy files, {len(pending)} to convert")
    store.ensure(db, table)
    started = time.time()

    def convert(year: int, path: Path) -> int:
        raw_df = pl.read_parquet(path)
        if raw_df.is_empty():
            return 0
        if table == "open_interest":
            frame_df = canonicalize_open_interest(raw_df)
        else:
            try:
                frame_df = canonicalize_chain(raw_df)
            except ValueError:
                frame_df = canonicalize_chain(raw_df, vega_scale=0.01)
        frame_df = frame_df.with_columns(
            pl.lit(path.stem).alias("symbol"), pl.lit(year, dtype=pl.Int32).alias("year")
        )
        store.write(db, table, frame_df)
        return frame_df.height

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(convert, year, path) for year, path in pending]
        for done, future in enumerate(as_completed(futures), start=1):
            future.result()
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} ({time.time() - started:.0f}s)", flush=True)


def migrate_tables(db, source: Path) -> None:
    def read(*names: str) -> pl.DataFrame:
        paths = [source / name for name in names if (source / name).exists()]
        return (
            pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed")
            if paths
            else pl.DataFrame()
        )

    universe_df = read("universe_history.parquet", "universe.parquet")
    if universe_df.height:
        universe_df = universe_df.with_columns(
            pl.col("ticker")
            .map_elements(lambda t: normalize_ticker(t, "option"), return_dtype=pl.String)
            .alias("symbol")
        )
        store.write(db, "universe", universe_df.select("date", "ticker", "symbol"))
        store.write(db, "calendar", universe_df.select("date").unique().sort("date"))
        print(f"universe: {universe_df.height:,} rows; calendar from its dates")

    stocks_df = read("underlying_history.parquet", "underlying_2025.parquet")
    if stocks_df.height:
        store.write(db, "underlying", canonicalize_stock(stocks_df))
        print(f"underlying: {stocks_df.height:,} rows")

    for legacy, table in (
        ("indices", "indices"),
        ("yields", "yields"),
        ("rates", "rates"),
        ("sectors", "sectors"),
        ("symbology_check", "symbology_check"),
    ):
        frame_df = read(f"{legacy}.parquet")
        if frame_df.height:
            store.write(db, table, frame_df)
            print(f"{table}: {frame_df.height:,} rows")

    actions_df = read("corporate_actions.parquet")
    if actions_df.height:
        store.write(
            db,
            "corporate_actions",
            actions_df.with_columns(pl.col("symbol").str.replace_all(r"\.", "")),
        )
        print(f"corporate_actions: {actions_df.height:,} rows")

    earnings_df = read("earnings.parquet")
    if earnings_df.height:
        store.write(db, "earnings", earnings_df)
        print(f"earnings: {earnings_df.height:,} rows")

    reference_dir = source / "vol_reference_returns"
    if reference_dir.is_dir():
        store.ensure(db, "reference_returns")
        for path in sorted(reference_dir.glob("*.parquet")):
            store.write(db, "reference_returns", pl.read_parquet(path))
        print(f"reference_returns: {len(list(reference_dir.glob('*.parquet')))} symbols")

    for legacy, table in (
        ("vol_factor_returns", "factor_returns"),
        ("vol_factor_loadings", "factor_loadings"),
        ("vol_factor_covariances", "factor_covariances"),
        ("vol_idio_vol", "idio_vol"),
    ):
        frame_df = read(f"{legacy}.parquet")
        if frame_df.height:
            store.write(db, table, frame_df)
            print(f"{table}: {frame_df.height:,} rows")


def run(source: str) -> None:
    source_path = Path(source).expanduser()
    db = store.connect()
    migrate_tables(db, source_path)
    migrate_chains(db, source_path, "open_interest", "open_interest")
    migrate_chains(db, source_path, "index_greeks", "index_greeks")
    migrate_chains(db, source_path, "option_greeks", "option_greeks")
    print(f"migrated into {store.store_path()}")
