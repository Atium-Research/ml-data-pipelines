"""The bear-lake store: table definitions, writes, and partition-aware reads.

Every table's schema, partition keys and primary keys are spelled once in
`TABLES`. `write` creates the table if needed, adds the `year` partition
column from `date`, casts to the schema and inserts. Reads go through
`scan`, which builds the parquet path list from the partition keys so a
per-symbol job opens one file instead of globbing every partition; the
layout is bear-lake's own, `<store>/<table>/<year>/<symbol>.parquet`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import bear_lake as bl
import polars as pl
from dotenv import load_dotenv

load_dotenv()

CANONICAL_OPTION_SCHEMA = {
    "date": pl.Date,
    "symbol": pl.String,
    "expiration": pl.Date,
    "strike": pl.Float64,
    "right": pl.String,
    "bid": pl.Float64,
    "ask": pl.Float64,
    "mid": pl.Float64,
    "iv": pl.Float64,
    "delta": pl.Float64,
    "gamma": pl.Float64,
    "theta": pl.Float64,
    "vega": pl.Float64,
    "underlying": pl.Float64,
    "volume": pl.Int64,
    "year": pl.Int32,
}

CONTRACT_KEYS = ["date", "symbol", "expiration", "strike", "right"]


@dataclass(frozen=True)
class Table:
    schema: dict[str, pl.DataType]
    partition_keys: list[str] | None
    primary_keys: list[str]


TABLES: dict[str, Table] = {
    # --- raw ---------------------------------------------------------------
    "calendar": Table({"date": pl.Date}, None, ["date"]),
    "universe": Table(
        {"date": pl.Date, "ticker": pl.String, "symbol": pl.String, "year": pl.Int32},
        ["year"],
        ["date", "ticker"],
    ),
    "sectors": Table(
        {"ticker": pl.String, "symbol": pl.String, "sector": pl.String, "sub_industry": pl.String},
        None,
        ["ticker"],
    ),
    "indices": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
        None,
        ["date", "symbol"],
    ),
    "yields": Table(
        {"date": pl.Date, "tenor": pl.String, "yield": pl.Float64}, None, ["date", "tenor"]
    ),
    "rates": Table(
        {"date": pl.Date, "symbol": pl.String, "rate": pl.Float64}, None, ["date", "symbol"]
    ),
    "corporate_actions": Table(
        {"symbol": pl.String, "date": pl.Date, "action": pl.String, "value": pl.Float64},
        None,
        ["symbol", "date", "action"],
    ),
    "earnings": Table(
        {
            "symbol": pl.String,
            "date": pl.Date,
            "session": pl.String,
            "announced_at": pl.Datetime("us", "America/New_York"),
            "eps_estimate": pl.Float64,
            "reported_eps": pl.Float64,
            "surprise_pct": pl.Float64,
        },
        None,
        ["symbol", "date"],
    ),
    "underlying": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Int64,
            "year": pl.Int32,
        },
        ["year"],
        ["date", "symbol"],
    ),
    "option_greeks": Table(CANONICAL_OPTION_SCHEMA, ["year", "symbol"], CONTRACT_KEYS),
    "index_greeks": Table(CANONICAL_OPTION_SCHEMA, ["year", "symbol"], CONTRACT_KEYS),
    "open_interest": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "expiration": pl.Date,
            "strike": pl.Float64,
            "right": pl.String,
            "oi": pl.Int64,
            "year": pl.Int32,
        },
        ["year", "symbol"],
        CONTRACT_KEYS,
    ),
    "symbology_check": Table(
        {
            "year": pl.Int32,
            "symbol": pl.String,
            "theta_days": pl.Int64,
            "overlap_days": pl.Int64,
            "return_correlation": pl.Float64,
            "median_return_difference": pl.Float64,
            "action_gap_days": pl.Int64,
            "status": pl.String,
        },
        None,
        ["year", "symbol"],
    ),
    # --- derived -----------------------------------------------------------
    "reference_returns": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "pnl_per_vega": pl.Float64,
            "cost_per_vega": pl.Float64,
            "exit_cost_per_vega": pl.Float64,
            "dollar_vega": pl.Float64,
            "entry_vega": pl.Float64,
            "dollar_theta": pl.Float64,
            "mid": pl.Float64,
            "underlying": pl.Float64,
            "dte": pl.Int64,
            "event": pl.String,
            "year": pl.Int32,
        },
        ["year", "symbol"],
        ["date", "symbol"],
    ),
    "factor_returns": Table(
        {"date": pl.Date, "factor": pl.String, "ret": pl.Float64}, None, ["date", "factor"]
    ),
    "factor_loadings": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "factor": pl.String,
            "loading": pl.Float64,
            "year": pl.Int32,
        },
        ["year"],
        ["date", "symbol", "factor"],
    ),
    "factor_covariances": Table(
        {"date": pl.Date, "factor_1": pl.String, "factor_2": pl.String, "covariance": pl.Float64},
        None,
        ["date", "factor_1", "factor_2"],
    ),
    "idio_vol": Table(
        {"date": pl.Date, "symbol": pl.String, "idio_vol": pl.Float64, "year": pl.Int32},
        ["year"],
        ["date", "symbol"],
    ),
    "surface": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "iv30": pl.Float64,
            "iv60": pl.Float64,
            "iv90": pl.Float64,
            "year": pl.Int32,
        },
        ["year"],
        ["date", "symbol"],
    ),
    "realized_vol": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "rv_1": pl.Float64,
            "rv_5": pl.Float64,
            "rv_22": pl.Float64,
            "rv_fwd": pl.Float64,
            "year": pl.Int32,
        },
        ["year"],
        ["date", "symbol"],
    ),
    "forecast": Table(
        {"date": pl.Date, "symbol": pl.String, "rv_fcst": pl.Float64, "year": pl.Int32},
        ["year"],
        ["date", "symbol"],
    ),
    "stock_features": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "ret": pl.Float64,
            "beta": pl.Float64,
            "idio_vol": pl.Float64,
            "gap_freq": pl.Float64,
            "year": pl.Int32,
        },
        ["year"],
        ["date", "symbol"],
    ),
    "signals": Table(
        {
            "date": pl.Date,
            "symbol": pl.String,
            "signal": pl.String,
            "score": pl.Float64,
            "year": pl.Int32,
        },
        ["year"],
        ["date", "symbol", "signal"],
    ),
}


def store_path() -> Path:
    path = os.getenv("ML_DATA_STORE")
    if not path:
        raise RuntimeError("set ML_DATA_STORE to the bear-lake directory (see .env.example)")
    return Path(path).expanduser()


def connect() -> bl.Database:
    return bl.connect(str(store_path()))


def ensure(db: bl.Database, name: str) -> None:
    table = TABLES[name]
    db.create(name, table.schema, table.partition_keys, table.primary_keys, mode="skip")


def conform(name: str, frame_df: pl.DataFrame) -> pl.DataFrame:
    """Add `year` where the table is partitioned on it, then select and cast to the schema."""
    table = TABLES[name]
    if "year" in table.schema and "year" not in frame_df.columns:
        frame_df = frame_df.with_columns(pl.col("date").dt.year().cast(pl.Int32).alias("year"))
    return frame_df.select([pl.col(column).cast(dtype) for column, dtype in table.schema.items()])


def write(db: bl.Database, name: str, frame_df: pl.DataFrame, mode: str = "overwrite") -> None:
    """Insert `frame_df` into `name`, creating the table on first use.

    `overwrite` replaces every partition the frame touches and leaves the
    rest alone; for an unpartitioned table it replaces the table.
    """
    if frame_df.is_empty():
        return
    ensure(db, name)
    db.insert(name, conform(name, frame_df), mode=mode)


def table_dir(name: str) -> Path:
    return store_path() / name


def partition_path(name: str, *values) -> Path:
    path = table_dir(name)
    for value in values:
        path = path / str(value)
    return path.with_suffix(".parquet")


def partition_exists(name: str, *values) -> bool:
    return partition_path(name, *values).exists()


def available_years(name: str) -> list[int]:
    directory = table_dir(name)
    if not directory.is_dir():
        return []
    return sorted(int(child.stem) for child in directory.iterdir() if child.stem.isdigit())


def available_symbols(name: str, year: int) -> list[str]:
    directory = table_dir(name) / str(year)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.parquet"))


def scan(
    name: str,
    years: list[int] | None = None,
    symbols: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy scan of a table, opening only the partitions asked for.

    Raises `FileNotFoundError` when nothing matches, naming the step that
    writes the table, since the usual cause is a step not yet run.
    """
    table = TABLES[name]
    keys = table.partition_keys or []
    if keys == ["year", "symbol"] and (years is not None or symbols is not None):
        years = years if years is not None else available_years(name)
        paths = []
        for year in years:
            names = symbols if symbols is not None else available_symbols(name, year)
            paths.extend(
                partition_path(name, year, symbol)
                for symbol in names
                if partition_exists(name, year, symbol)
            )
    elif keys == ["year"] and years is not None:
        paths = [partition_path(name, year) for year in years if partition_exists(name, year)]
    else:
        paths = sorted(table_dir(name).glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"no {name} partitions for years={years} symbols={symbols};"
            f" run `uv run pipelines {name.replace('_', '-')}`"
        )
    frame = pl.scan_parquet(paths)
    if symbols is not None and "symbol" in table.schema:
        frame = frame.filter(pl.col("symbol").is_in(symbols))
    return frame


def in_window(frame: pl.LazyFrame, start, end, column: str = "date") -> pl.LazyFrame:
    if start is not None:
        frame = frame.filter(pl.col(column) >= start)
    if end is not None:
        frame = frame.filter(pl.col(column) <= end)
    return frame
