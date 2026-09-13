"""One delta-hedged ATM straddle per name, marked daily: the reference unit.

Selection: the listed *monthly* expiration nearest `target_dte` calendar
days, the strike nearest spot with both a call and a put quoted. Marking:
the sum of the two mids. Hedge: `-delta × 100` shares, reset at every
close. Roll: at `roll_dte` or when the strike has drifted more than
`restrike_moneyness` from spot (a straddle 15% from the money is a gamma
position, not a vega position). A unit with no usable quote carries its
last mark; after `max_stale_days`, or at expiration, it is closed at the
last mark and reopened the next session a unit can be selected.

Costs: the full half-spread on both legs at open, close and roll, and
`hedge_cost_per_share` on every share of hedge traded.

The output is the unit's daily P&L and cost per dollar of its vega at
inception, which is the "return" everything downstream is estimated on.
"""

import datetime as dt
from dataclasses import dataclass

import polars as pl

CONTRACT_MULTIPLIER = 100

REFERENCE_COLUMNS = [
    "date",
    "symbol",
    "pnl_per_vega",
    "cost_per_vega",
    "exit_cost_per_vega",
    "dollar_vega",
    "entry_vega",
    "dollar_theta",
    "mid",
    "underlying",
    "dte",
    "event",
]


@dataclass(frozen=True)
class StraddleConfig:
    target_dte: int = 60
    roll_dte: int = 20
    min_dte: int = 25
    max_dte: int = 120
    max_moneyness: float = 0.10
    restrike_moneyness: float = 0.15
    max_stale_days: int = 5
    hedge_cost_per_share: float = 0.005
    chain_max_dte: int = 130
    chain_max_moneyness: float = 0.30


@dataclass
class Unit:
    expiration: dt.date
    strike: float
    inception_vega: float
    mid: float
    underlying: float
    delta: float
    hedge_shares: float = 0.0
    stale_days: int = 0


def select_unit(quotes_df: pl.DataFrame, date_: dt.date, config: StraddleConfig) -> tuple | None:
    """`(expiration, strike)` to enter on `date_`, or None if nothing qualifies.

    Ties are broken by explicit keys so the choice never depends on row
    order: nearest to the target DTE, then the longer series, then the
    strike nearest spot, then the lower strike.
    """
    candidates_df = (
        quotes_df.filter(pl.col("bid") > 0, pl.col("ask") > pl.col("bid"))
        .with_columns((pl.col("expiration") - pl.lit(date_)).dt.total_days().alias("dte"))
        .filter(
            pl.col("dte").is_between(config.min_dte, config.max_dte),
            pl.col("expiration").dt.day().is_between(15, 21),
            pl.col("expiration")
            .dt.weekday()
            .is_in([4, 5]),  # Friday, or Thursday before Good Friday
        )
    )
    if candidates_df.is_empty():
        return None
    pairs_df = (
        candidates_df.group_by("expiration", "strike", "dte")
        .agg(pl.col("right").n_unique().alias("rights"), pl.col("underlying").first())
        .filter(pl.col("rights") == 2)
        .with_columns((pl.col("strike") / pl.col("underlying") - 1.0).abs().alias("moneyness"))
        .filter(pl.col("moneyness") <= config.max_moneyness)
        .with_columns((pl.col("dte") - config.target_dte).abs().alias("dte_gap"))
        .sort(["dte_gap", "dte", "moneyness", "strike"], descending=[False, True, False, False])
    )
    if pairs_df.is_empty():
        return None
    return pairs_df["expiration"][0], float(pairs_df["strike"][0])


def mark_unit(quotes_df: pl.DataFrame, expiration: dt.date, strike: float) -> dict | None:
    """Per-share mid, bid, ask, delta, vega, theta and the spot of one straddle, or None."""
    legs_df = quotes_df.filter(
        pl.col("expiration") == expiration,
        pl.col("strike") == strike,
        pl.col("ask") > 0,
        pl.col("bid") >= 0,
        pl.col("ask") >= pl.col("bid"),
    )
    if legs_df.height < 2:
        return None
    total = legs_df.select(
        pl.col("mid").sum(),
        pl.col("bid").sum(),
        pl.col("ask").sum(),
        pl.col("delta").sum(),
        pl.col("vega").sum(),
        pl.col("theta").sum(),
        pl.col("underlying").first(),
    ).row(0, named=True)
    return {key: float(value) for key, value in total.items()}


def compute_reference_path(
    chain_df: pl.DataFrame,
    sessions: list[dt.date],
    config: StraddleConfig = StraddleConfig(),
) -> pl.DataFrame:
    """Hold one long unit through `sessions` on one symbol's canonical chain."""
    if chain_df.is_empty():
        return pl.DataFrame(schema=reference_schema())
    symbol = chain_df["symbol"][0]
    by_date = {key[0]: frame for key, frame in chain_df.partition_by("date", as_dict=True).items()}
    unit: Unit | None = None
    rows = []
    for date_ in sessions:
        quotes_df = by_date.get(date_)
        row = dict(
            date=date_,
            option_pnl=0.0,
            hedge_pnl=0.0,
            option_cost=0.0,
            hedge_cost=0.0,
            dollar_vega=0.0,
            dollar_theta=0.0,
            exit_cost=0.0,
            mid=None,
            underlying=None,
            dte=None,
            event="",
            inception_vega=None,
        )
        mark = None
        if unit is not None:
            mark = (
                mark_unit(quotes_df, unit.expiration, unit.strike)
                if quotes_df is not None
                else None
            )
            if mark is None:
                unit.stale_days += 1
                spot = unit.underlying
                if quotes_df is not None and not quotes_df.is_empty():
                    spot = float(quotes_df["underlying"][0])
                row["hedge_pnl"] = unit.hedge_shares * (spot - unit.underlying)
                unit.underlying = spot
                row.update(mid=unit.mid, underlying=spot, dte=(unit.expiration - date_).days)
                expired = (unit.expiration - date_).days <= 0
                if unit.stale_days > config.max_stale_days or expired:
                    row["event"] = "forced_close"
                    row["hedge_cost"] += abs(unit.hedge_shares) * config.hedge_cost_per_share
                    unit = None
                else:
                    row.update(dollar_vega=unit.inception_vega, inception_vega=unit.inception_vega)
                rows.append(row)
                continue
            unit.stale_days = 0
            row["option_pnl"] = (mark["mid"] - unit.mid) * CONTRACT_MULTIPLIER
            row["hedge_pnl"] = unit.hedge_shares * (mark["underlying"] - unit.underlying)
            unit.mid, unit.underlying, unit.delta = mark["mid"], mark["underlying"], mark["delta"]
            dte = (unit.expiration - date_).days
            drifted = abs(unit.strike / mark["underlying"] - 1.0) > config.restrike_moneyness
            if dte <= config.roll_dte or drifted:
                row["option_cost"] += half_spread_dollars(mark)
                selected = select_unit(quotes_df, date_, config)
                new_mark = mark_unit(quotes_df, *selected) if selected else None
                if new_mark is None:
                    row["event"] = "roll_failed_closed"
                    row["hedge_cost"] += abs(unit.hedge_shares) * config.hedge_cost_per_share
                    unit = None
                    rows.append(row)
                    continue
                row["option_cost"] += half_spread_dollars(new_mark)
                row["event"] = "roll"
                unit = Unit(
                    selected[0],
                    selected[1],
                    new_mark["vega"] * CONTRACT_MULTIPLIER,
                    new_mark["mid"],
                    new_mark["underlying"],
                    new_mark["delta"],
                    hedge_shares=unit.hedge_shares,
                )
                mark = new_mark
        else:
            selected = select_unit(quotes_df, date_, config) if quotes_df is not None else None
            mark = mark_unit(quotes_df, *selected) if selected else None
            if mark is None:
                continue
            row["option_cost"] += half_spread_dollars(mark)
            row["event"] = "open"
            unit = Unit(
                selected[0],
                selected[1],
                mark["vega"] * CONTRACT_MULTIPLIER,
                mark["mid"],
                mark["underlying"],
                mark["delta"],
            )
        # re-hedge on the final unit of the day
        target_shares = -mark["delta"] * CONTRACT_MULTIPLIER
        row["hedge_cost"] += abs(target_shares - unit.hedge_shares) * config.hedge_cost_per_share
        unit.hedge_shares = target_shares
        row.update(
            dollar_vega=mark["vega"] * CONTRACT_MULTIPLIER,
            dollar_theta=mark["theta"] * CONTRACT_MULTIPLIER,
            exit_cost=half_spread_dollars(mark),
            mid=mark["mid"],
            underlying=mark["underlying"],
            dte=(unit.expiration - date_).days,
            inception_vega=unit.inception_vega,
        )
        rows.append(row)
    if not rows:
        return pl.DataFrame(schema=reference_schema())
    return normalize_path(pl.DataFrame(rows), symbol)


def half_spread_dollars(mark: dict) -> float:
    return 0.5 * (mark["ask"] - mark["bid"]) * CONTRACT_MULTIPLIER


def normalize_path(records_df: pl.DataFrame, symbol: str) -> pl.DataFrame:
    """Scale P&L and costs by the inception vega of the unit held into each day."""
    scaled = pl.col("entry_vega") > 0
    return (
        records_df.sort("date")
        .with_columns(pl.col("inception_vega").shift(1).alias("entry_vega"))
        .with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.when(scaled)
            .then((pl.col("option_pnl") + pl.col("hedge_pnl")) / pl.col("entry_vega"))
            .alias("pnl_per_vega"),
            pl.when(scaled)
            .then((pl.col("option_cost") + pl.col("hedge_cost")) / pl.col("entry_vega"))
            .alias("cost_per_vega"),
            pl.when(pl.col("inception_vega") > 0)
            .then(pl.col("exit_cost") / pl.col("inception_vega"))
            .alias("exit_cost_per_vega"),
        )
        .select(REFERENCE_COLUMNS)
        .cast({"dte": pl.Int64, "mid": pl.Float64, "underlying": pl.Float64})
    )


def reference_schema() -> dict:
    return {
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
    }
