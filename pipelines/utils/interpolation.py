"""Constant-maturity implied vol by linear interpolation in total variance.

Total variance `iv^2 * t` is linear in time under flat forward variance,
which is the standard way to read a surface at a tenor no listed expiration
sits on. A target outside the pillar range is null unless `extrapolate`,
which holds the nearest pillar flat rather than extending a slope.
"""

import numpy as np
import polars as pl

DAYS_PER_YEAR = 365.0


def interpolate_total_variance(
    dte: np.ndarray, iv: np.ndarray, target_dte: float, extrapolate: bool = False
) -> float:
    dte = np.asarray(dte, dtype=float)
    iv = np.asarray(iv, dtype=float)
    keep = np.isfinite(dte) & np.isfinite(iv) & (dte > 0) & (iv > 0)
    dte, iv = dte[keep], iv[keep]
    if dte.size == 0:
        return float("nan")
    order = np.argsort(dte)
    dte, iv = dte[order], iv[order]
    unique_dte, inverse = np.unique(dte, return_inverse=True)
    if unique_dte.size != dte.size:
        iv = np.array([iv[inverse == i].mean() for i in range(unique_dte.size)])
        dte = unique_dte
    if dte.size == 1:
        return float(iv[0]) if extrapolate or dte[0] == target_dte else float("nan")
    if target_dte < dte[0] or target_dte > dte[-1]:
        if not extrapolate:
            return float("nan")
        return float(iv[0]) if target_dte < dte[0] else float(iv[-1])
    tau = dte / DAYS_PER_YEAR
    target_tau = target_dte / DAYS_PER_YEAR
    total_variance = np.interp(target_tau, tau, iv**2 * tau)
    return float(np.sqrt(max(total_variance, 0.0) / target_tau))


def build_tenor_columns(
    frame: pl.LazyFrame,
    group_by: list[str],
    dte_col: str,
    iv_col: str,
    target_dtes: tuple[int, ...],
    extrapolate: bool = False,
) -> pl.LazyFrame:
    """Collapse the pillars of each group into one row with an `iv{dte}` column per tenor."""
    names = [f"iv{t}" for t in target_dtes]

    def interpolate_row(row: dict) -> dict:
        dte = np.asarray(row[dte_col], dtype=float)
        iv = np.asarray(row[iv_col], dtype=float)
        return dict(
            zip(
                names,
                [interpolate_total_variance(dte, iv, t, extrapolate) for t in target_dtes],
                strict=True,
            )
        )

    return (
        frame.group_by(group_by)
        .agg(pl.col(dte_col), pl.col(iv_col))
        .with_columns(
            pl.struct(dte_col, iv_col)
            .map_elements(
                interpolate_row, return_dtype=pl.Struct({name: pl.Float64 for name in names})
            )
            .alias("tenors")
        )
        .unnest("tenors")
        .drop(dte_col, iv_col)
        .with_columns([pl.col(name).fill_nan(None) for name in names])
    )
