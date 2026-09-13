"""Exchange sessions, and cutting a window into requests ThetaData accepts."""

import datetime as dt

import polars as pl

from pipelines import store

# ThetaData rejects a history request spanning more than 365 days.
MAX_REQUEST_DAYS = 360


def trading_sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    """Sessions in the inclusive window, from the `calendar` table."""
    frame = store.in_window(store.scan("calendar"), start, end)
    return frame.collect()["date"].sort().to_list()


def date_chunks(
    start: dt.date, end: dt.date, span_days: int = MAX_REQUEST_DAYS
) -> list[tuple[dt.date, dt.date]]:
    """Split an inclusive window into pieces ThetaData will accept."""
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + dt.timedelta(days=span_days), end)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end + dt.timedelta(days=1)
    return chunks


def years_in(start: dt.date, end: dt.date) -> list[int]:
    return list(range(start.year, end.year + 1))


def year_window(year: int, start: dt.date, end: dt.date) -> tuple[dt.date, dt.date]:
    """The part of `[start, end]` that falls in `year`."""
    return max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31))


def sessions_frame(sessions: list[dt.date]) -> pl.DataFrame:
    return pl.DataFrame({"date": sessions}, schema={"date": pl.Date})
