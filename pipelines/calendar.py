"""Exchange sessions: weekdays minus ThetaData's full closures.

`calendar_year` is free at every tier and reaches back to 2016. Early closes
are trading days and are kept; only `full_close` rows are dropped.
"""

import datetime as dt

import polars as pl

from pipelines import store
from pipelines.utils.theta import make_client

START = dt.date(2016, 1, 1)
END = dt.date(2026, 12, 31)


def fetch_sessions(start: dt.date, end: dt.date) -> list[dt.date]:
    client = make_client()
    closed = set()
    for year in range(start.year, end.year + 1):
        calendar_df = client.calendar_year(str(year))
        closed |= set(
            calendar_df.filter(pl.col("type") == "full_close")["date"]
            .str.strptime(pl.Date, "%Y-%m-%d")
            .to_list()
        )
    sessions, day = [], start
    while day <= end:
        if day.weekday() < 5 and day not in closed:
            sessions.append(day)
        day += dt.timedelta(days=1)
    return sessions


def run(start: dt.date = START, end: dt.date = END) -> None:
    sessions = fetch_sessions(start, end)
    store.write(store.connect(), "calendar", pl.DataFrame({"date": sessions}))
    print(f"calendar: {len(sessions)} sessions, {sessions[0]} .. {sessions[-1]}")
