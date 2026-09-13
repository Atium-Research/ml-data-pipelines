"""EOD chains for the index roots (SPX, SPXW, XSP, VIX), in their own table.

Same pull as `option_greeks`; kept apart because index roots are not
universe members and SPX is the market factor.
"""

import datetime as dt

from pipelines import store
from pipelines.option_greeks import END, START, pull_year
from pipelines.utils.sessions import years_in

INDEX_ROOTS = ["SPX", "SPXW", "XSP", "VIX"]


def run(start: dt.date = START, end: dt.date = END) -> None:
    db = store.connect()
    for year in years_in(start, end):
        pull_year(db, "index_greeks", INDEX_ROOTS, year, start, end)
