"""How each vendor spells the universe, and who was in it when."""

import datetime as dt

import polars as pl

from pipelines import store

# Every vendor spells the universe differently. ThetaData stocks keep the dot
# (BRK.B); ThetaData options strip it (BRKB); Yahoo wants a dash (BRK-B).
#
# BNY is mapped for both ThetaData asset classes: the stock endpoint answers
# a BNY request with an unrelated small-cap rather than an error, and Bank of
# New York Mellon is BK there all year. Yahoo serves the real company under
# BNY, which is why the table is keyed by destination.
TICKER_OVERRIDES = {
    "stock": {"BNY": "BK"},
    "option": {"BNY": "BK"},
    "yahoo": {},
}

# In the Wikipedia universe but absent from ThetaData, so never requested.
UNAVAILABLE = {
    "stock": {"ECHO", "MRSH", "VMRK"},
    "option": {"ECHO", "MRSH", "VMRK", "NVR"},
    "yahoo": set(),
}


def normalize_ticker(ticker: str, asset: str) -> str:
    """Map a Wikipedia ticker onto the spelling `asset` expects."""
    ticker = TICKER_OVERRIDES[asset].get(ticker, ticker)
    if asset == "option":
        return ticker.replace(".", "")
    if asset == "yahoo":
        return ticker.replace(".", "-")
    return ticker


def universe_tickers(start: dt.date | None = None, end: dt.date | None = None) -> list[str]:
    """Wikipedia tickers that were index members at some point in the window."""
    years = list(range(start.year, end.year + 1)) if start and end else None
    frame = store.in_window(store.scan("universe", years=years), start, end)
    return frame.select("ticker").unique().collect()["ticker"].sort().to_list()


def universe_symbols(
    asset: str, start: dt.date | None = None, end: dt.date | None = None
) -> list[str]:
    """Universe members over the window, spelled the way `asset` expects."""
    return sorted(
        {
            normalize_ticker(ticker, asset)
            for ticker in universe_tickers(start, end)
            if ticker not in UNAVAILABLE[asset]
        }
    )


def symbol_map(start: dt.date | None = None, end: dt.date | None = None) -> pl.DataFrame:
    """Every universe name in all four spellings, one row each; nothing excluded."""
    tickers = universe_tickers(start, end)
    return pl.DataFrame(
        {
            "ticker": tickers,
            "stock_symbol": [normalize_ticker(t, "stock") for t in tickers],
            "option_symbol": [normalize_ticker(t, "option") for t in tickers],
            "yahoo_symbol": [normalize_ticker(t, "yahoo") for t in tickers],
        }
    )
