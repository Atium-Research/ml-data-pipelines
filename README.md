# ml-data-pipelines

The write side of the malatium data store. Pulls S&P 500 option chains, stock prices and reference data from ThetaData, Yahoo and Wikipedia into a [bear-lake](https://github.com/andrewhall1124/bear-lake) database, and builds the derived panels the vol books run on: the reference straddle returns, the factor risk model, the IV surface, realized vol and its forecast, stock features and the signals.

Every table is canonical on disk: option roots as `symbol`, `right` as `C`/`P`, `iv` null where the vendor's inversion failed, `vega` per vol point. Every other column the vendor delivers (higher-order greeks, trade OHLC, quote sizes, raw `implied_vol` and `iv_error`, timestamps) follows the canonical ones unchanged, so the store is a complete copy of the pull. [ml-data-access](https://github.com/Atium-Research/ml-data-access) reads the same tables; [malatium](https://github.com/Atium-Research/malatium) consumes them as frames.

## Setup

```bash
uv sync
cp .env.example .env     # THETADATA_API_KEY and ML_DATA_STORE
uv run pipelines --help
```

`ML_DATA_STORE` is the bear-lake directory. Nothing else is configured.

## Running

One command per table. Each takes at most `--start` and `--end`, and both default to the window the store is meant to hold, so a bare command reproduces the same data every time.

```bash
uv run pipelines calendar
uv run pipelines universe
uv run pipelines option-greeks                 # ~3 h per year at four workers, resumable per symbol-year
uv run pipelines reference-returns             # the straddle paths, ~25 min for the whole history
uv run pipelines build                         # everything, in order
uv run pipelines build --skip-to surface
uv run pipelines migrate --source ~/Projects/quant-research/data_store
```

Run one command at a time: ThetaData issues one session per account and two processes fight over it. Concurrency lives inside a step.

## Tables

Raw, from the vendors:

| table | partition | holds |
| --- | --- | --- |
| `calendar` | | exchange sessions |
| `universe` | year | point-in-time S&P 500 membership, `ticker` (Wikipedia) and `symbol` (option root) |
| `sectors` | | GICS sector per current constituent, a snapshot |
| `indices`, `yields`, `rates` | | SPX, RUT, OEX, XSP and the VIX complex; the CBOE yield curve; SOFR |
| `corporate_actions` | | splits and dividends from Yahoo, ex-date |
| `earnings` | | announcement dates with a `bmo` / `amc` / `unknown` session |
| `underlying` | year | EOD stock OHLCV, 2023-06 on (the free tier's floor) |
| `option_greeks` | year, symbol | EOD chains with greeks and IV, 2017 on |
| `index_greeks` | year, symbol | the same for SPX, SPXW, XSP, VIX, with the 15:59 quote repair |
| `open_interest` | year, symbol | EOD open interest, stamped pre-open |
| `symbology_check` | | per symbol-year: is the chain the company the universe names? |

Derived, in the order they are built:

| table | partition | holds |
| --- | --- | --- |
| `reference_returns` | year, symbol | one delta-hedged ATM straddle per name, rolled at 20 DTE or 15% drift: P&L, cost and exit cost per dollar of inception vega, plus SPX |
| `factor_returns`, `factor_loadings`, `factor_covariances`, `idio_vol` | (year) | market and GICS sector vol factors, trailing 250-session loadings, Ledoit-Wolf covariance, residual vol |
| `surface` | year | constant-maturity ATM IV at 30, 60 and 90 days |
| `realized_vol` | year | close-to-close RV over 1, 5 and 22 sessions and the forward 60 |
| `forecast` | year | pooled log-HAR forecast of the forward 60-session vol, refit monthly |
| `stock_features` | year | return, 250-session beta, 60-session idio vol, gap frequency |
| `signals` | year | richness scores: `vrp`, `iv_zscore`, `momentum` |

## Things to know before trusting a number

- **Delisted names carry a zero row** in `underlying`, kept so the store is complete; filter `close > 0` before computing a return.
- **`iv` is null, not wrong.** About 3% of contract-days fail to invert; the canonical table nulls them instead of carrying the vendor's pinned 0.5.
- **Open interest is one day stale by construction.** It reports the position after the previous close, which is what a trader at today's close knows, so it joins on the same `date`.
- **Nothing checks that a symbol is the company the universe names** except `symbology_check`. Eighteen symbol-years are another company's chain; screen on `status != 'wrong_instrument'`. `thin_overlap` means the check could not run, not that it failed.
- **Index sessions repaired from the 15:59 bar have null gamma** and an `underlying` from that bar.
- **The universe and corporate actions drift.** The Wikipedia walk starts from today's list and Yahoo back-adjusts to today, so those two tables differ slightly between rebuilds. The option data does not.

## Layout

`store.py` holds every table's schema, partition keys and primary keys, and the reads (`scan`) that open only the partitions a job needs. `canonical.py` is the only module that knows a ThetaData column name. One module per table with a `run`; `__main__.py` wires them into click. `utils/` holds the ThetaData session, ticker spellings, sessions, the Yahoo download, Black-Scholes and the total-variance interpolation.

```bash
uv run pytest        # synthetic data only, no store needed
uv run ruff check . && uv run ruff format .
```
