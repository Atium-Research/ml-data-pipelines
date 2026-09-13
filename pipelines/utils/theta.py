"""The shared ThetaData session, and running a pull over many symbols.

ThetaData issues one session per account, so every thread shares one client
and two pipeline processes must never run at once.
"""

import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from thetadata import ThetaClient

client_lock = threading.Lock()
shared_client: ThetaClient | None = None


def make_client() -> ThetaClient:
    """A single process-wide client; a second one invalidates the first."""
    global shared_client
    with client_lock:
        if shared_client is None:
            shared_client = ThetaClient(dataframe_type="polars")
    return shared_client


def reset_client() -> None:
    """Drop the shared client after the server answers UNAUTHENTICATED on an expired session."""
    global shared_client
    with client_lock:
        shared_client = None


def with_retries[T](fetch: Callable[[str], T], symbol: str, attempts: int = 8) -> T:
    """Exponential backoff on RESOURCE_EXHAUSTED and UNAVAILABLE; reconnect on UNAUTHENTICATED."""
    for attempt in range(attempts):
        try:
            return fetch(symbol)
        except Exception as error:
            message = str(error)
            expired = "UNAUTHENTICATED" in message
            retryable = expired or "RESOURCE_EXHAUSTED" in message or "UNAVAILABLE" in message
            if not retryable or attempt == attempts - 1:
                raise
            if expired:
                # every worker sees the dead session at once; wait longer and jitter
                reset_client()
                time.sleep(min(60, 5 * 2**attempt) * (0.5 + random.random()))
                continue
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_many[T](symbols: list[str], fetch: Callable[[str], T], workers: int) -> list[T]:
    """Run `fetch` over symbols with a thread pool, printing progress and an ETA."""
    results, failures = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(with_retries, fetch, symbol): symbol for symbol in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append((symbol, error))
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (len(symbols) - done)
            print(
                f"\r  {done}/{len(symbols)} | {elapsed:6.1f}s elapsed | {eta:7.1f}s eta"
                f" | {len(failures)} failed",
                end="",
                flush=True,
            )
    print()
    if failures:
        print(f"  failures: {[symbol for symbol, _ in failures][:20]}")
        print(f"  first error: {failures[0][1]}")
    return results


def is_missing_data(error: Exception) -> bool:
    """A symbol with no listed chain in the window: not a failure of the pull."""
    return "NoDataFound" in type(error).__name__ or "NOT_FOUND" in str(error)
