"""Black-Scholes prices and greeks, vectorised over numpy arrays.

Serves the vega-convention check and the synthetic chains in the tests.
`sigma` and `rate` are decimals, `tau` is years (calendar days / 365),
`vega` is per share per 1.00 of sigma, `theta` per share per calendar day.
"""

import numpy as np
from scipy.special import ndtr

DAYS_PER_YEAR = 365.0


def compute_d1_d2(spot, strike, tau, rate, sigma) -> tuple[np.ndarray, np.ndarray]:
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-12)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    root_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * root_tau)
    return d1, d1 - sigma * root_tau


def price(spot, strike, tau, rate, sigma, is_call) -> np.ndarray:
    d1, d2 = compute_d1_d2(spot, strike, tau, rate, sigma)
    discount = np.exp(-rate * tau)
    call = spot * ndtr(d1) - strike * discount * ndtr(d2)
    put = strike * discount * ndtr(-d2) - spot * ndtr(-d1)
    return np.where(is_call, call, put)


def delta(spot, strike, tau, rate, sigma, is_call) -> np.ndarray:
    d1, _ = compute_d1_d2(spot, strike, tau, rate, sigma)
    return np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)


def gamma(spot, strike, tau, rate, sigma) -> np.ndarray:
    d1, _ = compute_d1_d2(spot, strike, tau, rate, sigma)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-12)
    return normal_pdf(d1) / (spot * sigma * np.sqrt(tau))


def vega(spot, strike, tau, rate, sigma) -> np.ndarray:
    d1, _ = compute_d1_d2(spot, strike, tau, rate, sigma)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-12)
    return spot * normal_pdf(d1) * np.sqrt(tau)


def theta(spot, strike, tau, rate, sigma, is_call) -> np.ndarray:
    d1, d2 = compute_d1_d2(spot, strike, tau, rate, sigma)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-12)
    discount = np.exp(-rate * tau)
    common = -spot * normal_pdf(d1) * sigma / (2.0 * np.sqrt(tau))
    call = common - rate * strike * discount * ndtr(d2)
    put = common + rate * strike * discount * ndtr(-d2)
    return np.where(is_call, call, put) / DAYS_PER_YEAR


def normal_pdf(x) -> np.ndarray:
    return np.exp(-0.5 * np.square(x)) / np.sqrt(2.0 * np.pi)
