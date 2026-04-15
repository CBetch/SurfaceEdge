"""
baw_pricing.py

Barone-Adesi and Whaley (BAW) approximation for American option pricing.

NOTE: This model is kept for reference but was not used as a baseline in the
final SurfaceEdge evaluation. BSM-based models fail to beat the naive 0.0
baseline for next-day price change prediction because IV is derived from the
mark price — feeding IV back into BSM approximately recovers the current mark,
meaning the predicted change is close to zero with noise. The naive baseline
(baseline.py) is used instead.

Reference:
    Barone-Adesi, G. and Whaley, R.E. (1987). Efficient Analytic Approximation
    of American Option Values. Journal of Finance, 42(2), 301-320.

Usage:
    from baw_pricing import baw_price, baw_price_change

    price = baw_price(S=130.41, K=141.88, T=45/365, sigma=0.28,
                      is_call=True, q=0.005)

    pct_change = baw_price_change(S=130.41, K=141.88, T=45/365, sigma=0.28,
                                   is_call=True, mark=6.22, q=0.005)
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

RISK_FREE_RATE = 0.045


# ── Black-Scholes-Merton ──────────────────────────────────────────────────────

def _bs_call(S, K, T, r, sigma, q):
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2))


def _bs_put(S, K, T, r, sigma, q):
    if T <= 0 or sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (K * np.exp(-r * T) * norm.cdf(-d2)
            - S * np.exp(-q * T) * norm.cdf(-d1))


# ── BAW approximation ─────────────────────────────────────────────────────────

def _baw_call(S, K, T, r, sigma, q):
    """BAW American call. Falls back to BSM when q=0 (no dividends)."""
    if q == 0.0:
        return _bs_call(S, K, T, r, sigma, q)

    european = _bs_call(S, K, T, r, sigma, q)

    M  = 2 * r / sigma ** 2
    N  = 2 * (r - q) / sigma ** 2
    k  = 1 - np.exp(-r * T)
    q2 = (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / k)) / 2

    def _equation(Sstar):
        d1 = (np.log(Sstar / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return (Sstar - K) - (_bs_call(Sstar, K, T, r, sigma, q)
                               + (1 - np.exp(-q * T) * norm.cdf(d1)) * Sstar / q2)

    try:
        Sstar = brentq(_equation, K * 1e-6, K * 1000, maxiter=200)
    except ValueError:
        return european

    if S >= Sstar:
        return S - K

    d1star = (np.log(Sstar / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    A2     = (Sstar / q2) * (1 - np.exp(-q * T) * norm.cdf(d1star))
    d1     = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return european + A2 * (S / Sstar) ** q2


def _baw_put(S, K, T, r, sigma, q):
    """BAW American put."""
    european = _bs_put(S, K, T, r, sigma, q)

    M  = 2 * r / sigma ** 2
    N  = 2 * (r - q) / sigma ** 2
    k  = 1 - np.exp(-r * T)
    q1 = (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / k)) / 2

    def _equation(Sstar):
        d1 = (np.log(Sstar / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return (K - Sstar) - (_bs_put(Sstar, K, T, r, sigma, q)
                               - (1 - np.exp(-q * T) * norm.cdf(-d1)) * Sstar / q1)

    try:
        Sstar = brentq(_equation, K * 1e-6, K * 1000, maxiter=200)
    except ValueError:
        return european

    if S <= Sstar:
        return K - S

    d1star = (np.log(Sstar / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    A1     = -(Sstar / q1) * (1 - np.exp(-q * T) * norm.cdf(-d1star))
    d1     = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return european + A1 * (S / Sstar) ** q1


# ── Public API ────────────────────────────────────────────────────────────────

def baw_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    is_call: bool,
    q: float = 0.0,
    r: float = RISK_FREE_RATE,
) -> float:
    """
    Compute BAW American option price.

    Args:
        S:       Spot price
        K:       Strike price (split-adjusted)
        T:       Time to expiry in years (tau / 365)
        sigma:   Implied volatility (annualized)
        is_call: True for call, False for put
        q:       Annualized continuous dividend yield (default 0)
        r:       Risk-free rate (default 4.5%)

    Returns:
        Theoretical American option price.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    try:
        return _baw_call(S, K, T, r, sigma, q) if is_call else _baw_put(S, K, T, r, sigma, q)
    except Exception:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)


def baw_price_change(
    S: float,
    K: float,
    T: float,
    sigma: float,
    is_call: bool,
    mark: float,
    q: float = 0.0,
    r: float = RISK_FREE_RATE,
) -> float:
    """
    Predict next-day percentage price change as (BAW_price - mark) / mark.

    Args:
        S:       Spot price
        K:       Strike price (split-adjusted)
        T:       Time to expiry in years
        sigma:   Implied volatility
        is_call: True for call, False for put
        mark:    Today's observed mark price
        q:       Annualized dividend yield (default 0)
        r:       Risk-free rate (default 4.5%)

    Returns:
        Predicted percentage price change as a signed decimal.
    """
    if mark <= 0:
        return 0.0
    return (baw_price(S, K, T, sigma, is_call, q, r) - mark) / mark


if __name__ == "__main__":
    print("BAW sanity check — AAPL call, 2020-09-01, cell (32, 21):")
    price = baw_price(S=130.41, K=141.88, T=45/365, sigma=0.28, is_call=True, q=0.005)
    mark  = 6.22
    pct   = baw_price_change(S=130.41, K=141.88, T=45/365, sigma=0.28,
                              is_call=True, mark=mark, q=0.005)
    print(f"  BAW price          : ${price:.4f}")
    print(f"  Mark price         : ${mark:.4f}")
    print(f"  Predicted % change : {pct*100:.4f}%")
    print(f"  Actual % change    : +2.4293%")
