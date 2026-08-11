"""
Portfolio-fit metrics computed against the live IBKR account.

This is the module that turns a stock screen into a *portfolio* decision. A
name that looks excellent standalone can still be a poor addition: if it
correlates 0.95 with a position that is already 16% of the book, buying it
does not diversify anything, it just levers an existing bet.

Three metrics are produced:

``corr_to_portfolio``
    Weekly return correlation between the candidate and the existing book's
    aggregate return stream. Lower is better.

``diversification_benefit``
    The reduction in total portfolio volatility from allocating a marginal
    slice to the candidate, funded from cash. Positive = the addition lowers
    portfolio risk. This is the honest version of "does it diversify" --
    it accounts for the candidate's own volatility, not just correlation.

``existing_overlap``
    Current weight of the name in the book, as a share of net liquidation
    value. Penalizes doubling down.
"""

from __future__ import annotations

import numpy as np

from .metrics import align_returns, simple_returns

#: Size of the hypothetical marginal allocation used to measure the
#: diversification effect. 2% is a realistic starter position.
MARGINAL_ALLOCATION = 0.02


def _closes(instrument: dict) -> np.ndarray:
    hist = instrument.get("history") or {}
    closes = [c for c in (hist.get("close") or []) if isinstance(c, (int, float))]
    return np.asarray(closes, dtype=float)


def build_portfolio_return_series(
    positions: list[dict],
    history_by_ticker: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float], float]:
    """
    Construct the existing book's weekly return series.

    Only positions with usable price history contribute. Cash, bills and bonds
    are intentionally excluded from the *return series* (they are near-zero
    volatility and would mechanically damp every correlation toward zero) but
    they remain in net liquidation value for weight calculations.

    Returns ``(portfolio_returns, weights_by_ticker, covered_market_value)``.
    """
    usable: dict[str, np.ndarray] = {}
    exposures: dict[str, float] = {}

    for pos in positions:
        ticker = (pos.get("ticker") or "").upper()
        mv = float(pos.get("market_value") or 0.0)
        if not ticker or mv == 0:
            continue
        prices = history_by_ticker.get(ticker)
        if prices is None or prices.size < 13:
            continue
        rets = simple_returns(prices)
        if rets.size < 12:
            continue
        usable[ticker] = rets
        exposures[ticker] = exposures.get(ticker, 0.0) + mv

    if not usable:
        return np.asarray([]), {}, 0.0

    # Net exposure basis: shorts contribute negative weight, which is correct --
    # a short VTWO position genuinely reduces small-cap beta in the book.
    gross = sum(abs(v) for v in exposures.values())
    if gross == 0:
        return np.asarray([]), {}, 0.0
    weights = {t: mv / gross for t, mv in exposures.items()}

    n = min(r.size for r in usable.values())
    stacked = np.vstack([usable[t][-n:] for t in usable])
    w = np.asarray([weights[t] for t in usable], dtype=float)
    port_returns = stacked.T @ w

    return port_returns, weights, sum(abs(v) for v in exposures.values())


def portfolio_volatility(port_returns: np.ndarray, periods_per_year: int = 52) -> float | None:
    if port_returns.size < 3:
        return None
    return float(np.std(port_returns, ddof=1) * np.sqrt(periods_per_year))


def marginal_diversification_benefit(
    candidate_returns: np.ndarray,
    port_returns: np.ndarray,
    allocation: float = MARGINAL_ALLOCATION,
) -> float | None:
    """
    Change in annualized portfolio volatility from a marginal allocation.

    Returns a *positive* number when the addition reduces portfolio volatility
    (i.e. it diversifies), expressed in volatility points. Direction +1 in the
    factor model, so diversifiers score well.
    """
    c, p = align_returns(candidate_returns, port_returns)
    if c.size < 12:
        return None
    base_vol = float(np.std(p, ddof=1))
    if base_vol == 0:
        return None
    blended = (1.0 - allocation) * p + allocation * c
    new_vol = float(np.std(blended, ddof=1))
    # Positive => volatility fell => diversification benefit.
    return float((base_vol - new_vol) * np.sqrt(52))


def compute_portfolio_fit(
    instrument: dict,
    port_returns: np.ndarray,
    existing_weights_by_nlv: dict[str, float],
) -> dict[str, float | None]:
    """Portfolio-fit metrics for one candidate."""
    ticker = (instrument.get("ticker") or "").upper()
    prices = _closes(instrument)
    rets = simple_returns(prices)

    if port_returns.size < 12 or rets.size < 12:
        return {
            "corr_to_portfolio": None,
            "diversification_benefit": None,
            "existing_overlap": existing_weights_by_nlv.get(ticker, 0.0),
        }

    c, p = align_returns(rets, port_returns)
    corr = None
    if c.size >= 12 and np.std(c) > 0 and np.std(p) > 0:
        val = np.corrcoef(c, p)[0, 1]
        corr = float(val) if np.isfinite(val) else None

    return {
        "corr_to_portfolio": corr,
        "diversification_benefit": marginal_diversification_benefit(rets, port_returns),
        "existing_overlap": existing_weights_by_nlv.get(ticker, 0.0),
    }


def existing_weights(positions: list[dict], net_liquidation: float) -> dict[str, float]:
    """Absolute market value of each holding as a share of net liquidation value."""
    if net_liquidation <= 0:
        return {}
    out: dict[str, float] = {}
    for pos in positions:
        ticker = (pos.get("ticker") or "").upper()
        if not ticker:
            continue
        out[ticker] = out.get(ticker, 0.0) + abs(float(pos.get("market_value") or 0.0)) / net_liquidation
    return out


def concentration_stats(positions: list[dict], net_liquidation: float) -> dict[str, float]:
    """
    Descriptive concentration measures for the existing book.

    ``hhi`` is the Herfindahl-Hirschman index of position weights; the
    reciprocal is the 'effective number of positions', which is a far more
    honest read on diversification than a raw position count.
    """
    weights = existing_weights(positions, net_liquidation)
    if not weights:
        return {}
    w = np.asarray(list(weights.values()), dtype=float)
    invested = float(w.sum())
    hhi = float(np.sum(w ** 2))
    return {
        "positions": int(len(w)),
        "invested_share_of_nlv": invested,
        "largest_position": float(w.max()),
        "top5_concentration": float(np.sort(w)[::-1][:5].sum()),
        "hhi": hhi,
        "effective_positions": float(1.0 / hhi) if hhi > 0 else 0.0,
    }
