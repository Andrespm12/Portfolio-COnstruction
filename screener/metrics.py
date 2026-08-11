"""
Metric computation from raw IBKR price history and market-data snapshots.

Input contract
--------------
Every instrument arrives as a dict shaped like::

    {
      "ticker": "AAPL",
      "conid": 265598,
      "name": "APPLE INC",
      "asset_type": "STOCK" | "ETF",
      "exchange": "NASDAQ",
      "indices": ["SP500", "NDX", "DJIA"],
      "sector": "Information Technology",
      "snapshot": {...},   # normalized IBKR get_price_snapshot payload
      "history": {"time": [...], "close": [...], "high": [...],
                  "low": [...], "open": [...], "volume": [...]}
    }

All functions are defensive: IBKR returns ``{}`` for fields that are
unavailable for a given security type (e.g. ``cumulative-perf-*`` is
ETF-only, ``total-net-assets`` is empty for single stocks), so every
extractor returns ``None`` rather than raising when data is absent.
``None`` propagates through to the scorer, which re-normalizes weights.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .config import PERIODS_PER_YEAR, RISK_FREE_RATE


# --------------------------------------------------------------------------
# Safe extraction helpers
# --------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    """Coerce to float, returning None for anything non-numeric or non-finite."""
    if value is None or isinstance(value, (dict, list, str, bool)):
        if isinstance(value, str):
            try:
                value = float(value.replace(",", "").replace("%", ""))
            except ValueError:
                return None
        else:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def snap_get(snapshot: dict, field: str, key: str | None = None) -> float | None:
    """
    Read a value out of an IBKR snapshot.

    IBKR returns hyphenated keys for underscored request names, and returns an
    empty dict for unavailable fields, so both are handled here.
    """
    if not snapshot:
        return None
    node = snapshot.get(field)
    if node is None:
        node = snapshot.get(field.replace("_", "-"))
    if node is None:
        return None
    if isinstance(node, dict):
        if not node:
            return None
        if key is None:
            return None
        return _num(node.get(key))
    return _num(node)


def _closes(instrument: dict) -> np.ndarray:
    hist = instrument.get("history") or {}
    closes = [c for c in (hist.get("close") or []) if _num(c) is not None]
    return np.asarray(closes, dtype=float)


def _volumes(instrument: dict) -> np.ndarray:
    hist = instrument.get("history") or {}
    vols = [v for v in (hist.get("volume") or []) if _num(v) is not None]
    return np.asarray(vols, dtype=float)


def simple_returns(prices: np.ndarray) -> np.ndarray:
    """Period-over-period simple returns. Empty array if insufficient data."""
    if prices.size < 2:
        return np.asarray([], dtype=float)
    prev = prices[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.where(prev != 0, (prices[1:] - prev) / prev, np.nan)
    return rets[np.isfinite(rets)]


# --------------------------------------------------------------------------
# Momentum & trend
# --------------------------------------------------------------------------

def total_return(prices: np.ndarray, lookback: int, skip: int = 0) -> float | None:
    """
    Total return over `lookback` periods ending `skip` periods before the end.

    ``skip=4`` on weekly bars implements the standard 12M-1M momentum
    construction that excludes the short-term reversal month.
    """
    if prices.size < lookback + skip + 1:
        return None
    end = prices.size - 1 - skip
    start = end - lookback
    if start < 0 or prices[start] == 0:
        return None
    return float(prices[end] / prices[start] - 1.0)


def moving_average(prices: np.ndarray, window: int) -> float | None:
    if prices.size < window:
        return None
    return float(np.mean(prices[-window:]))


def pct_above_ma(prices: np.ndarray, window: int) -> float | None:
    ma = moving_average(prices, window)
    if ma is None or ma == 0:
        return None
    return float(prices[-1] / ma - 1.0)


def ma_slope(prices: np.ndarray, window: int, lookback: int) -> float | None:
    """Percentage change in the moving average itself over `lookback` periods."""
    if prices.size < window + lookback:
        return None
    ma_now = float(np.mean(prices[-window:]))
    ma_then = float(np.mean(prices[-window - lookback:-lookback]))
    if ma_then == 0:
        return None
    return float(ma_now / ma_then - 1.0)


# --------------------------------------------------------------------------
# Risk statistics
# --------------------------------------------------------------------------

def annualized_volatility(returns: np.ndarray) -> float | None:
    if returns.size < 3:
        return None
    return float(np.std(returns, ddof=1) * math.sqrt(PERIODS_PER_YEAR))


def annualized_return(prices: np.ndarray) -> float | None:
    if prices.size < 2 or prices[0] <= 0:
        return None
    periods = prices.size - 1
    years = periods / PERIODS_PER_YEAR
    if years <= 0:
        return None
    growth = prices[-1] / prices[0]
    if growth <= 0:
        return None
    return float(growth ** (1.0 / years) - 1.0)


def downside_deviation(returns: np.ndarray, mar: float = 0.0) -> float | None:
    """Annualized standard deviation of returns below the minimum acceptable return."""
    if returns.size < 3:
        return None
    downside = returns[returns < mar] - mar
    if downside.size == 0:
        return 0.0
    return float(math.sqrt(np.mean(downside ** 2)) * math.sqrt(PERIODS_PER_YEAR))


def max_drawdown(prices: np.ndarray) -> float | None:
    """Worst peak-to-trough decline as a negative fraction."""
    if prices.size < 2:
        return None
    running_peak = np.maximum.accumulate(prices)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_peak != 0, prices / running_peak - 1.0, np.nan)
    dd = dd[np.isfinite(dd)]
    return float(np.min(dd)) if dd.size else None


def ulcer_index(prices: np.ndarray) -> float | None:
    """
    Root-mean-square of percentage drawdowns.

    Unlike max drawdown this penalizes *time spent underwater*, not just the
    single worst point -- closer to how an allocator actually experiences pain.
    """
    if prices.size < 2:
        return None
    running_peak = np.maximum.accumulate(prices)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_peak != 0, (prices / running_peak - 1.0) * 100.0, np.nan)
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return None
    return float(math.sqrt(np.mean(dd ** 2)))


def sharpe_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float | None:
    if returns.size < 3:
        return None
    sd = np.std(returns, ddof=1)
    if sd == 0:
        return None
    excess = np.mean(returns) - rf / PERIODS_PER_YEAR
    return float(excess / sd * math.sqrt(PERIODS_PER_YEAR))


def sortino_ratio(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> float | None:
    if returns.size < 3:
        return None
    dd = downside_deviation(returns, mar=rf / PERIODS_PER_YEAR)
    if dd is None or dd == 0:
        return None
    excess = (np.mean(returns) - rf / PERIODS_PER_YEAR) * PERIODS_PER_YEAR
    return float(excess / dd)


def calmar_ratio(prices: np.ndarray) -> float | None:
    ann = annualized_return(prices)
    mdd = max_drawdown(prices)
    if ann is None or mdd is None or mdd == 0:
        return None
    return float(ann / abs(mdd))


def pct_positive(returns: np.ndarray) -> float | None:
    if returns.size < 3:
        return None
    return float(np.sum(returns > 0) / returns.size)


# --------------------------------------------------------------------------
# Market sensitivity (requires an aligned benchmark return series)
# --------------------------------------------------------------------------

def align_returns(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim both series to their common (most recent) length."""
    n = min(a.size, b.size)
    if n == 0:
        return np.asarray([]), np.asarray([])
    return a[-n:], b[-n:]


def beta_alpha(returns: np.ndarray, bench: np.ndarray,
               rf: float = RISK_FREE_RATE) -> tuple[float | None, float | None, float | None]:
    """
    OLS regression of asset excess returns on benchmark excess returns.

    Returns ``(beta, annualized_alpha, r_squared)``.
    """
    r, m = align_returns(returns, bench)
    if r.size < 12:
        return None, None, None
    rf_per = rf / PERIODS_PER_YEAR
    r_ex, m_ex = r - rf_per, m - rf_per
    var_m = np.var(m_ex, ddof=1)
    if var_m == 0:
        return None, None, None
    beta = float(np.cov(r_ex, m_ex, ddof=1)[0, 1] / var_m)
    alpha_per = float(np.mean(r_ex) - beta * np.mean(m_ex))
    alpha_ann = float((1.0 + alpha_per) ** PERIODS_PER_YEAR - 1.0)
    corr = np.corrcoef(r, m)[0, 1]
    r2 = float(corr ** 2) if np.isfinite(corr) else None
    return beta, alpha_ann, r2


def capture_ratios(returns: np.ndarray, bench: np.ndarray) -> tuple[float | None, float | None]:
    """
    Upside and downside capture vs the benchmark.

    Upside capture > 100% with downside capture < 100% is the convexity profile
    worth paying single-name risk for.
    """
    r, m = align_returns(returns, bench)
    if r.size < 12:
        return None, None
    up, down = m > 0, m < 0
    upside = float(np.mean(r[up]) / np.mean(m[up])) if up.sum() >= 3 and np.mean(m[up]) != 0 else None
    downside = float(np.mean(r[down]) / np.mean(m[down])) if down.sum() >= 3 and np.mean(m[down]) != 0 else None
    return upside, downside


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    x, y = align_returns(a, b)
    if x.size < 12 or np.std(x) == 0 or np.std(y) == 0:
        return None
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else None


# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------

def adv_usd(instrument: dict) -> float | None:
    """
    90-day average daily traded value in USD.

    Prefers IBKR's own ``avg-90d-usd-volume`` field; falls back to
    median(weekly volume)/5 * last price when that field is unavailable.
    """
    direct = snap_get(instrument.get("snapshot") or {}, "avg_90d_usd_volume", "volume")
    if direct is not None and direct > 0:
        return direct
    vols, closes = _volumes(instrument), _closes(instrument)
    if vols.size == 0 or closes.size == 0:
        return None
    recent = vols[-13:] if vols.size >= 13 else vols
    daily_shares = float(np.median(recent)) / 5.0
    return daily_shares * float(closes[-1])


def turnover_stability(instrument: dict) -> float | None:
    """
    Inverse coefficient of variation of weekly volume.

    Names whose liquidity appears only in bursts (earnings, index events) are
    penalized: average ADV overstates the depth available on an ordinary day.
    """
    vols = _volumes(instrument)
    if vols.size < 8:
        return None
    mean = float(np.mean(vols))
    sd = float(np.std(vols, ddof=1))
    if mean == 0 or sd == 0:
        return None
    return float(mean / sd)


def days_to_liquidate(adv: float | None, position_usd: float,
                      participation: float) -> float | None:
    if adv is None or adv <= 0 or participation <= 0:
        return None
    return float(position_usd / (adv * participation))


# --------------------------------------------------------------------------
# Snapshot-derived metrics
# --------------------------------------------------------------------------

def range_position(instrument: dict) -> float | None:
    """
    Where the last price sits inside the 52-week high-low band, 0.0 to 1.0.

    Scored with direction -1 (mean-reversion counterweight): a name pinned at
    the top of its range has already re-rated, all else equal.
    """
    snap = instrument.get("snapshot") or {}
    hi = snap_get(snap, "misc_statistics", "high_52w")
    lo = snap_get(snap, "misc_statistics", "low_52w")
    last = last_price(instrument)
    if hi is None or lo is None or last is None or hi <= lo:
        return None
    return float(max(0.0, min(1.0, (last - lo) / (hi - lo))))


def pct_from_52w_high(instrument: dict) -> float | None:
    """
    Percentage distance from the 52-week high, as a negative number.

    Direction +1: closer to zero (nearer the high) is better.
    """
    snap = instrument.get("snapshot") or {}
    hi = snap_get(snap, "misc_statistics", "high_52w")
    last = last_price(instrument)
    if hi is None or last is None or hi <= 0:
        closes = _closes(instrument)
        if closes.size < 2:
            return None
        hi, last = float(np.max(closes)), float(closes[-1])
    if hi <= 0:
        return None
    return float(last / hi - 1.0)


def last_price(instrument: dict) -> float | None:
    snap = instrument.get("snapshot") or {}
    price = snap_get(snap, "last", "price")
    if price is not None and price > 0:
        return price
    closes = _closes(instrument)
    return float(closes[-1]) if closes.size else None


def iv_hv_spread(instrument: dict) -> float | None:
    """Implied volatility minus 30-day realized volatility, both annualized."""
    snap = instrument.get("snapshot") or {}
    iv = snap_get(snap, "implied_vol_underlying", "annual_iv")
    hv = snap_get(snap, "historical_vol", "annual_pct")
    if iv is None or hv is None:
        return None
    return float(iv - hv)


def iv_percentile(instrument: dict) -> float | None:
    return snap_get(instrument.get("snapshot") or {}, "implied_volatility_percentile", "high_52w")


def dividend_yield(instrument: dict) -> float | None:
    y = snap_get(instrument.get("snapshot") or {}, "dividend_yield", "yield_pct")
    return None if y is None else float(y) / 100.0


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def compute_metrics(instrument: dict, bench_returns: np.ndarray,
                    net_liq: float, participation: float,
                    base_position_weight: float,
                    rf: float = RISK_FREE_RATE) -> dict[str, float | None]:
    """
    Compute every price/market-data-derived metric for one instrument.

    Portfolio-fit metrics are added separately by :mod:`screener.portfolio`,
    which needs the whole book rather than a single name.
    """
    prices = _closes(instrument)
    rets = simple_returns(prices)

    beta, alpha, r2 = beta_alpha(rets, bench_returns, rf=rf)
    up_cap, down_cap = capture_ratios(rets, bench_returns)
    capture_spread = (up_cap - down_cap) if (up_cap is not None and down_cap is not None) else None
    idio = (1.0 - r2) if r2 is not None else None

    adv = adv_usd(instrument)
    target_position_usd = net_liq * base_position_weight
    dtl = days_to_liquidate(adv, target_position_usd, participation)
    vol = annualized_volatility(rets)

    return {
        # Momentum & trend
        "mom_12_1": total_return(prices, lookback=48, skip=4),
        "mom_6m": total_return(prices, lookback=26),
        "mom_3m": total_return(prices, lookback=13),
        "pct_from_52w_high": pct_from_52w_high(instrument),
        "above_40w_ma": pct_above_ma(prices, 40),
        "ma_slope_13w": ma_slope(prices, window=40, lookback=13),
        # Risk-adjusted return
        "sharpe_1y": sharpe_ratio(rets, rf=rf),
        "sortino_1y": sortino_ratio(rets, rf=rf),
        "calmar_1y": calmar_ratio(prices),
        "pct_positive_periods": pct_positive(rets),
        # Volatility & drawdown
        "volatility_1y": vol,
        "max_drawdown": abs(max_drawdown(prices)) if max_drawdown(prices) is not None else None,
        "downside_deviation": downside_deviation(rets),
        "ulcer_index": ulcer_index(prices),
        # Market sensitivity
        "capture_spread": capture_spread,
        "alpha_annual": alpha,
        "idio_vol_share": idio,
        "beta_1y": beta,
        # Liquidity
        "adv_usd_log": math.log10(adv) if adv and adv > 0 else None,
        "days_to_liquidate": dtl,
        "turnover_stability": turnover_stability(instrument),
        # Valuation proxy & carry
        "dividend_yield": dividend_yield(instrument),
        "iv_hv_spread": iv_hv_spread(instrument),
        "iv_percentile": iv_percentile(instrument),
        "range_position": range_position(instrument),
    }


def diagnostics(instrument: dict, bench_returns: np.ndarray,
                rf: float = RISK_FREE_RATE) -> dict[str, Any]:
    """Raw descriptive values carried through to the report (not scored)."""
    prices = _closes(instrument)
    rets = simple_returns(prices)
    beta, alpha, r2 = beta_alpha(rets, bench_returns, rf=rf)
    up_cap, down_cap = capture_ratios(rets, bench_returns)
    mdd = max_drawdown(prices)
    return {
        "last_price": last_price(instrument),
        "bars": int(prices.size),
        "return_1y": total_return(prices, lookback=min(52, max(prices.size - 1, 0))),
        "ann_return": annualized_return(prices),
        "volatility": annualized_volatility(rets),
        "max_drawdown": mdd,
        "beta": beta,
        "alpha_annual": alpha,
        "r_squared": r2,
        "upside_capture": up_cap,
        "downside_capture": down_cap,
        "adv_usd": adv_usd(instrument),
        "dividend_yield": dividend_yield(instrument),
    }
