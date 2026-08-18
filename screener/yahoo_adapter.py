"""
Yahoo Finance adapter: builds ``market_data.json`` without an IBKR session.

Why this exists
---------------
The scoring engine consumes a broker-agnostic payload (see
:mod:`screener.metrics` for the schema). ``scripts/build_market_data.py``
produces that payload from a captured IBKR pull, which is exact but requires an
IBKR MCP session and is therefore frozen at capture time. This module produces
the *same payload shape* from Yahoo Finance, which needs nothing but an
internet connection -- so the engine can run in Google Colab, on a laptop, or
in CI, over a universe of any size.

What is faithful and what is not
--------------------------------
Faithful (computed from the same underlying quantities IBKR reports):

    last price, 52-week high/low, 90-day average daily traded value,
    weekly close history, weekly share volume, trailing dividend yield.

Degraded, and the engine is told so rather than being fed a guess:

    ``implied-vol-underlying``  -- available only with ``fetch_iv=True``, which
                                   costs two extra requests per ticker.
    ``implied-volatility-percentile`` -- NOT available at any setting. Yahoo
                                   exposes a current option chain, not a
                                   history of implied vol, so the 52-week IV
                                   percentile cannot be reconstructed.

Both fields are simply omitted from the snapshot. The scoring layer
re-normalizes block weights over the metrics that are present
(:func:`screener.scoring.score_universe`), so the Valuation & Carry block runs
on its remaining members rather than scoring the absent ones as zero. Call
:func:`coverage_report` to see exactly which metrics went missing.

Price series convention
-----------------------
Two different price series are used deliberately, and they are not
interchangeable:

    * **Adjusted close** feeds ``history.close``. Every return-based metric
      (momentum, Sharpe, drawdown, beta, correlation) must be computed on a
      total-return series or a dividend payment reads as a price decline.
    * **Raw close/high/low** feeds the snapshot -- last price, 52-week band and
      traded value. The eligibility screen's ``min_price`` test and the
      liquidity block are about the price a share actually trades at, not a
      back-adjusted one.

Mixing the two would corrupt ``pct_from_52w_high`` and ``range_position``,
which compare a last price against a 52-week band: both sides of that
comparison come from the raw series here.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import BENCHMARK_TICKER, RISK_FREE_RATE
from .universe import ETF_UNIVERSE, index_tags

#: Trading days used for each lookback. Yahoo returns calendar-indexed rows
#: with holidays absent, so these are counts of observations, not days.
BARS_52W = 252
BARS_ADV = 90
BARS_HV = 30

#: Weekly bars retained.
#:
#: Deliberately more than the engine needs. ``mom_12_1`` is the single
#: highest-weighted metric in the model (0.35 of the momentum block) and is a
#: 48-week return skipping the most recent 4, so it needs 53 observations
#: exactly. Retaining exactly 53 left it with no slack at all: one missing
#: Friday -- a holiday, a halt, a late file, a name that listed mid-window --
#: and ``total_return`` returns None, dropping a third of the momentum block
#: with nothing in the output to say it happened.
#:
#: 60 gives seven weeks of margin. The extra bars do not stretch the "1Y" risk
#: statistics, because :data:`screener.metrics.RISK_WINDOW_BARS` windows those
#: to a year explicitly -- which is the other half of this change and the part
#: that would have gone wrong silently.
DEFAULT_WEEKS = 60

#: Minimum weekly observations before an instrument is worth emitting at all.
#: Below the engine's ``min_history_bars`` the name would be screened out
#: anyway, but emitting it lets the run report *why* it was dropped.
MIN_WEEKLY_BARS = 12


# --------------------------------------------------------------------------
# Frame plumbing
# --------------------------------------------------------------------------

def extract_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """
    Pull one price field out of a ``yfinance.download`` frame as
    ``DataFrame[date x ticker]``.

    ``yf.download`` returns MultiIndex columns under its default
    ``group_by='column'``: level 0 is the price field, level 1 the ticker. A
    single-ticker download keeps that shape when ``multi_level_index=True``
    (the default), but callers passing a bare string get flat columns, so both
    layouts are handled.
    """
    if isinstance(df.columns, pd.MultiIndex):
        if field not in df.columns.get_level_values(0):
            raise KeyError(
                f"Field {field!r} not in downloaded frame. Present: "
                f"{sorted(set(df.columns.get_level_values(0)))}. "
                "Pass auto_adjust=False and actions=True to yf.download."
            )
        out = df.xs(field, axis=1, level=0)
    else:
        if field not in df.columns:
            raise KeyError(f"Field {field!r} not in downloaded frame.")
        out = df[[field]]
    return out.astype("float64")


def _clean(series: pd.Series) -> pd.Series:
    """Drop NaNs and non-positive prices, which Yahoo emits for halted names."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s[s > 0]


def _f(value: Any) -> float | None:
    """Coerce to a finite float or None. NaN never reaches the payload."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# --------------------------------------------------------------------------
# Per-instrument construction
# --------------------------------------------------------------------------

def weekly_bars(adj_close: pd.Series, volume: pd.Series,
                weeks: int = DEFAULT_WEEKS) -> tuple[list[float], list[float]]:
    """
    Resample daily observations to Friday-anchored weekly bars.

    Close takes the last observation of the week; volume *sums* the week --
    matching what the engine assumes downstream, where the ADV fallback in
    :func:`screener.metrics.adv_usd` divides weekly volume by 5 to recover a
    daily rate.
    """
    adj = _clean(adj_close)
    if adj.empty:
        return [], []

    wk_close = adj.resample("W-FRI").last().dropna()

    vol = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    wk_vol = vol.resample("W-FRI").sum() if not vol.empty else pd.Series(dtype="float64")
    # Align volume onto the close index so the two lists stay row-for-row
    # aligned; a week with a close but no volume row becomes 0.0, not a shift.
    wk_vol = wk_vol.reindex(wk_close.index).fillna(0.0)

    wk_close = wk_close.tail(weeks)
    wk_vol = wk_vol.tail(weeks)
    return [float(x) for x in wk_close], [float(x) for x in wk_vol]


def build_snapshot(raw_close: pd.Series, high: pd.Series, low: pd.Series,
                   volume: pd.Series, dividends: pd.Series | None = None,
                   implied_vol: float | None = None) -> dict:
    """
    Assemble the snapshot node in the hyphenated shape IBKR returns.

    Keys are omitted rather than set to ``None`` when a quantity is
    unavailable: :func:`screener.metrics.snap_get` treats a missing node and a
    null value identically, but omission keeps the payload honest about what
    was actually observed.
    """
    px = _clean(raw_close)
    if px.empty:
        return {}
    last = float(px.iloc[-1])

    node: dict[str, dict] = {"last": {"price": last}}

    hi = _clean(high).tail(BARS_52W)
    lo = _clean(low).tail(BARS_52W)
    if not hi.empty and not lo.empty:
        hi52, lo52 = float(hi.max()), float(lo.min())
        if hi52 > lo52:
            node["misc-statistics"] = {"high_52w": hi52, "low_52w": lo52}

    # Traded value, not share count: a $500 stock and a $5 stock with equal
    # share volume are two orders of magnitude apart in capacity.
    vol = pd.to_numeric(volume, errors="coerce")
    traded = (px * vol).dropna().tail(BARS_ADV)
    adv = _f(traded.mean()) if not traded.empty else None
    node["avg-90d-usd-volume"] = {"volume": adv} if adv and adv > 0 else {}

    if dividends is not None:
        d = pd.to_numeric(dividends, errors="coerce").fillna(0.0).tail(BARS_52W)
        paid = float(d.sum())
        if paid > 0 and last > 0:
            # IBKR reports this as a percentage; metrics.dividend_yield divides
            # by 100. Emitting a decimal here would understate carry 100x.
            node["dividend-yield"] = {"yield_pct": paid / last * 100.0}

    if implied_vol is not None:
        iv = _f(implied_vol)
        if iv and iv > 0:
            node["implied-vol-underlying"] = {"annual_iv": iv}
            hv = realized_vol(raw_close)
            if hv is not None:
                node["historical-vol"] = {"annual_pct": hv}

    return node


def realized_vol(close: pd.Series, bars: int = BARS_HV) -> float | None:
    """Annualized realized volatility over the trailing ``bars`` sessions."""
    px = _clean(close)
    if px.size < bars // 2:
        return None
    rets = px.pct_change().dropna().tail(bars)
    if rets.size < 5:
        return None
    return _f(float(rets.std(ddof=1)) * math.sqrt(252.0))


def classify(ticker: str) -> str:
    """
    ETF vs single stock, from the curated universe rather than a name regex.

    This drives peer-relative z-scoring, so a misclassification does real
    damage: it would score an ETF's structurally lower volatility against a
    cohort of single stocks and hand it a top risk score mechanically.
    Membership lookup is exact; description parsing is not.
    """
    return "ETF" if ticker.upper() in ETF_UNIVERSE else "STOCK"


def build_instrument(ticker: str, fields: dict[str, pd.Series], *,
                     name: str | None = None, sector: str | None = None,
                     implied_vol: float | None = None,
                     weeks: int = DEFAULT_WEEKS) -> dict | None:
    """
    Build one instrument node. Returns ``None`` when the name has too little
    usable history to be worth scoring.

    ``fields`` maps ``Close`` / ``Adj Close`` / ``High`` / ``Low`` / ``Volume``
    / ``Dividends`` to that ticker's daily series.
    """
    tkr = ticker.upper()
    adj = fields.get("Adj Close")
    raw = fields.get("Close")
    if adj is None or raw is None:
        return None
    # auto_adjust=True collapses the two into one column; fall back so the
    # adapter still works if a caller downloads that way.
    if adj is None or _clean(adj).empty:
        adj = raw

    closes, volumes = weekly_bars(adj, fields.get("Volume", pd.Series(dtype="float64")), weeks)
    if len(closes) < MIN_WEEKLY_BARS:
        return None

    snapshot = build_snapshot(
        raw,
        fields.get("High", raw),
        fields.get("Low", raw),
        fields.get("Volume", pd.Series(dtype="float64")),
        fields.get("Dividends"),
        implied_vol,
    )
    if not snapshot:
        return None

    return {
        "ticker": tkr,
        "name": name or tkr,
        "asset_type": classify(tkr),
        # The curated universe is US-listed by construction, so these satisfy
        # the eligibility screen without a per-ticker metadata request. A
        # custom ticker list should pass real metadata -- see fetch_metadata.
        "exchange": None,
        "country_code": "US",
        "sector": sector,
        "indices": index_tags(tkr),
        "snapshot": snapshot,
        "history": {"close": closes, "volume": volumes},
    }


# --------------------------------------------------------------------------
# Payload assembly
# --------------------------------------------------------------------------

def build_market_data(df: pd.DataFrame, tickers: Sequence[str], *,
                      benchmark: str = BENCHMARK_TICKER,
                      risk_free_rate: float = RISK_FREE_RATE,
                      as_of: str | None = None,
                      weeks: int = DEFAULT_WEEKS,
                      names: dict[str, str] | None = None,
                      sectors: dict[str, str] | None = None,
                      iv: dict[str, float] | None = None) -> dict:
    """
    Convert a ``yfinance.download`` frame into the engine's market-data payload.

    Pure function of the frame -- no network. This is the seam the tests exercise.
    """
    wanted = ["Close", "Adj Close", "High", "Low", "Volume", "Dividends"]
    available: dict[str, pd.DataFrame] = {}
    for field in wanted:
        try:
            available[field] = extract_field(df, field)
        except KeyError:
            continue

    if "Close" not in available:
        raise ValueError(
            "Downloaded frame has no Close column -- the download returned "
            "nothing. Check connectivity and the ticker list."
        )
    if "Adj Close" not in available:
        # auto_adjust=True was used: Close is already total-return adjusted.
        available["Adj Close"] = available["Close"]

    names = {k.upper(): v for k, v in (names or {}).items()}
    sectors = {k.upper(): v for k, v in (sectors or {}).items()}
    iv = {k.upper(): v for k, v in (iv or {}).items()}

    instruments: list[dict] = []
    dropped: list[tuple[str, str]] = []
    for ticker in tickers:
        tkr = ticker.upper()
        fields: dict[str, pd.Series] = {}
        for field, frame in available.items():
            if tkr in frame.columns:
                fields[field] = frame[tkr]
        if "Close" not in fields:
            dropped.append((tkr, "no data returned by Yahoo"))
            continue

        node = build_instrument(
            tkr, fields,
            name=names.get(tkr), sector=sectors.get(tkr),
            implied_vol=iv.get(tkr), weeks=weeks,
        )
        if node is None:
            dropped.append((tkr, f"fewer than {MIN_WEEKLY_BARS} usable weekly bars"))
            continue
        instruments.append(node)

    have = {i["ticker"] for i in instruments}
    if benchmark.upper() not in have:
        raise ValueError(
            f"Benchmark {benchmark} produced no usable history, so no "
            "market-relative metric (beta, alpha, capture) can be computed. "
            "Add it to the ticker list or change the benchmark."
        )

    return {
        "as_of": as_of or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "benchmark": benchmark.upper(),
        "risk_free_rate": float(risk_free_rate),
        "data_source": "Yahoo Finance (yfinance)",
        "history_desc": f"weekly bars (Fri), {weeks} max, adjusted close",
        "instruments": instruments,
        "dropped": dropped,
    }


def coverage_report(market_data: dict) -> pd.DataFrame:
    """
    Per-metric availability across the payload.

    Worth reading before trusting a run: a metric present on 10% of the
    universe is being z-scored against a nine-name cross-section while the rest
    of the book is scored without it.
    """
    from .config import FACTOR_MODEL

    instruments = market_data.get("instruments", [])
    n = len(instruments) or 1

    def _has(inst: dict, path: tuple[str, str]) -> bool:
        node = inst.get("snapshot", {}).get(path[0]) or {}
        return node.get(path[1]) is not None

    snapshot_backed = {
        "dividend_yield": ("dividend-yield", "yield_pct"),
        "iv_hv_spread": ("implied-vol-underlying", "annual_iv"),
        "iv_percentile": ("implied-volatility-percentile", "high_52w"),
        "range_position": ("misc-statistics", "high_52w"),
        "pct_from_52w_high": ("misc-statistics", "high_52w"),
        "adv_usd_log": ("avg-90d-usd-volume", "volume"),
    }

    rows = []
    for block in FACTOR_MODEL:
        for metric in block.metrics:
            path = snapshot_backed.get(metric.key)
            if path is None:
                available = n  # derived from price history, always present
                note = "computed from price history"
            else:
                available = sum(1 for i in instruments if _has(i, path))
                note = "from snapshot" if available else "UNAVAILABLE from Yahoo"
            rows.append({
                "block": block.label,
                "metric": metric.label,
                "coverage": available / n,
                "source": note,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Network layer
# --------------------------------------------------------------------------

def fetch_daily(tickers: Sequence[str], period: str = "2y",
                chunk_size: int = 100, progress: bool = False) -> pd.DataFrame:
    """
    Download daily OHLCV + dividends, in chunks.

    Chunked because a single request for several hundred symbols is the
    reliable way to get throttled; ``yfinance`` threads within each chunk.
    ``auto_adjust=False`` is required -- it is what splits ``Close`` (raw) from
    ``Adj Close`` (total return), and this adapter needs both.
    """
    import yfinance as yf

    syms = [t.upper() for t in dict.fromkeys(tickers)]
    frames: list[pd.DataFrame] = []
    for start in range(0, len(syms), chunk_size):
        chunk = syms[start:start + chunk_size]
        frame = yf.download(
            chunk, period=period, interval="1d",
            auto_adjust=False, actions=True, progress=progress,
            group_by="column", threads=True,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError(
            "Yahoo returned no data for any ticker. Usually connectivity or a "
            "rate limit -- wait a minute and retry."
        )
    return frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)


def fetch_metadata(tickers: Sequence[str],
                   max_workers: int = 8) -> tuple[dict[str, str], dict[str, str]]:
    """
    Fetch long names and sectors. Returns ``(names, sectors)``.

    One request per ticker, so this is the slow path -- only worth it for a
    custom ticker list, where ``name`` feeds
    :func:`screener.universe.is_excluded_product` and is the only thing
    standing between a leveraged or option-income wrapper and the scorer.
    The curated universe already excludes those by construction.
    """
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    def _one(ticker: str) -> tuple[str, str | None, str | None]:
        try:
            info = yf.Ticker(ticker).info or {}
            return ticker, info.get("longName") or info.get("shortName"), info.get("sector")
        except Exception:
            return ticker, None, None

    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, name, sector in pool.map(_one, [t.upper() for t in tickers]):
            if name:
                names[ticker] = name
            if sector:
                sectors[ticker] = sector
    return names, sectors


def fetch_atm_iv(tickers: Sequence[str], min_days: int = 20,
                 max_workers: int = 8) -> dict[str, float]:
    """
    At-the-money implied vol from the nearest expiry at least ``min_days`` out.

    Two requests per ticker and frequently empty for less liquid names, which
    is why it is opt-in. Note what this does and does not buy: it populates
    ``iv_hv_spread``, but ``iv_percentile`` needs a *history* of implied vol
    that Yahoo does not publish, so that metric stays unavailable either way.
    """
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    def _one(ticker: str) -> tuple[str, float | None]:
        try:
            handle = yf.Ticker(ticker)
            expiries = handle.options or []
            today = pd.Timestamp.now().normalize()
            target = next(
                (e for e in expiries
                 if (pd.Timestamp(e) - today).days >= min_days),
                expiries[-1] if expiries else None,
            )
            if target is None:
                return ticker, None

            chain = handle.option_chain(target)
            spot = float(handle.fast_info["last_price"])
            calls, puts = chain.calls, chain.puts
            if calls.empty and puts.empty:
                return ticker, None

            legs = []
            for frame in (calls, puts):
                if frame.empty:
                    continue
                near = frame.iloc[(frame["strike"] - spot).abs().argsort()[:1]]
                value = _f(near["impliedVolatility"].iloc[0])
                if value and 0.0 < value < 5.0:
                    legs.append(value)
            return ticker, (sum(legs) / len(legs) if legs else None)
        except Exception:
            return ticker, None

    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, value in pool.map(_one, [t.upper() for t in tickers]):
            if value is not None:
                out[ticker] = value
    return out


def daily_returns(df: pd.DataFrame,
                  tickers: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Daily total-return series as ``DataFrame[date x ticker]``.

    Uses adjusted close for the same reason ``history.close`` does: a dividend
    must not read as a price decline. The optimizer needs daily rather than
    weekly observations -- a covariance estimated from ~52 weekly bars over more
    than 52 names is singular, and shrinkage alone cannot rescue that few
    degrees of freedom.
    """
    try:
        prices = extract_field(df, "Adj Close")
    except KeyError:
        prices = extract_field(df, "Close")
    if tickers is not None:
        keep = [t for t in tickers if t in prices.columns]
        prices = prices[keep]
    return prices.pct_change().dropna(how="all")


def fetch_market_caps(tickers: Sequence[str],
                      max_workers: int = 8) -> dict[str, float]:
    """
    Market capitalization for stocks, total net assets for ETFs.

    Missing values are simply absent from the result rather than defaulted.
    :func:`screener.optimizer.market_weights` reports what is missing, because a
    silent constant here would mis-anchor the market equilibrium that the whole
    Black-Litterman model starts from.
    """
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    def _one(ticker: str) -> tuple[str, float | None]:
        try:
            info = yf.Ticker(ticker).info or {}
            value = info.get("marketCap") or info.get("totalAssets")
            return ticker, _f(value)
        except Exception:
            return ticker, None

    out: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, value in pool.map(_one, [t.upper() for t in tickers]):
            if value and value > 0:
                out[ticker] = value
    return out


def default_universe(include: Iterable[str] = ("SP500", "NDX", "DJIA", "ETF"),
                     benchmark: str = BENCHMARK_TICKER) -> list[str]:
    """The repo's curated lists, unioned and sorted, benchmark guaranteed in."""
    from .universe import all_index_members

    members = all_index_members()
    out: set[str] = set()
    for key in include:
        out |= set(members.get(key.upper(), frozenset()))
    out.add(benchmark.upper())
    # Yahoo uses a dash for share classes where IBKR uses a dot.
    return sorted(t.replace(".", "-") for t in out)


def fetch_market_data(tickers: Sequence[str] | None = None, *,
                      benchmark: str = BENCHMARK_TICKER,
                      risk_free_rate: float = RISK_FREE_RATE,
                      period: str = "2y",
                      weeks: int = DEFAULT_WEEKS,
                      with_metadata: bool = False,
                      with_iv: bool = False,
                      progress: bool = False,
                      with_frame: bool = False):
    """
    End-to-end: download, then convert. The one call the notebook makes.

    ``with_frame=True`` also returns the raw daily frame, which the optimizer
    needs for a covariance estimated on daily rather than weekly observations.
    """
    syms = list(tickers) if tickers else default_universe(benchmark=benchmark)
    if benchmark.upper() not in {t.upper() for t in syms}:
        syms.append(benchmark.upper())

    frame = fetch_daily(syms, period=period, progress=progress)
    names, sectors = fetch_metadata(syms) if with_metadata else ({}, {})
    iv = fetch_atm_iv(syms) if with_iv else {}

    payload = build_market_data(
        frame, syms,
        benchmark=benchmark, risk_free_rate=risk_free_rate,
        weeks=weeks, names=names, sectors=sectors, iv=iv,
    )
    return (payload, frame) if with_frame else payload
