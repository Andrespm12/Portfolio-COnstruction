"""
Tests for the Yahoo adapter.

The network call is one line (`yf.download`); everything that can actually be
wrong is in the conversion from a yfinance frame to the engine's payload. So
these tests build frames in yfinance's exact output shape and assert on the
payload -- no network, deterministic, and they run in CI.

The properties under test are the ones where a silent error would produce
plausible-looking but wrong rankings: the raw/adjusted price split, the
percent-vs-decimal dividend convention, weekly volume being a sum rather than a
mean, and unavailable metrics being *omitted* rather than zeroed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.metrics import compute_metrics, simple_returns  # noqa: E402
from screener.run_screen import run  # noqa: E402
from screener.yahoo_adapter import (  # noqa: E402
    build_instrument, build_market_data, build_snapshot, classify,
    coverage_report, default_universe, extract_field, weekly_bars,
)

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Fixtures: frames in yfinance's exact output shape
# --------------------------------------------------------------------------

def make_series(n: int, start: float, drift: float, seed: int) -> np.ndarray:
    """Deterministic geometric random walk."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, 0.012, size=n)
    return start * np.exp(np.cumsum(shocks))


def make_yf_frame(tickers: list[str], n_days: int = 520,
                  dividends: dict[str, float] | None = None,
                  seed_base: int = 7) -> pd.DataFrame:
    """
    Build a frame matching ``yf.download(..., auto_adjust=False, actions=True,
    group_by='column')``: MultiIndex columns with level 0 = price field and
    level 1 = ticker, over business-day rows.
    """
    idx = pd.bdate_range(end="2026-08-11", periods=n_days, name="Date")
    dividends = dividends or {}
    blocks: dict[tuple[str, str], np.ndarray] = {}

    for i, ticker in enumerate(tickers):
        raw = make_series(n_days, 100.0 + 10 * i, 0.0004, seed_base + i)
        # Adjusted close sits below raw by a constant factor -- the shape a
        # dividend-paying name actually has, and enough to catch code that
        # mixes the two series.
        adj = raw * 0.97
        vol = np.full(n_days, 5_000_000.0 + 100_000 * i)

        blocks[("Close", ticker)] = raw
        blocks[("Adj Close", ticker)] = adj
        blocks[("High", ticker)] = raw * 1.01
        blocks[("Low", ticker)] = raw * 0.99
        blocks[("Volume", ticker)] = vol

        div = np.zeros(n_days)
        if ticker in dividends:
            # Four quarterly payments inside the trailing 252 sessions.
            for k in range(4):
                div[n_days - 1 - k * 60] = dividends[ticker] / 4.0
        blocks[("Dividends", ticker)] = div

    frame = pd.DataFrame(blocks, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    return frame.sort_index(axis=1)


# --------------------------------------------------------------------------
# Frame plumbing
# --------------------------------------------------------------------------

def test_extract_field() -> None:
    df = make_yf_frame(["SPY", "AAPL"])
    close = extract_field(df, "Close")
    check("extract_field returns date x ticker",
          list(close.columns) == ["AAPL", "SPY"], f"got {list(close.columns)}")
    check("extract_field keeps row count", len(close) == 520, f"got {len(close)}")

    adj = extract_field(df, "Adj Close")
    check("Adj Close is distinct from Close",
          not np.allclose(adj["SPY"].to_numpy(), close["SPY"].to_numpy()))

    try:
        extract_field(df, "Nonexistent")
        check("missing field raises", False, "no exception")
    except KeyError as exc:
        check("missing field raises a KeyError naming what is present",
              "Close" in str(exc))


def test_weekly_resample() -> None:
    idx = pd.bdate_range("2026-01-05", periods=10, name="Date")  # 2 full weeks
    close = pd.Series(np.arange(1.0, 11.0), index=idx)
    volume = pd.Series(np.full(10, 100.0), index=idx)

    closes, volumes = weekly_bars(close, volume)
    check("weekly close takes the last observation of the week",
          closes == [5.0, 10.0], f"got {closes}")
    check("weekly volume SUMS the week (not mean)",
          volumes == [500.0, 500.0], f"got {volumes}")

    long_close = pd.Series(
        np.arange(1.0, 521.0), index=pd.bdate_range(end="2026-08-11", periods=520))
    trimmed, _ = weekly_bars(long_close, pd.Series(dtype="float64"), weeks=53)
    check("weekly history is trimmed to the requested window",
          len(trimmed) == 53, f"got {len(trimmed)}")


# --------------------------------------------------------------------------
# Snapshot conventions
# --------------------------------------------------------------------------

def test_snapshot_uses_raw_prices() -> None:
    df = make_yf_frame(["SPY"])
    raw = extract_field(df, "Close")["SPY"]
    snap = build_snapshot(
        raw, extract_field(df, "High")["SPY"], extract_field(df, "Low")["SPY"],
        extract_field(df, "Volume")["SPY"],
    )
    check("last price is the raw close, not the adjusted one",
          abs(snap["last"]["price"] - float(raw.iloc[-1])) < 1e-9)

    hi = snap["misc-statistics"]["high_52w"]
    lo = snap["misc-statistics"]["low_52w"]
    window = raw.tail(252)
    check("52w high comes from the intraday High series",
          abs(hi - float(window.max()) * 1.01) < 1e-6, f"{hi}")
    check("52w band brackets the last price",
          lo <= snap["last"]["price"] <= hi)

    expected_adv = float((raw * extract_field(df, "Volume")["SPY"]).tail(90).mean())
    check("ADV is traded VALUE (price x shares), not share count",
          abs(snap["avg-90d-usd-volume"]["volume"] - expected_adv) < 1e-3)


def test_dividend_yield_is_percent() -> None:
    df = make_yf_frame(["SPY"], dividends={"SPY": 6.0})
    raw = extract_field(df, "Close")["SPY"]
    snap = build_snapshot(
        raw, extract_field(df, "High")["SPY"], extract_field(df, "Low")["SPY"],
        extract_field(df, "Volume")["SPY"], extract_field(df, "Dividends")["SPY"],
    )
    last = float(raw.iloc[-1])
    expected_pct = 6.0 / last * 100.0
    got = snap["dividend-yield"]["yield_pct"]
    check("dividend yield is emitted in PERCENT (IBKR convention)",
          abs(got - expected_pct) < 1e-6, f"got {got}, expected {expected_pct}")

    # The convention only matters because the metrics layer divides by 100.
    # Verify the round trip rather than the intermediate.
    inst = {"snapshot": snap, "history": {"close": [1.0], "volume": [1.0]}}
    from screener.metrics import dividend_yield
    check("round trip through metrics gives a decimal yield",
          abs(dividend_yield(inst) - 6.0 / last) < 1e-9)


def test_no_dividends_omits_the_key() -> None:
    df = make_yf_frame(["NVDA"])
    snap = build_snapshot(
        extract_field(df, "Close")["NVDA"], extract_field(df, "High")["NVDA"],
        extract_field(df, "Low")["NVDA"], extract_field(df, "Volume")["NVDA"],
        extract_field(df, "Dividends")["NVDA"],
    )
    check("a non-payer omits dividend-yield rather than reporting 0%",
          "dividend-yield" not in snap)


def test_iv_omitted_by_default() -> None:
    df = make_yf_frame(["SPY"])
    fields = {f: extract_field(df, f)["SPY"]
              for f in ("Close", "Adj Close", "High", "Low", "Volume", "Dividends")}

    without = build_instrument("SPY", fields)
    check("implied vol keys are absent when IV was not fetched",
          "implied-vol-underlying" not in without["snapshot"])
    check("IV percentile is never emitted (Yahoo has no IV history)",
          "implied-volatility-percentile" not in without["snapshot"])

    with_iv = build_instrument("SPY", fields, implied_vol=0.22)
    check("IV is emitted when supplied",
          with_iv["snapshot"]["implied-vol-underlying"]["annual_iv"] == 0.22)
    check("realized vol accompanies IV so the spread is computable",
          with_iv["snapshot"]["historical-vol"]["annual_pct"] > 0)

    from screener.metrics import iv_hv_spread
    check("iv_hv_spread is None without IV, not 0.0",
          iv_hv_spread(without) is None)
    check("iv_hv_spread computes when IV is present",
          iv_hv_spread(with_iv) is not None)


# --------------------------------------------------------------------------
# Returns are computed on the adjusted series
# --------------------------------------------------------------------------

def test_history_uses_adjusted_close() -> None:
    df = make_yf_frame(["SPY"])
    fields = {f: extract_field(df, f)["SPY"]
              for f in ("Close", "Adj Close", "High", "Low", "Volume", "Dividends")}
    inst = build_instrument("SPY", fields)

    adj_weekly, _ = weekly_bars(fields["Adj Close"], fields["Volume"])
    check("history.close is the ADJUSTED series (total return)",
          np.allclose(inst["history"]["close"], adj_weekly))

    raw_weekly, _ = weekly_bars(fields["Close"], fields["Volume"])
    check("history.close is not the raw series",
          not np.allclose(inst["history"]["close"], raw_weekly))

    # The two differ by a constant factor here, so returns must be identical:
    # that is exactly the property that makes the adjusted series the right
    # input for momentum and risk.
    check("returns are invariant to the adjustment factor",
          np.allclose(simple_returns(np.array(inst["history"]["close"])),
                      simple_returns(np.array(raw_weekly))))


def test_asset_type_from_universe() -> None:
    check("ETF classified from the curated list", classify("SPY") == "ETF")
    check("single stock classified from the curated list", classify("AAPL") == "STOCK")
    check("classification is case-insensitive", classify("qqq") == "ETF")


# --------------------------------------------------------------------------
# Payload assembly and failure modes
# --------------------------------------------------------------------------

def test_build_market_data() -> None:
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GLD"]
    df = make_yf_frame(tickers, dividends={"SPY": 6.0, "AAPL": 1.0})
    payload = build_market_data(df, tickers, benchmark="SPY")

    check("every ticker becomes an instrument",
          len(payload["instruments"]) == 6, f"got {len(payload['instruments'])}")
    check("payload carries the keys the engine reads",
          {"as_of", "benchmark", "risk_free_rate", "instruments"} <= set(payload))
    check("data source is labelled Yahoo, not IBKR",
          "Yahoo" in payload["data_source"])

    spy = next(i for i in payload["instruments"] if i["ticker"] == "SPY")
    check("index tags are attached", "ETF" in spy["indices"])
    check("country is set so the US-listing filter passes",
          spy["country_code"] == "US")


def test_missing_ticker_is_dropped_with_a_reason() -> None:
    tickers = ["SPY", "QQQ", "AAPL", "DELISTED"]
    df = make_yf_frame(["SPY", "QQQ", "AAPL"])
    payload = build_market_data(df, tickers, benchmark="SPY")

    check("a ticker Yahoo returned nothing for is dropped",
          len(payload["instruments"]) == 3)
    dropped = dict(payload["dropped"])
    check("the drop is reported with a reason, not silent",
          "DELISTED" in dropped and "no data" in dropped["DELISTED"],
          f"got {payload['dropped']}")


def test_short_history_is_dropped() -> None:
    df = make_yf_frame(["SPY", "QQQ", "AAPL", "MSFT"], n_days=520)
    # Blank out one name's recent history so only a stub remains.
    df.loc[df.index[:-20], ("Close", "MSFT")] = np.nan
    df.loc[df.index[:-20], ("Adj Close", "MSFT")] = np.nan

    payload = build_market_data(df, ["SPY", "QQQ", "AAPL", "MSFT"], benchmark="SPY")
    dropped = dict(payload["dropped"])
    check("a name with too little history is dropped, not scored on noise",
          "MSFT" in dropped, f"dropped={payload['dropped']}")


def test_missing_benchmark_raises() -> None:
    df = make_yf_frame(["AAPL", "MSFT", "NVDA"])
    try:
        build_market_data(df, ["AAPL", "MSFT", "NVDA"], benchmark="SPY")
        check("a missing benchmark raises rather than scoring without beta", False)
    except ValueError as exc:
        check("a missing benchmark raises rather than scoring without beta",
              "benchmark" in str(exc).lower())


def test_default_universe() -> None:
    universe = default_universe()
    check("default universe is large", len(universe) > 400, f"got {len(universe)}")
    check("benchmark is guaranteed present", "SPY" in universe)
    check("share classes use Yahoo's dash convention",
          "BRK-B" in universe and "BRK.B" not in universe)


# --------------------------------------------------------------------------
# End to end through the real scoring engine
# --------------------------------------------------------------------------

def test_end_to_end_through_engine() -> None:
    tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "JPM", "LLY"]
    df = make_yf_frame(tickers, dividends={"SPY": 6.0, "JPM": 4.0})
    payload = build_market_data(df, tickers, benchmark="SPY")

    portfolio = {
        "net_liquidation": 10_000_000.0,
        "positions": [
            {"ticker": "SPY", "market_value": 2_000_000.0, "quantity": 2600, "asset_class": "STK"},
            {"ticker": "AAPL", "market_value": 500_000.0, "quantity": 1600, "asset_class": "STK"},
        ],
    }

    scored, meta = run(payload, portfolio)
    check("the engine scores the Yahoo payload end to end",
          len(scored) >= 8, f"scored {len(scored)}")
    check("every scored name has a recommendation",
          all(r.recommendation for r in scored))
    check("scores are finite",
          all(np.isfinite(r.score_0_100) for r in scored))
    check("scores are ranked descending",
          all(scored[i].score_0_100 >= scored[i + 1].score_0_100
              for i in range(len(scored) - 1)))
    check("ETFs and stocks are both present in the cross-section",
          {r.asset_type for r in scored} == {"ETF", "STOCK"})


def test_absent_metrics_renormalize_rather_than_zero() -> None:
    """
    The load-bearing claim of this adapter: metrics Yahoo cannot supply are
    omitted, and the engine re-weights the block over what remains. If they
    were scored as zero instead, every name would carry an identical artificial
    drag and the Valuation block would be noise.
    """
    tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "JPM", "LLY"]
    df = make_yf_frame(tickers, dividends={"SPY": 6.0, "JPM": 4.0})
    payload = build_market_data(df, tickers, benchmark="SPY")

    portfolio = {"net_liquidation": 10_000_000.0, "positions": []}
    scored, _ = run(payload, portfolio)

    spy = next(r for r in scored if r.ticker == "SPY")
    check("iv metrics are absent from raw_metrics, not present as 0.0",
          spy.raw_metrics.get("iv_hv_spread") is None
          and spy.raw_metrics.get("iv_percentile") is None)
    check("the Valuation & Carry block still scores on its surviving metrics",
          np.isfinite(spy.block_scores.get("valuation_carry", float("nan"))),
          f"got {spy.block_scores.get('valuation_carry')}")
    check("block coverage reports the shortfall honestly",
          0.0 < spy.block_coverage.get("valuation_carry", 0.0) < 1.0,
          f"coverage={spy.block_coverage.get('valuation_carry')}")
    check("blocks computed purely from price history stay fully covered",
          abs(spy.block_coverage.get("momentum", 0.0) - 1.0) < 1e-9)


def test_coverage_report() -> None:
    tickers = ["SPY", "QQQ", "AAPL", "MSFT"]
    df = make_yf_frame(tickers, dividends={"SPY": 6.0})
    payload = build_market_data(df, tickers, benchmark="SPY")
    report = coverage_report(payload)

    from screener.config import all_metrics
    check("coverage report covers every metric in the model",
          len(report) == len(all_metrics()),
          f"got {len(report)}, model has {len(all_metrics())}")
    iv_pct = report[report["metric"].str.contains("IV percentile")]
    check("IV percentile is flagged unavailable",
          float(iv_pct["coverage"].iloc[0]) == 0.0
          and "UNAVAILABLE" in iv_pct["source"].iloc[0])
    momentum = report[report["block"].str.contains("Momentum")]
    check("price-derived metrics report full coverage",
          bool((momentum["coverage"] == 1.0).all()))


def main() -> int:
    for fn in [
        test_extract_field,
        test_weekly_resample,
        test_snapshot_uses_raw_prices,
        test_dividend_yield_is_percent,
        test_no_dividends_omits_the_key,
        test_iv_omitted_by_default,
        test_history_uses_adjusted_close,
        test_asset_type_from_universe,
        test_build_market_data,
        test_missing_ticker_is_dropped_with_a_reason,
        test_short_history_is_dropped,
        test_missing_benchmark_raises,
        test_default_universe,
        test_end_to_end_through_engine,
        test_absent_metrics_renormalize_rather_than_zero,
        test_coverage_report,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
