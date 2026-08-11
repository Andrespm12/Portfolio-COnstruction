"""
Validation of the scoring engine on synthetic data with known properties.

These tests exist to catch sign errors and weighting bugs, which are the
failure mode that matters here: a screener with an inverted direction on a
risk metric will confidently recommend exactly the wrong names, and nothing
about the output will look obviously broken.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.config import MARKET_WEIGHT, OVERWEIGHT, UNDERWEIGHT
from screener.metrics import (
    beta_alpha, calmar_ratio, capture_ratios, compute_metrics, diagnostics,
    max_drawdown, sharpe_ratio, simple_returns, ulcer_index,
)
from screener.scoring import ScoredInstrument, run_scoring_pipeline, winsorize, zscore
from screener.universe import is_excluded_product


def make_series(n=60, drift=0.004, vol=0.02, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, vol, n)
    return start * np.cumprod(1.0 + shocks)


def make_instrument(ticker, prices, asset_type="STOCK", adv=5e8, div=1.0,
                    iv=0.25, hv=0.24):
    return {
        "ticker": ticker,
        "conid": abs(hash(ticker)) % 10**6,
        "name": f"{ticker} TEST",
        "asset_type": asset_type,
        "exchange": "NASDAQ",
        "country_code": "US",
        "indices": ["SP500"],
        "sector": "Test",
        "snapshot": {
            "last": {"price": float(prices[-1])},
            "misc-statistics": {"high_52w": float(np.max(prices)), "low_52w": float(np.min(prices))},
            "avg-90d-usd-volume": {"volume": adv},
            "dividend-yield": {"yield_pct": div},
            "implied-vol-underlying": {"annual_iv": iv},
            "historical-vol": {"annual_pct": hv},
            "implied-volatility-percentile": {"high_52w": 0.5},
        },
        "history": {
            "close": [float(p) for p in prices],
            "volume": [1e7] * len(prices),
        },
    }


# --------------------------------------------------------------------------
# Statistical primitives
# --------------------------------------------------------------------------

def test_max_drawdown_sign_and_magnitude():
    prices = np.array([100.0, 120.0, 60.0, 90.0])
    dd = max_drawdown(prices)
    assert dd is not None and math.isclose(dd, -0.5, rel_tol=1e-9), dd
    print("PASS max_drawdown returns -50% for a 120->60 decline")


def test_ulcer_penalizes_prolonged_underwater():
    quick = np.array([100.0, 80.0] + [100.0] * 20)          # one sharp dip, fast recovery
    prolonged = np.array([100.0] + [80.0] * 20 + [100.0])   # same depth, long underwater
    assert ulcer_index(prolonged) > ulcer_index(quick)
    print("PASS ulcer index penalizes time underwater, not just depth")


def test_beta_of_identical_series_is_one():
    bench = simple_returns(make_series(seed=1))
    beta, alpha, r2 = beta_alpha(bench, bench)
    assert math.isclose(beta, 1.0, abs_tol=1e-9), beta
    assert math.isclose(r2, 1.0, abs_tol=1e-9), r2
    print(f"PASS beta of a series against itself = {beta:.6f}, R2 = {r2:.6f}")


def test_beta_scales_with_leverage():
    bench_prices = make_series(seed=2)
    bench_rets = simple_returns(bench_prices)
    levered = bench_rets * 2.0
    beta, _, _ = beta_alpha(levered, bench_rets)
    assert math.isclose(beta, 2.0, rel_tol=1e-6), beta
    print(f"PASS 2x-levered return stream produces beta = {beta:.4f}")


def test_sharpe_higher_for_smoother_path():
    rng = np.random.default_rng(7)
    smooth = rng.normal(0.004, 0.01, 60)
    choppy = rng.normal(0.004, 0.04, 60)
    assert sharpe_ratio(smooth) > sharpe_ratio(choppy)
    print("PASS Sharpe rewards the lower-volatility path at equal drift")


def test_winsorize_clips_outlier():
    values = np.array([1.0, 2.0, 3.0, 4.0, 1000.0])
    assert winsorize(values).max() < 1000.0
    print("PASS winsorization clips the extreme value")


def test_zscore_is_standardized():
    z = zscore(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert math.isclose(float(np.mean(z)), 0.0, abs_tol=1e-9)
    print("PASS z-scores are mean-zero")


# --------------------------------------------------------------------------
# Direction correctness -- the tests that actually matter
# --------------------------------------------------------------------------

def test_strong_name_outranks_weak_name():
    """
    A high-return / low-volatility name must outscore a low-return /
    high-volatility one. If a direction sign is inverted anywhere in the
    risk or momentum blocks, this fails.
    """
    bench = make_series(n=60, drift=0.003, vol=0.015, seed=11)
    strong = make_series(n=60, drift=0.010, vol=0.012, seed=12)
    weak = make_series(n=60, drift=-0.006, vol=0.045, seed=13)
    middling = make_series(n=60, drift=0.002, vol=0.025, seed=14)
    middling2 = make_series(n=60, drift=0.003, vol=0.022, seed=15)

    instruments = [
        make_instrument("BENCH", bench, asset_type="ETF"),
        make_instrument("STRONG", strong),
        make_instrument("WEAK", weak),
        make_instrument("MID1", middling),
        make_instrument("MID2", middling2),
    ]
    bench_rets = simple_returns(bench)

    rows = []
    for inst in instruments:
        m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                            base_position_weight=0.03)
        m.update({"corr_to_portfolio": 0.5, "diversification_benefit": 0.0,
                  "existing_overlap": 0.0})
        rows.append(ScoredInstrument(
            ticker=inst["ticker"], name=inst["name"], asset_type=inst["asset_type"],
            indices=["SP500"], sector="Test", raw_metrics=m,
            diagnostics=diagnostics(inst, bench_rets),
        ))

    scored = run_scoring_pipeline(rows)
    by_ticker = {r.ticker: r for r in scored}

    assert by_ticker["STRONG"].composite_z > by_ticker["WEAK"].composite_z, (
        f"STRONG {by_ticker['STRONG'].composite_z:.3f} <= WEAK {by_ticker['WEAK'].composite_z:.3f}"
    )
    assert scored[0].ticker == "STRONG", f"top ranked was {scored[0].ticker}"
    print(f"PASS STRONG z={by_ticker['STRONG'].composite_z:+.2f} "
          f"ranks above WEAK z={by_ticker['WEAK'].composite_z:+.2f}")


def test_risk_block_prefers_lower_volatility():
    """Isolate the risk block: same drift, different vol."""
    bench = make_series(n=60, drift=0.003, vol=0.015, seed=21)
    bench_rets = simple_returns(bench)
    calm = make_series(n=60, drift=0.004, vol=0.010, seed=22)
    wild = make_series(n=60, drift=0.004, vol=0.050, seed=23)
    mid = make_series(n=60, drift=0.004, vol=0.025, seed=24)

    rows = []
    for tkr, series in (("CALM", calm), ("WILD", wild), ("MID", mid), ("BENCH", bench)):
        inst = make_instrument(tkr, series)
        m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                            base_position_weight=0.03)
        m.update({"corr_to_portfolio": 0.5, "diversification_benefit": 0.0,
                  "existing_overlap": 0.0})
        rows.append(ScoredInstrument(tkr, tkr, "STOCK", ["SP500"], "Test", m,
                                     diagnostics=diagnostics(inst, bench_rets)))

    run_scoring_pipeline(rows)
    by = {r.ticker: r for r in rows}
    assert by["CALM"].block_scores["risk"] > by["WILD"].block_scores["risk"]
    print(f"PASS risk block: CALM {by['CALM'].block_scores['risk']:+.2f} > "
          f"WILD {by['WILD'].block_scores['risk']:+.2f}")


def test_gates_only_downgrade():
    """A gate must never raise a recommendation."""
    from screener.scoring import _RANK, apply_risk_gates

    bench = make_series(n=60, seed=31)
    bench_rets = simple_returns(bench)
    inst = make_instrument("KNIFE", make_series(n=60, drift=-0.008, vol=0.03, seed=32))
    m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                        base_position_weight=0.03)
    m.update({"corr_to_portfolio": 0.9, "diversification_benefit": 0.0,
              "existing_overlap": 0.10})

    row = ScoredInstrument("KNIFE", "KNIFE", "STOCK", ["SP500"], "Test", m,
                           diagnostics=diagnostics(inst, bench_rets))
    row.composite_z = 2.0
    row.recommendation = OVERWEIGHT
    row.pre_gate_recommendation = OVERWEIGHT
    row.block_coverage = {k: 1.0 for k in
                          ("momentum", "risk_adjusted", "risk", "market_sensitivity",
                           "liquidity", "valuation_carry", "portfolio_fit")}

    apply_risk_gates([row])
    assert _RANK[row.recommendation] <= _RANK[OVERWEIGHT]
    assert row.recommendation != OVERWEIGHT, "trend/concentration gates should have capped this"
    print(f"PASS gates downgraded a falling knife from OVERWEIGHT to "
          f"{row.recommendation}: {row.gates_triggered}")


def test_missing_data_renormalizes_not_zeros():
    """
    A name missing an entire block must not be scored as if that block were
    average -- its remaining blocks should carry the composite.
    """
    bench = make_series(n=60, seed=41)
    bench_rets = simple_returns(bench)
    rows = []
    for i, tkr in enumerate(["A", "B", "C", "D"]):
        inst = make_instrument(tkr, make_series(n=60, drift=0.002 * (i + 1), seed=50 + i))
        m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                            base_position_weight=0.03)
        m.update({"corr_to_portfolio": 0.5, "diversification_benefit": 0.0,
                  "existing_overlap": 0.0})
        if tkr == "A":  # strip the entire carry block
            for k in ("dividend_yield", "iv_hv_spread", "iv_percentile", "range_position"):
                m[k] = None
        rows.append(ScoredInstrument(tkr, tkr, "STOCK", ["SP500"], "Test", m,
                                     diagnostics=diagnostics(inst, bench_rets)))

    run_scoring_pipeline(rows)
    a = next(r for r in rows if r.ticker == "A")
    assert "valuation_carry" not in a.block_scores
    assert math.isfinite(a.composite_raw)
    print(f"PASS missing block excluded from composite; A still scored "
          f"{a.composite_raw:+.3f} from {len(a.block_scores)} blocks")


# --------------------------------------------------------------------------
# Universe filtering
# --------------------------------------------------------------------------

def test_excludes_leveraged_and_income_products():
    should_exclude = [
        "GRANITESH 2X LNG NVDA ETF", "DIREXION DAILY NVDA BEAR 1X",
        "YIELDMAX NVDA OPTION INC ETF", "PROSHARES ULTRA NVDA",
        "TRADR 1.5X SHT NVDA ETF-USDI", "ROUNDHILL NVDA WEEKLYPAY ETF",
        "GRANITE YIELDBOOST NVDA", "REX NVDA GROWTH & INCOME ETF",
        "LEVERAGE SHARES 2X NVDA",
    ]
    should_keep = [
        "NVIDIA CORP", "APPLE INC", "SPDR S&P 500 ETF TRUST",
        "INVESCO QQQ TRUST", "ISHARES RUSSELL 2000 ETF",
        "VANGUARD TOTAL STOCK MARKET ETF", "SPDR GOLD SHARES",
    ]
    for desc in should_exclude:
        excluded, pattern = is_excluded_product(desc)
        assert excluded, f"failed to exclude: {desc}"
    for desc in should_keep:
        excluded, pattern = is_excluded_product(desc)
        assert not excluded, f"wrongly excluded {desc} via /{pattern}/"
    print(f"PASS excluded {len(should_exclude)} structured products, "
          f"kept {len(should_keep)} ordinary securities")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print("-" * 70)
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
