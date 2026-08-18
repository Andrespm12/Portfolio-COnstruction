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


def make_correlated(bench, beta=1.0, alpha=0.0, idio_vol=0.015, seed=0, start=100.0):
    """
    A price series that genuinely regresses on the benchmark.

    Fixtures built from independent random walks are uncorrelated with the
    benchmark by construction, which now means the market-sensitivity block is
    withheld from every one of them -- and with it enough block coverage to
    drop the whole cross-section below ``min_populated_blocks``. Every name
    would then come back Market Weight for want of data, and any test about
    ranking would pass or fail for the wrong reason. Real equities are
    correlated with the market; fixtures standing in for them should be too.
    """
    rng = np.random.default_rng(seed)
    b = simple_returns(np.asarray(bench, dtype=float))
    shocks = alpha + beta * b + rng.normal(0.0, idio_vol, b.size)
    return start * np.cumprod(1.0 + shocks)


def varied(i, **overrides):
    """
    Per-name market-data inputs that actually differ across the cross-section.

    Constant inputs are worse than missing ones here: ``zscore`` returns all-NaN
    on zero variance, so a fixture that gives every name the same ADV and
    dividend yield silently empties the liquidity and valuation blocks instead
    of scoring them.
    """
    return dict({"adv": 5e8 * (1.0 + 0.35 * i), "div": 0.5 + 0.4 * i,
                 "iv": 0.20 + 0.03 * i, "hv": 0.19 + 0.02 * i}, **overrides)


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
            # Jittered rather than flat: constant volume makes
            # turnover_stability undefined (sd == 0) for every name at once.
            "volume": [1e7 * (1.0 + 0.05 * ((j * 7 + hash(ticker)) % 11) / 11.0)
                       for j in range(len(prices))],
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


_MARKET_SENSITIVITY = ("beta_1y", "alpha_annual", "capture_spread", "idio_vol_share")


def test_extra_history_changes_nothing():
    """
    Retaining more bars must be margin, not a redefinition.

    The download now keeps 60 weekly bars where the engine's longest metric
    needs 53, so ``mom_12_1`` survives a missing Friday instead of silently
    returning None and dropping 35% of the momentum block. The hazard that
    creates is the mirror image: every "1Y" statistic is computed from whatever
    bars are present, so the extra history would quietly turn each of them into
    a 14-month figure while the label, the report column and the documentation
    all still said one year.

    The guarantee this test pins down is that the two histories -- 60 bars, and
    that same series truncated to its last 53 -- produce *identical* metrics.
    If any statistic silently widens with the window, this fails.
    """
    bench_long = make_series(n=60, drift=0.004, vol=0.02, seed=61)
    series_long = make_correlated(bench_long, beta=1.2, alpha=0.001,
                                  idio_vol=0.012, seed=62)

    long_inst = make_instrument("LONG", series_long, **varied(1))
    short_inst = make_instrument("SHORT", series_long[-53:], **varied(1))
    short_inst["history"]["volume"] = long_inst["history"]["volume"][-53:]

    bench_rets_long = simple_returns(bench_long)
    bench_rets_short = simple_returns(np.asarray(bench_long)[-53:])

    a = compute_metrics(long_inst, bench_rets_long, net_liq=1e7,
                        participation=0.2, base_position_weight=0.03)
    b = compute_metrics(short_inst, bench_rets_short, net_liq=1e7,
                        participation=0.2, base_position_weight=0.03)

    drifted = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x is None and y is None:
            continue
        if x is None or y is None or abs(float(x) - float(y)) > 1e-9:
            drifted.append(f"{key}: 60 bars {x} vs 53 bars {y}")
    assert not drifted, "metrics changed with history length: " + "; ".join(drifted)
    assert a["mom_12_1"] is not None, "the fixture never computed momentum"
    print(f"PASS all {len(a)} metrics identical on 60 bars and on the same "
          f"series cut to 53")


def test_momentum_survives_a_missing_bar():
    """
    The failure the wider window exists to prevent.

    ``mom_12_1`` is a 48-week return skipping the most recent 4, so it needs 53
    observations. At exactly 53 retained bars there is no slack: lose one and
    the model's highest-weighted metric disappears without a word in the output.
    """
    bench = make_series(n=60, drift=0.004, vol=0.02, seed=63)
    series = make_correlated(bench, beta=1.0, seed=64)

    tight = make_instrument("TIGHT", series[-53:][:-1])  # 52 bars: one short
    roomy = make_instrument("ROOMY", series[:-1])        # 59 bars: still fine

    tight_m = compute_metrics(tight, simple_returns(bench), net_liq=1e7,
                              participation=0.2, base_position_weight=0.03)
    roomy_m = compute_metrics(roomy, simple_returns(bench), net_liq=1e7,
                              participation=0.2, base_position_weight=0.03)

    assert tight_m["mom_12_1"] is None, (
        "fixture does not reproduce the shortfall it is testing"
    )
    assert roomy_m["mom_12_1"] is not None, (
        "the wider window still lost momentum to a single missing bar"
    )
    print("PASS one missing week kills mom_12_1 at 53 retained bars, "
          "survives at 60")


def test_market_model_significance_threshold():
    """
    The floor is the conventional t >= 2 on beta, evaluated at the sample size
    actually available -- not a fixed R-squared. Same explanatory power means
    something different on 20 observations than on 200.
    """
    from screener.metrics import market_model_holds

    # t^2 = (n-2) * R2/(1-R2); at n=52 the break-even R2 is 4/(50+4).
    breakeven = 4.0 / 54.0
    assert market_model_holds(breakeven + 1e-9, 52)
    assert not market_model_holds(breakeven - 1e-9, 52)

    # The same R-squared passes with more data and fails with less.
    assert not market_model_holds(0.05, 30), "0.05 should not clear on 30 bars"
    assert market_model_holds(0.05, 200), "0.05 should clear on 200 bars"

    for bad in (None, float("nan"), -0.1, 0.0):
        assert not market_model_holds(bad, 52), f"{bad} should not hold"
    assert market_model_holds(1.0, 52), "a perfect fit holds"
    assert not market_model_holds(0.9, 2), "n=2 has no degrees of freedom"
    print(f"PASS market-model floor is t>=2 on beta "
          f"(R2 {breakeven:.3f} at 52 bars, {4.0 / 202:.3f} at 200)")


def test_uncorrelated_name_drops_the_market_block():
    """
    An asset the benchmark does not explain must not be scored against it.

    Jensen's alpha at beta ~ 0 is the asset's own excess return, which the
    momentum block already scores -- so leaving it in pays for the same
    performance twice, under a label that reads as skill. Beta and
    idiosyncratic share are both functions of R-squared, so keeping only those
    two states one fact twice and leaves the block with no return term at all.
    The block is withheld entirely and the composite renormalizes.
    """
    bench = make_series(n=60, drift=0.004, vol=0.02, seed=71)
    bench_rets = simple_returns(bench)
    independent = make_series(n=60, drift=0.004, vol=0.02, seed=999)

    m = compute_metrics(make_instrument("INDEP", independent), bench_rets,
                        net_liq=1e7, participation=0.2, base_position_weight=0.03)
    d = diagnostics(make_instrument("INDEP", independent), bench_rets)

    assert d["r_squared"] < 0.07, f"fixture is not uncorrelated: R2={d['r_squared']:.3f}"
    for key in _MARKET_SENSITIVITY:
        assert m[key] is None, f"{key} survived at R2={d['r_squared']:.3f}: {m[key]}"
    assert d["alpha_annual"] is None, "the report would print an unscored alpha"
    assert d["upside_capture"] is None and d["downside_capture"] is None
    assert d["beta"] is not None, "beta is descriptive and should stay"
    assert d["r_squared"] is not None, "R2 must stay so the reason is visible"

    # And a correlated name keeps everything.
    levered = np.asarray([100.0], dtype=float)
    for x in simple_returns(bench) * 1.5:
        levered = np.append(levered, levered[-1] * (1 + x))
    m_ok = compute_metrics(make_instrument("LEVER", levered), bench_rets,
                           net_liq=1e7, participation=0.2, base_position_weight=0.03)
    for key in _MARKET_SENSITIVITY:
        assert m_ok[key] is not None, f"{key} was dropped from a correlated name"
    print(f"PASS market block withheld at R2={d['r_squared']:.3f}, kept at "
          f"R2={diagnostics(make_instrument('LEVER', levered), bench_rets)['r_squared']:.3f}")


def test_uncorrelated_loser_is_not_rewarded_for_being_uncorrelated():
    """
    The mis-ranking this fixes, stated as an outcome rather than a mechanism.

    On the IBKR snapshot the old model scored TLT fourth-highest on "Market
    Sensitivity & Alpha Quality" in a year it lost 4.9%, because beta near zero
    and idiosyncratic share near one are both just 'this does not track SPY'.
    A name that lost money must not out-score a name that made money on a block
    whose stated purpose is alpha quality.
    """
    bench = make_series(n=60, drift=0.005, vol=0.02, seed=81)
    bench_rets = simple_returns(bench)

    rows = []
    specs = [("LOSER", -0.004, 909), ("WINNER", 0.012, 77), ("MID1", 0.004, 78),
             ("MID2", 0.005, 79), ("MID3", 0.003, 80)]
    for tkr, drift, seed in specs:
        series = make_series(n=60, drift=drift, vol=0.02, seed=seed)
        inst = make_instrument(tkr, series)
        m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                            base_position_weight=0.03)
        m.update({"corr_to_portfolio": 0.3, "diversification_benefit": 0.0,
                  "existing_overlap": 0.0})
        rows.append(ScoredInstrument(tkr, tkr, "STOCK", ["SP500"], "Test", m,
                                     diagnostics=diagnostics(inst, bench_rets)))

    by = {r.ticker: r for r in run_scoring_pipeline(rows)}
    loser, winner = by["LOSER"], by["WINNER"]

    assert loser.raw_metrics["mom_12_1"] < 0, "fixture LOSER did not lose"
    assert winner.raw_metrics["mom_12_1"] > 0, "fixture WINNER did not win"

    loser_ms = loser.block_scores.get("market_sensitivity")
    if loser_ms is not None:
        assert loser_ms <= winner.block_scores.get("market_sensitivity", 99), (
            f"a losing name out-scored a winning one on alpha quality: "
            f"{loser_ms:+.2f} vs {winner.block_scores.get('market_sensitivity')}"
        )
    assert loser.composite_z < winner.composite_z, (
        f"LOSER z {loser.composite_z:+.2f} >= WINNER z {winner.composite_z:+.2f}"
    )
    populated = sum(1 for v in loser.block_coverage.values() if v > 0)
    print(f"PASS uncorrelated loser scored on {populated} blocks, z "
          f"{loser.composite_z:+.2f} below the winner's {winner.composite_z:+.2f}")


def test_dropping_the_block_still_leaves_a_scoreable_name():
    """
    Withholding one block must not silently force every such name to neutral.

    ``BANDS.min_populated_blocks`` is 5. The account-aware model declares 7
    blocks and the standalone profiles 6, so dropping market sensitivity leaves
    6 and 5 respectively -- at the threshold in standalone mode, which is worth
    a test rather than an assumption.
    """
    from screener.config import BANDS

    bench = make_series(n=60, drift=0.004, vol=0.02, seed=91)
    bench_rets = simple_returns(bench)

    rows = []
    for i, seed in enumerate((901, 902, 903, 904, 905)):
        # Independent of the benchmark on purpose -- these are the names whose
        # market block gets withheld. Everything else varies so the remaining
        # blocks are genuinely populated, as they are on real data.
        series = make_series(n=60, drift=0.002 * i, vol=0.02, seed=seed)
        inst = make_instrument(f"U{i}", series, **varied(i))
        m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                            base_position_weight=0.03)
        m.update({"corr_to_portfolio": 0.2 + 0.05 * i,
                  "diversification_benefit": 0.01 * (i % 4),
                  "existing_overlap": 0.005 * i})
        rows.append(ScoredInstrument(f"U{i}", f"U{i}", "STOCK", ["SP500"], "Test", m,
                                     diagnostics=diagnostics(inst, bench_rets)))

    scored = run_scoring_pipeline(rows)
    dropped = [r for r in scored if r.block_coverage.get("market_sensitivity", 0) == 0]
    assert dropped, "fixture produced no uncorrelated names, so it proves nothing"
    for row in dropped:
        populated = sum(1 for v in row.block_coverage.values() if v > 0)
        assert populated >= BANDS.min_populated_blocks, (
            f"{row.ticker} fell to {populated} blocks, below the "
            f"{BANDS.min_populated_blocks} needed for a non-neutral call"
        )
        assert np.isfinite(row.composite_z), f"{row.ticker} lost its composite"
    print(f"PASS {len(dropped)} name(s) kept a scoreable composite with the "
          f"market block withheld")


def _cross_section(drift_shift: float = 0.0, n: int = 8):
    """
    A scored cross-section whose whole return distribution can be shifted.

    ``drift_shift`` is added to every name's weekly drift, so the *ranking* is
    held fixed while the absolute quality of the entire universe moves. That
    separation is the point: it is what lets a test tell a relative statement
    from an absolute one.
    """
    bench = make_series(n=60, drift=0.003 + drift_shift, vol=0.015, seed=101)
    bench_rets = simple_returns(bench)

    rows = []
    for i in range(n):
        series = make_correlated(bench, beta=0.7 + 0.15 * i,
                                 alpha=0.001 * i + drift_shift,
                                 idio_vol=0.010 + 0.004 * (i % 3), seed=200 + i)
        inst = make_instrument(f"N{i}", series, **varied(i))
        m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                            base_position_weight=0.03)
        m.update({"corr_to_portfolio": 0.2 + 0.05 * i,
                  "diversification_benefit": 0.01 * (i % 4),
                  "existing_overlap": 0.005 * i})
        rows.append(ScoredInstrument(inst["ticker"], inst["ticker"], "STOCK",
                                     ["SP500"], "Test", m,
                                     diagnostics=diagnostics(inst, bench_rets)))
    return rows


def test_bands_are_purely_relative():
    """
    Standardizing the composite carries no information about absolute quality.

    This is the fact an earlier version of ``RecommendationBands.__doc__``
    denied -- it claimed that scoring on z rather than percentile stopped a
    mediocre universe from producing a top decile of Overweights. Here every
    single name in the cross-section loses money, and the bands still hand out
    Overweights, because z is scale-free. The test exists to keep that claim
    from being made again.
    """
    from screener.scoring import score_universe

    rows = score_universe(_cross_section(-0.010))
    returns = {r.ticker: r.raw_metrics["mom_12_1"] for r in rows}
    assert all(v < 0 for v in returns.values()), (
        f"fixture is not a losing universe: {returns}"
    )

    overweights = [r.ticker for r in rows if r.pre_gate_recommendation == OVERWEIGHT]
    assert overweights, "fixture produced no Overweights, so it proves nothing"
    worst = min(returns[t] for t in overweights)
    print(f"PASS bands are relative: every name lost money "
          f"({max(returns.values()):.0%} at best) yet the bands still rate "
          f"{len(overweights)} Overweight, one of them down {worst:.0%}")


def _repathed(market_data, annual_drift):
    """
    The real universe with every price series re-pathed to a given drift.

    Each name keeps its own volatility, its own shape and all of its
    non-price data; only the mean return is moved. That isolates the market
    regime from every other difference between names, which a freshly
    generated synthetic universe cannot do.
    """
    import copy

    out = copy.deepcopy(market_data)
    for inst in out.get("instruments", []):
        history = inst.get("history") or {}
        closes = history.get("close") or []
        if len(closes) < 10:
            continue
        prices = np.asarray(closes, dtype=float)
        rets = np.diff(prices) / prices[:-1]
        rets = rets - rets.mean() + annual_drift / 52.0
        path = [prices[0]]
        for r in rets:
            path.append(path[-1] * (1.0 + r))
        history["close"] = path
        if history.get("high"):
            history["high"] = [p * 1.01 for p in path]
        if history.get("low"):
            history["low"] = [p * 0.99 for p in path]
    return out


def test_quality_floor_across_market_regimes():
    """
    The finding, on the real universe rather than a synthetic one.

    A flat market is the case that mattered and the case a synthetic fixture
    misses. The trend gate needs a broken moving average *and* negative
    momentum, so it does nothing when prices go sideways -- yet at a 4.25%
    risk-free rate a flat year is a negative Sharpe, and overweighting equity
    that lost to Treasury bills is exactly what an absolute floor is for.

    The bull case is the control: a rail that also fires when the universe is
    genuinely good would be a bug, not a safeguard.
    """
    from screener import tuning
    from screener.run_screen import load_json, run

    root = Path(__file__).resolve().parents[1]
    market = load_json(root / "data" / "raw" / "market_data.json")
    book = load_json(root / "data" / "portfolio_ibkr.json")

    def overweights(data, **gate_changes):
        tuning.reset_all()
        if gate_changes:
            tuning.override("GATES", **gate_changes)
        try:
            rows, _ = run(data, book)
            return [r.ticker for r in rows if r.recommendation == OVERWEIGHT]
        finally:
            tuning.reset_all()

    off = dict(min_momentum_for_overweight=None, min_sharpe_for_overweight=None)
    results = {}
    for label, drift in (("alcista", 0.20), ("plano", 0.0), ("bajista", -0.10)):
        data = _repathed(market, drift)
        results[label] = (overweights(data, **off), overweights(data))

    bull_without, bull_with = results["alcista"]
    assert bull_without, "the bull fixture produced no Overweights at all"
    assert bull_with == bull_without, (
        f"the floor changed a rising market: {bull_without} -> {bull_with}"
    )

    for label in ("plano", "bajista"):
        without, with_floor = results[label]
        assert without, f"the {label} fixture produced no Overweights without the floor"
        assert not with_floor, (
            f"floor let {with_floor} through a {label} market"
        )

    print("PASS quality floor by regime: " + ", ".join(
        f"{label} {len(w)}→{len(f)}" for label, (w, f) in results.items()))


def test_quality_floor_does_not_touch_a_healthy_universe():
    """The mirror case: a rail that also fires when things are fine is a bug."""
    from screener import tuning

    def overweights(**gate_changes):
        tuning.reset_all()
        if gate_changes:
            tuning.override("GATES", **gate_changes)
        try:
            scored = run_scoring_pipeline(_cross_section(0.0))
            return [r.ticker for r in scored if r.recommendation == OVERWEIGHT]
        finally:
            tuning.reset_all()

    without = overweights(min_momentum_for_overweight=None,
                          min_sharpe_for_overweight=None)
    with_floor = overweights()
    assert without == with_floor, (
        f"floor changed a healthy universe: {without} -> {with_floor}"
    )
    assert with_floor, "fixture produced no Overweights at all"
    print(f"PASS quality floor is non-binding when the universe is good "
          f"({len(with_floor)} Overweight(s) either way)")


def test_quality_floor_catches_what_the_trend_gate_misses():
    """
    The specific gap: negative momentum while still above the 40-week average.

    The trend gate requires *both* a broken moving average and negative
    momentum, so a name that has fallen over the year but is bouncing above its
    average slips through it. That name is the whole reason the floor is not
    just a duplicate of the trend gate.
    """
    from screener.scoring import apply_risk_gates

    bench = make_series(n=60, seed=41)
    bench_rets = simple_returns(bench)

    # A crash early in the window, then a slow grind back up. The grind is
    # gentle enough that the 12M-1M return (bars 7 to 55) is still negative,
    # but the last price sits above the average of the trailing 40 bars -- so
    # the trend gate, which needs both conditions, cannot fire.
    series = np.concatenate([
        np.linspace(100.0, 50.0, 16),
        np.linspace(50.0, 70.0, 44),
    ])

    inst = make_instrument("BOUNCE", series)
    m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                        base_position_weight=0.03)
    m.update({"corr_to_portfolio": 0.2, "diversification_benefit": 0.0,
              "existing_overlap": 0.0})

    assert m["above_40w_ma"] > 0, f"fixture is not above its MA: {m['above_40w_ma']}"
    assert m["mom_12_1"] < 0, f"fixture momentum is not negative: {m['mom_12_1']}"

    row = ScoredInstrument("BOUNCE", "BOUNCE", "STOCK", ["SP500"], "Test", m,
                           diagnostics=diagnostics(inst, bench_rets))
    row.composite_z = 2.0
    row.recommendation = row.pre_gate_recommendation = OVERWEIGHT
    row.block_coverage = {k: 1.0 for k in
                          ("momentum", "risk_adjusted", "risk", "market_sensitivity",
                           "liquidity", "valuation_carry", "portfolio_fit")}

    apply_risk_gates([row])
    fired = [g for g in row.gates_triggered if g.startswith("CALIDAD ABSOLUTA")]
    assert not any(g.startswith("TREND") for g in row.gates_triggered), (
        f"trend gate should not fire here: {row.gates_triggered}"
    )
    assert fired, f"quality floor did not fire: {row.gates_triggered}"
    assert "momentum 12M-1M" in fired[0], (
        f"the momentum half of the floor is what should catch this: {fired[0]}"
    )
    assert row.recommendation == MARKET_WEIGHT, row.recommendation
    print(f"PASS quality floor caught a name the trend gate misses "
          f"(12M-1M {m['mom_12_1']:+.0%}, but {m['above_40w_ma']:+.0%} above its 40W MA)")


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
