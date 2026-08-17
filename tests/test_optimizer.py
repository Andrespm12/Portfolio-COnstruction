"""
Tests for the Black-Litterman posterior and the constrained optimizer.

Three of these exist because the original implementation got them wrong, and a
port that silently inherited the same defects would be worse than no port:

* the solver must not depend on an optional package (CCI's saved run died on a
  missing ECOS and produced no allocation at all),
* ``leverage_max`` must actually constrain something (it was declared and never
  applied), and
* the band audit must be able to fail (CCI's wrote "Auditoría OK"
  unconditionally, leaving a compliance assertion nothing had verified).

The rest target the places where a wrong number still looks plausible: that
views move the posterior in the direction they claim, that every regulatory
limit binds, and that a missing market cap is reported rather than defaulted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from screener.black_litterman import build_views  # noqa: E402
from screener.cci_regulation import (  # noqa: E402
    CLASE_COMMODITIES, EXCLUSIONES_DURAS, REGULACIONES, bands_for,
    classify_for_bands, unbanded_classes,
)
from screener.optimizer import (  # noqa: E402
    LEVERAGE_BUFFER, allocation_table, audit_bands, feasibility_report,
    implied_equilibrium, market_weights, optimize, posterior, select_basket,
    shrunk_covariance,
)
from screener.profiles import profile_for_strategy  # noqa: E402
from screener.run_screen import run_standalone  # noqa: E402
from screener.tuning import reset_all  # noqa: E402
from screener.yahoo_adapter import build_market_data, daily_returns  # noqa: E402
from test_yahoo_adapter import make_yf_frame  # noqa: E402

PASSED = 0
FAILED = 0

TICKERS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "LQD", "HYG", "BIL", "GLD",
           "XLE", "XLV", "XLF", "AAPL", "MSFT", "NVDA", "JPM", "LLY"]


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def world(strategy: str = "Moderado"):
    frame = make_yf_frame(TICKERS, dividends={"SPY": 6.0, "JPM": 4.0})
    data = build_market_data(frame, TICKERS, benchmark="SPY")
    returns = daily_returns(frame)
    cov = shrunk_covariance(returns)

    caps = {t: 1e10 + 1e9 * i for i, t in enumerate(cov.columns)}
    weights, _ = market_weights(caps, list(cov.columns))
    pi = implied_equilibrium(weights, cov)

    profile = profile_for_strategy(strategy)
    try:
        scored, meta = run_standalone(data, profile.key)
    finally:
        reset_all()

    types = {r.ticker: r.asset_type for r in scored}
    return data, scored, cov, pi, types, caps


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def test_covariance() -> None:
    frame = make_yf_frame(TICKERS)
    cov = shrunk_covariance(daily_returns(frame))

    check("covariance is square over the universe", cov.shape == (len(TICKERS),) * 2)
    check("covariance is symmetric", np.allclose(cov.values, cov.values.T))
    eigenvalues = np.linalg.eigvalsh(cov.values)
    check("shrinkage leaves it positive definite", bool(eigenvalues.min() > 0),
          f"min eigenvalue {eigenvalues.min():.2e}")
    check("annualized volatilities are plausible",
          bool((np.sqrt(np.diag(cov.values)) > 0.02).all()
               and (np.sqrt(np.diag(cov.values)) < 2.0).all()))

    thin = daily_returns(frame).head(10)
    try:
        shrunk_covariance(thin)
        check("too few observations raises", False, "no exception")
    except ValueError as exc:
        check("too few observations raises a clear error", "observaciones" in str(exc))


def test_market_weights_report_missing() -> None:
    caps = {"SPY": 5e11, "AAPL": 3e12, "TLT": None, "GLD": 0.0}
    weights, missing = market_weights(caps, ["SPY", "AAPL", "TLT", "GLD", "IEF"])

    check("weights sum to one", abs(weights.sum() - 1.0) < 1e-12)
    check("a missing cap is reported, never defaulted",
          set(missing) == {"TLT", "GLD", "IEF"}, f"got {missing}")
    check("missing names carry no weight", not {"TLT", "GLD", "IEF"} & set(weights.index))
    check("the larger company gets the larger weight",
          weights["AAPL"] > weights["SPY"])

    try:
        market_weights({}, ["SPY"])
        check("no caps at all raises", False, "no exception")
    except ValueError:
        check("no caps at all raises", True)


def test_equilibrium() -> None:
    _, _, cov, pi, _, _ = world()
    check("equilibrium is finite for every asset",
          bool(np.all(np.isfinite(pi.values))))
    check("equilibrium returns are of a plausible magnitude",
          bool((pi.abs() < 1.0).all()), f"max {pi.abs().max():.3f}")


# --------------------------------------------------------------------------
# Posterior
# --------------------------------------------------------------------------

def test_posterior_moves_in_the_direction_of_the_view() -> None:
    _, _, cov, pi, _, _ = world()

    bullish = [{"tipo": "absoluto", "activo": "AAPL", "Q": 0.05, "conviccion": 0.8}]
    bearish = [{"tipo": "absoluto", "activo": "AAPL", "Q": -0.05, "conviccion": 0.8}]

    up, _ = posterior(pi, cov, bullish)
    down, _ = posterior(pi, cov, bearish)

    check("a positive view raises the name's expected return",
          up["AAPL"] > pi["AAPL"], f"{up['AAPL']:.4f} vs {pi['AAPL']:.4f}")
    check("a negative view lowers it", down["AAPL"] < pi["AAPL"])
    check("no views leaves the equilibrium untouched",
          np.allclose(posterior(pi, cov, [])[0].values, pi.values))


def test_conviction_scales_the_shift() -> None:
    _, _, cov, pi, _, _ = world()
    strong = posterior(pi, cov, [{"tipo": "absoluto", "activo": "TLT",
                                  "Q": 0.04, "conviccion": 0.9}])[0]
    weak = posterior(pi, cov, [{"tipo": "absoluto", "activo": "TLT",
                                "Q": 0.04, "conviccion": 0.15}])[0]
    check("higher conviction moves the posterior further",
          abs(strong["TLT"] - pi["TLT"]) > abs(weak["TLT"] - pi["TLT"]))


def test_relative_view_tilts_both_legs() -> None:
    _, _, cov, pi, _, _ = world()
    view = [{"tipo": "relativo", "activo_long": "AAPL", "activo_short": "SPY",
             "Q": 0.04, "conviccion": 0.8}]
    post, _ = posterior(pi, cov, view)
    check("a relative view widens the spread between its legs",
          (post["AAPL"] - post["SPY"]) > (pi["AAPL"] - pi["SPY"]))


def test_unknown_ticker_in_a_view_is_skipped() -> None:
    _, _, cov, pi, _, _ = world()
    post, _ = posterior(pi, cov, [
        {"tipo": "absoluto", "activo": "NO_EXISTE", "Q": 0.05, "conviccion": 0.8},
    ])
    check("a view naming an unscored ticker is skipped, not raised on",
          np.allclose(post.values, pi.reindex(post.index).values))


# --------------------------------------------------------------------------
# Regulation
# --------------------------------------------------------------------------

def test_asset_classification() -> None:
    check("treasuries classify as sovereign IG",
          classify_for_bands("TLT", "ETF") == "RentaFija_Soberana_IG")
    check("investment-grade credit classifies correctly",
          classify_for_bands("LQD", "ETF") == "RentaFija_Corporativa_IG")
    check("high yield classifies as NoIG",
          classify_for_bands("HYG", "ETF") == "RentaFija_NoIG")
    check("EM sovereign debt is treated as NoIG",
          classify_for_bands("EMB", "ETF") == "RentaFija_NoIG")
    check("T-bill funds classify as cash",
          classify_for_bands("BIL", "ETF") == "Efectivo_MM")
    check("gold classifies as commodities",
          classify_for_bands("GLD", "ETF") == CLASE_COMMODITIES)
    check("an unmapped ETF falls back to equity ETF",
          classify_for_bands("SPY", "ETF") == "ETF_RentaVariable")
    check("a single stock classifies as Equity",
          classify_for_bands("AAPL", "STOCK") == "Equity")


def test_commodity_band_closes_the_hole() -> None:
    """
    CCI's REGULACIONES has no commodity class, and their optimizer only
    constrains classes present in `bandas` -- an unconstrained gold sleeve could
    take the entire book.
    """
    for strategy in REGULACIONES:
        bands = bands_for(strategy)
        check(f"{strategy}: commodities carry a band",
              CLASE_COMMODITIES in bands)
        check(f"{strategy}: the commodity ceiling is below 100%",
              bands[CLASE_COMMODITIES][1] < 1.0)

    check("commodity ceilings widen with risk tolerance",
          bands_for("Conservador_Defensivo")[CLASE_COMMODITIES][1]
          < bands_for("Conservador")[CLASE_COMMODITIES][1]
          < bands_for("Moderado")[CLASE_COMMODITIES][1]
          < bands_for("Agresivo")[CLASE_COMMODITIES][1])

    check("an unmapped class is reported rather than left silent",
          unbanded_classes({"XYZ": "Cripto"}, "Moderado") == {"Cripto"})


# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------

def test_optimization_respects_every_limit() -> None:
    for strategy in REGULACIONES:
        data, scored, cov, pi, types, _ = world(strategy)
        views = build_views(scored, data, strategy=strategy)
        er, post_cov = posterior(pi, cov, views)
        alloc = optimize(er, post_cov, types, strategy)

        check(f"{strategy}: the optimizer finds a solution",
              alloc.feasible, f"status {alloc.status}")
        check(f"{strategy}: the audit finds no breach",
              not alloc.breaches, "; ".join(alloc.breaches))
        check(f"{strategy}: no negative weights",
              bool((alloc.weights >= -1e-9).all()))

        budget = REGULACIONES[strategy]["leverage_max"] * LEVERAGE_BUFFER
        check(f"{strategy}: gross exposure respects the leverage budget",
              alloc.gross_exposure <= budget + 1e-6,
              f"{alloc.gross_exposure:.4f} > {budget:.4f}")
        check(f"{strategy}: the book stays invested",
              alloc.gross_exposure > 0.9, f"{alloc.gross_exposure:.4f}")


def test_leverage_is_actually_applied() -> None:
    """
    CCI's REGULACIONES declares leverage_max 1.25/1.50 and their optimizer
    hard-coded sum(w) == 1, so the field never constrained anything.
    """
    _, scored, cov, pi, types, _ = world("Moderado")

    conservative = optimize(pi, cov, types, "Conservador")
    moderate = optimize(pi, cov, types, "Moderado")
    aggressive = optimize(pi, cov, types, "Agresivo")

    check("an unlevered mandate is fully invested and no more",
          abs(conservative.gross_exposure - 1.0 * LEVERAGE_BUFFER) < 1e-4,
          f"{conservative.gross_exposure:.4f}")
    check("a levered mandate actually uses leverage",
          moderate.gross_exposure > 1.0, f"{moderate.gross_exposure:.4f}")
    check("more leverage allowance means more gross exposure",
          aggressive.gross_exposure > moderate.gross_exposure)
    check("the documented buffer keeps it off the regulatory margin",
          aggressive.gross_exposure <= 1.50 * LEVERAGE_BUFFER + 1e-6)


def test_hard_exclusions_hold() -> None:
    _, scored, cov, pi, types, _ = world("Agresivo")
    excluded = EXCLUSIONES_DURAS[0]

    cov2 = cov.copy()
    cov2.loc[excluded, :] = 0.0
    cov2.loc[:, excluded] = 0.0
    cov2.loc[excluded, excluded] = 0.01
    pi2 = pi.copy()
    pi2[excluded] = 0.50  # irresistible unless the exclusion binds
    types2 = dict(types, **{excluded: "ETF"})

    alloc = optimize(pi2, cov2, types2, "Agresivo")
    check("an Art. 170 exclusion holds against a dominant expected return",
          float(alloc.weights.get(excluded, 0.0)) < 1e-6,
          f"weight {alloc.weights.get(excluded)}")
    check("the audit confirms no breach", not alloc.breaches)


def test_single_name_cap_binds() -> None:
    _, scored, cov, pi, types, _ = world("Conservador")
    pi2 = pi.copy()
    pi2["AAPL"] = 0.60

    alloc = optimize(pi2, cov, types, "Conservador")
    cap = REGULACIONES["Conservador"]["max_equity_individual"]
    check("a single stock cannot exceed its individual cap",
          float(alloc.weights.get("AAPL", 0.0)) <= cap + 1e-6,
          f"{alloc.weights.get('AAPL'):.4f} > {cap}")


def test_equity_ceiling_binds() -> None:
    _, scored, cov, pi, types, _ = world("Conservador_Defensivo")
    pi2 = pi.copy()
    for t in ("AAPL", "MSFT", "NVDA", "SPY", "QQQ", "XLV"):
        if t in pi2.index:
            pi2[t] = 0.40

    alloc = optimize(pi2, cov, types, "Conservador_Defensivo")
    ceiling = REGULACIONES["Conservador_Defensivo"]["max_equity_total"]
    equity = sum(w for t, w in alloc.weights.items()
                 if classify_for_bands(t, types.get(t, "ETF"))
                 in ("Equity", "ETF_RentaVariable"))
    check("total equity respects the defensive ceiling",
          equity <= ceiling + 1e-6, f"{equity:.4f} > {ceiling}")
    check("the defensive mandate still holds fixed income",
          alloc.by_class.get("RentaFija_Soberana_IG", 0.0) > 0)


def test_commodity_ceiling_binds() -> None:
    _, scored, cov, pi, types, _ = world("Conservador")
    pi2 = pi.copy()
    pi2["GLD"] = 0.50  # would take the whole book unconstrained

    alloc = optimize(pi2, cov, types, "Conservador")
    ceiling = bands_for("Conservador")[CLASE_COMMODITIES][1]
    check("gold cannot exceed the commodity ceiling",
          float(alloc.by_class.get(CLASE_COMMODITIES, 0.0)) <= ceiling + 1e-6,
          f"{alloc.by_class.get(CLASE_COMMODITIES):.4f} > {ceiling}")


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def test_audit_can_actually_fail() -> None:
    """CCI's auditar_bandas wrote 'Auditoría OK' without comparing anything."""
    from screener.optimizer import Allocation

    types = {"AAPL": "STOCK", "TLT": "ETF"}
    classes = {t: classify_for_bands(t, a) for t, a in types.items()}

    bad = Allocation(
        weights=pd.Series({"AAPL": 0.90, "TLT": 0.60}),
        strategy="Conservador", status="optimal",
        gross_exposure=1.50, expected_return=0.1, volatility=0.2,
        by_class=pd.Series({"Equity": 0.90, "RentaFija_Soberana_IG": 0.60}),
    )
    breaches = audit_bands(bad, classes)

    check("the audit reports breaches instead of asserting compliance",
          len(breaches) >= 3, f"got {breaches}")
    check("the individual-stock cap breach is named",
          any("AAPL" in b for b in breaches))
    check("the leverage breach is named",
          any("bruta" in b.lower() for b in breaches))
    check("the equity ceiling breach is named",
          any("variable" in b.lower() for b in breaches))

    good = Allocation(
        weights=pd.Series({"AAPL": 0.05, "TLT": 0.90}),
        strategy="Conservador", status="optimal",
        gross_exposure=0.95, expected_return=0.05, volatility=0.1,
        by_class=pd.Series({"Equity": 0.05, "RentaFija_Soberana_IG": 0.90}),
    )
    check("a compliant allocation reports no breach", not audit_bands(good, classes))


def test_negative_and_excluded_weights_are_caught() -> None:
    from screener.optimizer import Allocation

    alloc = Allocation(
        weights=pd.Series({"AAPL": -0.10, EXCLUSIONES_DURAS[0]: 0.20, "TLT": 0.85}),
        strategy="Moderado", status="optimal",
        gross_exposure=0.95, expected_return=0.0, volatility=0.1,
        by_class=pd.Series({"RentaFija_Soberana_IG": 0.85}),
    )
    breaches = audit_bands(alloc, {"AAPL": "Equity", "TLT": "RentaFija_Soberana_IG",
                                   EXCLUSIONES_DURAS[0]: "ETF_RentaVariable"})
    check("a short position is caught", any("negativ" in b.lower() for b in breaches))
    check("a weight on an excluded issuer is caught",
          any(EXCLUSIONES_DURAS[0] in b for b in breaches))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_end_to_end_with_screener_views() -> None:
    frame = make_yf_frame(TICKERS, dividends={"SPY": 6.0, "JPM": 4.0})
    data = build_market_data(frame, TICKERS, benchmark="SPY")
    cov = shrunk_covariance(daily_returns(frame))
    caps = {t: 1e10 + 1e9 * i for i, t in enumerate(cov.columns)}
    weights, missing = market_weights(caps, list(cov.columns))
    pi = implied_equilibrium(weights, cov)

    try:
        scored, meta = run_standalone(data, "moderado")
    finally:
        reset_all()

    views = build_views(scored, data, strategy="Moderado",
                        reference_map={"AAPL": "QQQ", "JPM": "XLF"})
    check("the screener produced views to feed the optimizer", len(views) > 0)

    er, post_cov = posterior(pi, cov, views)
    types = {r.ticker: r.asset_type for r in scored}
    alloc = optimize(er, post_cov, types, "Moderado")

    check("the full chain solves", alloc.feasible, alloc.status)
    check("the allocation passes its own audit",
          not alloc.breaches, "; ".join(alloc.breaches))
    check("weights are finite", bool(np.all(np.isfinite(alloc.weights.values))))
    check("the book holds more than one name",
          int((alloc.weights > 0).sum()) > 1)
    check("class exposures sum to gross exposure",
          abs(alloc.by_class.sum() - alloc.gross_exposure) < 1e-6)
    check("reported volatility is positive", alloc.volatility > 0)

    # The views must actually change the allocation, or the bridge is decorative.
    baseline = optimize(pi, cov, types, "Moderado")
    moved = (alloc.weights - baseline.weights.reindex(alloc.weights.index)
             .fillna(0.0)).abs().max()
    check("the screener's views change the resulting portfolio",
          moved > 1e-4, f"max weight change {moved:.6f}")

    table = allocation_table(alloc, classes={
        t: classify_for_bands(t, types.get(t, "ETF")) for t in alloc.weights.index})
    check("the allocation table lists only held positions",
          len(table) == int((alloc.weights > 0).sum()))
    check("the table is sorted by weight",
          list(table["peso"]) == sorted(table["peso"], reverse=True))


def test_infeasible_problem_reports_rather_than_crashes() -> None:
    _, scored, cov, pi, types, _ = world("Conservador_Defensivo")

    # Genuinely infeasible: a basket of nothing but equities under the defensive
    # mandate, which caps total equity at 30% while the gross budget requires
    # 95% invested. Bond and cash tickers are excluded from the basket, because
    # the class map takes precedence over the asset_type label -- TLT is a bond
    # ETF whatever type you pass it, which is the correct behaviour and the
    # reason a naive "call everything a stock" fixture stays solvable.
    equities = [t for t in cov.columns
                if classify_for_bands(t, types.get(t, "ETF"))
                in ("Equity", "ETF_RentaVariable")]
    sub_cov = cov.loc[equities, equities]
    sub_pi = pi[equities]
    alloc = optimize(sub_pi, sub_cov, types, "Conservador_Defensivo")

    check("an infeasible problem returns a result instead of raising",
          isinstance(alloc.weights, pd.Series))
    check("infeasibility is reported as not feasible", not alloc.feasible,
          f"status {alloc.status}")
    check("the reason reaches the user",
          bool(alloc.breaches or alloc.notes))


def test_equity_only_basket_is_the_infeasibility_that_emptied_the_sheet() -> None:
    """
    Regression for a real failure: the notebook wrote an empty Cartera tab.

    The optimizer basket was the top-N by screener score. The screen ranks on
    momentum and risk-adjusted return, which equities dominate, so the basket
    came out all equity -- and every mandate caps total equity below the amount
    the book must invest. The solver said "infeasible" and the sheet came out
    blank with no stated cause.
    """
    equity_only = ["SPY", "QQQ", "IWM", "XLE", "XLV", "XLF",
                   "AAPL", "MSFT", "NVDA", "JPM", "LLY"]
    frame = make_yf_frame(equity_only)
    cov = shrunk_covariance(daily_returns(frame))
    caps = {t: 1e10 for t in cov.columns}
    weights, _ = market_weights(caps, list(cov.columns))
    pi = implied_equilibrium(weights, cov)
    types = {t: ("STOCK" if t in ("AAPL", "MSFT", "NVDA", "JPM", "LLY") else "ETF")
             for t in cov.columns}

    alloc = optimize(pi, cov, types, "Moderado")
    check("an all-equity basket is infeasible under the mandate",
          not alloc.feasible, f"status {alloc.status}")
    check("the empty result explains itself instead of going out blank",
          any("solo renta variable" in b for b in alloc.breaches),
          f"got {alloc.breaches}")
    check("the diagnosis names the classes that are missing",
          any("RentaFija" in b for b in alloc.breaches))
    check("allocation_table on an infeasible run is empty, so callers must handle it",
          allocation_table(alloc).empty)


def test_select_basket_spans_the_classes() -> None:
    tickers = ["SPY", "QQQ", "IWM", "XLE", "XLV", "XLF", "AAPL", "MSFT",
               "NVDA", "JPM", "LLY", "AMZN", "META", "MU", "TLT", "IEF",
               "LQD", "HYG", "BIL", "AGG", "GLD"]
    frame = make_yf_frame(tickers)
    data = build_market_data(frame, tickers, benchmark="SPY")
    try:
        scored, _ = run_standalone(data, "moderado")
    finally:
        reset_all()

    types = {r.ticker: r.asset_type for r in scored}
    basket = select_basket(scored, "Moderado", top_n=8, min_per_class=2)
    classes = {t: classify_for_bands(t, types.get(t, "ETF")) for t in basket}

    check("the basket keeps the top names by score",
          {r.ticker for r in scored[:8]} <= set(basket))
    check("the basket reaches beyond the top-N to cover classes",
          len(basket) > 8, f"got {len(basket)}")
    check("fixed income is represented",
          any(c.startswith("RentaFija") for c in classes.values()),
          f"classes {sorted(set(classes.values()))}")
    check("cash is represented", "Efectivo_MM" in set(classes.values()))
    check("no ticker is duplicated", len(basket) == len(set(basket)))

    universe_classes = {classify_for_bands(r.ticker, types.get(r.ticker, "ETF"))
                        for r in scored}
    check("every class available in the universe makes it into the basket",
          universe_classes == set(classes.values()),
          f"missing {universe_classes - set(classes.values())}")

    # And the whole point: it now solves.
    cov = shrunk_covariance(daily_returns(frame, basket))
    caps = {t: 1e10 + 1e9 * i for i, t in enumerate(cov.columns)}
    w, _ = market_weights(caps, list(cov.columns))
    alloc = optimize(implied_equilibrium(w, cov), cov, types, "Moderado")
    check("the class-aware basket produces a feasible portfolio",
          alloc.feasible, f"status {alloc.status}")
    check("and a non-empty allocation table", not allocation_table(alloc).empty)


def test_feasibility_report_is_quiet_when_reachable() -> None:
    healthy = {"TLT": "RentaFija_Soberana_IG", "AAPL": "Equity",
               "BIL": "Efectivo_MM", "SPY": "ETF_RentaVariable"}
    check("a basket that can fill the mandate reports no problem",
          feasibility_report(healthy, "Moderado") == [])
    check("an equity-only basket is flagged before the solver runs",
          feasibility_report({"AAPL": "Equity"}, "Moderado") != [])


def main() -> int:
    for fn in [
        test_covariance,
        test_market_weights_report_missing,
        test_equilibrium,
        test_posterior_moves_in_the_direction_of_the_view,
        test_conviction_scales_the_shift,
        test_relative_view_tilts_both_legs,
        test_unknown_ticker_in_a_view_is_skipped,
        test_asset_classification,
        test_commodity_band_closes_the_hole,
        test_optimization_respects_every_limit,
        test_leverage_is_actually_applied,
        test_hard_exclusions_hold,
        test_single_name_cap_binds,
        test_equity_ceiling_binds,
        test_commodity_ceiling_binds,
        test_audit_can_actually_fail,
        test_negative_and_excluded_weights_are_caught,
        test_end_to_end_with_screener_views,
        test_infeasible_problem_reports_rather_than_crashes,
        test_equity_only_basket_is_the_infeasibility_that_emptied_the_sheet,
        test_select_basket_spans_the_classes,
        test_feasibility_report_is_quiet_when_reachable,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
