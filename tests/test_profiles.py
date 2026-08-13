"""
Tests for risk profiles and the standalone (book-free) screen.

Two claims are load-bearing here and both are asserted against scoring output
rather than against configuration values.

**Independence.** No account data may reach a ranking. The subtle failure is
not "the book is wired in" -- it is that passing an *empty* book still emits
``existing_overlap = 0.0`` for every name, which the scorer would z-score and
count as a populated block. So these tests check that the Portfolio Fit block
and both book-relative metrics are absent from results, not merely zero.

**Profiles differ.** A profile that renamed the output without changing it
would be worse than no profile at all, so the tests assert that the three
produce different recommendations, different sizing and different gate
behaviour on identical data.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import screener.config as config  # noqa: E402
import screener.scoring as scoring  # noqa: E402
import screener.universe as universe  # noqa: E402
from screener.profiles import (  # noqa: E402
    AGRESIVO, CONSERVADOR, CONSERVADOR_DEFENSIVO, MODERADO, PROFILES,
    apply_profile, get_profile,
)
from screener.report import write_csv  # noqa: E402
from screener.run_screen import run, run_standalone  # noqa: E402
from screener.tuning import reset_all  # noqa: E402
from screener.yahoo_adapter import build_market_data  # noqa: E402
from test_yahoo_adapter import make_yf_frame  # noqa: E402

PASSED = 0
FAILED = 0

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "XLV",
           "AAPL", "MSFT", "NVDA", "JPM", "LLY", "AMZN", "META", "MU"]


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


def payload() -> dict:
    frame = make_yf_frame(TICKERS, dividends={"SPY": 6.0, "JPM": 4.0})
    return build_market_data(frame, TICKERS, benchmark="SPY")


# --------------------------------------------------------------------------
# Profile definitions
# --------------------------------------------------------------------------

def test_weights_are_valid() -> None:
    for key, profile in PROFILES.items():
        model = profile.model()
        total = sum(b.weight for b in model)
        check(f"{key}: block weights sum to 1.0", abs(total - 1.0) < 1e-9, f"got {total}")
        check(f"{key}: six blocks, Portfolio Fit removed",
              len(model) == 6 and all(b.key != "portfolio_fit" for b in model),
              f"got {[b.key for b in model]}")
        check(f"{key}: no block is negative", all(b.weight >= 0 for b in model))


def test_profiles_are_ordered_by_risk_tolerance() -> None:
    """The differences must run the direction the labels claim."""
    weights = {k: dict(p.block_weights) for k, p in PROFILES.items()}

    check("momentum rises with risk tolerance across all four profiles",
          weights["conservador_defensivo"]["momentum"]
          < weights["conservador"]["momentum"] < weights["moderado"]["momentum"]
          < weights["agresivo"]["momentum"])
    check("the volatility/drawdown brake falls as risk tolerance rises",
          weights["conservador_defensivo"]["risk"] > weights["conservador"]["risk"]
          > weights["moderado"]["risk"] > weights["agresivo"]["risk"])
    check("the defensive mandate is the strictest on every hard limit",
          CONSERVADOR_DEFENSIVO.gates.max_volatility_for_overweight
          < CONSERVADOR.gates.max_volatility_for_overweight
          and CONSERVADOR_DEFENSIVO.gates.beta_limit < CONSERVADOR.gates.beta_limit
          and CONSERVADOR_DEFENSIVO.sizing.max_weight < CONSERVADOR.sizing.max_weight
          and CONSERVADOR_DEFENSIVO.eligibility.min_adv_usd
          > CONSERVADOR.eligibility.min_adv_usd)
    check("the defensive mandate demands the most edge for an Overweight",
          CONSERVADOR_DEFENSIVO.bands.overweight_z > CONSERVADOR.bands.overweight_z)
    check("liquidity matters more when drawdown tolerance is lower",
          weights["conservador"]["liquidity"] > weights["agresivo"]["liquidity"])

    check("the Overweight bar falls as risk tolerance rises",
          CONSERVADOR.bands.overweight_z > MODERADO.bands.overweight_z
          > AGRESIVO.bands.overweight_z)
    check("the Underweight trigger is most sensitive for the conservative book",
          CONSERVADOR.bands.underweight_z > MODERADO.bands.underweight_z
          > AGRESIVO.bands.underweight_z)

    check("the volatility ceiling for an Overweight widens with risk tolerance",
          CONSERVADOR.gates.max_volatility_for_overweight
          < MODERADO.gates.max_volatility_for_overweight
          < AGRESIVO.gates.max_volatility_for_overweight)
    check("the aggressive profile still caps volatility (never None)",
          AGRESIVO.gates.max_volatility_for_overweight is not None)
    check("beta tolerance widens with risk tolerance",
          CONSERVADOR.gates.beta_limit < MODERADO.gates.beta_limit
          < AGRESIVO.gates.beta_limit)

    check("max position weight widens with risk tolerance",
          CONSERVADOR.sizing.max_weight < MODERADO.sizing.max_weight
          < AGRESIVO.sizing.max_weight)
    check("target position volatility widens with risk tolerance",
          CONSERVADOR.sizing.target_position_vol < MODERADO.sizing.target_position_vol
          < AGRESIVO.sizing.target_position_vol)
    check("the liquidity floor is highest for the conservative book",
          CONSERVADOR.eligibility.min_adv_usd > MODERADO.eligibility.min_adv_usd
          > AGRESIVO.eligibility.min_adv_usd)


def test_lookup() -> None:
    check("lookup by key", get_profile("agresivo") is AGRESIVO)
    check("lookup is case-insensitive", get_profile("Conservador") is CONSERVADOR)
    check("lookup tolerates a missing accent", get_profile("moderado") is MODERADO)
    try:
        get_profile("balanceado")
        check("unknown profile raises", False, "no exception")
    except KeyError as exc:
        check("unknown profile raises and lists the options", "moderado" in str(exc))


def test_describe_is_populated() -> None:
    text = AGRESIVO.describe()
    check("describe covers weights, bands, gates, sizing and eligibility",
          all(word in text for word in
              ("Pesos por bloque", "Overweight", "Volatilidad máxima",
               "Peso máximo", "Volumen diario")))
    check("describe does not mention Portfolio Fit", "Portfolio Fit" not in text)


# --------------------------------------------------------------------------
# Independence from any book
# --------------------------------------------------------------------------

def test_standalone_carries_no_book_data() -> None:
    data = payload()
    try:
        scored, meta = run_standalone(data, "moderado")

        check("no Portfolio Fit block in the scores",
              all("portfolio_fit" not in r.block_scores for r in scored))
        check("no Portfolio Fit block in coverage",
              all("portfolio_fit" not in r.block_coverage for r in scored))
        check("corr_to_portfolio is absent, not zero",
              all(r.raw_metrics.get("corr_to_portfolio") is None for r in scored))
        check("existing_overlap is absent, not zero",
              all(r.raw_metrics.get("existing_overlap") is None for r in scored),
              "an empty book still emits 0.0 -- the block must be removed, not blanked")
        check("no book metric was z-scored",
              all(not {"corr_to_portfolio", "diversification_benefit",
                       "existing_overlap"} & set(r.metric_z) for r in scored))

        check("meta flags the run as standalone", meta["standalone"] is True)
        check("meta reports no net liquidation", meta["net_liquidation"] == 0.0)
        check("meta carries no portfolio stats", meta["portfolio_stats"] == {})
        check("meta names the profile", meta["profile"] == "moderado")
    finally:
        reset_all()


def test_no_concentration_gate_can_fire() -> None:
    """The redundancy gate reads the book; without one it must be inert."""
    data = payload()
    try:
        scored, _ = run_standalone(data, "conservador")
        concentration = [g for r in scored for g in r.gates_triggered
                         if g.startswith("CONCENTRATION")]
        check("no book-relative gate fires in a standalone run",
              not concentration, f"got {concentration}")
    finally:
        reset_all()


def test_duplicate_gate_still_works() -> None:
    """
    The redundancy gate between *universe* members is not book-relative and
    must survive: recommending both IWM and VTWO is still double-counting one
    bet, book or no book.
    """
    tickers = TICKERS + ["VTWO"]
    frame = make_yf_frame(tickers, dividends={"SPY": 6.0})
    # Make VTWO a near-perfect clone of IWM so the duplicate detector fires.
    for field in ("Close", "Adj Close", "High", "Low", "Volume"):
        frame[(field, "VTWO")] = frame[(field, "IWM")].to_numpy() * 1.0001
    data = build_market_data(frame, tickers, benchmark="SPY")

    try:
        scored, _ = run_standalone(data, "agresivo")
        by_ticker = {r.ticker: r for r in scored}
        pair = [by_ticker["IWM"], by_ticker["VTWO"]]
        check("duplicate exposure is detected between universe members",
              any(r.duplicates for r in pair),
              f"duplicates: {[(r.ticker, r.duplicates) for r in pair]}")
        check("at most one of a duplicate pair holds an Overweight",
              sum(1 for r in pair if r.recommendation == "OVERWEIGHT") <= 1)
    finally:
        reset_all()


def test_standalone_requires_a_position_size() -> None:
    data = payload()
    try:
        run(data, {}, standalone=True, target_position_usd=0)
        check("standalone without a position size raises", False, "no exception")
    except ValueError as exc:
        check("standalone without a position size raises a clear error",
              "target_position_usd" in str(exc))
    finally:
        reset_all()


def test_position_size_changes_only_liquidity() -> None:
    """
    The assumed position size is a sizing input, not account data. It should
    move days_to_liquidate and nothing else about a name's own merits.
    """
    data = payload()
    try:
        small, _ = run_standalone(data, "moderado", position_usd=100_000)
        large, _ = run_standalone(data, "moderado", position_usd=50_000_000)

        s = {r.ticker: r for r in small}
        l = {r.ticker: r for r in large}
        check("a larger assumed position takes longer to liquidate",
              all(l[t].raw_metrics["days_to_liquidate"]
                  > s[t].raw_metrics["days_to_liquidate"] for t in s))
        check("momentum is unaffected by the assumed position size",
              all(abs(l[t].block_scores["momentum"]
                      - s[t].block_scores["momentum"]) < 1e-9 for t in s))
    finally:
        reset_all()


# --------------------------------------------------------------------------
# Profiles change the answer
# --------------------------------------------------------------------------

def results_by_profile() -> dict[str, list]:
    data = payload()
    out = {}
    try:
        for key in ("conservador", "moderado", "agresivo"):
            scored, _ = run_standalone(data, key)
            out[key] = scored
    finally:
        reset_all()
    return out


def test_profiles_produce_different_rankings() -> None:
    results = results_by_profile()

    z = {k: {r.ticker: r.composite_z for r in v} for k, v in results.items()}
    check("conservative and aggressive score the same names differently",
          any(abs(z["conservador"][t] - z["agresivo"][t]) > 0.1 for t in z["conservador"]))

    order = {k: [r.ticker for r in v] for k, v in results.items()}
    check("the ranking order itself differs between profiles",
          order["conservador"] != order["agresivo"], f"{order['conservador']}")

    counts = {k: sum(1 for r in v if r.recommendation == "OVERWEIGHT")
              for k, v in results.items()}
    check("the aggressive profile issues more Overweights than the conservative one",
          counts["agresivo"] > counts["conservador"], f"got {counts}")


def test_profiles_size_differently() -> None:
    results = results_by_profile()
    peak = {k: max(r.indicative_weight for r in v) for k, v in results.items()}
    check("the conservative profile sizes smaller than the aggressive one",
          peak["conservador"] < peak["agresivo"], f"got {peak}")
    for key, profile in (("conservador", CONSERVADOR), ("agresivo", AGRESIVO)):
        check(f"{key}: no weight exceeds the profile cap",
              peak[key] <= profile.sizing.max_weight + 1e-9,
              f"{peak[key]} > {profile.sizing.max_weight}")


def test_apply_profile_rebinds_every_holder() -> None:
    try:
        apply_profile(AGRESIVO)
        check("scoring uses the profile's bands", scoring.BANDS is AGRESIVO.bands)
        check("scoring uses the profile's gates", scoring.GATES is AGRESIVO.gates)
        check("scoring uses the profile's sizing", scoring.SIZING is AGRESIVO.sizing)
        check("universe uses the profile's eligibility",
              universe.ELIGIBILITY is AGRESIVO.eligibility)
        check("config and scoring agree on the model",
              scoring.FACTOR_MODEL is config.FACTOR_MODEL)
        check("the applied model has no Portfolio Fit block",
              all(b.key != "portfolio_fit" for b in config.FACTOR_MODEL))
    finally:
        reset_all()

    check("reset restores the seven-block model",
          len(config.FACTOR_MODEL) == 7
          and any(b.key == "portfolio_fit" for b in config.FACTOR_MODEL))
    check("reset restores the declared eligibility",
          universe.ELIGIBILITY is config.ELIGIBILITY
          and config.ELIGIBILITY.min_adv_usd == 20_000_000.0)


def test_book_path_still_works_after_a_profile_run() -> None:
    """A profile run must not leave the account-aware path broken."""
    data = payload()
    try:
        run_standalone(data, "agresivo")
    finally:
        reset_all()

    scored, meta = run(data, {
        "net_liquidation": 10_000_000.0,
        "positions": [{"ticker": "SPY", "market_value": 2_000_000.0,
                       "quantity": 2600, "asset_class": "STK"}],
    })
    check("the book-aware path returns the Portfolio Fit block again",
          all("portfolio_fit" in r.block_scores for r in scored))
    check("the book-aware path is not flagged standalone",
          meta["standalone"] is False)
    check("correlation to book is computed again",
          any(r.raw_metrics.get("corr_to_portfolio") is not None for r in scored))


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def test_standalone_csv_drops_book_columns() -> None:
    data = payload()
    try:
        scored, _ = run_standalone(data, "moderado")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_csv(scored, path, standalone=True)
            rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))

        header, first = rows[0], rows[1]
        check("no book columns in a standalone CSV",
              "corr_to_portfolio" not in header and "existing_overlap" not in header,
              f"got {header}")
        check("no Portfolio Fit block column",
              "block_portfolio_fit" not in header)
        check("header and rows stay aligned after dropping columns",
              len(header) == len(first), f"{len(header)} vs {len(first)}")
        check("the six block columns are present",
              sum(1 for h in header if h.startswith("block_")) == 6)
    finally:
        reset_all()


def main() -> int:
    for fn in [
        test_weights_are_valid,
        test_profiles_are_ordered_by_risk_tolerance,
        test_lookup,
        test_describe_is_populated,
        test_standalone_carries_no_book_data,
        test_no_concentration_gate_can_fire,
        test_duplicate_gate_still_works,
        test_standalone_requires_a_position_size,
        test_position_size_changes_only_liquidity,
        test_profiles_produce_different_rankings,
        test_profiles_size_differently,
        test_apply_profile_rebinds_every_holder,
        test_book_path_still_works_after_a_profile_run,
        test_standalone_csv_drops_book_columns,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
