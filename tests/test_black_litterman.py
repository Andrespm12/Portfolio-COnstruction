"""
Tests for the Black-Litterman bridge.

The load-bearing test here is not that the bridge produces well-formed dicts --
it is that **CCI's own solver accepts them**. ``cci_black_litterman_core`` below
is a verbatim copy of ``black_litterman_core`` from CCI's notebook, kept
unmodified on purpose: a test written against a paraphrase of the consumer
proves nothing about the consumer. If that function raises, or returns a
non-finite posterior, the export is broken regardless of how tidy the JSON is.

The remaining tests target the places where a silent error yields plausible but
wrong allocations: the sign of Q against the recommendation, the +/-5% cap CCI's
technical document specifies, the neutrality of a mid-cross-section name, and
the separation of conviction from signal magnitude.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from screener.black_litterman import (  # noqa: E402
    BASKET_COLUMNS, DEFAULT_PARAMS, ViewParams, block_agreement, build_basket,
    build_views, conviction, default_views_filename, expected_return,
    spread_volatility, views_payload, write_views,
)
from screener.profiles import (  # noqa: E402
    CCI_STRATEGIES, PROFILES, profile_for_strategy,
)
from screener.run_screen import run_standalone  # noqa: E402
from screener.tuning import reset_all  # noqa: E402
from screener.yahoo_adapter import build_market_data  # noqa: E402
from test_yahoo_adapter import make_yf_frame  # noqa: E402

PASSED = 0
FAILED = 0

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "XLV", "XLF",
           "AAPL", "MSFT", "NVDA", "JPM", "LLY", "AMZN", "META", "MU"]

#: Mirrors the activo_referencia column of CCI's basket sheet.
REFERENCIAS = {"AAPL": "QQQ", "MSFT": "QQQ", "NVDA": "QQQ",
               "JPM": "XLF", "LLY": "XLV"}


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Verbatim copy of CCI's solver -- do not "improve" it
# --------------------------------------------------------------------------

def cci_black_litterman_core(pesos_mkt, cov_matrix, views=None,
                            risk_aversion: float = 2.5, tau: float = 0.025):
    """Copied unchanged from CCI's notebook so the test exercises the real consumer."""
    import pandas as pd

    pi = risk_aversion * cov_matrix.dot(pesos_mkt)
    if not views:
        return pi, cov_matrix

    activos = list(cov_matrix.columns)
    k, n = len(views), len(activos)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, v in enumerate(views):
        Q[i] = v['Q']
        if v['tipo'] == 'absoluto':
            idx = activos.index(v['activo'])
            P[i, idx] = 1
        elif v['tipo'] == 'relativo':
            idx_long = activos.index(v['activo_long'])
            idx_short = activos.index(v['activo_short'])
            P[i, idx_long] = 1
            P[i, idx_short] = -1

        p_vec = P[i].reshape(1, -1)
        var_view = (p_vec @ cov_matrix.values @ p_vec.T)[0, 0] * tau
        convic = max(0.1, v.get('conviccion', 0.5))
        omega_diag[i] = var_view / convic

    Omega = np.diag(omega_diag)
    tau_cov_inv = np.linalg.inv(tau * cov_matrix.values)
    omega_inv = np.linalg.inv(Omega)
    term1 = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
    term2 = tau_cov_inv @ pi.values + P.T @ omega_inv @ Q
    posterior_er = term1 @ term2
    posterior_cov = cov_matrix.values + term1
    return (pd.Series(posterior_er, index=activos),
            pd.DataFrame(posterior_cov, index=activos, columns=activos))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def screen(strategy: str = "Moderado"):
    frame = make_yf_frame(TICKERS, dividends={"SPY": 6.0, "JPM": 4.0})
    data = build_market_data(frame, TICKERS, benchmark="SPY")
    profile = profile_for_strategy(strategy)
    try:
        scored, meta = run_standalone(data, profile.key)
    finally:
        reset_all()
    return scored, meta, data, profile


def covariance(data, tickers):
    import pandas as pd

    from screener.metrics import simple_returns
    series = {}
    for inst in data["instruments"]:
        if inst["ticker"] in tickers:
            series[inst["ticker"]] = simple_returns(
                np.asarray(inst["history"]["close"], dtype=float))
    n = min(len(v) for v in series.values())
    frame = pd.DataFrame({k: v[-n:] for k, v in series.items()})
    return frame.cov() * 52


# --------------------------------------------------------------------------
# The consumer accepts what the bridge produces
# --------------------------------------------------------------------------

def test_cci_solver_accepts_the_views() -> None:
    import pandas as pd

    scored, meta, data, profile = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)
    check("the bridge produced views at all", len(views) > 0, "none generated")

    tickers = [r.ticker for r in scored]
    cov = covariance(data, tickers)
    weights = pd.Series(1.0 / len(cov.columns), index=cov.columns)

    referenced = set()
    for v in views:
        referenced |= ({v["activo"]} if v["tipo"] == "absoluto"
                       else {v["activo_long"], v["activo_short"]})
    check("every ticker a view references is in the covariance universe",
          referenced <= set(cov.columns), f"missing {referenced - set(cov.columns)}")

    pi, _ = cci_black_litterman_core(weights, cov)
    posterior, post_cov = cci_black_litterman_core(weights, cov, views)

    check("CCI's solver runs on the exported views without raising", True)
    check("the posterior is finite everywhere",
          bool(np.all(np.isfinite(posterior.values))))
    check("the posterior covariance is finite",
          bool(np.all(np.isfinite(post_cov.values))))
    check("the views actually moved expected returns away from equilibrium",
          float(np.max(np.abs(posterior.values - pi.values))) > 1e-6)
    check("the posterior covariance stays symmetric",
          np.allclose(post_cov.values, post_cov.values.T))


def test_omega_is_invertible_for_every_view() -> None:
    """Conviction of exactly zero would make Omega singular; the floor prevents it."""
    scored, _, data, _ = screen("Agresivo")
    views = build_views(scored, data, strategy="Agresivo",
                        reference_map=REFERENCIAS)
    check("no view carries zero conviction",
          all(v["conviccion"] > 0 for v in views))
    check("no view exceeds the conviction ceiling",
          all(v["conviccion"] <= DEFAULT_PARAMS.max_conviction + 1e-9 for v in views))
    check("every view clears CCI's minimum-conviction filter",
          all(v["conviccion"] >= DEFAULT_PARAMS.min_conviction for v in views))


# --------------------------------------------------------------------------
# Signal translation
# --------------------------------------------------------------------------

def test_expected_return_is_centred_and_risk_scaled() -> None:
    p = DEFAULT_PARAMS
    check("a mid-cross-section name gets exactly zero expected return",
          expected_return(0.0, 0.25, p) == 0.0)
    check("sign follows the z-score",
          expected_return(1.0, 0.2, p) > 0 > expected_return(-1.0, 0.2, p))
    check("at equal rank, the more volatile name earns more expected return",
          expected_return(1.0, 0.40, p) > expected_return(1.0, 0.10, p))
    check("Q is capped at the documented +/-5%",
          expected_return(9.0, 2.0, p) == p.max_abs_q
          and expected_return(-9.0, 2.0, p) == -p.max_abs_q)
    check("a zero information coefficient produces no tilt",
          expected_return(2.0, 0.3, ViewParams(information_coefficient=0.0)) == 0.0)
    check("non-finite inputs degrade to zero rather than propagating NaN",
          expected_return(float("nan"), 0.2, p) == 0.0
          and expected_return(1.0, float("nan"), p) == 0.0)


def test_q_sign_matches_the_recommendation() -> None:
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado")  # absolute only
    by_ticker = {r.ticker: r for r in scored}

    mismatches = []
    for v in views:
        row = by_ticker[v["activo"]]
        if row.recommendation == "OVERWEIGHT" and v["Q"] <= 0:
            mismatches.append((v["activo"], row.recommendation, v["Q"]))
        if row.recommendation == "UNDERWEIGHT" and v["Q"] >= 0:
            mismatches.append((v["activo"], row.recommendation, v["Q"]))
    check("an Overweight never exports a negative Q, nor an Underweight a positive one",
          not mismatches, f"{mismatches}")


def test_conviction_is_independent_of_magnitude() -> None:
    """
    Two names with the same |z| must differ in conviction when their blocks
    disagree. If conviction tracked only |z|, Omega would carry no information
    the posterior does not already have from Q.
    """
    from screener.scoring import ScoredInstrument

    def make(z: float, blocks: dict[str, float], gates=()) -> ScoredInstrument:
        row = ScoredInstrument(ticker="TST", name="TST", asset_type="STOCK",
                               indices=[], sector=None, raw_metrics={})
        row.composite_z = z
        row.block_scores = blocks
        row.block_coverage = {k: 1.0 for k in blocks}
        row.gates_triggered = list(gates)
        return row

    unanimous = make(1.5, {f"b{i}": 0.9 for i in range(6)})
    split = make(1.5, {"b0": 2.5, "b1": -0.3, "b2": -0.4,
                       "b3": -0.2, "b4": -0.1, "b5": -0.5})

    check("full block agreement scores 1.0", block_agreement(unanimous) == 1.0)
    check("a lone supporting block scores low",
          block_agreement(split) < 0.25, f"{block_agreement(split)}")
    check("agreement raises conviction at identical |z|",
          conviction(unanimous, DEFAULT_PARAMS) > conviction(split, DEFAULT_PARAMS))

    gated = make(1.5, {f"b{i}": 0.9 for i in range(6)}, gates=["RISK: something"])
    check("a triggered risk gate halves conviction",
          abs(conviction(gated, DEFAULT_PARAMS)
              - conviction(unanimous, DEFAULT_PARAMS) * DEFAULT_PARAMS.gate_penalty)
          < 1e-9)

    thin = make(1.5, {f"b{i}": 0.9 for i in range(6)})
    thin.block_coverage = {k: 0.4 for k in thin.block_scores}
    check("thin data coverage lowers conviction",
          conviction(thin, DEFAULT_PARAMS) < conviction(unanimous, DEFAULT_PARAMS))

    neutral = make(float("nan"), {})
    check("an unscored name yields zero conviction",
          conviction(neutral, DEFAULT_PARAMS) == 0.0)


def test_neutral_names_produce_no_view() -> None:
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado")
    named = {v["activo"] for v in views if v["tipo"] == "absoluto"}
    middle = [r.ticker for r in scored
              if abs(r.composite_z) < DEFAULT_PARAMS.min_abs_z]
    check("no view is emitted for a name in the middle of the cross-section",
          not (named & set(middle)), f"{named & set(middle)}")


# --------------------------------------------------------------------------
# Relative views
# --------------------------------------------------------------------------

def test_relative_views_use_the_declared_reference() -> None:
    scored, _, data, _ = screen("Agresivo")
    views = build_views(scored, data, strategy="Agresivo",
                        reference_map=REFERENCIAS,
                        params=ViewParams(max_views=40, min_conviction=0.0,
                                          min_abs_z=0.0))
    relatives = [v for v in views if v["tipo"] == "relativo"]
    check("relative views are produced when a reference is declared",
          len(relatives) > 0)

    wrong = []
    for v in relatives:
        pair = {v["activo_long"], v["activo_short"]}
        match = [t for t, ref in REFERENCIAS.items()
                 if {t, ref} == pair]
        if not match:
            wrong.append(pair)
    check("every relative view pairs a name with its declared reference",
          not wrong, f"{wrong}")

    check("relative Q is always reported positive, with direction in the legs",
          all(v["Q"] > 0 for v in relatives))


def test_relative_views_need_both_legs_scored() -> None:
    """A reference that never entered the cross-section must not produce a view."""
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map={"AAPL": "NOT_IN_UNIVERSE"})
    legs = [v for v in views if v["tipo"] == "relativo"]
    check("an unscored reference falls back rather than emitting a broken pair",
          all("NOT_IN_UNIVERSE" not in (v["activo_long"], v["activo_short"])
              for v in legs))


def test_spread_volatility() -> None:
    _, _, data, _ = screen("Moderado")
    from screener.black_litterman import _returns_by_ticker
    returns = _returns_by_ticker(data)

    pair = spread_volatility("AAPL", "QQQ", returns)
    check("spread volatility is positive and finite",
          pair is not None and pair > 0 and np.isfinite(pair))
    check("a missing leg returns None",
          spread_volatility("AAPL", "NOPE", returns) is None)


# --------------------------------------------------------------------------
# Strategy mapping and schema
# --------------------------------------------------------------------------

def test_strategy_mapping() -> None:
    check("all four CCI strategies map to a profile",
          set(CCI_STRATEGIES) == {"Conservador_Defensivo", "Conservador",
                                  "Moderado", "Agresivo"})
    check("every mapped profile exists",
          all(v in PROFILES for v in CCI_STRATEGIES.values()))
    check("Conservador_Defensivo maps to the most defensive profile",
          profile_for_strategy("Conservador_Defensivo").key == "conservador_defensivo")
    try:
        profile_for_strategy("Balanceado")
        check("an unknown strategy raises", False, "no exception")
    except KeyError as exc:
        check("an unknown strategy raises and lists the valid ones",
              "Moderado" in str(exc))

    try:
        scored, _, data, _ = screen("Moderado")
        build_views(scored, data, strategy="Balanceado")
        check("build_views rejects an unknown strategy", False, "no exception")
    except KeyError:
        check("build_views rejects an unknown strategy", True)


def test_view_schema_matches_cci() -> None:
    scored, _, data, _ = screen("Conservador")
    views = build_views(scored, data, strategy="Conservador",
                        reference_map=REFERENCIAS)

    for v in views:
        base = {"estrategia", "tipo", "Q", "conviccion", "justificacion"}
        extra = ({"activo"} if v["tipo"] == "absoluto"
                 else {"activo_long", "activo_short"})
        if not (base | extra) <= set(v):
            check("view carries every field CCI's approval flow reads",
                  False, f"{sorted(v)}")
            return
    check("view carries every field CCI's approval flow reads", True)
    check("every view is tagged with its strategy",
          all(v["estrategia"] == "Conservador" for v in views))
    check("views are ordered by conviction, highest first",
          all(views[i]["conviccion"] >= views[i + 1]["conviccion"]
              for i in range(len(views) - 1)))
    check("the justification names the assumed IC so the manager sees it",
          all("IC supuesto" in v["justificacion"] for v in views))


def test_view_count_is_capped() -> None:
    scored, _, data, _ = screen("Agresivo")
    views = build_views(scored, data, strategy="Agresivo",
                        params=ViewParams(max_views=3, min_conviction=0.0,
                                          min_abs_z=0.0))
    check("the view count respects the cap", len(views) <= 3, f"got {len(views)}")


# --------------------------------------------------------------------------
# Basket export
# --------------------------------------------------------------------------

def test_basket_schema() -> None:
    scored, _, _, _ = screen("Moderado")
    rows = build_basket(scored, strategy="Moderado", reference_map=REFERENCIAS)

    check("a basket row per scored name", len(rows) == len(scored))
    check("columns match CCI's sheet exactly, in order",
          all(tuple(r) == BASKET_COLUMNS for r in rows),
          f"got {tuple(rows[0])}")

    by_ticker = {r["ticker"]: r for r in rows}
    check("ETFs are labelled with CCI's equity-ETF class",
          by_ticker["SPY"]["clase_activo"] == "ETF_RentaVariable")
    check("single stocks are labelled Equity",
          by_ticker["AAPL"]["clase_activo"] == "Equity")
    check("index membership fills clasificacion_sistema",
          by_ticker["AAPL"]["clasificacion_sistema"] == "indice_mayor")
    check("declared references are carried into activo_referencia",
          by_ticker["AAPL"]["activo_referencia"] == "QQQ")
    check("the screener's call is recorded in notas",
          "screener" in by_ticker["SPY"]["notas"])

    subset = build_basket(scored, strategy="Moderado", include=["AAPL", "SPY"])
    check("include filters the basket", {r["ticker"] for r in subset} == {"AAPL", "SPY"})


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_written_payload_round_trips() -> None:
    scored, meta, data, profile = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)

    with tempfile.TemporaryDirectory() as tmp:
        path = write_views(views, Path(tmp) / default_views_filename("Moderado"),
                           strategy="Moderado", profile=profile, meta=meta)
        payload = json.loads(path.read_text(encoding="utf-8"))

    check("the filename marks the file as a proposal, not approved views",
          path.name.startswith("Moderado_screener_propuestas_")
          and path.suffix == ".json", f"got {path.name}")
    check("the filename cannot be confused with the approval flow's output",
          path.name != f"Moderado_views_{__import__('datetime').date.today()}.json")
    check("payload records the profile used", payload["perfil_screener"] == "moderado")
    check("payload states no account data was read",
          "sin datos de cuenta" in payload["origen"])
    check("payload declares the IC as an assumption, not an estimate",
          "supuesto" in payload["calibracion"]["nota"])
    # Everything CCI reads survives the round trip; the only difference is the
    # internal `_q_bruto` diagnostic, which write_views strips on purpose so
    # the reviewed file carries no field their approval flow does not expect.
    from screener.black_litterman import public_view
    check("the views survive the round trip unchanged",
          payload["views"] == [public_view(v) for v in views])
    check("the round trip drops only the internal diagnostic key",
          all(set(v) - set(p) == {"_q_bruto"}
              for v, p in zip(views, payload["views"])),
          str([set(v) - set(p) for v, p in zip(views, payload["views"])]))

    # The nested list is what CCI's solver consumes; prove it still does.
    import pandas as pd
    cov = covariance(data, [r.ticker for r in scored])
    weights = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    posterior, _ = cci_black_litterman_core(weights, cov, payload["views"])
    check("CCI's solver accepts the views read back from disk",
          bool(np.all(np.isfinite(posterior.values))))


def test_refuses_to_write_into_the_approved_folder() -> None:
    """
    Governance guard. CCI's flujo_aprobacion writes a same-shaped file into
    aprobadas/ once a manager has reviewed and justified each view. Screener
    output landing there would replace a signed-off decision with unreviewed
    machine output, silently.
    """
    scored, meta, data, profile = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado")

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "CCI_BlackLitterman"
        for folder in ("propuestas", "aprobadas"):
            (base / folder).mkdir(parents=True)

        try:
            write_views(views, base / "aprobadas" / "Moderado_views_2026-08-17.json")
            check("writing into aprobadas/ is refused", False, "no exception")
        except ValueError as exc:
            check("writing into aprobadas/ is refused", True)
            check("the error explains why and names the right folder",
                  "propuestas" in str(exc) and "aprob" in str(exc).lower())

        check("no file was created in the protected folder",
              not list((base / "aprobadas").iterdir()))

        # Case variations must not slip past the guard.
        try:
            write_views(views, base / "APROBADAS" / "x.json")
            check("the guard is case-insensitive", False, "no exception")
        except ValueError:
            check("the guard is case-insensitive", True)

        ok = write_views(views, base / "propuestas"
                         / default_views_filename("Moderado"),
                         strategy="Moderado", profile=profile, meta=meta)
        check("writing into propuestas/ succeeds", ok.exists())


def test_cci_side_loader_snippet() -> None:
    """
    The snippet CCI pastes into their notebook is code that will run on their
    side, so it is exercised here against real bridge output rather than
    shipped unrun.
    """
    import datetime
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cci_loader", Path(__file__).resolve().parents[1]
        / "snippets" / "cci_bl_cargar_propuestas.py")
    loader = importlib.util.module_from_spec(spec)

    scored, meta, data, profile = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)

    with tempfile.TemporaryDirectory() as tmp:
        loader.BASE_DIR = tmp
        spec.loader.exec_module(loader)
        loader.CARPETA_PROPUESTAS = str(Path(tmp) / "propuestas")

        write_views(views, Path(loader.CARPETA_PROPUESTAS)
                    / default_views_filename("Moderado"),
                    strategy="Moderado", profile=profile, meta=meta)

        cargadas = loader.cargar_propuestas_screener("Moderado")
        check("the loader reads back every view the bridge wrote",
              len(cargadas) == len(views))
        check("the loader tags each view with its origin",
              all(v["origen"] == "screener" for v in cargadas))
        check("a strategy with no file returns empty rather than raising",
              loader.cargar_propuestas_screener("Agresivo") == [])

        stale = default_views_filename(
            "Conservador", datetime.date.today() - datetime.timedelta(days=30))
        write_views(views, Path(loader.CARPETA_PROPUESTAS) / stale,
                    strategy="Conservador", profile=profile, meta=meta)
        check("a stale file is still loaded (the warning is advisory)",
              len(loader.cargar_propuestas_screener("Conservador")) == len(views))

    # --- fusionar --------------------------------------------------------
    motor = [
        {"tipo": "absoluto", "activo": "SPY", "Q": 0.01, "conviccion": 0.90},
        {"tipo": "relativo", "activo_long": "QQQ", "activo_short": "AAPL",
         "Q": 0.02, "conviccion": 0.30},
        {"tipo": "absoluto", "activo": "GLD", "Q": 0.01, "conviccion": 0.40},
    ]
    screener_side = [
        {"tipo": "absoluto", "activo": "SPY", "Q": -0.02, "conviccion": 0.50,
         "origen": "screener"},
        # Same pair, legs reversed -- still the same bet, must dedupe.
        {"tipo": "relativo", "activo_long": "AAPL", "activo_short": "QQQ",
         "Q": 0.03, "conviccion": 0.70, "origen": "screener"},
        {"tipo": "absoluto", "activo": "TLT", "Q": 0.01, "conviccion": 0.60,
         "origen": "screener"},
    ]
    merged = loader.fusionar(motor, screener_side)

    check("duplicate absolute views collapse to one",
          sum(1 for v in merged if v.get("activo") == "SPY") == 1)
    check("the higher-conviction side wins a duplicate",
          next(v for v in merged if v.get("activo") == "SPY")["conviccion"] == 0.90)
    check("a reversed relative pair is recognised as the same bet",
          sum(1 for v in merged if v["tipo"] == "relativo") == 1)
    check("the reversed pair keeps the higher-conviction leg order",
          next(v for v in merged
               if v["tipo"] == "relativo")["activo_long"] == "AAPL")
    check("non-duplicate views from both sources survive",
          {"GLD", "TLT"} <= {v.get("activo") for v in merged})
    check("merged views are ordered by conviction",
          all(merged[i]["conviccion"] >= merged[i + 1]["conviccion"]
              for i in range(len(merged) - 1)))
    check("max_total caps the merged list",
          len(loader.fusionar(motor, screener_side, max_total=2)) == 2)

    # The merged set must still be something CCI's solver can consume.
    import pandas as pd
    cov = covariance(data, [r.ticker for r in scored])
    weights = pd.Series(1.0 / len(cov.columns), index=cov.columns)
    posterior, _ = cci_black_litterman_core(weights, cov, merged)
    check("CCI's solver accepts the merged view set",
          bool(np.all(np.isfinite(posterior.values))))


def test_no_portfolio_weights_are_exported() -> None:
    """
    The bridge must not ship weights. Black-Litterman owns allocation, and a
    second unconstrained set of weights alongside it invites the exact
    confusion a model-risk review exists to prevent.
    """
    scored, meta, data, profile = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado")
    payload = views_payload(views, strategy="Moderado", profile=profile, meta=meta)
    blob = json.dumps(payload)

    forbidden = ["indicative_weight", "peso_ind", "pesos_objetivo", "weight"]
    present = [t for t in forbidden if t in blob]
    check("no portfolio weight reaches the export", not present, f"found {present}")

    rows = build_basket(scored, strategy="Moderado")
    check("no portfolio weight reaches the basket either",
          not any(t in json.dumps(rows) for t in forbidden))


def main() -> int:
    for fn in [
        test_cci_solver_accepts_the_views,
        test_omega_is_invertible_for_every_view,
        test_expected_return_is_centred_and_risk_scaled,
        test_q_sign_matches_the_recommendation,
        test_conviction_is_independent_of_magnitude,
        test_neutral_names_produce_no_view,
        test_relative_views_use_the_declared_reference,
        test_relative_views_need_both_legs_scored,
        test_spread_volatility,
        test_strategy_mapping,
        test_view_schema_matches_cci,
        test_view_count_is_capped,
        test_basket_schema,
        test_written_payload_round_trips,
        test_refuses_to_write_into_the_approved_folder,
        test_cci_side_loader_snippet,
        test_no_portfolio_weights_are_exported,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
