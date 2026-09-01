"""
Tests for the model-risk diagnostics.

These functions exist to tell a reviewer something uncomfortable, so the tests
are built around whether they actually *fire*. A diagnostic that returns a tidy
number on healthy input and stays quiet on broken input is worse than no
diagnostic at all: it launders the problem it was added to surface.

So the cases below are mostly constructed pathologies -- six blocks that are
secretly one block, a clip that flattens a ranking the model made -- plus the
mirror cases proving the diagnostics stay quiet when nothing is wrong.

Two extra guards protect the boundary with CCI: the private ``_q_bruto`` key
must never reach the JSON a manager signs off on, and CCI's own solver must
still accept the in-memory dicts that carry it.
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
    DEFAULT_PARAMS, ViewParams, build_views, public_view, write_views,
)
from screener.config import FACTOR_MODEL  # noqa: E402
from screener.diagnostics import (  # noqa: E402
    MIN_OBSERVATIONS, REDUNDANT_CORR, block_correlation,
    format_block_correlation, format_view_saturation, run_diagnostics,
    view_label, view_saturation,
)
from screener.profiles import apply_profile, profile_for_strategy  # noqa: E402
from screener.run_screen import run_standalone  # noqa: E402
from screener.scoring import ScoredInstrument  # noqa: E402
from screener.tuning import reset_all  # noqa: E402
from screener.yahoo_adapter import build_market_data  # noqa: E402
from test_black_litterman import (  # noqa: E402
    REFERENCIAS, TICKERS, cci_black_litterman_core, covariance, screen,
)
from test_yahoo_adapter import make_yf_frame  # noqa: E402

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
# Fixtures
# --------------------------------------------------------------------------

def synthetic(block_values: dict[str, list[float]]) -> list[ScoredInstrument]:
    """
    Build scored rows whose block scores are exactly the values given.

    Bypasses the scoring engine on purpose: these tests are about the
    correlation measurement, and feeding it known series is the only way to
    know what the right answer is.
    """
    n = len(next(iter(block_values.values())))
    rows = []
    for i in range(n):
        row = ScoredInstrument(
            ticker=f"T{i:02d}", name=f"Name {i}", asset_type="STOCK",
            indices=[], sector=None, raw_metrics={},
        )
        row.block_scores = {k: v[i] for k, v in block_values.items()}
        row.block_coverage = {k: 1.0 for k in block_values}
        row.composite_z = float(np.mean(list(row.block_scores.values())))
        rows.append(row)
    return rows


def all_block_keys() -> list[str]:
    return [b.key for b in FACTOR_MODEL]


# --------------------------------------------------------------------------
# Block correlation
# --------------------------------------------------------------------------

def test_identical_blocks_collapse_to_one_factor() -> None:
    """Six blocks carrying the same series are one factor, and must say so."""
    rng = np.random.default_rng(7)
    common = rng.normal(size=40).tolist()
    diag = block_correlation(synthetic({k: list(common) for k in all_block_keys()}))

    check("identical blocks -> ~1 effective factor",
          abs(diag.effective_factors - 1.0) < 1e-6,
          f"effective_factors={diag.effective_factors}")

    expected_pairs = diag.n_blocks * (diag.n_blocks - 1) // 2
    check("identical blocks -> every pair flagged redundant",
          len(diag.redundant_pairs) == expected_pairs,
          f"{len(diag.redundant_pairs)} of {expected_pairs}")
    check("identical blocks -> the whole composite is flagged as overlapping",
          abs(diag.overlapping_weight - sum(diag.weights)) < 1e-9,
          f"overlapping={diag.overlapping_weight}, total={sum(diag.weights)}")


def test_independent_blocks_stay_quiet() -> None:
    """The mirror case: uncorrelated blocks must produce no warning."""
    rng = np.random.default_rng(11)
    keys = all_block_keys()
    diag = block_correlation(
        synthetic({k: rng.normal(size=400).tolist() for k in keys})
    )

    check("independent blocks -> no redundant pairs",
          diag.redundant_pairs == (),
          f"flagged {diag.redundant_pairs}")
    check("independent blocks -> effective factors near the block count",
          abs(diag.effective_factors - len(keys)) < 0.35,
          f"effective_factors={diag.effective_factors} for {len(keys)} blocks")
    check("independent blocks -> nothing reported as overlapping weight",
          diag.overlapping_weight == 0.0)


def test_correlation_sign_and_shape() -> None:
    """A perfectly inverted block is still redundant -- |r|, not r."""
    rng = np.random.default_rng(3)
    keys = all_block_keys()
    base = rng.normal(size=60)
    values = {k: rng.normal(size=60).tolist() for k in keys}
    values[keys[0]] = base.tolist()
    values[keys[1]] = (-base).tolist()

    diag = block_correlation(synthetic(values))
    pair = {frozenset((a, b)): c for a, b, c in diag.redundant_pairs}
    key = frozenset((keys[0], keys[1]))

    check("inverted blocks are flagged as redundant", key in pair)
    check("inverted blocks report a negative correlation",
          key in pair and pair[key] < -0.99, f"{pair.get(key)}")
    check("matrix is symmetric",
          np.allclose(diag.matrix, diag.matrix.T, equal_nan=True))
    check("matrix diagonal is 1.0",
          np.allclose(np.diag(diag.matrix), 1.0))


def test_constant_block_is_excluded_not_reported_as_noise() -> None:
    """
    A block with no cross-sectional variance has no ranking to correlate.

    The trap here is that ``np.std`` of fifty identical floats returns ~1.7e-16
    rather than 0.0, so a ``sd > 0`` test classifies the block as live and then
    publishes its floating-point noise as a correlation coefficient. Exactly
    the kind of number that survives review because it looks like data.
    """
    rng = np.random.default_rng(5)
    keys = all_block_keys()
    values = {k: rng.normal(size=50).tolist() for k in keys}
    values[keys[0]] = [0.4] * 50

    diag = block_correlation(synthetic(values))

    check("constant block -> excluded from the matrix",
          keys[0] not in diag.blocks, f"{diag.blocks}")
    check("constant block -> exclusion reported with a reason",
          (keys[0], "sin variación transversal") in diag.excluded_blocks,
          f"{diag.excluded_blocks}")
    check("constant block -> the surviving matrix is fully finite",
          np.isfinite(diag.matrix).all(), f"{diag.matrix}")
    check("constant block -> its noise is not published as a correlation",
          diag.redundant_pairs == (), f"{diag.redundant_pairs}")
    check("constant block -> effective factors still computed",
          np.isfinite(diag.effective_factors), f"{diag.effective_factors}")
    check("constant block -> formatter discloses the exclusion",
          "sin variación transversal" in format_block_correlation(diag))


def test_listwise_deletion_of_partial_rows() -> None:
    """Names missing any measured block are dropped, not filled with zero."""
    rng = np.random.default_rng(13)
    keys = all_block_keys()
    rows = synthetic({k: rng.normal(size=30).tolist() for k in keys})
    for row in rows[:10]:
        del row.block_scores[keys[-1]]

    diag = block_correlation(rows)
    check("a block above the coverage floor stays in the matrix",
          keys[-1] in diag.blocks, f"{diag.blocks}")
    check("partial rows are excluded from the correlation sample",
          diag.n_observations == 20, f"n={diag.n_observations}")


def test_a_thin_block_does_not_silence_the_whole_diagnostic() -> None:
    """
    The failure this coverage floor exists to prevent.

    ``portfolio_fit`` is never scored in standalone mode. Under plain listwise
    deletion over every declared block, requiring it leaves zero usable names
    and the diagnostic reports nothing at all -- a measurement that goes quiet
    precisely when it is being used.
    """
    rng = np.random.default_rng(23)
    keys = all_block_keys()
    rows = synthetic({k: rng.normal(size=40).tolist() for k in keys})
    for row in rows:
        del row.block_scores[keys[-1]]

    diag = block_correlation(rows)
    check("an unscored block is dropped instead of emptying the sample",
          keys[-1] not in diag.blocks, f"{diag.blocks}")
    check("the remaining blocks are still measured on every name",
          diag.n_observations == 40, f"n={diag.n_observations}")
    check("the diagnostic stays reliable", diag.reliable)
    check("the dropped block is disclosed, with its coverage",
          any(k == keys[-1] and "0 de 40" in why
              for k, why in diag.excluded_blocks),
          f"{diag.excluded_blocks}")
    check("declared block count is retained for context",
          diag.n_declared == len(keys) and diag.n_blocks == len(keys) - 1)


def test_thin_sample_refuses_to_report() -> None:
    """Below three usable names there is nothing to correlate."""
    rng = np.random.default_rng(2)
    keys = all_block_keys()
    diag = block_correlation(synthetic({k: rng.normal(size=2).tolist() for k in keys}))

    check("two names -> matrix is all NaN",
          not np.isfinite(diag.matrix).any())
    check("two names -> effective factors is NaN",
          not np.isfinite(diag.effective_factors))
    check("two names -> reported as unreliable", not diag.reliable)
    check("two names -> formatter says so rather than printing a matrix",
          "No se reporta" in format_block_correlation(diag))

    thin = block_correlation(synthetic({k: rng.normal(size=MIN_OBSERVATIONS - 1).tolist()
                                        for k in keys}))
    check(f"below {MIN_OBSERVATIONS} names -> still withheld", not thin.reliable)


def test_empty_input() -> None:
    diag = block_correlation([])
    check("empty input does not raise", diag.n_observations == 0)
    check("empty input formats cleanly",
          isinstance(format_block_correlation(diag), str))


def test_weights_follow_the_active_profile() -> None:
    """
    The overlapping-weight figure must reflect the profile in force.

    ``apply_profile`` rebinds ``FACTOR_MODEL`` through :mod:`screener.tuning`;
    if this module had captured the default model at import, an Aggressive run
    would be reported against Moderate weights and nobody would notice.
    """
    rng = np.random.default_rng(17)
    try:
        profile = apply_profile(profile_for_strategy("Agresivo"))
        keys = [b.key for b in profile.model()]
        common = rng.normal(size=40).tolist()
        diag = block_correlation(synthetic({k: list(common) for k in keys}))

        check("profile model drops portfolio_fit from the diagnostic",
              "portfolio_fit" not in diag.blocks, f"{diag.blocks}")
        check("profile block weights are the ones reported",
              all(abs(w - profile.block_weights[k]) < 1e-9
                  for k, w in zip(diag.blocks, diag.weights)),
              f"{dict(zip(diag.blocks, diag.weights))}")
        check("overlapping weight sums the profile's weights, not config's",
              abs(diag.overlapping_weight - 1.0) < 1e-9,
              f"{diag.overlapping_weight}")
    finally:
        reset_all()


def test_real_screen_produces_a_usable_matrix() -> None:
    """End to end on the same fixture the rest of the suite screens."""
    scored, _, _, _ = screen("Moderado")
    diag = block_correlation([r for r in scored if r.eligible])

    check("real screen -> correlation reported on enough names", diag.reliable,
          f"n={diag.n_observations}")
    finite = diag.matrix[np.isfinite(diag.matrix)]
    check("real screen -> every correlation is inside [-1, 1]",
          finite.size > 0 and np.all(np.abs(finite) <= 1.0 + 1e-9))
    check("real screen -> effective factors between 1 and the block count",
          1.0 - 1e-9 <= diag.effective_factors <= diag.n_blocks + 1e-9,
          f"{diag.effective_factors}")
    text = format_block_correlation(diag)
    check("real screen -> formatter names every block",
          all(k[:7] in text for k in diag.blocks))
    print("    " + text.replace("\n", "\n    "))


# --------------------------------------------------------------------------
# View saturation
# --------------------------------------------------------------------------

def test_no_clipping_reports_nothing() -> None:
    params = ViewParams(information_coefficient=0.001)
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS, params=params)
    diag = view_saturation(views, params)

    check("tiny IC -> no view reaches the cap", diag.n_at_cap == 0,
          f"{diag.n_at_cap} of {diag.n_views}")
    check("tiny IC -> no overshoot", abs(diag.max_overshoot - 1.0) < 1e-9,
          f"{diag.max_overshoot}")
    check("tiny IC -> no rank collapse", not diag.rank_collapse)
    check("tiny IC -> formatter says the cap is not binding",
          "no está limitando" in format_view_saturation(diag))


def test_large_ic_saturates_and_is_reported() -> None:
    params = ViewParams(information_coefficient=5.0)
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS, params=params)

    check("fixture generated views to measure", len(views) > 2, f"{len(views)}")
    diag = view_saturation(views, params)

    check("absurd IC -> views pile onto the cap", diag.n_at_cap == diag.n_views,
          f"{diag.n_at_cap} of {diag.n_views}")
    check("absurd IC -> share at cap is 1.0", abs(diag.share_at_cap - 1.0) < 1e-9)
    check("absurd IC -> overshoot is reported as a multiple of the cap",
          diag.max_overshoot > 2.0, f"{diag.max_overshoot}")
    check("absurd IC -> rank collapse detected", diag.rank_collapse)
    text = format_view_saturation(diag)
    check("absurd IC -> formatter raises the alarm", "ATENCIÓN" in text)
    check("absurd IC -> formatter names the correction",
          "information coefficient" in text)
    check("absurd IC -> formatter shows the pre-clip value",
          "sin recorte" in text)


def test_rank_collapse_needs_a_lost_ranking() -> None:
    """Views clipped from near-identical raw values lost nothing."""
    cap = DEFAULT_PARAMS.max_abs_q
    together = [
        {"tipo": "absoluto", "activo": "AAA", "Q": cap, "_q_bruto": cap * 1.02},
        {"tipo": "absoluto", "activo": "BBB", "Q": cap, "_q_bruto": cap * 1.03},
    ]
    apart = [
        {"tipo": "absoluto", "activo": "AAA", "Q": cap, "_q_bruto": cap * 1.02},
        {"tipo": "absoluto", "activo": "BBB", "Q": cap, "_q_bruto": cap * 4.00},
    ]
    check("two views clipped from the same place is not a collapse",
          not view_saturation(together, DEFAULT_PARAMS).rank_collapse)
    check("two views clipped from far apart is a collapse",
          view_saturation(apart, DEFAULT_PARAMS).rank_collapse)
    check("a single clipped view is never a collapse",
          not view_saturation(apart[:1], DEFAULT_PARAMS).rank_collapse)


def test_negative_views_count_against_the_cap() -> None:
    """The cap is two-sided; an underweight at -5% is just as saturated."""
    cap = DEFAULT_PARAMS.max_abs_q
    views = [{"tipo": "absoluto", "activo": "AAA", "Q": -cap, "_q_bruto": -cap * 3}]
    diag = view_saturation(views, DEFAULT_PARAMS)
    check("a view at the negative cap is counted", diag.n_at_cap == 1)
    check("overshoot uses magnitude", abs(diag.max_overshoot - 3.0) < 1e-9,
          f"{diag.max_overshoot}")


def test_foreign_views_do_not_invent_an_overshoot() -> None:
    """
    Views from CCI's own generator carry no ``_q_bruto``. The diagnostic must
    understate the clipping rather than fabricate a pre-clip number.
    """
    cap = DEFAULT_PARAMS.max_abs_q
    views = [{"tipo": "absoluto", "activo": "AAA", "Q": cap},
             {"tipo": "relativo", "activo_long": "BBB", "activo_short": "CCC",
              "Q": 0.01}]
    diag = view_saturation(views, DEFAULT_PARAMS)
    check("foreign views -> saturation still counted", diag.n_at_cap == 1)
    check("foreign views -> no overshoot invented",
          abs(diag.max_overshoot - 1.0) < 1e-9, f"{diag.max_overshoot}")
    check("relative views are labelled by both legs",
          view_label(views[1]) == "BBB/CCC", view_label(views[1]))
    check("absolute views are labelled by their ticker",
          view_label(views[0]) == "AAA")


def test_empty_views() -> None:
    diag = view_saturation([], DEFAULT_PARAMS)
    check("no views -> share at cap is 0, not a division by zero",
          diag.share_at_cap == 0.0)
    check("no views -> formatter says so",
          "no se generó" in format_view_saturation(diag))


# --------------------------------------------------------------------------
# The private diagnostic key must not cross the CCI boundary
# --------------------------------------------------------------------------

def test_q_bruto_is_attached_in_memory() -> None:
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)
    check("build_views attaches the pre-clip Q",
          all("_q_bruto" in v for v in views))
    check("the pre-clip Q has the same sign as Q",
          all(np.sign(v["Q"]) == np.sign(v["_q_bruto"]) or v["Q"] == 0
              for v in views))
    check("the pre-clip Q is never smaller in magnitude than Q",
          all(abs(v["_q_bruto"]) >= abs(v["Q"]) - 1e-12 for v in views),
          str([(v["Q"], v["_q_bruto"]) for v in views]))


def test_q_bruto_never_reaches_the_json() -> None:
    scored, meta, data, profile = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)

    check("public_view strips private keys",
          all(not k.startswith("_") for k in public_view(views[0])))
    check("public_view keeps everything else",
          set(public_view(views[0]))
          == set(views[0]) - {"_q_bruto", "_pairing"})

    with tempfile.TemporaryDirectory() as tmp:
        path = write_views(views, Path(tmp) / "propuestas" / "p.json",
                           strategy="Moderado", profile=profile, meta=meta)
        payload = json.loads(path.read_text(encoding="utf-8"))
    check("written JSON carries no private keys",
          all(not k.startswith("_") for v in payload["views"] for k in v),
          str(payload["views"][:1]))
    check("written JSON still carries every view", len(payload["views"]) == len(views))

    with tempfile.TemporaryDirectory() as tmp:
        bare = write_views(views, Path(tmp) / "bare.json")
        bare_payload = json.loads(bare.read_text(encoding="utf-8"))
    check("the no-provenance write path also strips private keys",
          all(not k.startswith("_") for v in bare_payload["views"] for k in v))


def test_cci_solver_accepts_views_carrying_diagnostics() -> None:
    """The in-memory dicts keep the extra key -- CCI's solver must not care."""
    import pandas as pd

    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)
    tickers = sorted({t for v in views
                      for t in ([v["activo"]] if v["tipo"] == "absoluto"
                                else [v["activo_long"], v["activo_short"]])})
    cov = covariance(data, tickers)
    weights = pd.Series(1.0 / len(tickers), index=cov.columns)

    posterior, _ = cci_black_litterman_core(weights, cov, views=views)
    check("CCI's solver runs on views that carry _q_bruto",
          bool(np.isfinite(posterior.values).all()),
          str(posterior.to_dict()))


def test_combined_report() -> None:
    scored, _, data, _ = screen("Moderado")
    views = build_views(scored, data, strategy="Moderado",
                        reference_map=REFERENCIAS)
    text = run_diagnostics(scored, views, DEFAULT_PARAMS)
    check("combined report contains both measurements",
          "Correlación entre bloques" in text and "Saturación de views" in text)
    print("    " + text.replace("\n", "\n    "))


def main() -> int:
    for fn in [
        test_identical_blocks_collapse_to_one_factor,
        test_independent_blocks_stay_quiet,
        test_correlation_sign_and_shape,
        test_constant_block_is_excluded_not_reported_as_noise,
        test_listwise_deletion_of_partial_rows,
        test_a_thin_block_does_not_silence_the_whole_diagnostic,
        test_thin_sample_refuses_to_report,
        test_empty_input,
        test_weights_follow_the_active_profile,
        test_real_screen_produces_a_usable_matrix,
        test_no_clipping_reports_nothing,
        test_large_ic_saturates_and_is_reported,
        test_rank_collapse_needs_a_lost_ranking,
        test_negative_views_count_against_the_cap,
        test_foreign_views_do_not_invent_an_overshoot,
        test_empty_views,
        test_q_bruto_is_attached_in_memory,
        test_q_bruto_never_reaches_the_json,
        test_cci_solver_accepts_views_carrying_diagnostics,
        test_combined_report,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
