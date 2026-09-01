"""
Bridge from the screener to CCI's Black-Litterman tactical allocation system.

The boundary
------------
The two systems are complementary, not substitutes, and the line between them
is sharp: **the screener decides what deserves a view and how strong it is; the
Black-Litterman engine decides the weights.**

Concretely, the screener supplies the tactical inputs -- which names to hold a
view on, the expected excess return ``Q``, and the confidence in it -- while
CCI's system keeps ownership of the market equilibrium ``pi``, the Bayesian
posterior, the regulatory bands and the constrained optimization. Nothing here
proposes a portfolio weight. The screener's own ``indicative_weight`` is
deliberately *not* exported: under Black-Litterman the weights are an output of
the optimizer subject to CCI's Investment Procedure, and shipping a second,
unconstrained set of weights alongside them would invite exactly the confusion
a model-risk review exists to prevent.

Why the screener's signal and not the existing factor module
------------------------------------------------------------
CCI's factor dimension scores momentum with a three-valued step function
(``+0.5`` above 5%, ``-0.5`` below -5%, else 0) and computes a volatility ratio
it never uses. The screener scores the same names on 25 metrics across six
blocks, winsorized and z-scored *cross-sectionally*, with ETF and single-stock
cohorts standardized separately, and with risk gates applied afterwards. The
composite z is a materially richer ranking of the same universe.

Translating a ranking into a return forecast
--------------------------------------------
A cross-sectional z-score is a *ranking*, not an expected return, and the two
have different units. The conversion here is explicit:

    Q_i = ic * z_i * sigma_i        (clipped to +/- max_abs_q)

where ``sigma_i`` is the name's annualized volatility and ``ic`` is an assumed
information coefficient. Two properties make this the right shape for a
mean-variance optimizer:

* It is **risk-scaled**. At equal ranking, a more volatile name carries a larger
  expected return. A flat mapping would make the optimizer systematically
  overweight low-volatility names for the same conviction, since it divides
  expected return by risk.
* It is **centered**. A name at the middle of the cross-section (``z = 0``) gets
  ``Q = 0`` exactly, so a neutral screen produces no tilt at all.

``ic`` is a judgement call, not an estimate: it is the assumed correlation
between the screener's ranking and realized returns. The default of 0.08 is
deliberately modest and produces views inside the +/-5% band CCI's technical
document specifies. It is not calibrated against any backtest, and the
docstring says so because a hand-picked constant presented as an estimate is
how model risk enters a firm.

Conviction is not signal strength
---------------------------------
The magnitude of the call belongs in ``Q``. Conviction feeds ``Omega``, the
variance of the view, and so must express *confidence in the estimate*:

* how many of the six blocks agree in sign with the composite,
* how much data actually backed those blocks (coverage),
* whether a risk gate fired on the name.

A name at ``z = +1.5`` carried by one block is a different object from one where
all six agree, even though the two would have identical ``Q``. Collapsing that
distinction into a single number is what makes ``Omega`` meaningless.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metrics import align_returns, simple_returns
from .profiles import CCI_STRATEGIES, RiskProfile, profile_for_strategy
from .scoring import ScoredInstrument

#: Column set CCI's Google Sheet tabs require, in their order.
BASKET_COLUMNS: tuple[str, ...] = (
    "ticker", "nombre", "clase_activo", "categoria_banda",
    "clasificacion_sistema", "min_30_posiciones", "activo_referencia", "notas",
)

#: Asset-class labels used by CCI's REGULACIONES bands.
CLASE_ETF_RV = "ETF_RentaVariable"
CLASE_EQUITY = "Equity"


@dataclass(frozen=True)
class ViewParams:
    """
    Calibration of the ranking-to-view translation.

    Every field here is a judgement call. None is estimated from data.
    """

    #: Assumed correlation between the screener's ranking and realized returns.
    #: Higher values produce larger tactical tilts.
    information_coefficient: float = 0.08

    #: Hard cap on |Q|. CCI's technical document specifies views calibrated to
    #: +/-5%; this enforces it rather than trusting the scaling to stay inside.
    max_abs_q: float = 0.05

    #: Views below this conviction are dropped, matching the filter CCI's
    #: generator already applies to avoid churn and transaction costs.
    min_conviction: float = 0.20

    #: Ceiling on conviction. Never 1.0: a screen built on ~52 weekly bars does
    #: not justify claiming a view is certain.
    max_conviction: float = 0.85

    #: Most views to emit, highest conviction first. CCI's own generator caps
    #: at 8, or 3 when the macro module flags divergence.
    max_views: int = 8

    #: Only names whose |composite z| clears this get a view at all. Names in
    #: the middle of the cross-section carry no information.
    min_abs_z: float = 0.35

    #: Multiplier applied to conviction when any risk gate fired on the name.
    gate_penalty: float = 0.50

    #: Look for a peer when ``reference_map`` does not name one. Off restores
    #: the previous behaviour exactly: a name with no declared reference gets
    #: an absolute view.
    auto_pair: bool = True

    #: A pairing is only accepted if expressing the call as a spread cuts the
    #: volatility of the expression by at least this fraction against holding
    #: the leg outright. This is the whole test: a relative view exists to
    #: remove common risk, and if the spread is not quieter than the leg, the
    #: pairing added risk instead of hedging it -- while *raising* Q, because Q
    #: scales with the volatility it is given.
    min_hedge_benefit: float = 0.15

    #: Peers correlated above this are the same bet in two wrappers. Their
    #: spread is noise around zero, and a view on it asserts a distinction the
    #: data does not support.
    max_pair_corr: float = 0.95


DEFAULT_PARAMS = ViewParams()


# --------------------------------------------------------------------------
# Signal translation
# --------------------------------------------------------------------------

def raw_expected_return(z: float, volatility: float, params: ViewParams) -> float:
    """
    The ``ic * z * sigma`` product *before* the ``max_abs_q`` clip.

    Kept separate so :mod:`screener.diagnostics` can report how hard the clip
    is biting. A clip that binds on most views has quietly replaced the
    screener's ranking with a constant, and that is invisible if only the
    post-clip number is ever recorded.
    """
    if not np.isfinite(z) or not np.isfinite(volatility) or volatility <= 0:
        return 0.0
    return float(params.information_coefficient * z * volatility)


def expected_return(z: float, volatility: float, params: ViewParams) -> float:
    """
    Convert a cross-sectional z-score into an expected excess return.

    Risk-scaled and centered -- see the module docstring for why both matter.
    """
    raw = raw_expected_return(z, volatility, params)
    return float(np.clip(raw, -params.max_abs_q, params.max_abs_q))


def block_agreement(row: ScoredInstrument) -> float:
    """
    Share of populated blocks whose sign matches the composite, in [0, 1].

    This is the part of conviction that cannot be read off the composite: two
    names at the same z are not equally reliable if one is carried by a single
    block and the other has six blocks pointing the same way.
    """
    scores = [v for v in row.block_scores.values() if np.isfinite(v)]
    if not scores:
        return 0.0
    direction = np.sign(row.composite_z) if np.isfinite(row.composite_z) else 0.0
    if direction == 0:
        return 0.0
    return float(sum(1 for s in scores if np.sign(s) == direction) / len(scores))


def mean_coverage(row: ScoredInstrument) -> float:
    """Average share of each block's metric weight that was actually available."""
    values = [v for v in row.block_coverage.values() if np.isfinite(v)]
    return float(np.mean(values)) if values else 0.0


def conviction(row: ScoredInstrument, params: ViewParams) -> float:
    """
    Confidence in the view, in ``[0, max_conviction]``.

    Deliberately independent of ``Q``: magnitude belongs in the return, not in
    the variance of the view.

    Built as a ceiling scaled by factors that are each bounded by 1.0, rather
    than as a product clipped at the end. The difference is not cosmetic: with a
    trailing clip, a name with a large ``|z|`` and unanimous blocks saturates the
    ceiling, and every penalty below it -- a fired risk gate, missing data --
    stops changing the answer. A gate must always reduce conviction, including
    on the strongest names, which are precisely the ones a gate exists to
    restrain.
    """
    if not np.isfinite(row.composite_z):
        return 0.0

    # Saturating in |z| -- a z of 3 is not three times as trustworthy as a z
    # of 1; both say "clearly one tail", and the extra magnitude is already
    # expressed in Q.
    strength = float(np.tanh(abs(row.composite_z) / 1.5))

    # Unanimity keeps the full ceiling; a lone supporting block halves it.
    agreement = 0.5 + 0.5 * block_agreement(row)

    # Full coverage keeps the ceiling; a fully missing block set costs 40%.
    coverage = 0.6 + 0.4 * mean_coverage(row)

    gate = params.gate_penalty if row.gates_triggered else 1.0

    value = params.max_conviction * strength * agreement * coverage * gate
    return float(np.clip(value, 0.0, params.max_conviction))


# --------------------------------------------------------------------------
# Relative views
# --------------------------------------------------------------------------

def _returns_by_ticker(market_data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for inst in market_data.get("instruments", []):
        ticker = (inst.get("ticker") or "").upper()
        closes = [c for c in ((inst.get("history") or {}).get("close") or [])
                  if isinstance(c, (int, float))]
        if ticker and len(closes) > 2:
            out[ticker] = simple_returns(np.asarray(closes, dtype=float))
    return out


def spread_volatility(ticker: str, reference: str,
                      returns: Mapping[str, np.ndarray],
                      periods_per_year: int = 52) -> float | None:
    """
    Annualized volatility of the long/short return spread.

    A relative view's expected return must be scaled by the volatility of the
    *pair*, not of the leg. Two names that move together have a thin spread and
    cannot support the same Q as two unrelated ones at the same rank distance.
    """
    a, b = returns.get(ticker), returns.get(reference)
    if a is None or b is None:
        return None
    x, y = align_returns(a, b)
    if x.size < 12:
        return None
    spread = x - y
    sd = float(np.std(spread, ddof=1))
    return sd * float(np.sqrt(periods_per_year)) if sd > 0 else None


def pair_correlation(ticker: str, reference: str,
                     returns: Mapping[str, np.ndarray]) -> float | None:
    """Correlation of the two legs' return series, or None if not computable."""
    a, b = returns.get(ticker), returns.get(reference)
    if a is None or b is None:
        return None
    x, y = align_returns(a, b)
    if x.size < 12 or np.std(x) == 0 or np.std(y) == 0:
        return None
    c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else None


def hedge_benefit(row: ScoredInstrument, peer: ScoredInstrument,
                  returns: Mapping[str, np.ndarray]) -> float | None:
    """
    Fraction of the leg's volatility that pairing removes, or None.

    ``1 - sigma_spread / sigma_leg``. Positive means the spread is quieter than
    holding ``row`` outright, which is the only reason to express a call as a
    pair. Negative means the pairing *added* variance -- two loosely related
    names bundled together rather than hedged -- and that is the case worth
    refusing, because ``Q = ic * z * sigma`` would then hand the optimizer a
    larger number for a worse-founded opinion.
    """
    leg_vol = row.raw_metrics.get("volatility_1y")
    if leg_vol is None or not np.isfinite(leg_vol) or leg_vol <= 0:
        return None
    spread_vol = spread_volatility(row.ticker, peer.ticker, returns)
    if spread_vol is None or spread_vol <= 0:
        return None
    return float(1.0 - spread_vol / leg_vol)


def find_peer(row: ScoredInstrument, candidates: Sequence[ScoredInstrument],
              returns: Mapping[str, np.ndarray],
              params: ViewParams = DEFAULT_PARAMS) -> ScoredInstrument | None:
    """
    Best counterparty for a relative view on ``row``, or None to stay absolute.

    Every candidate must clear three tests, and each rules out a distinct way an
    automatic pairing goes wrong:

    * **It has to hedge.** ``hedge_benefit`` at or above ``min_hedge_benefit``.
      Without this the search would happily pair unrelated names, and because
      variance adds, the spread would be *more* volatile than the leg -- which
      inflates ``Q`` rather than shrinking it. The safeguard has to come first
      or the search optimizes against it.
    * **It cannot be the same bet.** Correlation at or below ``max_pair_corr``,
      and never a name already flagged in ``row.duplicates``. A spread between
      two wrappers of one index has no signal to express.
    * **The ranking has to survive the subtraction.** ``|z_row - z_peer|`` must
      still clear ``min_abs_z`` -- the same bar an absolute view has to clear.
      Pairing two names the screener ranked alike says nothing, however well the
      pair hedges.

    Among survivors, the peer with the **largest hedge benefit** wins, with ETFs
    preferred over single stocks at equal benefit. Selecting on hedge quality is
    deliberate: ranking by ``Q`` instead would systematically pick the loosest
    pairing available, since ``Q`` grows with spread volatility. The tie-break
    towards ETFs follows what the hand-written ``reference_map`` already does --
    a sector ETF is a cleaner statement of "the market leg" than another single
    name carrying its own idiosyncratic risk.
    """
    if not np.isfinite(row.composite_z):
        return None

    excluded = {row.ticker, *row.duplicates}
    best: tuple[float, int, str] | None = None
    winner: ScoredInstrument | None = None

    for cand in candidates:
        if cand.ticker in excluded or not np.isfinite(cand.composite_z):
            continue
        if abs(row.composite_z - cand.composite_z) < params.min_abs_z:
            continue

        benefit = hedge_benefit(row, cand, returns)
        if benefit is None or benefit < params.min_hedge_benefit:
            continue

        corr = pair_correlation(row.ticker, cand.ticker, returns)
        if corr is None or corr > params.max_pair_corr:
            continue

        # Sort key: benefit, then ETF preference, then ticker for determinism.
        key = (benefit, 1 if cand.asset_type == "ETF" else 0, cand.ticker)
        if best is None or key[:2] > best[:2] or (key[:2] == best[:2] and key[2] < best[2]):
            best, winner = key, cand

    return winner


# --------------------------------------------------------------------------
# View construction
# --------------------------------------------------------------------------

def build_views(scored: Sequence[ScoredInstrument],
                market_data: Mapping[str, Any], *,
                strategy: str,
                reference_map: Mapping[str, str] | None = None,
                pair_pool: Iterable[str] | None = None,
                params: ViewParams = DEFAULT_PARAMS) -> list[dict]:
    """
    Translate a scored cross-section into Black-Litterman views.

    Emits dicts in exactly the shape CCI's ``flujo_aprobacion`` and
    ``black_litterman_core`` already consume, so the receiving notebook needs no
    change to accept them.

    ``reference_map`` pairs a name with the ETF it should be scored against,
    matching the ``activo_referencia`` column of CCI's basket sheet. A declared
    reference always wins. When there is none -- or the declared one is not in
    the cross-section -- :func:`find_peer` looks for one, and falls back to an
    absolute view if nothing qualifies. Set ``params.auto_pair = False`` to
    restore the previous behaviour, where only declared pairs went relative.

    ``pair_pool`` limits which tickers may serve as the found peer, and callers
    should pass the optimizer's basket. :func:`screener.optimizer.posterior`
    silently drops any view naming a ticker outside the covariance universe, so
    a peer that never reaches the basket does not merely weaken the view -- it
    deletes it, and the name burns one of the ``max_views`` slots on the way
    out. Left as None the whole cross-section is eligible, which is only safe
    when the caller intends to optimize over all of it.
    """
    if strategy not in CCI_STRATEGIES:
        raise KeyError(
            f"Estrategia CCI desconocida: {strategy!r}. "
            f"Opciones: {sorted(CCI_STRATEGIES)}"
        )

    reference_map = {k.upper(): v.upper()
                     for k, v in (reference_map or {}).items()}
    returns = _returns_by_ticker(market_data)
    by_ticker = {r.ticker: r for r in scored}

    if pair_pool is None:
        pool = list(scored)
    else:
        wanted = {t.upper() for t in pair_pool}
        pool = [r for r in scored if r.ticker in wanted]

    candidates: list[dict] = []
    for row in scored:
        if not np.isfinite(row.composite_z) or abs(row.composite_z) < params.min_abs_z:
            continue

        conv = conviction(row, params)
        if conv < params.min_conviction:
            continue

        vol = row.raw_metrics.get("volatility_1y")
        reference = reference_map.get(row.ticker)
        peer = by_ticker.get(reference) if reference else None
        pairing = "declarado" if peer is not None else ""

        if peer is None and params.auto_pair:
            peer = find_peer(row, pool, returns, params)
            if peer is not None:
                pairing = "automatico"

        if peer is not None and peer.ticker != row.ticker:
            spread_vol = spread_volatility(row.ticker, peer.ticker, returns)
            if spread_vol is None:
                continue
            z_spread = row.composite_z - peer.composite_z
            q = expected_return(z_spread, spread_vol, params)
            if q == 0.0:
                continue
            long_leg, short_leg = ((row.ticker, peer.ticker) if q > 0
                                   else (peer.ticker, row.ticker))
            candidates.append({
                "estrategia": strategy,
                "tipo": "relativo",
                "activo_long": long_leg,
                "activo_short": short_leg,
                "Q": abs(q),
                "conviccion": round(conv, 4),
                "justificacion": _rationale(row, params, peer=peer,
                                            q=abs(q), conv=conv,
                                            pairing=pairing,
                                            benefit=hedge_benefit(row, peer, returns)),
                "origen": "screener",
                "_q_bruto": abs(raw_expected_return(z_spread, spread_vol, params)),
                "_pairing": pairing,
            })
        else:
            if vol is None:
                continue
            q = expected_return(row.composite_z, vol, params)
            if q == 0.0:
                continue
            candidates.append({
                "estrategia": strategy,
                "tipo": "absoluto",
                "activo": row.ticker,
                "Q": q,
                "conviccion": round(conv, 4),
                "justificacion": _rationale(row, params, q=q, conv=conv),
                "origen": "screener",
                "_q_bruto": raw_expected_return(row.composite_z, vol, params),
                "_pairing": "",
            })

    candidates.sort(key=lambda v: v["conviccion"], reverse=True)
    return candidates[:params.max_views]


def _rationale(row: ScoredInstrument, params: ViewParams, *,
               peer: ScoredInstrument | None = None,
               q: float = 0.0, conv: float = 0.0,
               pairing: str = "", benefit: float | None = None) -> str:
    """
    Human-readable justification, written for the approval screen.

    States the drivers *and* what weakened the view, because a manager choosing
    between approve and edit needs the second more than the first.
    """
    drivers = sorted((kv for kv in row.block_scores.items() if np.isfinite(kv[1])),
                     key=lambda kv: abs(kv[1]), reverse=True)[:3]
    detail = ", ".join(f"{k} {v:+.2f}" for k, v in drivers)

    head = (f"Screener {row.recommendation}: score {row.score_0_100:.0f}/100, "
            f"z {row.composite_z:+.2f}")
    if peer is not None:
        head += f", vs {peer.ticker} (z {peer.composite_z:+.2f})"

    parts = [f"{head}. Bloques dominantes: {detail}."]

    # A manager approving a pair needs to know who chose it. A declared
    # reference is a house decision; a found one is the model's, and the
    # number that justified it belongs on the same screen.
    if peer is not None and pairing == "automatico":
        extra = (f" El par reduce la volatilidad de la expresión {benefit:.0%} "
                 f"frente a tomar {row.ticker} suelto." if benefit is not None else "")
        parts.append(
            f"PAR AUTOMÁTICO: {peer.ticker} no está en REFERENCIAS; lo eligió el "
            f"modelo por cobertura entre los nombres de la cesta.{extra}"
        )
    elif peer is not None and pairing == "declarado":
        parts.append(f"Par declarado en REFERENCIAS: {peer.ticker}.")
    parts.append(
        f"Q = {params.information_coefficient:.2f} (IC supuesto) x z x volatilidad "
        f"=> {q:+.2%}; convicción {conv:.2f}."
    )

    agreement = block_agreement(row)
    if agreement < 0.6:
        parts.append(
            f"ATENCIÓN: solo {agreement:.0%} de los bloques coincide con el signo "
            "del compuesto; convicción reducida."
        )
    coverage = mean_coverage(row)
    if coverage < 0.9:
        parts.append(f"Cobertura media de datos {coverage:.0%}.")
    if row.gates_triggered:
        parts.append("Gate de riesgo activo: " + "; ".join(row.gates_triggered))
    if row.duplicates:
        parts.append("Exposición duplicada con " + ", ".join(row.duplicates) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Basket export
# --------------------------------------------------------------------------

def build_basket(scored: Sequence[ScoredInstrument], *,
                 strategy: str,
                 reference_map: Mapping[str, str] | None = None,
                 include: Iterable[str] | None = None) -> list[dict]:
    """
    Rows for a CCI basket tab, in the sheet's exact column order.

    The screener has already applied the eligibility rules of the profile that
    matches this strategy -- liquidity floor, minimum price, index membership,
    and the leveraged / inverse / option-income product exclusions -- so every
    row emitted here has cleared them. ``clasificacion_sistema`` is filled from
    the screener's index membership, which is an exact lookup rather than the
    scraped ``Index`` field CCI's Finviz check parses.
    """
    reference_map = {k.upper(): v.upper()
                     for k, v in (reference_map or {}).items()}
    wanted = {t.upper() for t in include} if include is not None else None

    rows: list[dict] = []
    for row in scored:
        if wanted is not None and row.ticker not in wanted:
            continue
        clase = CLASE_ETF_RV if row.asset_type == "ETF" else CLASE_EQUITY
        indices = [i for i in row.indices if i != "ETF"]
        rows.append({
            "ticker": row.ticker,
            "nombre": row.name,
            "clase_activo": clase,
            "categoria_banda": clase,
            "clasificacion_sistema": "indice_mayor" if indices else "fuera_indice",
            "min_30_posiciones": "",
            "activo_referencia": reference_map.get(row.ticker, ""),
            "notas": (f"screener {row.recommendation}; score "
                      f"{row.score_0_100:.0f}/100; z {row.composite_z:+.2f}"),
        })
    return rows


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def public_view(view: Mapping[str, Any]) -> dict:
    """
    A view with its internal diagnostic keys removed.

    Keys prefixed with ``_`` (currently ``_q_bruto`` and ``_pairing``) exist for
    :mod:`screener.diagnostics` and never leave the process. The file CCI reads
    must contain exactly the fields their approval flow and
    ``black_litterman_core`` expect -- an unexpected key in a JSON a manager
    signs off on is a question nobody should have to answer.
    """
    return {k: v for k, v in view.items() if not k.startswith("_")}


def views_payload(views: Sequence[dict], *, strategy: str,
                  profile: RiskProfile, meta: Mapping[str, Any],
                  params: ViewParams = DEFAULT_PARAMS) -> dict:
    """Wrap views with the provenance a model-risk review will ask for."""
    return {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "estrategia": strategy,
        "perfil_screener": profile.key,
        "origen": "screener standalone (sin datos de cuenta)",
        "fuente_datos": meta.get("data_source"),
        "benchmark": meta.get("benchmark"),
        "universo_puntuado": meta.get("n_eligible"),
        "calibracion": {
            "information_coefficient": params.information_coefficient,
            "max_abs_q": params.max_abs_q,
            "min_conviccion": params.min_conviction,
            "max_conviccion": params.max_conviction,
            "min_abs_z": params.min_abs_z,
            "nota": ("IC es un supuesto declarado, no una estimación "
                     "calibrada contra backtest."),
        },
        "views": [public_view(v) for v in views],
    }


#: Subfolder of CCI's Drive tree that receives screener output.
#:
#: ``propuestas``, never ``aprobadas``. The distinction is the governance
#: boundary of the whole system: the screener emits *proposals*, and only the
#: manager's approval flow may write to ``aprobadas``. Their
#: ``flujo_aprobacion`` writes ``{estrategia}_views_{fecha}.json`` there after a
#: human has reviewed, edited and justified each view -- an unreviewed file
#: landing on that path would replace signed-off decisions with machine output
#: and leave no trace that it happened.
DRIVE_PROPOSALS_DIR = "propuestas"

#: Directory name that screener output must never be written into.
PROTECTED_DIR = "aprobadas"


def write_views(views: Sequence[dict], path: str | Path, **kwargs: Any) -> Path:
    """
    Write the proposals JSON.

    The ``views`` list inside is exactly what CCI's ``black_litterman_core``
    accepts, so the file can be read straight into that function.

    Refuses to write anywhere under an ``aprobadas`` directory. That is a
    guard, not a convention: CCI's approval flow writes a file of the same
    shape there once a manager has reviewed and justified each view, and a
    silent overwrite would substitute unreviewed machine output for a signed-off
    decision with nothing in the record to show it.
    """
    path = Path(path)
    if PROTECTED_DIR in {p.lower() for p in path.parts}:
        raise ValueError(
            f"Se intentó escribir propuestas del screener en '{PROTECTED_DIR}/'. "
            "Esa carpeta es solo para views que un gestor ya aprobó; escribir "
            f"ahí sobrescribiría decisiones firmadas. Usa '{DRIVE_PROPOSALS_DIR}/'."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (views_payload(views, **kwargs) if kwargs
               else {"views": [public_view(v) for v in views]})
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def default_views_filename(strategy: str, when: date | None = None) -> str:
    """
    Filename for a proposals file.

    Deliberately *not* CCI's ``{estrategia}_views_{fecha}.json``. That name
    belongs to the approval flow's output; reusing it invites exactly the
    overwrite :func:`write_views` refuses to perform, and makes the two files
    indistinguishable in a Drive listing.
    """
    return f"{strategy}_screener_propuestas_{when or date.today()}.json"
