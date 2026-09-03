"""
Black-Litterman posterior and constrained optimization, in-process.

This is the mathematics half of CCI's system, ported so the screener and the
allocation can run in one notebook with the views passed in memory. The formulas
follow CCI's technical document exactly -- inverse optimization for the
equilibrium, Ledoit-Wolf shrinkage on the covariance, the Bayesian posterior,
and mean-variance maximization under the Investment Procedure's bands.

Three defects found in the original implementation are not carried over, and
each is a behaviour change worth stating plainly:

1. **The solver.** CCI's code requests ECOS, which is not installed in a stock
   Colab, and the saved run died there having produced no allocation at all.
   This uses CLARABEL, which ships with CVXPY, and falls back across whatever
   else is installed rather than failing on a missing optional dependency.

2. **Leverage and derivatives were declared but never applied.** ``REGULACIONES``
   carries ``leverage_max`` of 1.25 and 1.50, and the technical document
   specifies a 95% buffer, but the optimizer hard-coded ``sum(w) == 1``. Here
   the gross exposure budget is real, with the documented buffer.

3. **The band audit asserted conformity without checking it.** CCI's
   ``auditar_bandas`` wrote "Auditoría OK" unconditionally. :func:`audit_bands`
   compares the solved weights against every limit and returns the breaches.
   An audit trail that cannot fail is worse than none: it leaves a document
   claiming compliance that nothing verified.

A fourth change comes out of the external model review, and is the largest of
the four in its effect on the final weights:

4. **The equilibrium anchor.** ``pi = lambda * Sigma * w`` passes whatever
   ``w`` it is given straight through, and with a handful of views over a few
   dozen assets the anchor decides roughly three quarters of the book. Anchoring
   on market capitalization normalized against ETF AUM mixes incompatible units
   and lands near 95% equity, which no mandate here permits, so the bands ended
   up doing the asset allocation and the model was left arguing with them.
   :func:`policy_weights` anchors on the mandate's own neutral portfolio
   instead, and :func:`market_weights` is kept only for comparison.

The views themselves come from :mod:`screener.black_litterman`, not from CCI's
hybrid signal combiner, which mixes three signals on incompatible scales and
hands a fully neutral asset a -1.4% expected return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cci_regulation import (
    CLASE_EQUITY, CLASE_ETF_RV, EXCLUSIONES_DURAS, EXPOSICIONES_NUCLEO,
    GRUPOS_ASIGNACION, MODELO_ASIGNACION, REGULACIONES, RISK_TARGETS,
    SECTOR_CAPS, bands_for, clase_a_grupo, classify_for_bands,
    risk_aversion_for, unbanded_classes,
)

#: Risk aversion of the **market**, for the equilibrium ``pi = delta * Sigma * w``.
#: CCI's document parameterizes it at 2.5.
#:
#: This is not the client's. ``pi`` says what returns would make the market
#: portfolio optimal for the average investor; it does not change because this
#: particular mandate is conservative. The client's appetite belongs in the
#: objective function, and lives in
#: :data:`~screener.cci_regulation.RISK_AVERSION_BY_STRATEGY`.
#:
#: Running one value through both is a modelling error with a visible symptom:
#: since ``pi`` scales linearly with it, an Aggressive book solved with a low
#: lambda comes back with a *lower* expected return than the Moderate one -- an
#: artifact of scale that reads as the aggressive mandate being worse.
RISK_AVERSION = 2.5

#: Uncertainty in the equilibrium itself. Fixed low, per CCI's document.
TAU = 0.025

#: Fraction of the leverage limit actually used, per CCI's technical document,
#: so a solution never sits exactly on the regulatory margin.
LEVERAGE_BUFFER = 0.95

#: Whether the desk uses the leverage its mandates permit. Off: every strategy
#: solves fully invested at 100% gross, regardless of what ``leverage_max``
#: allows.
#:
#: This is a **desk decision, not a regulatory limit**, and it lives here rather
#: than in :data:`~screener.cci_regulation.REGULACIONES` on purpose. That table
#: is a verbatim copy of the Investment Procedure; editing it to read 1.0 would
#: destroy the record of what the mandate actually permits and leave nothing
#: showing that a choice was ever made. Keeping the two apart means the run can
#: report both -- what is allowed, and what was used.
#:
#: With leverage off the budget is 1.0 exactly, not ``1.0 * LEVERAGE_BUFFER``.
#: The buffer exists so a solution never sits on a *regulatory* margin; with no
#: leverage there is no margin to stand off from, and applying it anyway would
#: strand 5% of every book in nothing at all -- not even cash, which is already
#: an asset class here with its own band and weight.
ALLOW_LEVERAGE = False

#: Gross exposure when leverage is off: fully invested, long only.
NO_LEVERAGE_BUDGET = 1.0


def gross_budget(strategy: str, *, allow_leverage: bool = ALLOW_LEVERAGE) -> float:
    """
    Gross exposure the book must reach, given the mandate and desk policy.

    Single source of truth: the solver, the feasibility check and the reports
    all read this, so none of them can disagree about how much has to be
    invested.
    """
    if not allow_leverage:
        return NO_LEVERAGE_BUDGET
    return float(REGULACIONES[strategy]["leverage_max"]) * LEVERAGE_BUFFER

#: Smallest holding worth executing, as a fraction of the portfolio.
#:
#: A mean-variance optimizer has no notion of what is worth trading. Left alone
#: it returns whatever weight improves the objective, including 0.16% of the
#: book -- a position that costs a ticket, a line on every report and a
#: reconciliation forever, in exchange for a risk contribution that rounds to
#: nothing. On a US$5MM book that is US$8,000, less than one share of some of
#: the names being screened.
#:
#: This is a portfolio-construction judgement, not a regulatory limit: CCI's
#: Investment Procedure sets ceilings, never floors. 1% is a common desk
#: convention and nothing more; it is exposed as a parameter for that reason.
MIN_POSITION = 0.01

#: Cap on the drop-and-re-solve loop that enforces :data:`MIN_POSITION`.
#: Each pass can only remove names, so it terminates on its own; the cap is
#: there so a pathological basket cannot spin.
MAX_MIN_POSITION_PASSES = 12

#: Weights below this are solver noise rather than positions.
WEIGHT_EPS = 1e-6

TRADING_DAYS = 252


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def shrunk_covariance(returns: pd.DataFrame,
                      periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """
    Annualized covariance with Ledoit-Wolf shrinkage.

    Shrinkage is not a refinement here, it is a requirement: with more names
    than observations the sample covariance is singular and the optimization
    has no unique solution. Shrinkage also pulls in the extreme pairwise
    correlations that would otherwise drive the optimizer into concentrated,
    unstable corners.
    """
    from sklearn.covariance import LedoitWolf

    clean = returns.dropna(axis=1, how="all").dropna()
    if clean.shape[0] < 20:
        raise ValueError(
            f"Solo {clean.shape[0]} observaciones comunes de retornos; "
            "insuficiente para estimar covarianza."
        )
    estimator = LedoitWolf().fit(clean.values)
    return pd.DataFrame(estimator.covariance_ * periods_per_year,
                        index=clean.columns, columns=clean.columns)


def market_weights(caps: Mapping[str, float],
                   tickers: Sequence[str]) -> tuple[pd.Series, list[str]]:
    """
    Normalized market-capitalization weights, and the names that were missing.

    Kept for comparison, and no longer the recommended anchor: see
    :func:`policy_weights` for why normalizing single-stock market cap against
    ETF assets under management is the wrong neutral portfolio for a
    band-constrained mandate. Use this to show what changed, not to allocate.

    Returns the missing list rather than silently substituting a constant. CCI's
    implementation falls back to ``1e9`` on any lookup failure, which feeds
    straight into the equilibrium: an $80bn ETF that failed to download would
    quietly weigh the same as a billion-dollar one, mis-anchoring pi with no
    warning printed.
    """
    values, missing = {}, []
    for ticker in tickers:
        cap = caps.get(ticker)
        if cap is None or not np.isfinite(cap) or cap <= 0:
            missing.append(ticker)
        else:
            values[ticker] = float(cap)

    if not values:
        raise ValueError("Ninguna capitalización de mercado disponible; "
                         "no se puede calcular el equilibrio.")

    series = pd.Series(values)
    return series / series.sum(), missing


#: Classes the ``max_equity_total`` ceiling applies to.
EQUITY_CLASSES: tuple[str, ...] = (CLASE_EQUITY, CLASE_ETF_RV)


def _model_targets(strategy: str, present: list[str],
                   classes: Mapping[str, str],
                   caps: Mapping[str, float] | None,
                   notes: list[str]) -> dict[str, float]:
    """
    Per-class targets from the Procedimiento's Modelo de Asignación.

    The Procedimiento allocates to four lines; the engine carries eight classes,
    so each line's budget is split across the classes of its group by the market
    value actually in the basket -- the same rule already used inside a class,
    one level up. Two things constrain that split:

    * A class with nothing in the basket takes none of the line. The budget goes
      to the rest of its own group, never to another line: a line is a policy
      decision and must not leak across.
    * A class cannot exceed its regulatory band. This is what keeps the
      corporate line from becoming all high yield -- ``RentaFija_NoIG`` holds
      its 5 / 10 / 15 / 25% ceiling inside the group, and what it cannot take
      goes back to investment grade.

    A group whose classes are all absent leaves its budget unassignable; it is
    renormalized away by the caller along with everything else, and reported.
    """
    model = MODELO_ASIGNACION[strategy]
    bands = bands_for(strategy)
    caps = {k.upper(): v for k, v in (caps or {}).items()}

    def value_of(clase: str) -> float:
        total = 0.0
        for ticker, c in classes.items():
            if c != clase:
                continue
            raw = caps.get(ticker.upper())
            try:
                v = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                v = None
            total += v if v is not None and np.isfinite(v) and v > 0 else 0.0
        return total

    class_target = {c: 0.0 for c in present}
    empty: list[str] = []

    for grupo, budget in model.items():
        members = [c for c in GRUPOS_ASIGNACION[grupo] if c in present]
        if not members:
            empty.append(grupo)
            continue

        values = {c: value_of(c) for c in members}
        if sum(values.values()) <= 0:
            # No usable market values anywhere in the group: split evenly rather
            # than dropping the line, and say so.
            share = {c: 1.0 / len(members) for c in members}
            notes.append(
                f"Sin capitalización utilizable en «{grupo}»; su asignación se "
                "repartió en partes iguales entre las clases presentes."
            )
        else:
            share = {c: values[c] / sum(values.values()) for c in members}

        # Split, then pull any class back to its band and give the excess to the
        # others in the same line. Repeats because clamping one raises the rest.
        free = list(members)
        remaining = budget
        for _ in range(len(members) + 1):
            if not free or remaining <= 1e-12:
                break
            scale = sum(share[c] for c in free)
            if scale <= 0:
                break
            over = [c for c in free
                    if bands.get(c) is not None
                    and remaining * share[c] / scale > bands[c][1] + 1e-12]
            if not over:
                for c in free:
                    class_target[c] += remaining * share[c] / scale
                remaining = 0.0
                break
            for c in over:
                ceiling = bands[c][1]
                class_target[c] += ceiling
                remaining -= ceiling
                free.remove(c)
                notes.append(
                    f"{c} topado en su banda de {ceiling:.0%} dentro de "
                    f"«{grupo}»; el resto de esa línea fue a las demás clases "
                    "del mismo grupo."
                )
        if remaining > 1e-12:
            notes.append(
                f"«{grupo}» no puede colocar {remaining:.2%}: sus clases ya "
                "están en sus bandas. Ese porcentaje queda sin asignar en el "
                "ancla y se reparte al renormalizar."
            )

    if empty:
        notes.append(
            f"Líneas del Modelo de Asignación sin ninguna clase en la cesta: "
            f"{sorted(empty)}. Su porcentaje no se pudo colocar; el ancla se "
            "renormaliza sobre lo que sí está, así que las demás líneas suben "
            "en proporción."
        )

    sin_linea = sorted(c for c in present if clase_a_grupo(c) is None)
    if sin_linea:
        notes.append(
            f"Clases sin línea en el Modelo de Asignación: {sin_linea}. Quedan "
            "en 0 en el ancla — el Procedimiento no les asigna nada — así que "
            "el optimizador solo las toma si una view las empuja."
        )

    notes.append(
        "Ancla construida con el Modelo de Asignación de Mercado Internacional "
        "del Procedimiento de Inversión. Dentro de cada línea el reparto es por "
        "valor de mercado, con las bandas y el tope por nombre aplicados."
    )
    return class_target


def policy_weights(asset_types: Mapping[str, str], strategy: str, *,
                   caps: Mapping[str, float] | None = None,
                   targets: Mapping[str, float] | None = None,
                   total: float = 1.0) -> tuple[pd.Series, list[str]]:
    """
    The mandate's own neutral portfolio, for use as the Black-Litterman anchor.

    Why not market-capitalization weights
    -------------------------------------
    ``pi = lambda * Sigma * w`` is a linear map: whatever ``w`` is handed to it
    *is* the neutral portfolio, and with a handful of views over a few dozen
    assets it determines most of the answer. Measured on this system, views move
    about 27% of the book -- the other ~73% is the anchor. So the anchor is not
    a technicality, it is the largest single decision in the allocation, and
    :func:`market_weights` gets it wrong here for two reasons.

    *The units do not match.* It normalizes single-stock market capitalization
    against ETF assets under management. A company's market cap is the value of
    the company; a fund's AUM is how much money happens to sit in that wrapper,
    and for an equity ETF it double-counts shares already priced elsewhere in
    the basket. Dividing one by the sum of both is not a market portfolio in any
    sense the CAPM would recognize.

    *It ignores the mandate.* Market-cap weighting a basket of mega-caps and
    bond ETFs anchors near 95% equity. A Moderado mandate caps equity at 60%.
    The optimizer then spends its whole budget dragging the solution back to the
    ceiling, so the allocation is decided by the constraint rather than by the
    model, and inside equity the weights are just market cap. Under that setup
    the screener's ranking barely reaches the portfolio.

    What this does instead
    ----------------------
    Cross-class allocation comes from policy; within-class allocation comes from
    market value, where comparing market values is actually meaningful:

    1. Each asset class present takes the midpoint of its regulatory band.
    2. Midpoints are renormalized over the classes actually in the basket --
       CCI's bands are ceilings and sum well above 1.0, so they are read as
       relative preferences among what is held.
    3. Total equity is scaled to respect ``max_equity_total``, with the shortfall
       redistributed across the non-equity classes. An anchor sitting outside
       the feasible set is the defect being fixed; it must not be reintroduced
       here.
    4. Inside each class, names split by market cap when available and equally
       otherwise. Missing caps degrade one class's split rather than
       mis-anchoring the whole portfolio.

    With no views, reverse optimization on this anchor returns this portfolio.
    That is the property that makes it the right neutral: the model's output
    with nothing to say is the mandate's own strategic allocation, and views
    tilt away from it.

    A stand-in, and labelled as one
    -------------------------------
    **Band midpoints are not CCI's strategic asset allocation.** A real SAA is
    an Investment Committee decision, and CCI's documents supply bands, not
    targets. The midpoint is a defensible reading of a band and a far better
    anchor than mixed market caps, but it is still an inference. Pass ``targets``
    to supply the committee's real numbers; they are used as given (renormalized
    to ``total``) and the equity ceiling is still enforced.

    One consequence of renormalizing follows directly and is worth knowing
    before reading the output: **the anchor depends on which classes are in the
    basket.** Midpoints across CCI's classes sum well above 1.0, so the
    normalization divides by whatever is present. Screen a basket spanning seven
    classes and neutral equity comes out near 19%; screen the same mandate with
    only equity and bonds in the basket and it comes out far higher, with no
    change in policy. That is honest -- the optimizer can only hold what it is
    given -- but it means the neutral portfolio moves with basket construction,
    which a genuine SAA would not. Supplying ``targets`` removes the effect
    entirely, and is the reason the parameter exists.

    Returns the weights and a list of notes describing every substitution made.
    """
    tickers = list(asset_types)
    if not tickers:
        raise ValueError("No hay activos para anclar el equilibrio.")

    notes: list[str] = []
    classes = {t: classify_for_bands(t, asset_types[t]) for t in tickers}
    present = sorted(set(classes.values()))
    bands = bands_for(strategy)

    if targets is not None:
        unknown = set(targets) - set(present)
        if unknown:
            notes.append(
                f"Objetivos de política para clases que no están en la cesta, "
                f"ignorados: {sorted(unknown)}."
            )
        class_target = {c: float(targets.get(c, 0.0)) for c in present}
        missing = [c for c in present if c not in targets]
        if missing:
            notes.append(
                f"Sin objetivo explícito para {sorted(missing)}; quedan en 0 en "
                "el ancla, así que el optimizador solo los tomará si una view "
                "los empuja."
            )
    elif strategy in MODELO_ASIGNACION:
        class_target = _model_targets(strategy, present, classes, caps, notes)
    else:
        class_target = {}
        for clase in present:
            band = bands.get(clase)
            if band is None:
                class_target[clase] = 0.0
                notes.append(
                    f"La clase {clase} no tiene banda declarada, así que no "
                    "tiene punto medio y queda en 0 en el ancla. Sin banda no "
                    "hay política que leer: confirmar con Compliance."
                )
            else:
                class_target[clase] = (float(band[0]) + float(band[1])) / 2.0
        notes.append(
            "Ancla de política construida con los puntos medios de las bandas. "
            "Eso es una lectura de los límites, NO la asignación estratégica "
            "del Comité de Inversiones, que no está en los documentos."
        )

    gross = sum(class_target.values())
    if gross <= 0:
        raise ValueError(
            f"Los objetivos de política suman {gross}; no se puede anclar el "
            "equilibrio en una cartera vacía."
        )
    class_target = {c: v / gross * total for c, v in class_target.items()}

    # The ceiling on total equity, applied to the anchor itself.
    #
    # Deliberately NOT scaled by `total`. `optimize` constrains
    # `sum(w[equity]) <= max_equity_total` in absolute weight, while the gross
    # budget may exceed 1.0 under leverage -- "máximo 60% en renta variable"
    # reads against the portfolio's value, not against gross exposure. Scaling
    # the ceiling here by the budget would make the anchor more permissive than
    # the solver and could place it outside the feasible set, which is the exact
    # defect this function exists to remove.
    max_equity = float(REGULACIONES[strategy]["max_equity_total"])
    equity_now = sum(v for c, v in class_target.items() if c in EQUITY_CLASSES)
    other_now = total - equity_now
    if equity_now > max_equity + 1e-12:
        if other_now <= 1e-12:
            notes.append(
                f"La cesta es toda renta variable, así que el ancla no puede "
                f"respetar el techo de {max_equity:.0%}. La cesta no es "
                "compatible con el mandato."
            )
        else:
            scale_eq = max_equity / equity_now
            scale_other = (total - max_equity) / other_now
            class_target = {
                c: v * (scale_eq if c in EQUITY_CLASSES else scale_other)
                for c, v in class_target.items()
            }
            notes.append(
                f"Renta variable neutral bajada de {equity_now:.0%} a "
                f"{max_equity:.0%} (peso absoluto) por el techo del mandato; "
                "el resto se repartió entre las demás clases."
            )

    # Within each class, split by market value when it is comparable.
    caps = {k.upper(): v for k, v in (caps or {}).items()}

    def usable_cap(ticker: str) -> float | None:
        """A positive, finite market value, or None. ``None`` is a real input
        here -- callers pass a caps mapping straight from a download that may
        have failed for some names."""
        raw = caps.get(ticker.upper())
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    # Per-name ceiling, applied inside the class the same way the solver applies
    # it. Only single stocks carry one: `max_equity_individual` is a limit on
    # one company, and an ETF is a basket rather than a name. That means an
    # equity ETF has no per-instrument ceiling here, which matches `optimize`
    # exactly -- the anchor and the solver must agree or the anchor is outside
    # the feasible set again, which is the whole defect this function exists to
    # remove.
    name_cap = float(REGULACIONES[strategy]["max_equity_individual"])

    def split_class(members: list[str], budget: float,
                    capped: bool) -> tuple[dict[str, float], float]:
        """
        Weights for one class, and whatever budget it could not absorb.

        Market value decides the split, but a name at its ceiling stops taking
        more and its excess goes to the others -- which can push a second name
        onto the ceiling, so this repeats until nothing moves. Water-filling,
        not a single pass: one pass would leave the second name over the limit.
        """
        if not members or budget <= 0:
            return {t: 0.0 for t in members}, max(budget, 0.0)

        usable = {t: c for t in members if (c := usable_cap(t)) is not None}
        if len(usable) == len(members):
            share = {t: usable[t] / sum(usable.values()) for t in members}
        else:
            if usable:
                notes.append(
                    f"Capitalización faltante en {clase} para "
                    f"{sorted(set(members) - set(usable))}; esa clase se reparte "
                    "en partes iguales."
                )
            share = {t: 1.0 / len(members) for t in members}

        if not capped:
            return {t: budget * share[t] for t in members}, 0.0

        out: dict[str, float] = {}
        free = list(members)
        remaining = budget
        for _ in range(len(members) + 1):
            if not free or remaining <= 1e-12:
                break
            scale = sum(share[t] for t in free)
            if scale <= 0:
                break
            hit = [t for t in free if remaining * share[t] / scale > name_cap + 1e-12]
            if not hit:
                for t in free:
                    out[t] = remaining * share[t] / scale
                remaining = 0.0
                free = []
                break
            for t in hit:
                out[t] = name_cap
                remaining -= name_cap
                free.remove(t)
        for t in free:
            out[t] = 0.0
        return {t: out.get(t, 0.0) for t in members}, max(remaining, 0.0)

    weights: dict[str, float] = {}
    unabsorbed: dict[str, float] = {}
    for clase in present:
        members = [t for t in tickers if classes[t] == clase]
        part, left = split_class(members, class_target[clase],
                                 capped=clase == CLASE_EQUITY)
        weights.update(part)
        if left > 1e-12:
            unabsorbed[clase] = left

    # A class that cannot hold its own budget hands the remainder back, so the
    # anchor still spends exactly `total`. Only reachable when the basket has
    # too few single stocks to absorb the equity allocation at the per-name
    # ceiling -- select_basket's three-per-class floor normally prevents it, but
    # this function is public and cannot assume its caller.
    for clase, left in unabsorbed.items():
        # Prefer the rest of the same Modelo de Asignación line. A line is a
        # policy decision, so weight the per-name cap pushes out of single
        # stocks belongs with the index ETFs it was allocated alongside -- not
        # in fixed income. Only a fully saturated line leaks to other lines.
        grupo = clase_a_grupo(clase)
        siblings = [c for c in present
                    if c != clase and clase_a_grupo(c) == grupo
                    and class_target[c] > 0] if grupo else []
        takers = siblings or [c for c in present
                              if c != clase and class_target[c] > 0]
        capacity = sum(class_target[c] for c in takers)
        if capacity <= 0:
            # Nowhere to put it. Such a basket is already infeasible -- an
            # all-equity cesta cannot meet the equity ceiling either -- so
            # enforcing the per-name cap here would only add a second failure
            # to a portfolio that cannot exist, and would break the invariant
            # that the anchor spends its budget. Fall back to the uncapped
            # split and report both problems instead.
            members = [t for t in tickers if classes[t] == clase]
            part, _ = split_class(members, class_target[clase], capped=False)
            weights.update(part)
            notes.append(
                f"{clase} no puede absorber {left:.2%} bajo el tope por nombre "
                f"de {name_cap:.0%}, y no hay otra clase donde ponerlo. El ancla "
                "reparte sin el tope: la cesta no es compatible con el mandato, "
                "y el optimizador lo va a reportar como infactible."
            )
            continue
        for c in takers:
            extra = left * class_target[c] / capacity
            members = [t for t in tickers if classes[t] == c]
            part, _ = split_class(members, class_target[c] + extra,
                                  capped=c == CLASE_EQUITY)
            weights.update(part)
        notes.append(
            f"El tope por nombre de {name_cap:.0%} deja {left:.2%} que {clase} "
            "no puede sostener; se repartió entre las demás clases. Faltan "
            "nombres de esa clase en la cesta."
        )

    hit_cap = sorted(t for t, w in weights.items()
                     if classes[t] == CLASE_EQUITY and w >= name_cap - 1e-9)
    if hit_cap:
        notes.append(
            f"Tope por nombre de {name_cap:.0%} aplicado en el ancla a: "
            f"{', '.join(hit_cap)}. El exceso se repartió por capitalización "
            "entre las demás acciones de la clase."
        )

    return pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0), notes


def implied_equilibrium(weights: pd.Series, covariance: pd.DataFrame,
                        risk_aversion: float = RISK_AVERSION) -> pd.Series:
    """Reverse optimization: ``pi = lambda * Sigma * w_anchor``."""
    common = [t for t in covariance.columns if t in weights.index]
    return risk_aversion * covariance.loc[common, common].dot(weights[common])


# --------------------------------------------------------------------------
# Bayesian posterior
# --------------------------------------------------------------------------

def posterior(pi: pd.Series, covariance: pd.DataFrame,
              views: Sequence[Mapping[str, Any]],
              tau: float = TAU) -> tuple[pd.Series, pd.DataFrame]:
    """
    Combine the equilibrium with the views.

    Follows CCI's ``black_litterman_core``: ``Omega`` is diagonal with each
    element the view's own variance scaled by conviction, and the posterior is
    the precision-weighted blend of prior and views.

    Views naming a ticker outside the covariance universe are skipped with the
    reason recorded, rather than raising -- a manager editing views by hand will
    eventually type a name the screen dropped.
    """
    assets = list(covariance.columns)
    if not views:
        return pi.reindex(assets), covariance

    usable = []
    for view in views:
        legs = ([view.get("activo")] if view.get("tipo") == "absoluto"
                else [view.get("activo_long"), view.get("activo_short")])
        if all(leg in assets for leg in legs):
            usable.append(view)
    if not usable:
        return pi.reindex(assets), covariance

    k, n = len(usable), len(assets)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega = np.zeros(k)

    for i, view in enumerate(usable):
        Q[i] = float(view["Q"])
        if view["tipo"] == "absoluto":
            P[i, assets.index(view["activo"])] = 1.0
        else:
            P[i, assets.index(view["activo_long"])] = 1.0
            P[i, assets.index(view["activo_short"])] = -1.0

        row = P[i].reshape(1, -1)
        variance = float((row @ covariance.values @ row.T).item()) * tau
        conviction = max(0.1, float(view.get("conviccion", 0.5)))
        omega[i] = variance / conviction

    tau_sigma_inv = np.linalg.inv(tau * covariance.values)
    omega_inv = np.linalg.inv(np.diag(omega))

    blended = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    combined = tau_sigma_inv @ pi.reindex(assets).values + P.T @ omega_inv @ Q

    return (pd.Series(blended @ combined, index=assets),
            pd.DataFrame(covariance.values + blended,
                         index=assets, columns=assets))


# --------------------------------------------------------------------------
# Constrained optimization
# --------------------------------------------------------------------------

def core_vehicles(scored: Sequence[Any],
                  exposures: Mapping[str, Sequence[str]] | None = None,
                  ) -> tuple[dict[str, str], list[str]]:
    """
    One vehicle per core exposure: the best-scoring eligible candidate.

    Splits a decision the old basket ran together. *Which exposures the core can
    hold* is allocation policy and lives in
    :data:`screener.cci_regulation.EXPOSICIONES_NUCLEO`. *Which fund delivers an
    exposure* is a comparison of near-identical products, which is exactly what
    the screener's score is for -- so policy names the line and the model picks
    the wrapper.

    Returns ``({exposure: ticker}, notes)``. An exposure whose candidates are
    all missing or ineligible is reported, never substituted: dropping in a
    different fund because the intended one failed a liquidity gate would put a
    product in the core that no rule selected.
    """
    exposures = EXPOSICIONES_NUCLEO if exposures is None else exposures
    by_ticker = {r.ticker.upper(): r for r in scored}
    eligible = {t: r for t, r in by_ticker.items()
                if getattr(r, "eligible", True)}

    chosen: dict[str, str] = {}
    notes: list[str] = []
    for exposure, candidates in exposures.items():
        ranked = [t for t in (c.upper() for c in candidates) if t in eligible]
        if not ranked:
            presentes = [t for t in (c.upper() for c in candidates)
                         if t in by_ticker]
            motivo = ("ninguno pasó los filtros de elegibilidad"
                      if presentes else "ninguno llegó al universo puntuado")
            notes.append(
                f"Exposición núcleo «{exposure}» sin vehículo: {motivo} "
                f"({', '.join(candidates)}). El optimizador no podrá tomarla."
            )
            continue
        best = min(ranked, key=lambda t: _rank_of(eligible[t]))
        chosen[exposure] = best
        if len(ranked) > 1:
            resto = ", ".join(f"{t} ({_fmt_score(eligible[t])})"
                              for t in ranked if t != best)
            notes.append(
                f"Núcleo «{exposure}»: {best} "
                f"(score {_fmt_score(eligible[best])}) sobre {resto}."
            )
    return chosen, notes


def _score_of(row: Any) -> float | None:
    """
    The 0-100 composite of a :class:`ScoredInstrument`.

    Named for the attribute, not for the spreadsheet: the workbook exports the
    column as ``score`` while the object carries ``score_0_100``, and reading
    the sheet and assuming the field is what broke this in Colab.
    """
    value = getattr(row, "score_0_100", None)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value else value      # descarta NaN


def _fmt_score(row: Any) -> str:
    score = _score_of(row)
    return "sin score" if score is None else f"{score:.0f}"


def _rank_of(row: Any) -> float:
    """Sort key that prefers a higher score; missing scores sort last."""
    score = _score_of(row)
    return float("inf") if score is None else -score


def select_basket(scored: Sequence[Any], strategy: str, top_n: int = 25,
                  min_per_class: int = 3,
                  core_exposures: Mapping[str, Sequence[str]] | None = None,
                  include_core: bool = True) -> tuple[list[str], list[str]]:
    """
    Choose the optimizer's basket so the mandate's bands are actually reachable
    and the core exposures are actually on offer.

    Returns ``(tickers, notes)``.

    Three rules, in order:

    1. **Top-N by score.** The screener's opinion.
    2. **Core index exposures**, one vehicle each, whether or not they scored.
    3. **A floor per asset class**, so every class the bands need is reachable.

    Rule 2 is the one added after a live run. Top-N plus the class floor is not
    enough, and the failure is invisible in the output. The factor model weights
    momentum at 25-36%, momentum clusters by industry, so the top of the ranking
    is whatever ran hardest. In an Agresivo run that produced a basket whose only
    equity ETFs were XBI, EWT and EWY -- biotech, Taiwan, Korea -- while SPY sat
    at #149, IWM at #115 and EEM at #119. The optimizer then held one of the
    three and the run read as "the model rejected broad index exposure". It had
    never been offered any. A momentum ranking decides what runs *well*; it must
    not also decide what is *available*, or the core is a momentum artifact.

    Rule 3 exists for a different failure. Under Moderado total equity caps at
    60% while the book must be roughly fully invested, so an all-equity basket
    makes the solver return infeasible and the portfolio comes out empty with no
    obvious cause. CCI's own system never hits it because its basket comes from
    a hand-maintained sheet spanning bonds, credit, cash and equity; replacing
    that sheet means reproducing the property.
    """
    ranked = [r for r in scored if getattr(r, "eligible", True)]
    picked = [r.ticker for r in ranked[:max(top_n, 0)]]
    chosen = set(picked)
    notes: list[str] = []

    if include_core:
        vehicles, core_notes = core_vehicles(scored, core_exposures)
        notes.extend(core_notes)
        added = []
        for exposure, ticker in vehicles.items():
            if ticker not in chosen:
                chosen.add(ticker)
                picked.append(ticker)
                added.append(f"{ticker} ({exposure})")
        if added:
            notes.append(
                f"Núcleo agregado a la cesta por construcción, fuera del "
                f"top-{top_n}: {', '.join(added)}."
            )

    by_class: dict[str, list[str]] = {}
    for row in ranked:
        clase = classify_for_bands(row.ticker, getattr(row, "asset_type", "ETF"))
        by_class.setdefault(clase, []).append(row.ticker)

    for clase, members in by_class.items():
        if clase not in bands_for(strategy):
            continue
        present = sum(1 for t in members if t in chosen)
        for ticker in members:
            if present >= min_per_class:
                break
            if ticker not in chosen:
                chosen.add(ticker)
                picked.append(ticker)
                present += 1

    return picked, notes


def feasibility_report(classes: Mapping[str, str], strategy: str,
                       budget: float | None = None) -> list[str]:
    """
    Why a basket cannot fill the mandate, stated before the solver says
    "infeasible".

    A solver status is not a diagnosis. This adds the upper bands of every class
    actually present in the basket: if that ceiling sits below the amount the
    book must invest, no allocation exists and the reason is a missing asset
    class, not a numerical problem.
    """
    rules = REGULACIONES[strategy]
    required = budget if budget is not None else gross_budget(strategy)
    bands = bands_for(strategy)

    present = set(classes.values())
    reachable = sum(bands[c][1] for c in present if c in bands)

    equity_only = present <= {CLASE_EQUITY, CLASE_ETF_RV}
    problems: list[str] = []

    if equity_only and rules["max_equity_total"] < required:
        missing = [c for c in bands if c not in present
                   and c not in (CLASE_EQUITY, CLASE_ETF_RV)]
        problems.append(
            f"La cesta es solo renta variable, y {strategy} la limita a "
            f"{rules['max_equity_total']:.0%} del libro, pero hay que invertir "
            f"{required:.0%}. Faltan clases como: {', '.join(missing[:4])}. "
            "Amplía el universo o baja el top-N para que entren."
        )
    elif reachable < required:
        problems.append(
            f"Las bandas de las clases presentes suman como máximo "
            f"{reachable:.0%}, por debajo del {required:.0%} que hay que "
            f"invertir. Clases en la cesta: {', '.join(sorted(present))}."
        )

    return problems


@dataclass
class Allocation:
    """Solved weights plus everything a reviewer needs to challenge them."""

    weights: pd.Series
    strategy: str
    status: str
    gross_exposure: float
    expected_return: float
    volatility: float
    by_class: pd.Series
    breaches: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Look-through exposure by sector. Empty when no sector map was supplied,
    #: which is not the same as "no concentration" and must not be read as it.
    sector_exposure: dict[str, float] = field(default_factory=dict)
    #: Findings against the desk's risk expectations for the mandate, kept
    #: **out** of ``breaches`` on purpose. ``breaches`` means the Investment
    #: Procedure was violated; a book outside the desk's volatility range is a
    #: different animal -- the range is not in that document, and it is an
    #: expectation rather than a limit. Mixing them would make a compliance
    #: signal fire on a number the Committee has not even approved yet.
    risk_findings: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return self.status in {"optimal", "optimal_inaccurate"}


def relative_view_pairs(views: Sequence[Mapping[str, Any]],
                        universe: Iterable[str] | None = None,
                        ) -> list[tuple[str, str]]:
    """
    ``(largo, corto)`` de cada view relativa cuyas dos patas están en el problema.

    Una view relativa que nombra un ticker fuera de la covarianza no existe para
    el optimizador — :func:`posterior` la descarta — así que tampoco puede
    restringirlo.
    """
    disponibles = None if universe is None else {str(t).upper() for t in universe}
    out: list[tuple[str, str]] = []
    for view in views or ():
        if view.get("tipo") not in ("relativa", "relativo"):
            continue
        largo = str(view.get("activo_long") or "").upper()
        corto = str(view.get("activo_short") or "").upper()
        if not largo or not corto or largo == corto:
            continue
        if disponibles is not None and not {largo, corto} <= disponibles:
            continue
        # Q negativo invierte la dirección: la view dice que gana la otra pata.
        try:
            q = float(view.get("Q", 0.0))
        except (TypeError, ValueError):
            q = 0.0
        out.append((corto, largo) if q < 0 else (largo, corto))
    return out


def view_coherence_breaches(weights: pd.Series | Mapping[str, float],
                            pairs: Sequence[tuple[str, str]],
                            tolerance: float = 1e-6) -> list[str]:
    """Views relativas que la cartera contradice: pesa más la pata perdedora."""
    held = dict(weights.items()) if hasattr(weights, "items") else dict(weights)
    fuera = []
    for largo, corto in pairs:
        w_l = float(held.get(largo, 0.0) or 0.0)
        w_s = float(held.get(corto, 0.0) or 0.0)
        if w_s > w_l + tolerance:
            fuera.append(
                f"View {largo} sobre {corto}: la cartera lleva {w_s:.2%} de "
                f"{corto} y {w_l:.2%} de {largo}, al revés de lo que dice la view"
            )
    return fuera


def sector_exposures(weights: pd.Series | Mapping[str, float],
                     sector_weights: Mapping[str, Mapping[str, float]],
                     ) -> dict[str, float]:
    """Look-through sector exposure of a solved book, largest first."""
    held = dict(weights.items()) if hasattr(weights, "items") else dict(weights)
    out: dict[str, float] = {}
    for sector, members in sector_weights.items():
        total = sum(float(held.get(t, 0.0) or 0.0) * float(share)
                    for t, share in members.items())
        if total > 1e-12:
            out[sector] = total
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def audit_sectors(weights: pd.Series | Mapping[str, float],
                  sector_weights: Mapping[str, Mapping[str, float]],
                  cap: float | None,
                  tolerance: float = 1e-4) -> list[str]:
    """Sector concentrations above ``cap``. Empty when ``cap`` is None."""
    if cap is None:
        return []
    return [f"Sector {sector}: {value:.2%} excede el tope de {cap:.0%}"
            for sector, value in sector_exposures(weights, sector_weights).items()
            if value > cap + tolerance]


def optimize(expected_returns: pd.Series, covariance: pd.DataFrame,
             asset_types: Mapping[str, str], strategy: str,
             risk_aversion: float | None = None,
             solver: str | None = None,
             min_position: float | None = MIN_POSITION,
             allow_leverage: bool = ALLOW_LEVERAGE,
             sector_weights: Mapping[str, Mapping[str, float]] | None = None,
             sector_cap: float | None | str = "auto",
             views: Sequence[Mapping[str, Any]] | None = None,
             enforce_view_coherence: bool = True,
             risk_budget: tuple[float, float] | None | str = "auto",
             anchor: pd.Series | None = None,
             prior: pd.Series | None = None) -> Allocation:
    """
    Maximize ``w'mu - (lambda/2) w'Sigma w`` under CCI's Investment Procedure.

    Long-only, per-class bands, a total-equity ceiling, a per-name cap on single
    stocks, an optional look-through sector ceiling, hard exclusions, and a
    gross-exposure budget.

    ``sector_weights`` maps ``{sector: {ticker: fraction of that ticker}}``, as
    :func:`screener.lookthrough.sector_map` produces it. Pass it and the solver
    constrains industry concentration *through* the funds, so a sector ETF and a
    single name in the same industry compete for one ceiling. Without it the
    sleeve is unconstrained and the run says so -- CCI's bands are by asset
    class and stop nothing here, which is how a live Agresivo book reached ~35%
    in one semiconductor chain and passed its audit clean.

    ``sector_cap`` defaults to the strategy's entry in
    :data:`screener.cci_regulation.SECTOR_CAPS`; pass a number to override or
    ``None`` to measure sector exposure without constraining it.

    ``views``, with ``enforce_view_coherence``, stops the book from being
    positioned *against* its own relative views: for each one, the long leg is
    constrained to weigh at least as much as the short leg. A live run held MU
    at 5.84% and LRCX at 5.85% on a view saying MU beats LRCX -- the portfolio
    was marginally short its own call. The constraint is a floor and not a
    margin on purpose: ``0 >= 0`` satisfies it, so it never forces a position
    into the book, it only forbids the contradiction. Long-only cannot express
    the short leg, and pretending otherwise by sizing to the spread would put
    on a bet nobody approved.

    ``risk_budget`` is the ``(floor, ceiling)`` annual volatility the mandate
    implies, defaulting to :data:`screener.cci_regulation.RISK_TARGETS`. It is
    deliberately **not** a constraint on the primary solve. Two reasons.

    A ceiling the basket cannot meet turns into no portfolio at all, and "this
    basket cannot build a defensive book" is information the desk needs in its
    hands, not an empty result. And a hard ceiling breaks the property this
    whole design rests on -- that with no views the optimizer returns the
    mandate's own neutral portfolio -- whenever that neutral portfolio is itself
    above the ceiling, which is a contradiction in the policy that must be
    reported rather than silently resolved.

    So the budget does two other things. The floor drives a **second pass**: a
    book that solves below its own floor is re-solved maximizing return subject
    to ``w'Sigma w <= ceiling**2``, which is convex where a floor is not, so the
    mandate's risk budget becomes something the portfolio is built to use and
    not merely to respect. And both ends are **audited**, so a book outside its
    range says so.

    ``allow_leverage`` is off by default: the book solves fully invested at 100%
    gross for every strategy, whatever ``leverage_max`` permits. See
    :data:`ALLOW_LEVERAGE` for why that lives here and not in the regulation
    table. Pass ``True`` to use the mandate's leverage with its documented
    buffer.

    ``min_position`` drops holdings too small to be worth executing; see
    :data:`MIN_POSITION`. Pass ``None`` to keep whatever the solver produces.
    """
    import cvxpy as cp

    if strategy not in REGULACIONES:
        raise KeyError(f"Estrategia desconocida: {strategy!r}")

    tickers = [t for t in covariance.columns if t in expected_returns.index]
    if len(tickers) < 2:
        raise ValueError("Se necesitan al menos 2 activos para optimizar.")

    mu = expected_returns[tickers].values
    sigma = covariance.loc[tickers, tickers].values
    rules = REGULACIONES[strategy]
    classes = {t: classify_for_bands(t, asset_types.get(t, "ETF")) for t in tickers}

    # El ancla del mandato y su equilibrio, para penalizar el riesgo ACTIVO.
    # Sin ellos se cae al objetivo de riesgo total con el lambda global, que es
    # el comportamiento anterior: así un llamador que no los pase no cambia de
    # resultado por esta versión.
    w_ancla = pi_prior = None
    if anchor is not None and prior is not None:
        w_ancla = np.asarray([float(anchor.get(t, 0.0)) for t in tickers])
        pi_prior = np.asarray([float(prior.get(t, 0.0)) for t in tickers])

    notes: list[str] = []
    # El lambda del mandato, no uno global. Las cuatro estrategias resolvían con
    # el mismo 2.5 y se diferenciaban solo por el ancho de sus bandas, que son
    # techos: nada obligaba a la Agresiva a usarlos.
    if risk_aversion is None:
        risk_aversion = (risk_aversion_for(strategy, RISK_AVERSION)
                         if w_ancla is not None else RISK_AVERSION)
        if w_ancla is not None:
            notes.append(
                f"Aversión al riesgo λ={risk_aversion:g} para {strategy} "
                f"(Moderado usa {risk_aversion_for('Moderado'):g}), aplicada al "
                "riesgo ACTIVO contra el Modelo de Asignación. Sin views la "
                "cartera es el Modelo; el λ decide cuánto se aparta de él una "
                "view."
            )
    budget = gross_budget(strategy, allow_leverage=allow_leverage)
    infeasible_reasons = feasibility_report(classes, strategy, budget)
    notes.extend(infeasible_reasons)

    permitted = float(rules["leverage_max"])
    if not allow_leverage and permitted > 1.0:
        notes.append(
            f"Apalancamiento desactivado por política de mesa: la cartera "
            f"resuelve invertida al {budget:.0%}. El mandato {strategy} permite "
            f"hasta {permitted:.0%}, y ese límite sigue vigente en el "
            f"Procedimiento — solo no se está usando."
        )

    orphaned = unbanded_classes(classes, strategy)
    if orphaned:
        notes.append(
            f"Clases sin banda declarada: {sorted(orphaned)}. Quedan sin techo; "
            "confirmar con Compliance antes de operar."
        )

    # --- Sector ceiling --------------------------------------------------
    #
    # Restricted to the tickers actually in this problem, so a sector map built
    # for a wider universe cannot smuggle in names the optimizer never sees.
    if sector_cap == "auto":
        sector_cap = SECTOR_CAPS.get(strategy)
    sectors: dict[str, dict[str, float]] = {}
    if sector_weights:
        for sector, members in sector_weights.items():
            fila = {t: float(s) for t, s in members.items()
                    if t in set(tickers) and float(s) > 0}
            if fila:
                sectors[sector] = fila

    if not sectors:
        notes.append(
            "Sin desglose sectorial: la concentración por industria queda SIN "
            "restringir. Las bandas del Procedimiento son por clase de activo y "
            "no limitan sector. Corre bajar_tenencias.py para habilitarla."
        )
    else:
        cubierto = {t for fila in sectors.values() for t in fila}
        sin_dato = [t for t in tickers if t not in cubierto]
        equity_sin_dato = [t for t in sin_dato
                           if classes[t] in (CLASE_EQUITY, CLASE_ETF_RV)]
        if sector_cap is None:
            notes.append(
                f"Exposición sectorial medida sobre {len(cubierto)} de "
                f"{len(tickers)} instrumentos, pero SIN tope: "
                f"SECTOR_CAPS[{strategy!r}] es None."
            )
        else:
            notes.append(
                f"Tope sectorial {sector_cap:.0%} (mirando a través de los "
                f"fondos) sobre {len(sectors)} sector(es) y {len(cubierto)} de "
                f"{len(tickers)} instrumentos. Número de la mesa, no del "
                "Procedimiento: pendiente de confirmar con el Comité."
            )
        if equity_sin_dato:
            notes.append(
                f"Renta variable sin sector conocido, fuera del tope: "
                f"{', '.join(sorted(equity_sin_dato)[:10])}"
                f"{' ...' if len(equity_sin_dato) > 10 else ''}. Ese peso puede "
                "concentrarse sin que la restricción lo vea."
            )

    # --- View coherence --------------------------------------------------
    pairs = (relative_view_pairs(views or (), tickers)
             if enforce_view_coherence else [])
    if pairs:
        notes.append(
            f"Coherencia con {len(pairs)} view(s) relativa(s): la pata larga no "
            "puede pesar menos que la corta. "
            + "; ".join(f"{a}>={b}" for a, b in pairs)
        )

    # --- Risk budget -----------------------------------------------------
    if risk_budget == "auto":
        risk_budget = RISK_TARGETS.get(strategy)
    piso_vol, techo_vol = risk_budget if risk_budget else (None, None)
    if piso_vol is not None:
        notes.append(
            f"Presupuesto de riesgo {piso_vol:.1%}–{techo_vol:.1%} de "
            f"volatilidad anual para {strategy}. Números de la mesa, no del "
            "Procedimiento: pendientes de confirmar con el Comité."
        )

    # CLARABEL ships with CVXPY. CCI's code asked for ECOS, which is optional
    # and absent from a stock Colab -- that is what killed their saved run.
    order = ([solver] if solver else
             [s for s in ("CLARABEL", "SCS", "OSQP", "ECOS")
              if s in cp.installed_solvers()])

    def solve(banned: frozenset[str], with_sectors: bool = True,
              with_views: bool = True, with_vol: bool = True,
              mode: str = "utility") -> tuple[np.ndarray | None, str]:
        """Solve the mandate's problem with ``banned`` names forced to zero."""
        w = cp.Variable(len(tickers))
        constraints = [w >= 0, cp.sum(w) <= budget]
        if not allow_leverage or rules["leverage_max"] <= 1.0:
            # Fully invested at the budget. Long-only plus an equality on the
            # sum is the whole of "no leverage": nothing can be borrowed and
            # nothing can sit unallocated.
            constraints.append(cp.sum(w) == budget)
        else:
            # Stay invested: without a floor the optimizer can sit in cash and
            # report a technically optimal empty book.
            constraints.append(cp.sum(w) >= 1.0)

        for i, ticker in enumerate(tickers):
            if ticker in EXCLUSIONES_DURAS or ticker in banned:
                constraints.append(w[i] == 0)
            if classes[ticker] == CLASE_EQUITY:
                constraints.append(w[i] <= rules["max_equity_individual"])

        equity_idx = [i for i, t in enumerate(tickers)
                      if classes[t] in (CLASE_EQUITY, CLASE_ETF_RV)]
        if equity_idx:
            constraints.append(cp.sum(w[equity_idx]) <= rules["max_equity_total"])

        for clase, (low, high) in bands_for(strategy).items():
            idx = [i for i, t in enumerate(tickers) if classes[t] == clase]
            if idx:
                constraints.append(cp.sum(w[idx]) >= low)
                constraints.append(cp.sum(w[idx]) <= high)

        # One row per sector: the look-through weight of the book in that
        # industry, counting a fund by its own breakdown rather than as a
        # single undifferentiated "equity" position.
        posicion = {t: i for i, t in enumerate(tickers)}

        if with_sectors and sectors and sector_cap is not None:
            for fila in sectors.values():
                idx = [posicion[t] for t in fila]
                shares = np.array([fila[t] for t in fila])
                constraints.append(shares @ w[idx] <= sector_cap)

        # La pata larga de una view relativa no puede pesar menos que la corta.
        # No fuerza a tener la posición: 0 >= 0 se cumple.
        if with_views:
            for largo, corto in pairs:
                constraints.append(w[posicion[largo]] >= w[posicion[corto]])

        # El techo solo restringe la segunda pasada, la que gasta el
        # presupuesto de riesgo. En la primera no va: un techo que la cesta no
        # puede cumplir devolvería cero cartera, y ademas rompe la propiedad de
        # que sin views el optimizador reproduce el ancla del mandato.
        if with_vol and mode == "max_return" and techo_vol is not None:
            constraints.append(
                cp.quad_form(w, cp.psd_wrap(sigma)) <= float(techo_vol) ** 2)

        if mode == "utility" and w_ancla is not None:
            # Riesgo ACTIVO contra el ancla, no riesgo total.
            #
            # Con riesgo total y un lambda por mandato, la cartera sin views
            # deja de ser el ancla: sale el ancla mezclada con la de mínima
            # varianza, y en un mandato defensivo esa mezcla se aleja 54 puntos
            # de lo que el Comité aprobó. El modelo estaría sobrescribiendo la
            # asignación estratégica con un parámetro nuestro.
            #
            # Penalizando la desviación se obtienen las dos cosas: sin views el
            # término activo es cero y la cartera ES el Modelo de Asignación,
            # sea cual sea el lambda; con views, el lambda decide cuánto se
            # permite el mandato apartarse de él. Que una cartera conservadora
            # se aparte menos de su asignación estratégica por una view es
            # exactamente lo que significa ser conservadora.
            activo = w - w_ancla
            objective = cp.Maximize(
                (mu - pi_prior) @ w
                - (risk_aversion / 2) * cp.quad_form(activo, cp.psd_wrap(sigma)))
        elif mode == "max_return":
            # Segunda pasada: gastar el presupuesto de riesgo del mandato en vez
            # de buscar el óptimo de utilidad. Solo se usa cuando la cartera
            # sale POR DEBAJO del piso de su perfil, que es el caso en que el
            # cliente está recibiendo menos riesgo del que contrató. El techo de
            # volatilidad y todas las demás restricciones siguen puestas.
            objective = cp.Maximize(mu @ w)
        else:
            objective = cp.Maximize(
                mu @ w - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(sigma)))
        problem = cp.Problem(objective, constraints)

        status = "unsolved"
        for candidate in order:
            try:
                problem.solve(solver=candidate)
                status = problem.status
                if w.value is not None:
                    return np.asarray(w.value).ravel(), status
            except Exception as exc:  # noqa: BLE001 - try the next solver
                notes.append(f"Solver {candidate} falló: {exc}")
        return None, status

    solution, status = solve(frozenset())

    if solution is None:
        # Name the constraint that caused it instead of leaving a bare
        # "infeasible". Dropping one at a time is the only way to know: if the
        # same problem solves once a constraint is gone, that constraint is the
        # reason, and the desk can relax it or widen the basket knowing which
        # of the two to do.
        culpables: list[tuple[str, dict]] = []
        if sectors and sector_cap is not None:
            culpables.append((
                f"El tope sectorial de {sector_cap:.0%} es lo que deja la "
                "cartera sin solución: sin él sí resuelve. La cesta no tiene "
                "suficientes industrias distintas para llenar el libro bajo ese "
                "techo — amplía la cesta o sube el tope, pero decídelo, no lo "
                "descubras por un resultado vacío.", {"with_sectors": False}))
        if pairs:
            culpables.append((
                f"La coherencia con las {len(pairs)} view(s) relativa(s) es lo "
                "que deja la cartera sin solución: sin ella sí resuelve. Alguna "
                "pata larga no puede alcanzar a su corta bajo las bandas — "
                "revisa esa view antes de apagar la restricción.",
                {"with_views": False}))

        for mensaje, kwargs in culpables:
            if solve(frozenset(), **kwargs)[0] is not None:
                infeasible_reasons.insert(0, mensaje)
                break

    if solution is None:
        # A bare "infeasible" is not a diagnosis. Lead with the structural
        # reason when there is one, so the empty result explains itself.
        reasons = infeasible_reasons or [
            "La optimización no encontró solución factible. Revisa que la "
            "cesta cubra las clases de activo que exigen las bandas."
        ]
        return Allocation(pd.Series(0.0, index=tickers), strategy,
                          status or "infeasible", 0.0, 0.0, 0.0,
                          pd.Series(dtype=float),
                          breaches=reasons, notes=notes)

    # --- Minimum position size ------------------------------------------
    #
    # Enforced by re-solving with the offending names forced to zero, not by
    # zeroing them in the answer. Deleting a weight after the fact would leave
    # the book short of its budget and could push a surviving name past its
    # individual cap or a class past its band -- the weights would no longer be
    # the solution to any stated problem, while still being presented as one.
    # Each pass here is a genuine constrained optimization, so every limit in
    # the Investment Procedure still holds exactly.
    def apply_min_position(sol: np.ndarray, st: str, mode: str
                           ) -> tuple[np.ndarray, str, set[str], list[str]]:
        avisos: list[str] = []
        banned: set[str] = set()
        if not (min_position and min_position > 0):
            return sol, st, banned, avisos

        for _ in range(MAX_MIN_POSITION_PASSES):
            held = pd.Series(sol, index=tickers)
            too_small = [t for t, v in held.items()
                         if t not in banned and WEIGHT_EPS < v < min_position]
            if not too_small:
                break
            candidate_ban = banned | set(too_small)
            retry, retry_status = solve(frozenset(candidate_ban), mode=mode)
            if retry is None:
                avisos.append(
                    f"No se pudo aplicar el mínimo de {min_position:.1%} a "
                    f"{sorted(too_small)}: sin ellos la cartera no tiene "
                    "solución factible, así que se conservan. Revisa si la "
                    "cesta da para el número de posiciones que exige el mandato."
                )
                break
            banned, sol, st = candidate_ban, retry, retry_status
        else:
            avisos.append(
                f"El mínimo de {min_position:.1%} no convergió en "
                f"{MAX_MIN_POSITION_PASSES} pasadas; se reporta la última "
                "solución factible."
            )
        if banned:
            avisos.append(
                f"{len(banned)} posición(es) descartada(s) por quedar debajo "
                f"del mínimo de {min_position:.1%}: {', '.join(sorted(banned))}. "
                "El peso se redistribuyó re-optimizando, no repartiendo a mano."
            )
        return sol, st, banned, avisos

    def annual_vol(sol: np.ndarray) -> float:
        return float(np.sqrt(max(float(sol @ sigma @ sol), 0.0)))

    solution, status, _banned, _avisos = apply_min_position(
        solution, status, "utility")
    notes.extend(_avisos)

    # --- Risk floor: spend the mandate's budget --------------------------
    #
    # The utility optimum can land below the floor the mandate implies, and no
    # constraint can pull it up: ``w'Sigma w >= min**2`` is reverse convex. What
    # *is* convex is maximizing return under the ceiling, so when the book comes
    # in short of its own floor it is re-solved that way -- the risk budget
    # becomes something the portfolio is built to use rather than merely to
    # respect. Every other limit stays in force, and the swap only happens if it
    # actually raises the volatility.
    # Solo con views. Sin nada que decir, la cartera ES el ancla del mandato --
    # esa propiedad es la que hace del ancla un neutral de verdad y no se
    # sacrifica por el piso. Si la asignación estratégica del Procedimiento
    # produce menos riesgo del que el rango dice, eso es una contradicción entre
    # dos documentos del Comité, y se reporta como incumplimiento en vez de
    # taparse cambiando la asignación estratégica por una cartera de retorno
    # máximo que nadie aprobó.
    if views and piso_vol is not None and annual_vol(solution) < piso_vol - 1e-6:
        alterna, alterna_status = solve(frozenset(), mode="max_return")
        if alterna is not None:
            alterna, alterna_status, _, avisos_alt = apply_min_position(
                alterna, alterna_status, "max_return")
            if annual_vol(alterna) > annual_vol(solution) + 1e-9:
                notes.append(
                    f"La cartera de utilidad máxima salía en "
                    f"{annual_vol(solution):.2%} de volatilidad, por debajo del "
                    f"piso de {piso_vol:.1%} de {strategy}. Se resolvió otra vez "
                    f"maximizando retorno contra el techo de "
                    f"{float(techo_vol):.1%}, y quedó en {annual_vol(alterna):.2%}. "
                    "El presupuesto de riesgo del mandato es parte del objetivo, "
                    "no solo un límite."
                )
                solution, status = alterna, alterna_status
                notes.extend(avisos_alt)

    weights = pd.Series(solution, index=tickers).clip(lower=0.0)
    weights[weights < WEIGHT_EPS] = 0.0

    by_class = weights.groupby(pd.Series(classes)).sum().sort_values(ascending=False)
    variance = float(weights.values @ sigma @ weights.values)

    allocation = Allocation(
        weights=weights.round(6),
        strategy=strategy,
        status=status,
        gross_exposure=float(weights.sum()),
        expected_return=float(mu @ weights.values),
        volatility=float(np.sqrt(max(variance, 0.0))),
        by_class=by_class,
        notes=notes,
    )
    allocation.breaches = audit_bands(allocation, classes)
    # Checked after the fact even though the solver was given the same ceiling:
    # an audit that only repeats what the solver was told cannot catch a bad
    # sector map, a solver that returned "optimal_inaccurate", or a constraint
    # that never got built.
    allocation.breaches += audit_sectors(allocation.weights, sectors, sector_cap)
    allocation.breaches += view_coherence_breaches(allocation.weights, pairs)

    # El piso de volatilidad no se puede imponer -- es convexo al revés -- así
    # que aquí es donde se detecta. Una cartera por debajo del piso de su
    # mandato no es prudente, es otro mandato.
    if piso_vol is not None and allocation.volatility < piso_vol - 1e-6:
        allocation.risk_findings.append(
            f"Volatilidad {allocation.volatility:.2%} por DEBAJO del piso de "
            f"{piso_vol:.1%} que la mesa fija para {strategy}: la cartera "
            "asume menos riesgo del que el mandato contrató, y ni siquiera "
            "maximizando retorno contra el techo se alcanza. La cesta no da "
            "para este mandato — amplíala o revisa el rango con el Comité."
        )
    if techo_vol is not None and allocation.volatility > techo_vol + 1e-6:
        allocation.risk_findings.append(
            f"Volatilidad {allocation.volatility:.2%} por ENCIMA del techo de "
            f"{techo_vol:.1%} de {strategy}. No se impone como restricción a "
            "propósito: un techo que la cesta no puede cumplir devolvería cero "
            "cartera en vez de esta advertencia."
        )
    if sectors:
        allocation.sector_exposure = sector_exposures(allocation.weights, sectors)
    return allocation


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def audit_bands(allocation: Allocation,
                classes: Mapping[str, str],
                tolerance: float = 1e-4,
                allow_leverage: bool = ALLOW_LEVERAGE) -> list[str]:
    """
    Check solved weights against every limit. Returns the breaches found.

    Replaces CCI's ``auditar_bandas``, which wrote "Auditoría OK" to disk
    without comparing anything. An audit that cannot fail leaves a document
    asserting compliance that nothing verified -- in a regulatory examination
    that is worse than having no document.
    """
    rules = REGULACIONES[allocation.strategy]
    weights = allocation.weights
    breaches: list[str] = []

    # Audited against the budget actually in force, not just the regulatory
    # ceiling. Checking only the mandate would let a book solved under a desk
    # policy of no leverage come back at 113% and still pass, since 113% clears
    # a 125% limit -- the audit would be true and useless.
    budget = gross_budget(allocation.strategy, allow_leverage=allow_leverage)
    if allocation.gross_exposure > budget + tolerance:
        if allow_leverage:
            detalle = (f"(apalancamiento {rules['leverage_max']:.2f} con buffer "
                       f"{LEVERAGE_BUFFER:.0%})")
        else:
            detalle = ("(apalancamiento desactivado por política de mesa; el "
                       f"mandato permitiría {rules['leverage_max']:.0%})")
        breaches.append(
            f"Exposición bruta {allocation.gross_exposure:.2%} excede el "
            f"presupuesto {budget:.2%} {detalle}"
        )

    negative = weights[weights < -tolerance]
    if not negative.empty:
        breaches.append(f"Pesos negativos (no se permiten cortos): {list(negative.index)}")

    for ticker in EXCLUSIONES_DURAS:
        if float(weights.get(ticker, 0.0)) > tolerance:
            breaches.append(f"{ticker} está excluido por Art. 170 RIV y tiene peso")

    for ticker, weight in weights.items():
        if classes.get(ticker) == CLASE_EQUITY and weight > rules["max_equity_individual"] + tolerance:
            breaches.append(
                f"{ticker} pesa {weight:.2%}, sobre el máximo individual "
                f"{rules['max_equity_individual']:.2%}"
            )

    equity_total = sum(w for t, w in weights.items()
                       if classes.get(t) in (CLASE_EQUITY, CLASE_ETF_RV))
    if equity_total > rules["max_equity_total"] + tolerance:
        breaches.append(
            f"Renta variable total {equity_total:.2%} sobre el máximo "
            f"{rules['max_equity_total']:.2%}"
        )

    for clase, (low, high) in bands_for(allocation.strategy).items():
        exposure = float(allocation.by_class.get(clase, 0.0))
        if exposure > high + tolerance:
            breaches.append(f"{clase}: {exposure:.2%} sobre la banda máxima {high:.2%}")
        if exposure < low - tolerance:
            breaches.append(f"{clase}: {exposure:.2%} bajo la banda mínima {low:.2%}")

    return breaches


def drawdown_metrics(weights: pd.Series | Mapping[str, float],
                     returns: pd.DataFrame | None,
                     periods_per_year: int = TRADING_DAYS) -> dict[str, float]:
    """
    Caída de la cartera **en la muestra**, aplicando los pesos de hoy al pasado.

    No es un backtest y no debe presentarse como uno: estos pesos no existían
    entonces, salieron de un modelo que vio ese mismo período. Es la respuesta a
    "cuánto habría caído esta cartera si la hubieras tenido puesta", que sigue
    siendo la pregunta que un comité hace, con la advertencia puesta al lado.

    Devuelve ``max_drawdown`` (la peor caída pico a valle) y ``peor_12m`` (el
    peor retorno móvil de doce meses), los dos como números negativos.
    """
    if returns is None or returns.empty:
        return {}
    held = {t: float(w) for t, w in
            (weights.items() if hasattr(weights, "items") else weights)
            if float(w) > WEIGHT_EPS and t in returns.columns}
    if not held:
        return {}

    serie = returns[list(held)].mul(pd.Series(held), axis=1).sum(axis=1).dropna()
    if len(serie) < periods_per_year // 4:
        return {}

    curva = (1.0 + serie).cumprod()
    caida = curva / curva.cummax() - 1.0
    out = {"max_drawdown": float(caida.min())}

    if len(curva) > periods_per_year:
        rodante = curva / curva.shift(periods_per_year) - 1.0
        out["peor_12m"] = float(rodante.min())
    return out


def risk_profile_table(covariance: pd.DataFrame,
                       asset_types: Mapping[str, str],
                       caps: Mapping[str, float] | None,
                       views: Sequence[Mapping[str, Any]],
                       *, returns: pd.DataFrame | None = None,
                       sector_weights: Mapping[str, Mapping[str, float]] | None = None,
                       strategies: Sequence[str] = tuple(REGULACIONES),
                       min_position: float | None = MIN_POSITION,
                       ) -> tuple[pd.DataFrame, list[str]]:
    """
    Retorno y riesgo esperados de los cuatro mandatos, resueltos de verdad.

    Cada estrategia se resuelve entera — su propia ancla del Modelo de
    Asignación, su propio lambda, sus bandas, su tope sectorial y su techo de
    volatilidad — con la **misma cesta y las mismas views**. Así lo único que
    separa una fila de otra es el mandato.

    Sirve para una pregunta que el sistema no podía contestar: *¿la cartera
    Agresiva asume más riesgo y espera más retorno que la Moderada?* Nada lo
    garantizaba, porque las cuatro optimizaban la misma función y las bandas son
    techos. Las notas devuelven cada inversión del orden que se encuentre.
    """
    filas: list[dict[str, Any]] = []
    for estrategia in strategies:
        presupuesto = gross_budget(estrategia)
        ancla, _ = policy_weights(asset_types, estrategia, caps=caps,
                                  total=presupuesto)
        # El equilibrio usa el lambda del MERCADO, no el del cliente. Son dos
        # cosas distintas y confundirlas rompe la comparación: pi = delta*Sigma*w
        # describe qué retornos hacen del portafolio de mercado un óptimo para
        # el inversionista promedio, y no cambia porque este cliente sea
        # conservador. Al meterle el lambda del mandato, la Agresiva salía con
        # retornos esperados mecánicamente más bajos que la Moderada — un
        # artefacto de escala, no una propiedad de la cartera. El apetito del
        # cliente vive en la función objetivo, que es donde lo aplica optimize().
        pi = implied_equilibrium(ancla, covariance, risk_aversion=RISK_AVERSION)
        er, cov_post = posterior(pi, covariance, views)
        lam = risk_aversion_for(estrategia, RISK_AVERSION)
        alloc = optimize(er, cov_post, asset_types, estrategia,
                         min_position=min_position,
                         sector_weights=sector_weights, views=views,
                         anchor=ancla, prior=pi)

        piso, techo = RISK_TARGETS.get(estrategia, (float("nan"),) * 2)
        fila = {
            "estrategia": estrategia,
            "lambda": lam,
            "estado": alloc.status,
            "retorno_esperado": alloc.expected_return if alloc.feasible else float("nan"),
            "volatilidad": alloc.volatility if alloc.feasible else float("nan"),
            "vol_min_objetivo": piso,
            "vol_max_objetivo": techo,
            "posiciones": int((alloc.weights > WEIGHT_EPS).sum()),
            "incumplimientos": " | ".join(alloc.breaches),
            "riesgo_vs_mandato": " | ".join(alloc.risk_findings) or "dentro del rango",
        }
        fila.update(drawdown_metrics(alloc.weights, returns))
        # Pérdida anual que solo se supera 1 año de cada 20, bajo normalidad.
        # La normalidad es falsa en las colas y por eso el número va etiquetado
        # en la hoja: subestima lo que pasa en un mercado malo de verdad.
        if alloc.feasible:
            fila["caida_1a_95"] = (alloc.expected_return
                                   - 1.645 * alloc.volatility)
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    return tabla, coherence_notes(tabla)


def coherence_notes(tabla: pd.DataFrame) -> list[str]:
    """
    Lo que hay que mirar en la tabla de riesgo antes de creérsela.

    Separado de :func:`risk_profile_table` para poder verificarlo sin resolver
    cuatro optimizaciones: una comprobación que solo se ejerce a través de un
    solver acaba sin probarse en el caso que importa, que es cuando falla.
    """
    notas: list[str] = []
    if tabla.empty or "retorno_esperado" not in tabla:
        return notas

    viables = tabla[tabla["retorno_esperado"].notna()]
    for columna, etiqueta in (("volatilidad", "riesgo"),
                              ("retorno_esperado", "retorno esperado")):
        valores = list(viables[columna])
        if valores != sorted(valores):
            orden = ", ".join(f"{e} {v:.2%}" for e, v in
                              zip(viables["estrategia"], valores))
            notas.append(
                f"El {etiqueta} NO crece con el perfil: {orden}. Un mandato más "
                "agresivo que asume menos que uno más conservador es una "
                "contradicción con lo que el cliente firmó."
            )

    fuera = viables[(viables["volatilidad"] < viables["vol_min_objetivo"] - 1e-6)
                    | (viables["volatilidad"] > viables["vol_max_objetivo"] + 1e-6)]
    for _, f in fuera.iterrows():
        notas.append(
            f"{f['estrategia']}: volatilidad {f['volatilidad']:.2%} fuera de su "
            f"rango objetivo {f['vol_min_objetivo']:.1%}–{f['vol_max_objetivo']:.1%}."
        )
    return notas


def allocation_table(allocation: Allocation,
                     names: Mapping[str, str] | None = None,
                     classes: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Held positions, largest first, ready to display or export."""
    held = allocation.weights[allocation.weights > 0].sort_values(ascending=False)
    names = names or {}
    classes = classes or {}
    return pd.DataFrame({
        "ticker": held.index,
        "nombre": [names.get(t, t) for t in held.index],
        "clase_activo": [classes.get(t, "") for t in held.index],
        "peso": held.values,
    })
