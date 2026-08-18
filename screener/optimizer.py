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
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cci_regulation import (
    CLASE_EQUITY, CLASE_ETF_RV, EXCLUSIONES_DURAS, REGULACIONES, bands_for,
    classify_for_bands, unbanded_classes,
)

#: Global risk-aversion coefficient. CCI's document parameterizes it at 2.5 for
#: a risk-neutral investor; the same value drives the equilibrium and the
#: objective so the two stay consistent.
RISK_AVERSION = 2.5

#: Uncertainty in the equilibrium itself. Fixed low, per CCI's document.
TAU = 0.025

#: Fraction of the leverage limit actually used, per CCI's technical document,
#: so a solution never sits exactly on the regulatory margin.
LEVERAGE_BUFFER = 0.95

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

    weights: dict[str, float] = {}
    for clase in present:
        members = [t for t in tickers if classes[t] == clase]
        budget = class_target[clase]
        usable = {t: c for t in members if (c := usable_cap(t)) is not None}
        if len(usable) == len(members) and members:
            scale = sum(usable.values())
            for t in members:
                weights[t] = budget * usable[t] / scale
        else:
            if members and usable:
                notes.append(
                    f"Capitalización faltante en {clase} para "
                    f"{sorted(set(members) - set(usable))}; esa clase se reparte "
                    "en partes iguales."
                )
            for t in members:
                weights[t] = budget / len(members) if members else 0.0

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

def select_basket(scored: Sequence[Any], strategy: str, top_n: int = 25,
                  min_per_class: int = 3) -> list[str]:
    """
    Choose the optimizer's basket so the mandate's bands are actually reachable.

    Top-N by score alone is not enough, and the failure is silent. The screener
    ranks on momentum and risk-adjusted return, which equities dominate, so the
    top of the list is routinely all equity. Under Moderado total equity caps at
    60% while the book must be roughly fully invested -- the solver returns
    infeasible and the portfolio comes out empty with no obvious cause.

    CCI's system never hits this because its basket comes from a hand-maintained
    sheet that deliberately spans bonds, credit, cash and equity. Replacing that
    sheet means reproducing that property: take the top names by score, then
    ensure every asset class available in the universe has at least a few
    representatives, chosen by score within the class.
    """
    ranked = [r for r in scored if getattr(r, "eligible", True)]
    picked = [r.ticker for r in ranked[:max(top_n, 0)]]
    chosen = set(picked)

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

    return picked


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
    required = budget if budget is not None else rules["leverage_max"] * LEVERAGE_BUFFER
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

    @property
    def feasible(self) -> bool:
        return self.status in {"optimal", "optimal_inaccurate"}


def optimize(expected_returns: pd.Series, covariance: pd.DataFrame,
             asset_types: Mapping[str, str], strategy: str,
             risk_aversion: float = RISK_AVERSION,
             solver: str | None = None) -> Allocation:
    """
    Maximize ``w'mu - (lambda/2) w'Sigma w`` under CCI's Investment Procedure.

    Long-only, per-class bands, a total-equity ceiling, a per-name cap on single
    stocks, hard exclusions, and a gross-exposure budget that honours
    ``leverage_max`` with the documented buffer.
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

    notes: list[str] = []
    budget_required = float(rules["leverage_max"]) * LEVERAGE_BUFFER
    infeasible_reasons = feasibility_report(classes, strategy, budget_required)
    notes.extend(infeasible_reasons)

    orphaned = unbanded_classes(classes, strategy)
    if orphaned:
        notes.append(
            f"Clases sin banda declarada: {sorted(orphaned)}. Quedan sin techo; "
            "confirmar con Compliance antes de operar."
        )

    w = cp.Variable(len(tickers))

    # Gross exposure. leverage_max of 1.0 collapses this to the fully invested
    # long-only case; above 1.0 it is a real budget, which CCI's original code
    # never applied.
    budget = float(rules["leverage_max"]) * LEVERAGE_BUFFER
    constraints = [w >= 0, cp.sum(w) <= budget]
    if rules["leverage_max"] <= 1.0:
        constraints.append(cp.sum(w) == budget)
    else:
        # Stay invested: without a floor the optimizer can sit in cash and
        # report a technically optimal empty book.
        constraints.append(cp.sum(w) >= 1.0)

    for i, ticker in enumerate(tickers):
        if ticker in EXCLUSIONES_DURAS:
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

    objective = cp.Maximize(mu @ w - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, constraints)

    # CLARABEL ships with CVXPY. CCI's code asked for ECOS, which is optional
    # and absent from a stock Colab -- that is what killed their saved run.
    order = ([solver] if solver else
             [s for s in ("CLARABEL", "SCS", "OSQP", "ECOS")
              if s in cp.installed_solvers()])
    status = "unsolved"
    for candidate in order:
        try:
            problem.solve(solver=candidate)
            status = problem.status
            if w.value is not None:
                break
        except Exception as exc:  # noqa: BLE001 - try the next solver
            notes.append(f"Solver {candidate} falló: {exc}")

    if w.value is None:
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

    weights = pd.Series(np.asarray(w.value).ravel(), index=tickers).clip(lower=0.0)
    weights[weights < 1e-6] = 0.0

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
    return allocation


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------

def audit_bands(allocation: Allocation,
                classes: Mapping[str, str],
                tolerance: float = 1e-4) -> list[str]:
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

    budget = rules["leverage_max"] * LEVERAGE_BUFFER
    if allocation.gross_exposure > budget + tolerance:
        breaches.append(
            f"Exposición bruta {allocation.gross_exposure:.2%} excede el "
            f"presupuesto {budget:.2%} (apalancamiento {rules['leverage_max']:.2f} "
            f"con buffer {LEVERAGE_BUFFER:.0%})"
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
