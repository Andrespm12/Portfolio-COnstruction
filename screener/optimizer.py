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


def implied_equilibrium(weights: pd.Series, covariance: pd.DataFrame,
                        risk_aversion: float = RISK_AVERSION) -> pd.Series:
    """Reverse optimization: ``pi = lambda * Sigma * w_mkt``."""
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
        return Allocation(pd.Series(0.0, index=tickers), strategy,
                          status or "infeasible", 0.0, 0.0, 0.0,
                          pd.Series(dtype=float),
                          breaches=["La optimización no encontró solución "
                                    "factible; revisa bandas y cesta."],
                          notes=notes)

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
