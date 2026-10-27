"""
Sensibilidad de la cartera a los parámetros que nadie ha calibrado.

El review lo pidió así: "si cambios pequeños de parámetro voltean la
asignación, el Comité tiene que verlo". Tres parámetros mandan sobre las views
y ninguno está estimado contra datos:

* **IC** (0,08): convierte el ranking en retorno esperado, ``Q = IC·z·σ``. Es
  un supuesto declarado; sin backtest no hay forma de saber si es 0,04 o 0,12.
* **τ** (0,025): cuánta incertidumbre se le asigna al equilibrio. Más τ, más
  pesan las views.
* **λ** del mandato: cuánto riesgo activo se permite contra el ancla.

Cada fila resuelve la misma cartera con un parámetro movido y lo demás
quieto, y mide la **rotación** contra la cartera base: ``½·Σ|w − w_base|``, la
fracción del libro que habría que operar para pasar de una a otra. Una
rotación de 2% es ruido; una de 20% significa que la cartera la decide el
parámetro, no las views.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .black_litterman import DEFAULT_PARAMS
from .cci_regulation import risk_aversion_for
from .optimizer import RISK_AVERSION, TAU, Allocation, optimize, posterior

#: Rotación a partir de la cual la corrida avisa que la asignación depende del
#: parámetro más que de las views.
ROTACION_SENSIBLE = 0.10


def rotacion(a: pd.Series, b: pd.Series) -> float:
    """``½·Σ|a − b|`` sobre la unión de nombres: la fracción del libro a operar."""
    idx = a.index.union(b.index)
    return float(0.5 * (a.reindex(idx, fill_value=0.0)
                        - b.reindex(idx, fill_value=0.0)).abs().sum())


def escalar_ic(views: Sequence[Mapping[str, Any]], factor: float,
               max_abs_q: float = DEFAULT_PARAMS.max_abs_q) -> list[dict]:
    """
    Las mismas views con el IC multiplicado por ``factor``.

    ``Q = IC·z·σ`` es lineal en el IC, pero después se recorta a ``max_abs_q``.
    Escalar la Q ya recortada subestimaría el efecto en las views que tocaron el
    techo, así que se escala el valor previo al recorte (``_q_bruto``) cuando
    la view lo trae, y se vuelve a recortar.
    """
    out = []
    for v in views:
        q = float(v["Q"])
        bruto = abs(float(v.get("_q_bruto", q)))
        nuevo = min(bruto * factor, max_abs_q)
        out.append({**v, "Q": float(np.sign(q) * nuevo) if q else 0.0})
    return out


def tabla_sensibilidad(pi: pd.Series, covariance: pd.DataFrame,
                       views: Sequence[Mapping[str, Any]],
                       asset_types: Mapping[str, str], strategy: str, *,
                       base: Allocation,
                       anchor: pd.Series | None = None,
                       factores_ic: Iterable[float] = (0.5, 1.5),
                       taus: Iterable[float] = (TAU / 2, TAU * 2),
                       factores_lambda: Iterable[float] = (0.5, 2.0),
                       **optimize_kwargs: Any) -> tuple[pd.DataFrame, list[str]]:
    """
    Una fila por variante: el parámetro, su valor, y cómo quedó la cartera.

    ``base`` es la cartera de la corrida, resuelta con los parámetros vigentes.
    ``optimize_kwargs`` son los mismos que recibió esa cartera (tope sectorial,
    satélite, mínimo de posición...), para que lo único distinto sea el
    parámetro.
    """
    lam_base = risk_aversion_for(strategy, RISK_AVERSION)
    variantes: list[tuple[str, str, list, float, float]] = []
    for k in factores_ic:
        variantes.append(("IC", f"x{k:g}", escalar_ic(views, k), TAU, lam_base))
    for t in taus:
        variantes.append(("τ", f"{t:g}", list(views), t, lam_base))
    for k in factores_lambda:
        variantes.append(("λ", f"{lam_base * k:g} (x{k:g})", list(views), TAU,
                          lam_base * k))

    filas = [{
        "parametro": "base", "valor": f"IC x1, τ {TAU:g}, λ {lam_base:g}",
        "estado": base.status, "retorno_esperado": base.expected_return,
        "volatilidad": base.volatility,
        "satelite": base.satellite_exposure if base.satellite else float("nan"),
        "posiciones": int((base.weights > 1e-6).sum()), "rotacion_vs_base": 0.0,
    }]
    notas: list[str] = []
    for parametro, valor, vistas, tau, lam in variantes:
        er, cov_post = posterior(pi, covariance, vistas, tau=tau)
        alloc = optimize(er, cov_post, asset_types, strategy, views=vistas,
                         anchor=anchor, prior=pi, risk_aversion=lam,
                         **optimize_kwargs)
        rot = (rotacion(alloc.weights, base.weights)
               if alloc.feasible and base.feasible else float("nan"))
        filas.append({
            "parametro": parametro, "valor": valor, "estado": alloc.status,
            "retorno_esperado": alloc.expected_return if alloc.feasible else float("nan"),
            "volatilidad": alloc.volatility if alloc.feasible else float("nan"),
            "satelite": (alloc.satellite_exposure
                         if alloc.feasible and alloc.satellite else float("nan")),
            "posiciones": int((alloc.weights > 1e-6).sum()),
            "rotacion_vs_base": rot,
        })
        if np.isfinite(rot) and rot > ROTACION_SENSIBLE:
            notas.append(
                f"Con {parametro} = {valor} la cartera rota {rot:.0%} del libro "
                "contra la base: la asignación depende de ese parámetro más que "
                "de las views. El Comité debería verlo antes de aprobarla.")
    return pd.DataFrame(filas), notas

