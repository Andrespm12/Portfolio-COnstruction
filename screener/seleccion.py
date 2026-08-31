"""
Política de selección del universo: qué instrumentos entran a evaluarse.

El problema que resuelve
------------------------
Antes de esto había tres capas de decisión y solo dos estaban escritas:

1. Exclusión por tipo de producto (regex sobre el nombre) — escrita.
2. Elegibilidad por precio, volumen y liquidación — escrita.
3. **Cuáles nombres del índice llegaron siquiera a la lista** — no escrita.
   Era una lista tecleada a mano en :mod:`screener.universe`.

La tercera capa es la peor de las tres, y no por ser subjetiva: las otras dos
también encierran juicio. Es la peor porque es **invisible**. Un nombre que la
capa 2 rechaza sale en el reporte con su motivo — ``ADV $39.3MM below $50MM
minimum`` — y cualquiera puede discutirlo. Un nombre que nunca se tecleó no
aparece en ningún lado; para auditarlo hay que abrir el código y contar.

Este módulo convierte la selección en una regla declarada, reproducible y con
rastro. No elimina el juicio: lo pone donde se puede discutir.

La regla que no se puede romper
-------------------------------
**No se selecciona sobre nada que el modelo puntúe como desempeño.**

Si el universo se filtrara por momentum, por Sharpe o por retorno, el puntaje
dejaría de significar algo: como es transversal, quitar a los perdedores antes
de puntuar hace que todos los que quedan parezcan promedio. Se estaría metiendo
la respuesta en la pregunta.

Por eso los criterios admisibles son de tres tipos, y ninguno mira el
desempeño del activo:

* **Tipo de producto.** Un ETF apalancado tiene retorno dependiente del camino;
  las métricas calculadas sobre esa serie no son comparables con las de una
  acción. Es incomparabilidad, no preferencia.
* **Suficiencia de datos.** Un nombre sin historia suficiente no es peor: es
  *no medible* por este modelo. Ver :func:`barras_requeridas`.
* **Negociabilidad.** Precio, volumen y días para liquidar. Es la restricción
  institucional real.

El tercero tiene una consecuencia que conviene declarar en vez de esconder:
la liquidez **también se puntúa** (bloque de liquidez, 11% en Moderado), así
que filtrar por volumen trunca la distribución de ese bloque. El efecto es de
segundo orden y el filtro es innegociable en una mesa, pero el sesgo existe.

Lo que este módulo NO arregla
-----------------------------
El conjunto de candidatos. Sigue saliendo del snapshot estático de
:mod:`screener.universe`, con su sesgo de supervivencia. Esta política decide
bien sobre lo que se le da; no puede inventar lo que nunca le llegó.
:func:`screener.universe.load_membership_override` es el punto de enganche.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import ELIGIBILITY, FACTOR_MODEL, EligibilityRules
from .universe import is_excluded_product, screen_eligibility

# --------------------------------------------------------------------------
# Cuántas barras necesita cada métrica
# --------------------------------------------------------------------------
#
# Se declara aquí en vez de inferirse, porque los requisitos viven implícitos
# en los argumentos de las funciones de metrics.py y una tabla que se puede
# desincronizar en silencio no sirve de nada. `tests/test_seleccion.py` verifica
# cada entrada contra la función real: en el número declarado la métrica se
# calcula, y una barra menos devuelve None. Si alguien cambia un lookback, la
# prueba falla.

BARRAS_REQUERIDAS: dict[str, int | None] = {
    # Momentum y tendencia
    "mom_12_1": 53,          # total_return(lookback=48, skip=4)
    "mom_6m": 27,            # total_return(lookback=26)
    "mom_3m": 14,            # total_return(lookback=13)
    "above_40w_ma": 40,      # pct_above_ma(window=40)
    "ma_slope_13w": 53,      # ma_slope(window=40, lookback=13)
    "pct_from_52w_high": None,   # del snapshot
    # Retorno ajustado por riesgo
    "sharpe_1y": 4, "sortino_1y": 4, "calmar_1y": 4, "pct_positive_periods": 4,
    # Volatilidad y caídas
    "volatility_1y": 4, "max_drawdown": 3, "downside_deviation": 4, "ulcer_index": 3,
    # Sensibilidad al mercado
    "beta_1y": 13, "alpha_annual": 13, "capture_spread": 13, "idio_vol_share": 13,
    # Liquidez
    "adv_usd_log": None, "days_to_liquidate": None, "turnover_stability": 8,
    # Valuación y carry
    "dividend_yield": None, "iv_hv_spread": None, "iv_percentile": None,
    "range_position": None,
    # Ajuste al portafolio (solo vía IBKR)
    "corr_to_portfolio": 13, "diversification_benefit": 13, "existing_overlap": None,
}


def barras_requeridas(cobertura_objetivo: float = 1.0,
                      modelo: Sequence[Any] | None = None) -> int:
    """
    Barras necesarias para que el modelo vigente alcance ``cobertura_objetivo``.

    Se deriva del modelo de factores en uso, no de una constante: si mañana se
    cambian los pesos o se quita un bloque, este número se mueve solo.

    Con el modelo por defecto, la cobertura total exige 53 barras — que es lo
    que pide ``mom_12_1``, la métrica de mayor peso individual de todo el
    modelo (0.35 dentro de un bloque que pesa 25% en Moderado).
    """
    modelo = modelo if modelo is not None else FACTOR_MODEL
    candidatas = sorted({b for b in BARRAS_REQUERIDAS.values() if b} | {0})

    for barras in candidatas:
        num = den = 0.0
        for bloque in modelo:
            pesos = bloque.normalized_metric_weights()
            for m in bloque.metrics:
                req = BARRAS_REQUERIDAS.get(m.key)
                aporte = bloque.weight * pesos[m.key]
                den += aporte
                if req is None or barras >= req:
                    num += aporte
        if den > 0 and num / den >= cobertura_objetivo - 1e-9:
            return barras
    return max(candidatas)


# --------------------------------------------------------------------------
# Criterios declarados
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Criterio:
    clave: str
    titulo: str
    razon: str
    #: True si el criterio filtra por algo que el modelo además puntúa. Solo
    #: la negociabilidad lo hace, y se declara para que el sesgo quede dicho.
    trunca_un_bloque: bool = False


CRITERIOS: tuple[Criterio, ...] = (
    Criterio(
        "producto",
        "Tipo de producto",
        "Apalancados, inversos, de un solo subyacente, covered-call y ETNs. "
        "Su retorno depende del camino del precio, no del punto de llegada, "
        "así que las métricas no son comparables con las de una acción normal.",
    ),
    Criterio(
        "mercado",
        "Listado y pertenencia",
        "Listado en EE.UU. y miembro de al menos un índice del universo, o "
        "ETF de la lista. Es la definición de alcance del mandato.",
    ),
    Criterio(
        "historia",
        "Suficiencia de datos",
        "Historia suficiente para que el modelo se pueda calcular completo. "
        "Un nombre corto de historia no es peor: es no medible sobre la misma "
        "base que el resto, y meterlo compara un compuesto de tres métricas "
        "contra uno de seis.",
    ),
    Criterio(
        "negociabilidad",
        "Negociabilidad",
        "Precio mínimo, volumen diario mínimo y días para liquidar. Define si "
        "la posición se puede construir y deshacer sin pagar impacto.",
        trunca_un_bloque=True,
    ),
)

#: Criterios que quedan explícitamente prohibidos, y por qué. Está escrito
#: porque un criterio ausente no se puede auditar: si alguien agrega mañana un
#: filtro por momentum, esta lista es contra qué se compara.
PROHIBIDOS: tuple[tuple[str, str], ...] = (
    ("desempeño",
     "Retorno, momentum, Sharpe, volatilidad, drawdown o beta. Quitar a los "
     "perdedores antes de puntuar hace que los que quedan parezcan promedio: "
     "es meter la respuesta en la pregunta."),
    ("sector",
     "Balancear el universo por sector. La nota es transversal, así que la "
     "composición define contra quién se mide cada nombre; equilibrarla a mano "
     "es decidir el resultado por la puerta de atrás."),
    ("tamaño",
     "Capitalización, más allá del piso de liquidez que ya impone la "
     "negociabilidad."),
)


# --------------------------------------------------------------------------
# Decisión por nombre
# --------------------------------------------------------------------------

@dataclass
class Decision:
    ticker: str
    admitido: bool
    criterio: str = ""
    detalle: str = ""
    barras: int | None = None


def _barras(instrumento: Mapping[str, Any]) -> int:
    historia = instrumento.get("history") or {}
    cierres = [c for c in (historia.get("close") or [])
               if isinstance(c, (int, float))]
    return len(cierres)


def evaluar(instrumento: Mapping[str, Any], metricas: Mapping[str, Any], *,
            reglas: EligibilityRules | None = None,
            minimo_barras: int | None = None) -> Decision:
    """
    Decide si un instrumento entra al universo evaluado, y deja dicho por qué.

    ``minimo_barras`` por defecto es lo que el modelo vigente necesita para
    cobertura completa (:func:`barras_requeridas`), no la constante suelta de
    ``EligibilityRules.min_history_bars`` — que está en 30 y deja pasar nombres
    con la mitad del bloque de momentum sin calcular.
    """
    reglas = reglas or ELIGIBILITY
    ticker = str(instrumento.get("ticker", "")).upper()
    barras = _barras(instrumento)

    excluido, patron = is_excluded_product(instrumento.get("name", "") or "")
    if excluido:
        return Decision(ticker, False, "producto",
                        f"producto excluido (coincide /{patron}/)", barras)

    minimo = barras_requeridas() if minimo_barras is None else minimo_barras
    if barras and barras < minimo:
        return Decision(ticker, False, "historia",
                        f"{barras} barras, el modelo necesita {minimo} para "
                        "calcularse completo", barras)

    ok, motivos = screen_eligibility(dict(instrumento), dict(metricas), reglas)
    if not ok:
        de_mercado = ("not US-listed", "exchange", "not a member")
        criterio = ("mercado" if any(m.startswith(de_mercado) for m in motivos)
                    else "negociabilidad")
        return Decision(ticker, False, criterio, "; ".join(motivos), barras)

    return Decision(ticker, True, "", "", barras)


def seleccionar(instrumentos: Sequence[Mapping[str, Any]],
                metricas_por_ticker: Mapping[str, Mapping[str, Any]], *,
                reglas: EligibilityRules | None = None,
                minimo_barras: int | None = None) -> tuple[list[str], list[Decision]]:
    """Aplica la política a un conjunto de candidatos. Devuelve admitidos y rastro."""
    decisiones = [
        evaluar(inst, metricas_por_ticker.get(str(inst.get("ticker", "")).upper(), {}),
                reglas=reglas, minimo_barras=minimo_barras)
        for inst in instrumentos
    ]
    return [d.ticker for d in decisiones if d.admitido], decisiones


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------

def resumen(decisiones: Sequence[Decision]) -> dict[str, int]:
    """Cuántos entraron y cuántos cayó cada criterio."""
    out = {"candidatos": len(decisiones),
           "admitidos": sum(1 for d in decisiones if d.admitido)}
    for c in CRITERIOS:
        out[c.clave] = sum(1 for d in decisiones
                           if not d.admitido and d.criterio == c.clave)
    return out


def tabla(decisiones: Sequence[Decision]):
    """El rastro completo, listo para la hoja «Universo» del Excel."""
    import pandas as pd

    return pd.DataFrame([{
        "ticker": d.ticker,
        "admitido": "sí" if d.admitido else "no",
        "criterio": d.criterio or "",
        "motivo": d.detalle,
        "barras": d.barras,
    } for d in sorted(decisiones, key=lambda x: (x.admitido, x.ticker))])


def politica_declarada() -> str:
    """La política en texto, para el reporte y para la hoja de parámetros."""
    lineas = [f"Política de selección — {barras_requeridas()} barras mínimas "
              "para cobertura completa del modelo vigente", ""]
    lineas.append("Criterios aplicados:")
    for c in CRITERIOS:
        marca = "  (trunca un bloque puntuado)" if c.trunca_un_bloque else ""
        lineas.append(f"  · {c.titulo}{marca}")
        lineas.append(f"      {c.razon}")
    lineas.append("")
    lineas.append("Criterios explícitamente prohibidos:")
    for clave, razon in PROHIBIDOS:
        lineas.append(f"  · {clave}: {razon}")
    return "\n".join(lineas)
