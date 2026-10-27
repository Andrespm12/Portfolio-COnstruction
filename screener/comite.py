"""
Todo lo que el modelo usa y el Procedimiento de Inversión no dice, en una lista.

El review lo pidió así: el techo de materias primas, los topes sectoriales,
los rangos de volatilidad y el lambda por perfil estaban marcados "de la
mesa, no del Procedimiento" cada uno en su celda, y nada impedía que uno
saliera a producción sin que nadie lo aprobara. Esta lista se arma leyendo las
constantes vivas del motor, no copiándolas: si alguien cambia un número en el
código, la hoja del Comité cambia con él.

Cada fila dice de dónde sale el número:

* **mesa** — lo puso la mesa como criterio; es una decisión, no un cálculo.
* **derivado** — sale de una fórmula sobre otros supuestos; se discute el
  supuesto, no el resultado.
* **supuesto** — un parámetro de mercado o de modelo sin calibrar.
* **asesoría** — no es un número del modelo sino algo que el modelo no ve y
  que necesita opinión fiscal o legal. No es asesoría fiscal.
"""

from __future__ import annotations

from .black_litterman import DEFAULT_PARAMS, HORIZONTE_VIEWS_MESES
from .cci_regulation import (
    COMMODITY_BANDS, DRAWDOWN_TOLERADO, EXPOSICIONES_NUCLEO, RETORNO_ESPERADO,
    RISK_AVERSION_BY_STRATEGY, RISK_TARGETS, SATELITE_MAX, SECTOR_CAPS,
    SUPUESTOS_CLASE, SUPUESTOS_CORRELACION, TRACKING_ERROR_OBJETIVO,
    VOL_ACTIVA_SATELITE,
)
from .optimizer import ALLOW_LEVERAGE, MIN_POSITION, TAU
from .profiles import profile_for_strategy

ESTRATEGIAS = ("Conservador_Defensivo", "Conservador", "Moderado", "Agresivo")
_CORTO = {"Conservador_Defensivo": "CD", "Conservador": "C",
          "Moderado": "M", "Agresivo": "A"}


def _por_mandato(valores, fmt="{:.1%}") -> str:
    return " · ".join(f"{_CORTO[e]} {fmt.format(valores[e])}"
                      for e in ESTRATEGIAS if e in valores)


def _rango(valores) -> str:
    return " · ".join(f"{_CORTO[e]} {lo:.1%}–{hi:.1%}"
                      for e in ESTRATEGIAS for lo, hi in [valores[e]])


def pendientes() -> list[dict[str, str]]:
    """Una fila por decisión abierta, con su valor vigente y su origen."""
    perfiles = {e: profile_for_strategy(e) for e in ESTRATEGIAS}
    filas: list[dict[str, str]] = []

    def fila(tema, parametro, valor, origen, nota=""):
        filas.append({"tema": tema, "parametro": parametro, "valor": valor,
                      "origen": origen, "nota": nota,
                      "estado": "pendiente del Comité"})

    # --- Cartera -------------------------------------------------------
    fila("Cartera", "Techo del satélite (SATELITE_MAX)",
         _por_mandato(SATELITE_MAX), "derivado",
         f"TE máximo del mandato / {VOL_ACTIVA_SATELITE:.0%} de volatilidad "
         "activa supuesta del satélite.")
    fila("Cartera", "Volatilidad activa del satélite (VOL_ACTIVA_SATELITE)",
         f"{VOL_ACTIVA_SATELITE:.0%}", "supuesto",
         "Canasta concentrada de acciones contra el índice que reemplaza.")
    fila("Cartera", "Vehículos núcleo de renta variable",
         " | ".join(f"{k}: {', '.join(v)}" for k, v in EXPOSICIONES_NUCLEO.items()),
         "mesa", "Qué exposiciones indexadas forman el núcleo.")
    fila("Cartera", "Techo de materias primas (COMMODITY_BANDS)",
         _rango(COMMODITY_BANDS), "mesa",
         "El Procedimiento no tiene clase de materias primas.")
    fila("Cartera", "Tope sectorial, fracción del sleeve de RV (SECTOR_CAPS)",
         _por_mandato({e: v for e, v in SECTOR_CAPS.items() if v is not None},
                      "{:.0%}"), "mesa", "")
    fila("Cartera", "Posición mínima ejecutable", f"{MIN_POSITION:.1%}", "mesa", "")
    fila("Cartera", "Apalancamiento", "apagado" if not ALLOW_LEVERAGE else "encendido",
         "mesa", "El Procedimiento permite 125% / 150% en Moderado / Agresivo.")

    # --- Riesgo ----------------------------------------------------------
    fila("Riesgo", "Rango de volatilidad anual (RISK_TARGETS)",
         _rango(RISK_TARGETS), "derivado",
         "Piso 80% de la volatilidad neutral; techo en el tope regulatorio de RV.")
    fila("Riesgo", "Retorno estratégico esperado (RETORNO_ESPERADO)",
         _por_mandato(RETORNO_ESPERADO), "derivado",
         "Supuestos por clase aplicados al Modelo de Asignación.")
    fila("Riesgo", "Drawdown tolerado (DRAWDOWN_TOLERADO)",
         _por_mandato(DRAWDOWN_TOLERADO, "{:.0%}"), "derivado",
         "2,5σ en los conservadores, 3,0σ en los agresivos.")
    fila("Riesgo", "Tracking error contra el Modelo (TRACKING_ERROR_OBJETIVO)",
         _rango(TRACKING_ERROR_OBJETIVO), "mesa",
         "También limita la segunda pasada de retorno máximo.")
    fila("Riesgo", "Aversión al riesgo λ (RISK_AVERSION_BY_STRATEGY)",
         _por_mandato(RISK_AVERSION_BY_STRATEGY, "{:g}"), "mesa",
         "Moderado 2,5 viene del documento de CCI; el resto lo puso la mesa.")
    fila("Riesgo", "Supuestos por clase",
         " | ".join(f"{c}: retorno {ret:.1%}, volatilidad {vol:.1%}"
                    for c, (vol, ret) in SUPUESTOS_CLASE.items()),
         "supuesto", "De ellos salen las bandas, el retorno y el drawdown.")
    fila("Riesgo", "Correlaciones entre clases",
         " | ".join(f"{'–'.join(sorted(par))}: {rho:+.2f}"
                    for par, rho in SUPUESTOS_CORRELACION.items()),
         "supuesto", "2022 mostró que acciones-bonos puede volverse positiva.")

    # --- Screener ------------------------------------------------------
    for e in ESTRATEGIAS:
        p = perfiles[e]
        fila("Screener", f"Pesos por bloque — {p.label}",
             " · ".join(f"{k} {v:.0%}" for k, v in p.block_weights.items()),
             "mesa", "Momentum y retorno ajustado por riesgo correlacionan "
             "+0,90: es una apuesta contada dos veces.")
    fila("Screener", "Beta máxima para Overweight",
         _por_mandato({e: perfiles[e].gates.beta_limit for e in ESTRATEGIAS},
                      "{:.2f}"), "mesa", "Gate independiente del drawdown.")
    fila("Screener", "Drawdown de 1 año que capa a Market Weight",
         _por_mandato({e: perfiles[e].gates.max_drawdown_limit
                       for e in ESTRATEGIAS}, "{:.0%}"), "mesa", "")
    fila("Screener", "Volatilidad máxima para Overweight",
         _por_mandato({e: perfiles[e].gates.max_volatility_for_overweight
                       for e in ESTRATEGIAS}, "{:.0%}"), "mesa", "")

    # --- Views ---------------------------------------------------------
    fila("Views", "Coeficiente de información (IC)",
         f"{DEFAULT_PARAMS.information_coefficient:g}", "supuesto",
         "Sin calibrar: requiere backtest walk-forward con datos sin sesgo "
         "de supervivencia. La hoja Sensibilidad muestra IC ±50%.")
    fila("Views", "Q máxima por view", f"±{DEFAULT_PARAMS.max_abs_q:.0%}", "mesa", "")
    fila("Views", "τ (incertidumbre del equilibrio)", f"{TAU:g}", "supuesto", "")
    fila("Views", "Horizonte de las views", f"{HORIZONTE_VIEWS_MESES} meses",
         "mesa", "La convicción decae a cero al vencer (aplicar_vigencia).")

    # --- Cliente y jurisdicción ---------------------------------------
    fila("Jurisdicción", "Retención sobre dividendos de EE.UU.",
         "30% (sin tratado RD–EE.UU., según el review)", "asesoría",
         "El modelo trata el dividend yield como carry positivo; para estos "
         "clientes parte es costo fiscal. Los UCITS irlandeses de acumulación "
         "suelen sufrir ~15% a nivel fondo. Confirmar con asesoría fiscal.")
    fila("Jurisdicción", "Impuesto sucesoral de EE.UU.",
         "exposición sobre ~USD 60.000 en activos situs EE.UU.", "asesoría",
         "Afecta el domicilio de los vehículos núcleo (ETFs de EE.UU. vs "
         "UCITS). Es una pregunta de idoneidad; confirmar con asesoría legal.")
    fila("Jurisdicción", "Política cambiaria (DOP)", "sin definir", "asesoría",
         "Todo está en USD. Clientes que miden su patrimonio en pesos "
         "necesitan una política de moneda por perfil.")
    fila("Jurisdicción", "Restricciones por cliente (IPS)", "sin definir", "mesa",
         "Necesidades de ingreso, liquidez y horizonte distintos dentro de un "
         "mismo perfil.")
    return filas
