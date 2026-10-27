"""
Estrés histórico: qué le habría pasado a la cartera en 2008, 2020 y 2022.

Lo pidió el review de las carteras de largo plazo, y con razón. La
``caida_1a_95`` de la tabla de riesgo es ``μ − 1.645σ`` bajo normalidad, y la
normalidad es justo lo que falla en un mercado malo. Además 2022 es el
escenario que más daña a un perfil conservador y el que una covarianza de dos
años no ve: acciones y bonos cayendo juntos, con la correlación que el
modelo estima negativa volviéndose positiva.

Cómo se calcula, y lo que no es
-------------------------------
Cada clase de activo se representa con un ETF índice que existía en las tres
ventanas (:data:`PROXY_POR_CLASE`), y la cartera se estresa por **clase**:
``Σ peso_clase × retorno_del_proxy``, comprando al inicio de la ventana y
vendiendo al final, sin rebalancear. Es una aproximación de primer orden:

* No son los instrumentos de la cartera. Un ETF corporativo de duración corta
  cae menos que LQD; una acción individual no es SPY. Para la renta variable
  se usa la beta del sleeve contra SPY cuando se puede medir, que corrige lo
  más grueso (un satélite de beta 1,5 cae más que el índice).
* No es la caída máxima dentro de la ventana, es el resultado de punta a
  punta. Las ventanas van de pico a valle del S&P 500, así que para la renta
  variable coinciden; para una cartera mixta la caída intermedia puede ser
  algo mayor.
* No es un backtest. Los pesos son los de hoy aplicados a ventanas pasadas.

Lo que sí es: un orden de magnitud honesto, comparable entre mandatos y
contra el drawdown tolerado de cada uno.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cci_regulation import (
    CLASE_COMMODITIES, CLASE_EQUITY, CLASE_ETF_RV, GRUPOS_ASIGNACION,
    MODELO_ASIGNACION, drawdown_tolerado_for,
)

#: Ventanas de pico a valle del S&P 500, con lo que cada una pone a prueba.
ESCENARIOS: dict[str, tuple[str, str, str]] = {
    "Crisis 2008": ("2007-10-09", "2009-03-09",
                    "crisis de crédito: acciones -55%, Tesoro sube"),
    "COVID 2020": ("2020-02-19", "2020-03-23",
                   "caída de cinco semanas; el crédito corporativo se congela"),
    "Inflación 2022": ("2022-01-03", "2022-10-12",
                       "acciones y bonos caen juntos: la diversificación falla"),
}

#: Un ETF por clase, todos con historia desde antes de octubre de 2007.
PROXY_POR_CLASE: dict[str, str] = {
    CLASE_EQUITY: "SPY",
    CLASE_ETF_RV: "SPY",
    "RentaFija_Soberana_IG": "IEF",
    "ETF_RentaFija": "AGG",
    "RentaFija_Corporativa_IG": "LQD",
    "RentaFija_NoIG": "HYG",
    "Efectivo_MM": "BIL",
    CLASE_COMMODITIES: "GLD",
}

#: Rango a descargar para cubrir las tres ventanas con margen.
INICIO_DESCARGA = "2007-09-01"
FIN_DESCARGA = "2022-12-31"

#: Una ventana sin dato a menos de esto de su inicio o su fin no se evalúa:
#: medir 2008 desde 2009 no es medir 2008.
HOLGURA_DIAS = 7

_CLASES_RV = (CLASE_EQUITY, CLASE_ETF_RV)

#: La clase que representa a cada línea del Modelo de Asignación.
_CLASE_DE_LINEA = {grupo: clases[0] for grupo, clases in GRUPOS_ASIGNACION.items()}


def proxies() -> list[str]:
    """Los tickers que hay que bajar para estresar cualquier cartera."""
    return sorted(set(PROXY_POR_CLASE.values()))


def retornos_escenarios(precios: pd.DataFrame,
                        escenarios: Mapping[str, tuple[str, str, str]] = ESCENARIOS,
                        ) -> tuple[pd.DataFrame, list[str]]:
    """
    Retorno de punta a punta de cada proxy en cada ventana.

    ``precios`` son cierres **ajustados** (incluyen dividendos y cupones), con
    fechas en el índice y un ticker por columna. Devuelve ``(escenario ×
    ticker, notas)``; una celda sin datos suficientes queda en NaN y se dice.
    """
    precios = precios.sort_index()
    notas: list[str] = []
    filas: dict[str, dict[str, float]] = {}
    holgura = pd.Timedelta(days=HOLGURA_DIAS)
    for nombre, (inicio, fin, _desc) in escenarios.items():
        t0, t1 = pd.Timestamp(inicio), pd.Timestamp(fin)
        fila: dict[str, float] = {}
        for ticker in precios.columns:
            serie = pd.to_numeric(precios[ticker], errors="coerce").dropna()
            antes = serie[serie.index >= t0]
            hasta = serie[serie.index <= t1]
            if (antes.empty or hasta.empty or antes.index[0] - t0 > holgura
                    or t1 - hasta.index[-1] > holgura or antes.index[0] >= hasta.index[-1]):
                fila[ticker] = float("nan")
                notas.append(f"{ticker}: sin datos para {nombre} ({inicio} a {fin}).")
                continue
            fila[ticker] = float(hasta.iloc[-1] / antes.iloc[0] - 1.0)
        filas[nombre] = fila
    return pd.DataFrame.from_dict(filas, orient="index"), notas


def pesos_modelo(estrategia: str) -> dict[str, float]:
    """El Modelo de Asignación del mandato, expresado por clase."""
    return {_CLASE_DE_LINEA[grupo]: float(peso)
            for grupo, peso in MODELO_ASIGNACION[estrategia].items()}


def beta_del_sleeve(pesos: Mapping[str, float], clases: Mapping[str, str],
                    retornos: pd.DataFrame | None,
                    referencia: str = "SPY") -> tuple[float, str]:
    """
    Beta del sleeve de renta variable contra ``referencia``, o 1.0 si no se
    puede medir. Devuelve ``(beta, cómo se obtuvo)``.
    """
    if retornos is None or referencia not in getattr(retornos, "columns", ()):
        return 1.0, f"beta 1.0 supuesta: {referencia} no está en los retornos"
    rv = {t: float(w) for t, w in pesos.items()
          if float(w) > 0 and clases.get(t) in _CLASES_RV and t in retornos.columns}
    total = sum(rv.values())
    if total <= 0:
        return 1.0, "beta 1.0 supuesta: la cartera no tiene renta variable"
    datos = retornos[list(rv) + ([referencia] if referencia not in rv else [])].dropna()
    if len(datos) < 60:
        return 1.0, "beta 1.0 supuesta: menos de 60 observaciones comunes"
    sleeve = datos[list(rv)].mul(pd.Series(rv) / total, axis=1).sum(axis=1)
    m = datos[referencia]
    var = float(np.var(m, ddof=1))
    if var <= 0:
        return 1.0, f"beta 1.0 supuesta: {referencia} sin varianza"
    beta = float(np.cov(sleeve, m, ddof=1)[0, 1] / var)
    return beta, f"beta del sleeve de renta variable contra {referencia}: {beta:.2f}"


def estresar(pesos_por_clase: Mapping[str, float], retornos_esc: pd.DataFrame,
             beta_rv: float = 1.0) -> tuple[dict[str, float], list[str]]:
    """
    Retorno de la cartera en cada escenario, por clase.

    La renta variable entra como ``beta_rv × retorno de SPY``. Una clase sin
    proxy o con el proxy sin datos deja el escenario en NaN: rellenarla con
    cero diría que esa parte del libro no se movió.
    """
    notas: list[str] = []
    sin_proxy = [c for c, w in pesos_por_clase.items()
                 if float(w) > 1e-9 and c not in PROXY_POR_CLASE]
    if sin_proxy:
        notas.append(f"Clases sin proxy de estrés: {sorted(sin_proxy)}.")

    out: dict[str, float] = {}
    for escenario in retornos_esc.index:
        total = 0.0
        for clase, peso in pesos_por_clase.items():
            peso = float(peso)
            if peso <= 1e-9:
                continue
            proxy = PROXY_POR_CLASE.get(clase)
            r = (float(retornos_esc.at[escenario, proxy])
                 if proxy in retornos_esc.columns else float("nan"))
            if clase in _CLASES_RV:
                r *= beta_rv
            total += peso * r
        out[escenario] = float(total) if np.isfinite(total) else float("nan")
    return out, notas


def tabla_estres(carteras: Mapping[str, tuple[str, Mapping[str, float], float]],
                 retornos_esc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Una fila por cartera: su retorno en cada escenario, el peor, y si ese peor
    pasa el drawdown tolerado de su mandato.

    ``carteras`` mapea ``nombre -> (estrategia, pesos_por_clase, beta_rv)``.
    """
    filas: list[dict] = []
    notas: list[str] = []
    for nombre, (estrategia, pesos, beta) in carteras.items():
        res, n = estresar(pesos, retornos_esc, beta)
        notas.extend(f"{nombre}: {x}" for x in n)
        validos = [v for v in res.values() if np.isfinite(v)]
        peor = min(validos) if validos else float("nan")
        tolerado = drawdown_tolerado_for(estrategia)
        excede = bool(validos and tolerado is not None and peor < tolerado)
        filas.append({"cartera": nombre, "estrategia": estrategia,
                      "beta_rv": beta, **res, "peor": peor,
                      "dd_tolerado": tolerado if tolerado is not None else float("nan"),
                      "excede_tolerancia": excede})
        if excede:
            notas.append(
                f"{nombre}: en el peor escenario cae {peor:.1%}, más que el "
                f"drawdown tolerado de {tolerado:.0%} de {estrategia}.")
    return pd.DataFrame(filas), notas


def escenarios_descritos(nombres: Sequence[str] | None = None) -> list[str]:
    """Una línea por escenario, para imprimir sobre la tabla."""
    return [f"{n}: {ESCENARIOS[n][0]} a {ESCENARIOS[n][1]} — {ESCENARIOS[n][2]}"
            for n in (nombres or ESCENARIOS)]
