"""
Ratios fundamentales point-in-time: hechos de EDGAR + precio de la misma fecha.

Qué resuelve
------------
La fase 2 dejó un almacén de hechos con su fecha de presentación. Esto lo
convierte en números comparables entre emisores, que es el único formato que el
scoring transversal sabe leer.

La regla que gobierna el módulo entero: **un ratio mezcla un fundamental con un
precio, y los dos tienen que ser del mismo día.** Casar el balance de 2023 con
el precio de hoy produce un P/B que no existió nunca; casar el precio de 2023
con el balance restatado en 2025 produce un backtest que sabía cosas que nadie
sabía. Por eso ``ratios()`` pide una fecha y usa ``edgar.as_of`` para no ver
nada presentado después.

Rendimientos, no múltiplos
--------------------------
Todos los ratios de valuación se calculan **invertidos**: ``utilidad/precio`` en
vez de ``precio/utilidad``. No es estilo, es lo único que ordena bien.

Un P/E se rompe en el cero. Una empresa que gana 1 centavo por acción a $100
cotiza a 10.000x; si pierde 1 centavo cotiza a -10.000x, y en un ranking donde
"P/E bajo es mejor" ese -10.000 queda **primero**, por delante de cualquier
empresa sana. El múltiplo no es monótono en la calidad del negocio: salta a
infinito y vuelve por abajo. Un z-score sobre eso premia sistemáticamente a las
empresas que pierden dinero, y lo hace en silencio.

El rendimiento no tiene ese problema: +10% es mejor que +2%, que es mejor que
-3%, en una sola recta y sin discontinuidades. Los múltiplos clásicos se
calculan igual, pero **solo para mostrar** — nunca entran al score. Van en
:func:`multiplos_legibles` y salen en NaN cuando el rendimiento no es positivo,
porque un P/E negativo no es una valuación barata: es la ausencia de una
valuación.

Lo que este módulo NO hace
--------------------------
No imputa. Un emisor sin EBITDA no recibe el EBITDA de su sector: se queda sin
ese ratio y el bloque se renormaliza sobre lo que sí tiene, que es lo que
``scoring.py`` ya hace con cualquier métrica ausente.

No estima trimestres. Solo lee períodos anuales, por la razón que explica
``edgar.anuales``: un TTM mal armado desde XBRL es peor que no tener número.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .edgar import anuales

#: Días de tolerancia entre el cierre del ejercicio y la fecha de un saldo.
#: Un balance de cierre cae el mismo día que termina el ejercicio, pero el
#: conteo de acciones de la portada del 10-K lleva la fecha de la presentación,
#: que son unos dos meses después. Con una tolerancia estrecha se perdería justo
#: el dato más fresco de acciones, que es el que hace falta para una
#: capitalización.
TOLERANCIA_SALDO_DIAS = 120

#: Un ejercicio cerrado hace más de esto no describe a la empresa de hoy. Se
#: deja holgado a propósito: un 10-K se presenta hasta 90 días después del
#: cierre, así que en el peor momento del calendario el último ejercicio
#: disponible tiene 15 meses. Más allá de 550 días el emisor dejó de reportar, y
#: usar sus números viejos como si fueran actuales es el error que este tope
#: existe para evitar.
MAX_ANTIGUEDAD_DIAS = 550

#: Qué fracción del universo con fundamentales tiene que cubrir un ratio para
#: entrar al scoring.
#:
#: El umbral es **relativo y no absoluto**, y la diferencia importa. Lo que hay
#: que proteger no es un tamaño de muestra en abstracto: es la comparabilidad
#: entre las métricas de un mismo bloque. ``scoring.py`` renormaliza el bloque
#: sobre las métricas presentes, así que un ratio medido sobre 12 emisores de
#: 300 entra al promedio con el mismo peso que uno medido sobre 290 — el z de
#: una anécdota discutiendo de igual a igual con el z de una distribución.
#:
#: Un mínimo fijo de 30 nombres resolvía eso en el universo grande y rompía el
#: caso chico: en una lista de 12 nombres ningún ratio llegaría nunca, y el
#: bloque fundamental no existiría por una constante pensada para otro tamaño.
FRACCION_COHORTE = 0.50

#: Piso absoluto: por debajo de esto ``scoring.zscore`` devuelve NaN de todos
#: modos, así que emitir el ratio solo agregaría ruido.
MIN_COHORTE_ABS = 3


def cohorte_minima(n: int) -> int:
    """Cuántos emisores necesita un ratio en un universo de ``n``."""
    import math

    return max(MIN_COHORTE_ABS, math.ceil(FRACCION_COHORTE * max(n, 0)))


# --------------------------------------------------------------------------
# Derivadas — lo que no viene reportado pero se puede reconstruir
# --------------------------------------------------------------------------
#
# Cada una de estas existe porque la etiqueta directa tiene mala cobertura y la
# identidad contable no. `Liabilities` es un subtotal OPCIONAL en US GAAP y en
# nuestra corrida lo reportaron 195 de 280; activos y patrimonio los reportaron
# los 280, y la resta entre ellos ES el pasivo, no una aproximación suya.
#
# La regla: derivar solo por identidad contable. Nada de rellenar con el
# promedio del sector — eso produce un número con apariencia de medición, que es
# exactamente lo que este proyecto lleva meses sacando.

@dataclass(frozen=True)
class Derivada:
    """Una magnitud reconstruida, con la identidad que la justifica."""

    clave: str
    formula: str
    descripcion: str


DERIVADAS: tuple[Derivada, ...] = (
    Derivada("pasivos_der", "activos - patrimonio",
             "Identidad del balance. Sube la cobertura del pasivo total de 70% "
             "a 100% sin suponer nada: no es una estimación, es la resta."),
    Derivada("pasivos_no_corrientes_der", "pasivos_der - pasivos_corrientes",
             "LiabilitiesNoncurrent lo reporta el 17%. Esta resta llega al 85%, "
             "que es la cobertura de pasivos_corrientes."),
    Derivada("ebitda", "ebit + depreciacion",
             "Resultado operativo antes de D&A. Un banco no reporta EBIT y ahí "
             "el EBITDA no significa nada de todos modos."),
    Derivada("flujo_libre", "flujo_operativo - capex",
             "Caja que queda después de mantener el negocio en pie."),
    Derivada("deuda_total", "deuda_corto_plazo + deuda_largo_plazo",
             "Con cualquiera de las dos ausente no se calcula: sumar solo la "
             "larga y llamarla deuda total subestima el apalancamiento justo "
             "en los emisores que más se financian a corto."),
    Derivada("deuda_neta", "deuda_total - efectivo",
             "Lo que se sumaría a la capitalización para comprar la empresa "
             "entera. Puede ser negativa, y eso es información, no un error."),
    Derivada("acciones_efectivas", "utilidad_neta / eps_diluido",
             "El conteo diluido que la propia empresa usó para reportar su EPS. "
             "Se prefiere al reportado porque CommonStockSharesOutstanding es "
             "POR CLASE: en una empresa con acciones A y B trae una sola clase "
             "y la capitalización sale a la mitad."),
)


# --------------------------------------------------------------------------
# Ratios
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Ratio:
    """Un ratio, con lo que necesita y para qué sirve."""

    clave: str
    etiqueta: str
    formula: str
    #: Los nombres que tienen que estar presentes y ser finitos.
    requiere: tuple[str, ...]
    #: Denominadores que además tienen que ser estrictamente positivos. Es la
    #: diferencia entre un ratio y un sinsentido con signo.
    positivos: tuple[str, ...] = ()
    familia: str = "valuacion"
    #: +1 si más alto es mejor. Solo informativo aquí; el modelo lo declara en
    #: config.py, que es donde vive la orientación de cada métrica.
    direccion: int = 1
    descripcion: str = ""


RATIOS: tuple[Ratio, ...] = (
    # ---- Valuación: todos rendimientos, todos monótonos ------------------
    Ratio("earnings_yield", "Rendimiento de utilidades",
          "eps_diluido / precio", ("eps_diluido", "precio"), ("precio",),
          descripcion="El inverso del P/E. Negativo cuando la empresa pierde "
                      "dinero, que es exactamente donde debe quedar."),
    Ratio("fcf_yield", "Rendimiento de flujo libre",
          "flujo_libre / capitalizacion", ("flujo_libre", "capitalizacion"),
          ("capitalizacion",),
          descripcion="Más difícil de maquillar que la utilidad contable."),
    Ratio("ebitda_ev", "EBITDA sobre valor de empresa",
          "ebitda / valor_empresa", ("ebitda", "valor_empresa"),
          ("valor_empresa",),
          descripcion="El inverso del EV/EBITDA. Compara empresas con "
                      "estructuras de capital distintas."),
    Ratio("book_yield", "Patrimonio sobre capitalización",
          "patrimonio / capitalizacion", ("patrimonio", "capitalizacion"),
          ("capitalizacion",),
          descripcion="El inverso del P/B. Patrimonio negativo da rendimiento "
                      "negativo, que es la lectura correcta."),
    Ratio("sales_yield", "Ingresos sobre capitalización",
          "ingresos / capitalizacion", ("ingresos", "capitalizacion"),
          ("capitalizacion",),
          descripcion="El inverso del P/S. Sobrevive a un año de pérdidas, que "
                      "es cuando los otros rendimientos dejan de ordenar."),
    # ---- Calidad: no entran al score todavía -----------------------------
    Ratio("roe", "Retorno sobre patrimonio", "utilidad_neta / patrimonio",
          ("utilidad_neta", "patrimonio"), ("patrimonio",), familia="calidad",
          descripcion="Con patrimonio negativo no se calcula: el cociente sale "
                      "positivo por doble signo y leería como excelente."),
    Ratio("margen_operativo", "Margen operativo", "ebit / ingresos",
          ("ebit", "ingresos"), ("ingresos",), familia="calidad"),
    Ratio("margen_neto", "Margen neto", "utilidad_neta / ingresos",
          ("utilidad_neta", "ingresos"), ("ingresos",), familia="calidad"),
    Ratio("accruals", "Devengos sobre activos",
          "(utilidad_neta - flujo_operativo) / activos",
          ("utilidad_neta", "flujo_operativo", "activos"), ("activos",),
          familia="calidad", direccion=-1,
          descripcion="Utilidad que no llegó como caja. Alto y persistente es "
                      "la firma clásica de contabilidad agresiva."),
    Ratio("deuda_neta_ebitda", "Deuda neta / EBITDA",
          "deuda_neta / ebitda", ("deuda_neta", "ebitda"), ("ebitda",),
          familia="calidad", direccion=-1),
    Ratio("crecimiento_ingresos", "Crecimiento de ingresos",
          "ingresos / ingresos_previo - 1", ("ingresos", "ingresos_previo"),
          ("ingresos_previo",), familia="calidad",
          descripcion="Un solo año. No es una tendencia y no pretende serlo."),
    Ratio("payout", "Payout sobre utilidad",
          "dividendos_pagados / utilidad_neta",
          ("dividendos_pagados", "utilidad_neta"), ("utilidad_neta",),
          familia="calidad", direccion=-1,
          descripcion="Sobre utilidad negativa no se calcula. Los dividendos "
                      "vienen con signo negativo en el flujo de caja, así que "
                      "se toma el valor absoluto."),
)

RATIO_POR_CLAVE: dict[str, Ratio] = {r.clave: r for r in RATIOS}

#: Los que hoy entran al modelo. La calidad se calcula y se reporta, pero
#: ponerla a puntuar exige decidir su peso, y un peso es una decisión del
#: Comité, no un detalle de implementación.
RATIOS_PUNTUADOS: tuple[str, ...] = tuple(
    r.clave for r in RATIOS if r.familia == "valuacion")


# --------------------------------------------------------------------------
# Panel anual: un ejercicio coherente por emisor
# --------------------------------------------------------------------------

def panel(hechos, fecha: str | None = None, *, atras: int = 0,
          tolerancia_dias: int = TOLERANCIA_SALDO_DIAS,
          max_antiguedad_dias: int | None = MAX_ANTIGUEDAD_DIAS):
    """
    Una fila por emisor con las magnitudes de **un mismo ejercicio**.

    El detalle que decide si el número significa algo: cada métrica se toma del
    mismo período, no la más reciente de cada una por separado. Un EBITDA con el
    EBIT de 2024 y la depreciación de 2023 no es el EBITDA de ningún año, y sale
    de un ``groupby(...).last()`` inocente.

    Así que primero se fija el **ejercicio de referencia** del emisor — el cierre
    anual más reciente entre sus flujos — y después todo se ancla ahí: los flujos
    con ese cierre exacto, los saldos con el corte más cercano dentro de
    ``tolerancia_dias``.

    ``atras=1`` devuelve el ejercicio anterior, que es lo que necesita un
    crecimiento.
    """
    import numpy as np
    import pandas as pd

    from .edgar import CONCEPTO_POR_CLAVE

    vista = anuales(hechos, fecha)
    columnas = ["ticker", "periodo", "filed", *sorted(CONCEPTO_POR_CLAVE)]
    if vista is None or len(vista) == 0:
        return pd.DataFrame(columns=columnas).set_index("ticker")

    vista = vista.copy()
    vista["_fin"] = pd.to_datetime(vista["fin"], errors="coerce")
    vista = vista[vista["_fin"].notna()]
    es_flujo = ~vista["metrica"].map(
        lambda m: CONCEPTO_POR_CLAVE[m].instantaneo
        if m in CONCEPTO_POR_CLAVE else False)

    # Referencia: el cierre anual `atras`-ésimo contando desde el más reciente.
    cierres = (vista[es_flujo].groupby("ticker")["_fin"]
               .apply(lambda s: sorted(set(s), reverse=True)))
    referencia = {t: fechas[atras] for t, fechas in cierres.items()
                  if len(fechas) > atras}
    if not referencia:
        return pd.DataFrame(columns=columnas).set_index("ticker")

    if max_antiguedad_dias is not None:
        corte = (pd.Timestamp(fecha) if fecha else pd.Timestamp.today())
        referencia = {t: f for t, f in referencia.items()
                      if (corte - f).days <= max_antiguedad_dias}

    vista = vista[vista["ticker"].isin(referencia)]
    vista["_ref"] = pd.to_datetime(vista["ticker"].map(referencia))
    vista["_delta"] = (vista["_fin"] - vista["_ref"]).dt.days.abs()

    # Flujos: el cierre exacto. Saldos: el más cercano dentro de la tolerancia.
    flujo = ~vista["metrica"].map(
        lambda m: CONCEPTO_POR_CLAVE[m].instantaneo
        if m in CONCEPTO_POR_CLAVE else False)
    sirve = np.where(flujo, vista["_delta"] == 0,
                     vista["_delta"] <= tolerancia_dias)
    vista = vista[sirve]
    if vista.empty:
        return pd.DataFrame(columns=columnas).set_index("ticker")

    vista = (vista.sort_values(["ticker", "metrica", "_delta", "filed"],
                               kind="stable")
                  .drop_duplicates(subset=["ticker", "metrica"], keep="first"))

    ancho = vista.pivot(index="ticker", columns="metrica", values="valor")
    for clave in CONCEPTO_POR_CLAVE:
        if clave not in ancho.columns:
            ancho[clave] = np.nan
    ancho = ancho[sorted(CONCEPTO_POR_CLAVE)]
    ancho.insert(0, "filed", vista.groupby("ticker")["filed"].max())
    ancho.insert(0, "periodo",
                 pd.Series(referencia).reindex(ancho.index)
                 .dt.strftime("%Y-%m-%d"))
    ancho.columns.name = None
    return ancho


def obsoletos(hechos, fecha: str | None = None, *,
              max_antiguedad_dias: int = MAX_ANTIGUEDAD_DIAS,
              ) -> dict[str, str]:
    """
    Emisores cuyo último ejercicio anual quedó fuera del tope de antigüedad.

    Existe para que la caída no sea silenciosa. Un almacén entero puede quedar
    obsoleto sin que nada falle — pasó en las pruebas: el fixture traía FY2024,
    el tope son 550 días, y ``panel`` devolvió cero filas sin decir por qué.
    Un cero sin explicación se lee como "no hay datos" cuando lo que hay es
    "los datos son viejos", y son dos problemas distintos.
    """
    import pandas as pd

    vista = anuales(hechos, fecha)
    if vista is None or len(vista) == 0:
        return {}

    fin = pd.to_datetime(vista["fin"], errors="coerce")
    ultimo = pd.DataFrame({"ticker": vista["ticker"], "fin": fin}).dropna()
    if ultimo.empty:
        return {}

    ultimo = ultimo.groupby("ticker")["fin"].max()
    corte = pd.Timestamp(fecha) if fecha else pd.Timestamp.today()
    viejos = ultimo[(corte - ultimo).dt.days > max_antiguedad_dias]
    return {t: f.strftime("%Y-%m-%d") for t, f in viejos.items()}


def derivar(tabla):
    """
    Agrega las magnitudes de :data:`DERIVADAS` sobre un panel.

    ``pasivos`` y ``pasivos_no_corrientes`` quedan como columnas nuevas con
    sufijo ``_der`` en vez de tapar a las reportadas: cuando las dos existen, la
    diferencia entre ellas es un control de consistencia gratis, y taparlas lo
    perdería.
    """
    import numpy as np

    if len(tabla) == 0:
        return tabla

    import pandas as pd

    t = tabla.copy()
    vacia = pd.Series(np.nan, index=t.index, dtype=float)

    def col(nombre):
        return (pd.to_numeric(t[nombre], errors="coerce")
                if nombre in t.columns else vacia)

    t["pasivos_der"] = col("pasivos").where(
        col("pasivos").notna(), col("activos") - col("patrimonio"))
    t["pasivos_no_corrientes_der"] = col("pasivos_no_corrientes").where(
        col("pasivos_no_corrientes").notna(),
        t["pasivos_der"] - col("pasivos_corrientes"))
    t["ebitda"] = col("ebit") + col("depreciacion")
    t["flujo_libre"] = col("flujo_operativo") - col("capex")
    t["deuda_total"] = col("deuda_corto_plazo") + col("deuda_largo_plazo")
    t["deuda_neta"] = t["deuda_total"] - col("efectivo")

    # Conteo de acciones. El implícito por EPS gana porque el reportado es por
    # clase; si el EPS no está o es cero, queda el reportado y se anota cuál se
    # usó, porque una capitalización a la mitad no se ve en el resultado.
    eps = col("eps_diluido")
    implicito = (col("utilidad_neta") / eps).where(eps.abs() > 1e-9)
    implicito = implicito.where(implicito > 0)
    reportado = col("acciones_en_circulacion")
    t["acciones_efectivas"] = implicito.where(implicito.notna(), reportado)
    t["acciones_fuente"] = np.where(
        implicito.notna(), "eps",
        np.where(reportado.notna(), "reportadas", ""))
    # Cuánto se separan las dos. Un 2x aquí es casi siempre una empresa con dos
    # clases de acciones donde la etiqueta reportada trae una sola.
    with np.errstate(divide="ignore", invalid="ignore"):
        t["acciones_discrepancia"] = (implicito / reportado) - 1.0
    return t


# --------------------------------------------------------------------------
# Ratios
# --------------------------------------------------------------------------

def ratios(hechos, precios: Mapping[str, float], fecha: str | None = None, *,
           tolerancia_dias: int = TOLERANCIA_SALDO_DIAS,
           max_antiguedad_dias: int | None = MAX_ANTIGUEDAD_DIAS):
    """
    Los ratios de :data:`RATIOS` para lo que se sabía en ``fecha``.

    ``precios`` es el precio de **esa misma fecha**, no el de hoy. El módulo no
    lo puede verificar — un número es un número — así que la responsabilidad es
    de quien llama, y por eso ``adjuntar()`` existe: para que la corrida normal
    no tenga que acordarse.
    """
    import numpy as np
    import pandas as pd

    actual = derivar(panel(hechos, fecha, tolerancia_dias=tolerancia_dias,
                           max_antiguedad_dias=max_antiguedad_dias))
    if len(actual) == 0:
        return pd.DataFrame(columns=["periodo", "filed", "precio",
                                     *[r.clave for r in RATIOS]])

    previo = panel(hechos, fecha, atras=1, tolerancia_dias=tolerancia_dias,
                   max_antiguedad_dias=None)
    actual["ingresos_previo"] = (previo["ingresos"].reindex(actual.index)
                                 if "ingresos" in getattr(previo, "columns", [])
                                 else np.nan)

    actual["precio"] = pd.Series(
        {str(k).upper(): float(v) for k, v in precios.items()
         if v is not None and np.isfinite(float(v))},
        dtype=float).reindex(actual.index)
    actual["capitalizacion"] = actual["precio"] * actual["acciones_efectivas"]
    actual["valor_empresa"] = actual["capitalizacion"] + actual["deuda_neta"]

    # Los dividendos salen del flujo de caja, donde una salida es negativa.
    if "dividendos_pagados" in actual.columns:
        actual["dividendos_pagados"] = actual["dividendos_pagados"].abs()

    for ratio in RATIOS:
        actual[ratio.clave] = _evaluar(actual, ratio)

    columnas = ["periodo", "filed", "precio", "acciones_efectivas",
                "acciones_fuente", "acciones_discrepancia", "capitalizacion",
                "valor_empresa", *[r.clave for r in RATIOS]]
    return actual[[c for c in columnas if c in actual.columns]]


def _evaluar(tabla, ratio: Ratio):
    """Un ratio, o NaN donde no se puede calcular sin inventar."""
    import numpy as np

    faltan = [n for n in ratio.requiere if n not in tabla.columns]
    if faltan:
        return np.full(len(tabla), np.nan)

    valido = np.ones(len(tabla), dtype=bool)
    for nombre in ratio.requiere:
        valido &= np.isfinite(tabla[nombre].astype(float))
    for nombre in ratio.positivos:
        valido &= tabla[nombre].astype(float) > 0

    with np.errstate(divide="ignore", invalid="ignore"):
        crudo = _formula(tabla, ratio.clave)
    return np.where(valido, crudo, np.nan)


def _formula(t, clave: str):
    """Las fórmulas, en un solo sitio y en el mismo orden que RATIOS."""
    return {
        "earnings_yield": lambda: t["eps_diluido"] / t["precio"],
        "fcf_yield": lambda: t["flujo_libre"] / t["capitalizacion"],
        "ebitda_ev": lambda: t["ebitda"] / t["valor_empresa"],
        "book_yield": lambda: t["patrimonio"] / t["capitalizacion"],
        "sales_yield": lambda: t["ingresos"] / t["capitalizacion"],
        "roe": lambda: t["utilidad_neta"] / t["patrimonio"],
        "margen_operativo": lambda: t["ebit"] / t["ingresos"],
        "margen_neto": lambda: t["utilidad_neta"] / t["ingresos"],
        "accruals": lambda: (t["utilidad_neta"] - t["flujo_operativo"])
        / t["activos"],
        "deuda_neta_ebitda": lambda: t["deuda_neta"] / t["ebitda"],
        "crecimiento_ingresos": lambda: t["ingresos"] / t["ingresos_previo"] - 1,
        "payout": lambda: t["dividendos_pagados"] / t["utilidad_neta"],
    }[clave]()


def multiplos_legibles(tabla):
    """
    P/E, EV/EBITDA, P/B y P/S para leer, **no para puntuar**.

    Salen en NaN donde el rendimiento no es positivo, que es donde el múltiplo
    deja de ser una valuación. Un P/E de -8x no es más barato que uno de 30x: es
    una empresa que pierde dinero, y el múltiplo no sabe decirlo.
    """
    import numpy as np
    import pandas as pd

    if len(tabla) == 0:
        return pd.DataFrame(index=getattr(tabla, "index", None))

    def invertir(serie):
        s = pd.to_numeric(serie, errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            return (1.0 / s).where(s > 0)

    return pd.DataFrame({
        "pe": invertir(tabla.get("earnings_yield")),
        "ev_ebitda": invertir(tabla.get("ebitda_ev")),
        "pb": invertir(tabla.get("book_yield")),
        "ps": invertir(tabla.get("sales_yield")),
    }, index=tabla.index)


# --------------------------------------------------------------------------
# Cohorte y cobertura
# --------------------------------------------------------------------------

def cohortes(tabla, tickers: Sequence[str] | None = None,
             *, minimo: int | None = None):
    """
    Cuántos emisores tienen cada ratio, y si eso alcanza para compararlos.

    Es el entregable que decide qué entra al modelo. ``puntuable=False`` no
    significa que el ratio esté mal: significa que su cohorte es tan chica que
    un z-score contra ella no es comparable con el de un ratio medido sobre
    trescientos nombres, y mezclarlos en el mismo bloque le daría a doce
    emisores un peso que no ganaron.
    """
    import numpy as np
    import pandas as pd

    universo = ([str(t).upper() for t in tickers] if tickers is not None
                else list(getattr(tabla, "index", [])))
    minimo = cohorte_minima(len(tabla)) if minimo is None else minimo
    filas = []
    for ratio in RATIOS:
        serie = (pd.to_numeric(tabla[ratio.clave], errors="coerce")
                 .reindex(universo) if ratio.clave in getattr(
                     tabla, "columns", []) else pd.Series(dtype=float,
                                                          index=universo))
        con_dato = int(np.isfinite(serie).sum())
        filas.append({
            "ratio": ratio.clave,
            "familia": ratio.familia,
            "etiqueta": ratio.etiqueta,
            "formula": ratio.formula,
            "cobertura": con_dato / len(universo) if universo else 0.0,
            "con_dato": con_dato,
            "sin_dato": len(universo) - con_dato,
            "mediana": float(serie.median()) if con_dato else float("nan"),
            "puntuable": bool(con_dato >= minimo
                              and ratio.clave in RATIOS_PUNTUADOS),
        })
    return pd.DataFrame(filas)


def metricas_para_scoring(tabla, *, minimo: int | None = None,
                          claves: Sequence[str] = RATIOS_PUNTUADOS,
                          ) -> dict[str, dict[str, float]]:
    """
    ``{ticker: {metrica: valor}}`` con lo que puede entrar a ``raw_metrics``.

    Un ratio cuya cohorte no llega a ``minimo`` no se emite para nadie. Emitirlo
    solo para los pocos que lo tienen sería peor que omitirlo: ``scoring.py``
    renormaliza el bloque sobre las métricas presentes, así que esos pocos
    tendrían un bloque construido sobre una comparación que el resto no tuvo.
    """
    import numpy as np
    import pandas as pd

    if len(tabla) == 0:
        return {}
    minimo = cohorte_minima(len(tabla)) if minimo is None else minimo

    usables = []
    for clave in claves:
        if clave not in tabla.columns:
            continue
        serie = pd.to_numeric(tabla[clave], errors="coerce")
        if int(np.isfinite(serie).sum()) >= minimo:
            usables.append(clave)

    fuera: dict[str, dict[str, float]] = {}
    for ticker, fila in tabla[usables].iterrows() if usables else []:
        valores = {c: float(fila[c]) for c in usables
                   if np.isfinite(pd.to_numeric(fila[c], errors="coerce"))}
        if valores:
            fuera[str(ticker).upper()] = valores
    return fuera


# --------------------------------------------------------------------------
# Enganche con la corrida
# --------------------------------------------------------------------------

def adjuntar(market_data: dict, hechos, fecha: str | None = None, *,
             minimo: int | None = None) -> dict:
    """
    Calcula los ratios con el precio de cada instrumento y los cuelga del payload.

    Ésta es la función que garantiza la regla del módulo. El precio sale del
    ``snapshot`` del propio instrumento, así que fundamental y precio vienen del
    mismo payload y no hay forma de casar el balance de un año con la cotización
    de otro por descuido.

    Los ETF no tienen CIK y salen sin ratios. No es un hueco a tapar: un ETF no
    tiene estados financieros, y el bloque se renormaliza sobre lo que sí hay,
    igual que hace con cualquier métrica ausente.

    Devuelve el mismo ``market_data``, con ``instruments[i]["fundamentals"]``
    puesto donde corresponde y un nodo ``fundamentals_meta`` con la fecha y la
    cobertura, para que el reporte pueda decir sobre qué se calculó.
    """
    instrumentos = market_data.get("instruments") or []
    precios = {}
    for inst in instrumentos:
        ticker = (inst.get("ticker") or "").upper()
        precio = (inst.get("snapshot") or {}).get("last", {}).get("price")
        if ticker and precio:
            precios[ticker] = precio

    tabla = ratios(hechos, precios, fecha)
    minimo = cohorte_minima(len(tabla)) if minimo is None else minimo
    metricas = metricas_para_scoring(tabla, minimo=minimo)
    cobertura = cohortes(tabla, list(precios), minimo=minimo)

    for inst in instrumentos:
        ticker = (inst.get("ticker") or "").upper()
        if ticker in metricas:
            inst["fundamentals"] = {
                "metricas": metricas[ticker],
                "periodo": _celda(tabla, ticker, "periodo"),
                "filed": _celda(tabla, ticker, "filed"),
                "acciones_fuente": _celda(tabla, ticker, "acciones_fuente"),
            }

    market_data["fundamentals_meta"] = {
        "as_of": fecha,
        "cohorte_minima": minimo,
        "con_ratios": len(metricas),
        "obsoletos": obsoletos(hechos, fecha),
        "puntuados": sorted({m for v in metricas.values() for m in v}),
        "cobertura": cobertura.to_dict(orient="records"),
    }
    return market_data


def _celda(tabla, ticker: str, columna: str) -> Any:
    if columna not in getattr(tabla, "columns", []) or ticker not in tabla.index:
        return None
    valor = tabla.at[ticker, columna]
    return None if valor is None or valor != valor else valor


def metricas_del_instrumento(inst: Mapping[str, Any]) -> dict[str, float]:
    """Lo que ``run_screen`` mezcla en ``raw_metrics``. Vacío si no hay nada."""
    nodo = inst.get("fundamentals") or {}
    metricas = nodo.get("metricas") or {}
    return {str(k): float(v) for k, v in metricas.items()
            if v is not None and float(v) == float(v)}
