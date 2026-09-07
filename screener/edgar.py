"""
Fundamentales desde SEC EDGAR, con fecha de presentación.

Por qué EDGAR y no un proveedor
-------------------------------
El bloque ``valuation_carry`` pesa 10-12% del compuesto y su propio texto admite
que corre con **proxies**: "IBKR's market-data surface exposes no fundamental
valuation ratios, so this block uses market-implied proxies rather than
pretending to have P/E or EV/EBITDA". En una corrida real dos de sus tres
métricas salieron ``UNAVAILABLE from Yahoo``.

Cualquier proveedor de múltiplos resuelve eso. Ninguno resuelve el problema que
importa, que es **point-in-time**: un vendor te da el dato de hoy, ya
restatado, y con eso no se puede calibrar nada. Preguntarle a un backtest "¿qué
P/E tenía esta empresa en marzo de 2024?" y recibir el número corregido en 2025
es meterle al modelo información que no existía, y el IC que salga de ahí está
inflado por construcción.

EDGAR no tiene ese problema porque no es un proveedor: es el archivo. Cada dato
viene con el ``filed`` de la presentación que lo trajo, y las **versiones
sucesivas conviven** — el original y cada restatement son entradas separadas.
Filtrar ``filed <= fecha`` reconstruye lo que se sabía ese día, por
construcción y no por promesa de nadie.

Es gratis, no pide llave, y es la fuente primaria de la que los vendors
revenden.

Lo que este módulo NO hace
--------------------------
No calcula ratios. Baja hechos, los mapea a conceptos declarados y reporta
cobertura. Un P/E necesita casar un fundamental con un precio y alinear las dos
fechas, y eso es una decisión aparte que merece su propio archivo.

Tampoco estima. Un emisor cuyo concepto no está en :data:`CONCEPTOS` queda
declarado como cobertura faltante. Rellenar con el promedio del sector daría un
número con apariencia de medición, que es exactamente lo que este proyecto
lleva meses sacando.

Cortesía con la SEC
-------------------
Piden un ``User-Agent`` con contacto real y no más de 10 peticiones por
segundo. Las dos cosas se cumplen aquí; :data:`SEC_MAX_RPS` es el límite y
``fetch_json`` lo respeta.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SEC_DATA = "https://data.sec.gov"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"

#: Tope de la SEC. Pasarse es la forma de que te bloqueen la IP.
SEC_MAX_RPS = 8.0

#: Formularios que traen estados financieros. Un 8-K puede traer cifras, pero
#: no auditadas y sin la estructura de un reporte periódico.
FORMAS_VALIDAS = ("10-K", "10-Q", "20-F", "40-F", "10-K/A", "10-Q/A", "20-F/A")


# --------------------------------------------------------------------------
# Mapeo de conceptos — la parte que decide si el número significa algo
# --------------------------------------------------------------------------
#
# XBRL no es un esquema, es un vocabulario, y los emisores eligen palabras
# distintas para la misma idea. "Ingresos" es ``Revenues`` en unos,
# ``RevenueFromContractWithCustomerExcludingAssessedTax`` en otros desde ASC
# 606, y ``SalesRevenueNet`` en los que no actualizaron la etiqueta. Un mapeo
# flojo produce un P/E precioso y falso, que es peor que no tener P/E.
#
# La regla: los candidatos van **en orden de preferencia**, se toma el primero
# que el emisor haya reportado, y la corrida registra cuál se usó. Así un
# número raro se puede rastrear hasta la etiqueta que lo produjo en vez de
# quedar como un misterio en una hoja de Excel.
#
# ``instantaneo`` distingue saldos de flujos, y no es cosmética: un saldo
# (activos, deuda) se reporta a una fecha y un flujo (ingresos, utilidad) sobre
# un período. Sumar cuatro trimestres tiene sentido para un flujo y ninguno
# para un saldo, y confundirlos es la forma más rápida de calcular un EV/EBITDA
# cuatro veces más grande de lo que es.


@dataclass(frozen=True)
class Concepto:
    """Una métrica nuestra y las etiquetas XBRL que pueden traerla."""

    clave: str
    etiquetas: tuple[str, ...]
    unidad: str = "USD"
    instantaneo: bool = False
    descripcion: str = ""


CONCEPTOS: tuple[Concepto, ...] = (
    # ---- Flujos: estado de resultados y flujo de caja --------------------
    Concepto(
        "ingresos",
        ("RevenueFromContractWithCustomerExcludingAssessedTax",
         "RevenueFromContractWithCustomerIncludingAssessedTax",
         "Revenues",
         "SalesRevenueNet",
         "SalesRevenueGoodsNet",
         "RevenuesNetOfInterestExpense"),
        descripcion="Ingresos. Las tres primeras son post-ASC 606; las demás "
                    "sobreviven en emisores que no reetiquetaron.",
    ),
    Concepto(
        "utilidad_neta",
        ("NetIncomeLoss",
         "ProfitLoss",
         "NetIncomeLossAvailableToCommonStockholdersBasic"),
        descripcion="ProfitLoss incluye participación no controladora; "
                    "NetIncomeLoss no. Por eso va primero el segundo.",
    ),
    Concepto(
        "ebit",
        ("OperatingIncomeLoss",),
        descripcion="Resultado operativo. Un banco no lo reporta, y ahí el "
                    "EV/EBITDA no significa nada de todos modos.",
    ),
    Concepto(
        "depreciacion",
        ("DepreciationDepletionAndAmortization",
         "DepreciationAmortizationAndAccretionNet",
         "DepreciationAndAmortization",
         "Depreciation"),
        descripcion="Para reconstruir EBITDA = EBIT + D&A.",
    ),
    Concepto(
        "flujo_operativo",
        ("NetCashProvidedByUsedInOperatingActivities",
         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
        descripcion="Más difícil de maquillar que la utilidad contable.",
    ),
    Concepto(
        "capex",
        ("PaymentsToAcquirePropertyPlantAndEquipment",
         "PaymentsToAcquireProductiveAssets"),
        descripcion="Para flujo libre = operativo - capex.",
    ),
    Concepto(
        "dividendos_pagados",
        ("PaymentsOfDividendsCommonStock",
         "PaymentsOfDividends",
         "PaymentsOfDividendsMinorityInterest"),
    ),
    Concepto(
        "eps_diluido",
        ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"),
        unidad="USD/shares",
        descripcion="El denominador del P/E que reporta la propia empresa.",
    ),
    # ---- Saldos: balance -------------------------------------------------
    Concepto(
        "activos", ("Assets",), instantaneo=True,
    ),
    Concepto(
        "pasivos", ("Liabilities",), instantaneo=True,
        descripcion="El total. Es OPCIONAL en US GAAP: un emisor con balance "
                    "clasificado puede publicar solo los componentes y el "
                    "renglón LiabilitiesAndStockholdersEquity. En una corrida "
                    "real ABBV no lo trae. Para esos, los dos conceptos de "
                    "abajo permiten reconstruirlo sumando.",
    ),
    # Componentes, NO sinónimos. Ponerlos como candidatos de `pasivos` haría
    # que un emisor que reporta los dos quedara con solo el corriente
    # etiquetado como "pasivos totales" -- una subestimación silenciosa del
    # apalancamiento, que es peor que el hueco que vendría a tapar.
    Concepto(
        "pasivos_corrientes", ("LiabilitiesCurrent",), instantaneo=True,
    ),
    Concepto(
        "pasivos_no_corrientes",
        ("LiabilitiesNoncurrent",),
        instantaneo=True,
        descripcion="Solo el subtotal. `OtherLiabilitiesNoncurrent` estuvo aquí "
                    "como respaldo y era un error: es el renglón residual "
                    "«otros», no el total. En una corrida de 280 nombres ganó "
                    "en 173 de 221, o sea que la mayoría habría quedado con una "
                    "partida menor etiquetada como su pasivo no corriente "
                    "completo. Pocos emisores presentan este subtotal, y 17% de "
                    "cobertura honesta vale más que 79% de un número que mide "
                    "otra cosa.",
    ),
    Concepto(
        "patrimonio",
        ("StockholdersEquity",
         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        instantaneo=True,
        descripcion="El denominador del P/B.",
    ),
    Concepto(
        "efectivo",
        ("CashAndCashEquivalentsAtCarryingValue",
         "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
         "CashAndCashEquivalentsAtCarryingValueIncludingDiscontinuedOperations"),
        instantaneo=True,
        descripcion="Resta en el enterprise value.",
    ),
    Concepto(
        "deuda_largo_plazo",
        ("LongTermDebtNoncurrent",
         "LongTermDebt",
         "LongTermDebtAndCapitalLeaseObligations"),
        instantaneo=True,
    ),
    Concepto(
        "deuda_corto_plazo",
        ("LongTermDebtCurrent",
         "ShortTermBorrowings",
         "DebtCurrent",
         "LongTermDebtAndCapitalLeaseObligationsCurrent"),
        instantaneo=True,
    ),
    Concepto(
        "acciones_en_circulacion",
        ("CommonStockSharesOutstanding",
         "EntityCommonStockSharesOutstanding",
         "WeightedAverageNumberOfDilutedSharesOutstanding"),
        unidad="shares",
        instantaneo=True,
        descripcion="Las dos primeras son un saldo; la tercera es un promedio "
                    "del período y solo se usa si no hay saldo.",
    ),
)

CONCEPTO_POR_CLAVE: dict[str, Concepto] = {c.clave: c for c in CONCEPTOS}

#: Todas las etiquetas XBRL que nos interesan, para descartar el resto al vuelo.
#: Un ``companyfacts`` de una empresa grande pesa 10-15 MB y el 95% no se usa.
ETIQUETAS_DE_INTERES: dict[str, str] = {
    etiqueta: c.clave for c in CONCEPTOS for etiqueta in c.etiquetas
}

COLUMNAS = ("ticker", "cik", "metrica", "etiqueta", "valor", "unidad",
            "inicio", "fin", "filed", "forma", "fy", "fp", "accn")


@dataclass
class Hecho:
    """Un dato reportado, con la presentación que lo trajo."""

    ticker: str
    cik: str
    metrica: str
    etiqueta: str
    valor: float
    unidad: str
    inicio: str | None
    fin: str
    filed: str
    forma: str
    fy: int | None
    fp: str | None
    accn: str

    def fila(self) -> list[Any]:
        return [self.ticker, self.cik, self.metrica, self.etiqueta, self.valor,
                self.unidad, self.inicio or "", self.fin, self.filed,
                self.forma, self.fy or "", self.fp or "", self.accn]


# --------------------------------------------------------------------------
# Red — aislada en una función para poder sustituirla en las pruebas
# --------------------------------------------------------------------------

class Limitador:
    """Espaciador de peticiones. La SEC bloquea al que se pasa de 10/s."""

    def __init__(self, rps: float | None = None):
        # Se lee al construir y no en la firma: con el valor enlazado en el
        # `def`, una prueba no puede bajar el ritmo y cada corrida tardaría
        # segundos de espera artificial.
        rps = SEC_MAX_RPS if rps is None else rps
        self.intervalo = 1.0 / rps if rps > 0 else 0.0
        self._ultima = 0.0

    def esperar(self) -> None:
        if self.intervalo <= 0:
            return
        falta = self.intervalo - (time.monotonic() - self._ultima)
        if falta > 0:
            time.sleep(falta)
        self._ultima = time.monotonic()


def user_agent(contacto: str) -> str:
    """
    La SEC exige identificarse con un correo real.

    No es burocracia: peticiones anónimas se bloquean, y el bloqueo es por IP.
    """
    contacto = (contacto or "").strip()
    if "@" not in contacto:
        raise ValueError(
            "La SEC exige un User-Agent con correo de contacto real. "
            "Pasa algo como 'CCI Puesto de Bolsa tucorreo@dominio.com'."
        )
    return contacto


def fetch_json(url: str, *, contacto: str,
               limitador: Limitador | None = None) -> dict:
    """La única línea que toca la red. Sustituible en pruebas."""
    import requests

    if limitador is not None:
        limitador.esperar()
    respuesta = requests.get(
        url, timeout=30,
        headers={"User-Agent": user_agent(contacto),
                 "Accept-Encoding": "gzip, deflate"})
    respuesta.raise_for_status()
    return respuesta.json()


# --------------------------------------------------------------------------
# Mapa ticker -> CIK
# --------------------------------------------------------------------------

def parse_ticker_map(payload: Mapping[str, Any]) -> dict[str, str]:
    """
    ``{"0": {"cik_str": 320193, "ticker": "AAPL", ...}, ...}`` -> ``{AAPL: CIK...}``.

    El CIK va a diez dígitos con ceros a la izquierda, que es como lo quiere la
    API. Sin el relleno, ``companyfacts`` responde 404 sin explicar por qué.
    """
    out: dict[str, str] = {}
    filas = payload.values() if isinstance(payload, Mapping) else payload
    for fila in filas:
        try:
            ticker = str(fila["ticker"]).strip().upper()
            cik = f"{int(fila['cik_str']):010d}"
        except (KeyError, TypeError, ValueError):
            continue
        if ticker:
            out.setdefault(ticker, cik)
    return out


def load_ticker_map(*, contacto: str, cache: Path | str | None = None,
                    fetch: Callable[..., dict] | None = None,
                    limitador: Limitador | None = None) -> dict[str, str]:
    """Mapa ticker->CIK, del disco si ya está y de la SEC si no."""
    fetch = fetch_json if fetch is None else fetch
    cache = Path(cache) if cache else None
    if cache and cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))

    mapa = parse_ticker_map(fetch(SEC_TICKERS, contacto=contacto,
                                  limitador=limitador))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(mapa, indent=0, sort_keys=True),
                         encoding="utf-8")
    return mapa


# --------------------------------------------------------------------------
# Extracción
# --------------------------------------------------------------------------

def _numero(valor: Any) -> float | None:
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    return None if v != v else v          # descarta NaN


def extract_facts(payload: Mapping[str, Any], ticker: str,
                  *, formas: Sequence[str] = FORMAS_VALIDAS,
                  ) -> tuple[list[Hecho], dict[str, str]]:
    """
    Hechos de un ``companyfacts``, y qué etiqueta ganó cada métrica.

    Se queda con **todas las versiones** de cada período, no solo la última.
    Ese es el punto entero del ejercicio: el original y cada corrección son
    entradas distintas con su propio ``filed``, y :func:`as_of` elige después.
    Quedarse solo con la última convertiría a EDGAR en un proveedor más.
    """
    cik = str(payload.get("cik") or "")
    if cik:
        cik = f"{int(cik):010d}"
    facts = payload.get("facts") or {}
    validas = {f.upper() for f in formas}

    disponibles: dict[str, dict] = {}
    for taxonomia in ("us-gaap", "ifrs-full", "dei"):
        for etiqueta, cuerpo in (facts.get(taxonomia) or {}).items():
            if etiqueta in ETIQUETAS_DE_INTERES:
                disponibles.setdefault(etiqueta, cuerpo)

    hechos: list[Hecho] = []
    elegidas: dict[str, str] = {}

    for concepto in CONCEPTOS:
        etiqueta = next((e for e in concepto.etiquetas if e in disponibles), None)
        if etiqueta is None:
            continue
        unidades = (disponibles[etiqueta].get("units") or {})
        serie = unidades.get(concepto.unidad)
        if not serie:
            # Un emisor puede reportar en otra unidad (CAD, EUR). Mezclarla con
            # USD en un z-score transversal sería comparar monedas distintas.
            continue

        elegidas[concepto.clave] = etiqueta
        for dato in serie:
            forma = str(dato.get("form") or "").upper()
            if forma not in validas:
                continue
            valor = _numero(dato.get("val"))
            fin, filed = dato.get("end"), dato.get("filed")
            if valor is None or not fin or not filed:
                continue
            hechos.append(Hecho(
                ticker=ticker.upper(), cik=cik, metrica=concepto.clave,
                etiqueta=etiqueta, valor=valor, unidad=concepto.unidad,
                inicio=dato.get("start"), fin=str(fin), filed=str(filed),
                forma=forma, fy=dato.get("fy"), fp=dato.get("fp"),
                accn=str(dato.get("accn") or "")))

    return hechos, elegidas


def company_facts(cik: str, *, contacto: str,
                  fetch: Callable[..., dict] | None = None,
                  limitador: Limitador | None = None) -> dict:
    """
    El ``companyfacts`` de un emisor: todo lo que ha reportado.

    ``fetch`` se resuelve al llamar y no al definir. Con un valor por defecto
    enlazado en la firma, una prueba no puede sustituirlo parcheando el módulo,
    y la única forma de ejercitar el guion completo sería salir a la red.
    """
    fetch = fetch_json if fetch is None else fetch
    url = f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json"
    return fetch(url, contacto=contacto, limitador=limitador)


# --------------------------------------------------------------------------
# Almacén en disco — un CSV por ticker
# --------------------------------------------------------------------------
#
# Por ticker y no consolidado para que actualizar sea reescribir un archivo.
# Es la misma forma que ``tenencias/``, y por la misma razón: lo que se puede
# rehacer por partes se rehace por partes.

def escribir_hechos(destino: Path | str, ticker: str,
                    hechos: Sequence[Hecho]) -> Path:
    import csv

    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"{ticker.upper()}.csv"
    with archivo.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(COLUMNAS)
        for h in sorted(hechos, key=lambda x: (x.metrica, x.fin, x.filed)):
            escritor.writerow(h.fila())
    return archivo


def leer_hechos(directorio: Path | str,
                tickers: Iterable[str] | None = None) -> "pd.DataFrame":
    """Todo el almacén como una tabla. Vacía si no hay nada bajado."""
    import pandas as pd

    directorio = Path(directorio)
    if not directorio.is_dir():
        return pd.DataFrame(columns=list(COLUMNAS))

    querido = None if tickers is None else {str(t).upper() for t in tickers}
    marcos = []
    for archivo in sorted(directorio.glob("*.csv")):
        if archivo.name.startswith("_"):
            continue
        if querido is not None and archivo.stem.upper() not in querido:
            continue
        try:
            marcos.append(pd.read_csv(archivo, dtype={"cik": str, "fy": str}))
        except Exception:                  # noqa: BLE001 - un archivo malo no
            continue                       # puede tumbar el almacén entero
    if not marcos:
        return pd.DataFrame(columns=list(COLUMNAS))
    return pd.concat(marcos, ignore_index=True)


# --------------------------------------------------------------------------
# Point-in-time
# --------------------------------------------------------------------------

def as_of(hechos: "pd.DataFrame", fecha: str | None = None,
          *, metricas: Sequence[str] | None = None) -> "pd.DataFrame":
    """
    Lo que se sabía en ``fecha``: última versión de cada período presentada
    **en o antes** de ese día.

    Esta función es la razón de ser del módulo. Sin ella el almacén es otro
    proveedor de datos restatados; con ella, un backtest puede preguntar qué
    veía el mercado un martes de 2023 y recibir la respuesta correcta, con los
    errores que después se corrigieron incluidos. Esos errores son parte de lo
    que el modelo tenía delante y quitarlos infla el IC.

    ``fecha=None`` devuelve todo lo conocido hoy, que es lo que quiere una
    corrida normal.
    """
    import pandas as pd

    if hechos.empty:
        return hechos

    vista = hechos
    if fecha is not None:
        vista = vista[vista["filed"].astype(str) <= str(fecha)]
    if metricas is not None:
        vista = vista[vista["metrica"].isin(list(metricas))]
    if vista.empty:
        return vista.copy()

    # Orden estable: dentro de un mismo período gana el filed más reciente, y
    # si dos presentaciones comparten fecha, la de accn mayor (la posterior).
    vista = vista.sort_values(["ticker", "metrica", "fin", "inicio",
                              "filed", "accn"], kind="stable")
    return (vista.drop_duplicates(subset=["ticker", "metrica", "fin", "inicio"],
                                  keep="last")
                 .reset_index(drop=True))


def ultimo_anual(hechos: "pd.DataFrame", fecha: str | None = None,
                 ) -> "pd.DataFrame":
    """
    Una fila por ticker y métrica: el período anual más reciente conocible.

    Para un saldo se toma el corte más reciente. Para un flujo se exige un
    período de entre 300 y 400 días — es decir, un año — porque sumar cuatro
    trimestres desde XBRL tiene trampas (los 10-Q traen acumulados de ejercicio
    en unos emisores y trimestres sueltos en otros) y un TTM mal armado es un
    número peor que no tener número.
    """
    import pandas as pd

    vista = as_of(hechos, fecha)
    if vista.empty:
        return vista

    inicio = pd.to_datetime(vista["inicio"], errors="coerce")
    fin = pd.to_datetime(vista["fin"], errors="coerce")
    dias = (fin - inicio).dt.days

    es_instantaneo = vista["metrica"].map(
        lambda m: CONCEPTO_POR_CLAVE[m].instantaneo
        if m in CONCEPTO_POR_CLAVE else False)

    anual = (es_instantaneo & inicio.isna()) | (~es_instantaneo
                                                & dias.between(300, 400))
    vista = vista[anual.fillna(False)]
    if vista.empty:
        return vista

    return (vista.sort_values(["ticker", "metrica", "fin"], kind="stable")
                 .drop_duplicates(subset=["ticker", "metrica"], keep="last")
                 .reset_index(drop=True))


# --------------------------------------------------------------------------
# Cobertura — el entregable de la fase 2
# --------------------------------------------------------------------------

def coverage_report(hechos: "pd.DataFrame", tickers: Sequence[str],
                    fecha: str | None = None) -> "pd.DataFrame":
    """
    Qué porcentaje del universo tiene cada métrica de verdad.

    Es la pregunta que decide si vale la pena construir el bloque fundamental.
    Con 60% de cobertura un z-score transversal compara a los que reportaron
    contra un hueco, y eso no es una medición.

    ``etiqueta_principal`` es la **más usada**, no la primera del alfabeto.
    Ordenar alfabéticamente parecía inofensivo y no lo era: en una corrida real
    la columna reportó ``Depreciation``, que es el último recurso de la lista de
    candidatos, cuando la etiqueta que trajo casi todos los números era otra. Una
    columna de trazabilidad que apunta a la etiqueta equivocada es peor que no
    tenerla, porque invita a buscar el problema donde no está.
    """
    import pandas as pd

    pedidos = [str(t).upper() for t in tickers]
    anual = ultimo_anual(hechos, fecha)

    filas = []
    for concepto in CONCEPTOS:
        sub = anual[anual["metrica"] == concepto.clave] if not anual.empty \
            else pd.DataFrame(columns=list(COLUMNAS))
        con_dato = set(sub["ticker"]) & set(pedidos)

        # Cuántos emisores trajo cada etiqueta. Se desempata por el orden
        # declarado en CONCEPTOS, que es el orden de preferencia real.
        conteo = (sub[sub["ticker"].isin(pedidos)]
                  .drop_duplicates(subset=["ticker"])["etiqueta"]
                  .value_counts() if not sub.empty else pd.Series(dtype=int))
        prioridad = {e: i for i, e in enumerate(concepto.etiquetas)}
        etiquetas = sorted(conteo.index,
                           key=lambda e: (-int(conteo[e]),
                                          prioridad.get(e, len(prioridad))))
        filas.append({
            "metrica": concepto.clave,
            "instantaneo": concepto.instantaneo,
            "cobertura": len(con_dato) / len(pedidos) if pedidos else 0.0,
            "con_dato": len(con_dato),
            "sin_dato": len(pedidos) - len(con_dato),
            "etiquetas_usadas": len(etiquetas),
            "etiqueta_principal": etiquetas[0] if etiquetas else "",
            "etiquetas_detalle": " | ".join(f"{e}×{int(conteo[e])}"
                                            for e in etiquetas),
            "faltan": ", ".join(sorted(set(pedidos) - con_dato)[:12]),
        })
    return pd.DataFrame(filas).sort_values("cobertura", ascending=False,
                                           ignore_index=True)


def historia_por_ticker(hechos: "pd.DataFrame") -> "pd.DataFrame":
    """Cuánta historia hay por nombre: períodos, versiones y desde cuándo."""
    import pandas as pd

    if hechos.empty:
        return pd.DataFrame(columns=["ticker", "metricas", "periodos",
                                     "versiones", "desde", "hasta"])
    grupo = hechos.groupby("ticker")
    return pd.DataFrame({
        "metricas": grupo["metrica"].nunique(),
        "periodos": grupo.apply(
            lambda g: g.drop_duplicates(subset=["metrica", "fin", "inicio"]).shape[0],
            include_groups=False),
        "versiones": grupo.size(),
        "desde": grupo["fin"].min(),
        "hasta": grupo["fin"].max(),
    }).reset_index()


def restatements(hechos: "pd.DataFrame", minimo: float = 0.02) -> "pd.DataFrame":
    """
    Períodos que se reportaron más de una vez con cifras distintas.

    No es una curiosidad: es la prueba de que el point-in-time hace falta. Cada
    fila aquí es un dato que un proveedor te habría dado ya corregido, y que en
    su momento el mercado no tenía.
    """
    import pandas as pd

    if hechos.empty:
        return pd.DataFrame(columns=["ticker", "metrica", "fin", "versiones",
                                     "primero", "ultimo", "cambio"])
    orden = hechos.sort_values("filed", kind="stable")
    grupo = orden.groupby(["ticker", "metrica", "fin"], as_index=False)
    tabla = grupo.agg(versiones=("valor", "size"),
                      primero=("valor", "first"),
                      ultimo=("valor", "last"))
    tabla = tabla[tabla["versiones"] > 1].copy()
    if tabla.empty:
        return tabla
    base = tabla["primero"].abs().replace(0.0, float("nan"))
    tabla["cambio"] = (tabla["ultimo"] - tabla["primero"]) / base
    tabla = tabla[tabla["cambio"].abs() >= minimo]
    return tabla.sort_values("cambio", key=abs, ascending=False,
                             ignore_index=True)
