"""
Pruebas de screener/tenencias_yahoo.py.

Ninguna toca la red: se sustituye ``obtener``, que es exactamente la línea que
llama a Yahoo. Lo que se verifica es lo que puede corromper un número de
cumplimiento — la fila ``_RESTO``, la escala fracción/por ciento, y que un
fondo que falla no tumbe la corrida.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.lookthrough import parse_holdings_csv
from screener.tenencias_yahoo import (a_por_ciento, bajar_uno, bajar_varios,
                                      escribir_auxiliares, filas_con_resto,
                                      tabla_a_dict)


class FondoFalso:
    """Lo que devuelve ``yf.Ticker(tk).funds_data``, en lo que usamos."""

    def __init__(self, tenencias=None, sectores=None, canasta=None):
        self.top_holdings = tenencias
        self.sector_weightings = sectores or {}
        self.equity_holdings = canasta


def tabla(pesos: dict[str, float], columna: str = "Holding Percent"):
    return pd.DataFrame({columna: list(pesos.values())},
                        index=list(pesos.keys()))


# ------------------------------------------------------------------ tabla
def test_tabla_lee_la_columna_de_peso():
    df = tabla({"AAPL": 0.07, "MSFT": 0.06})
    assert tabla_a_dict(df) == {"AAPL": 0.07, "MSFT": 0.06}


def test_tabla_acepta_el_nombre_alterno_de_la_columna():
    df = tabla({"AAPL": 7.0}, columna="weight")
    assert tabla_a_dict(df) == {"AAPL": 7.0}


def test_tabla_cae_a_la_primera_columna_numerica():
    df = pd.DataFrame({"pct_del_fondo": [7.0, 6.0]}, index=["AAPL", "MSFT"])
    assert tabla_a_dict(df) == {"AAPL": 7.0, "MSFT": 6.0}


def test_tabla_descarta_el_relleno_de_yahoo():
    # Yahoo mete '-' cuando no publica el nombre. Contarlo como emisor
    # inventa una posición que no existe.
    df = tabla({"AAPL": 7.0, "-": 3.0, "N/A": 2.0})
    assert tabla_a_dict(df) == {"AAPL": 7.0}


def test_tabla_vacia_no_es_error():
    assert tabla_a_dict(None) == {}
    assert tabla_a_dict(pd.DataFrame()) == {}


def test_tabla_sin_columnas_numericas_devuelve_vacio():
    df = pd.DataFrame({"nombre": ["Apple"]}, index=["AAPL"])
    assert tabla_a_dict(df) == {}


def test_tabla_suma_el_ticker_repetido():
    df = pd.DataFrame({"weight": [4.0, 3.0]}, index=["AAPL", "AAPL"])
    assert tabla_a_dict(df) == {"AAPL": 7.0}


# ------------------------------------------------------------------ escala
def test_fracciones_se_vuelven_por_ciento():
    assert a_por_ciento({"AAPL": 0.07, "MSFT": 0.06}) == pytest.approx(
        {"AAPL": 7.0, "MSFT": 6.0})


def test_por_cientos_se_dejan_como_estan():
    assert a_por_ciento({"AAPL": 7.0, "MSFT": 6.0}) == pytest.approx(
        {"AAPL": 7.0, "MSFT": 6.0})


def test_una_sola_posicion_grande_en_fraccion_se_escala():
    # 1.2 no puede ser 1.2% de un fondo con una sola línea listada; es 120%
    # imposible, así que el umbral de 1.5 lo trata como fracción.
    assert a_por_ciento({"AGG": 1.2}) == pytest.approx({"AGG": 120.0})


# ------------------------------------------------------------------ _RESTO
def test_resto_cubre_lo_no_detallado():
    filas = filas_con_resto({"AAPL": 7.0, "MSFT": 6.0})
    assert filas[0] == ("AAPL", 7.0)
    assert filas[-1][0] == "_RESTO"
    assert filas[-1][1] == pytest.approx(87.0)
    assert sum(p for _, p in filas) == pytest.approx(100.0)


def test_sin_resto_cuando_el_fondo_viene_completo():
    filas = filas_con_resto({"A": 60.0, "B": 40.0})
    assert [tk for tk, _ in filas] == ["A", "B"]


def test_no_hay_resto_negativo_si_yahoo_pasa_de_cien():
    filas = filas_con_resto({"A": 70.0, "B": 40.0})
    assert all(tk != "_RESTO" for tk, _ in filas)


# ------------------------------------------------------------------ bajar
def test_bajar_uno_escribe_csv_que_el_lookthrough_entiende(tmp_path):
    fondo = FondoFalso(tabla({"AAPL": 0.07, "MSFT": 0.06}),
                       {"technology": 0.30, "healthcare": 0.13})
    exito, detalle, sectores, _ = bajar_uno("SPY", tmp_path,
                                            obtener=lambda t: fondo)

    assert exito
    assert "2 posiciones" in detalle
    assert sectores == {"technology": 0.30, "healthcare": 0.13}

    pesos, _ = parse_holdings_csv(tmp_path / "SPY.csv")
    assert pesos["AAPL"] == pytest.approx(0.07)
    assert pesos["_RESTO"] == pytest.approx(0.87)


def test_bajar_uno_reporta_el_fallo_sin_levantar(tmp_path):
    def explota(_):
        raise RuntimeError("Yahoo dijo que no")

    exito, detalle, _, _ = bajar_uno("XXX", tmp_path, obtener=explota)
    assert not exito
    assert "RuntimeError" in detalle and "Yahoo dijo que no" in detalle
    assert not list(tmp_path.iterdir())


def test_una_accion_no_es_un_fondo(tmp_path):
    exito, detalle, _, _ = bajar_uno("AAPL", tmp_path,
                                     obtener=lambda t: FondoFalso())
    assert not exito
    assert "composición" in detalle


def test_fondo_con_sectores_pero_sin_tenencias_sirve(tmp_path):
    # Yahoo cubre el desglose sectorial de muchos fondos cuyas posiciones no
    # publica. Ese fondo aporta sectores aunque no aporte emisores.
    exito, detalle, sectores, _ = bajar_uno(
        "EFA", tmp_path, obtener=lambda t: FondoFalso(None, {"financials": 0.2}))
    assert exito
    assert sectores == {"financials": 0.2}
    assert not (tmp_path / "EFA.csv").exists()


def test_destino_none_no_escribe_nada(tmp_path):
    exito, _, _, _ = bajar_uno("SPY", None,
                               obtener=lambda t: FondoFalso(tabla({"AAPL": 7.0})))
    assert exito
    assert not list(tmp_path.iterdir())


def test_canasta_se_extrae_cuando_la_hay(tmp_path):
    canasta = pd.DataFrame({"SPY": [24.5, 4.1]}, index=["priceToEarnings",
                                                        "priceToBook"])
    _, _, _, salida = bajar_uno(
        "SPY", tmp_path,
        obtener=lambda t: FondoFalso(tabla({"AAPL": 7.0}), {}, canasta))
    assert salida == pytest.approx({"priceToEarnings": 24.5,
                                    "priceToBook": 4.1})


def test_canasta_con_texto_no_tumba_la_bajada(tmp_path):
    canasta = pd.DataFrame({"SPY": ["n/a", 4.1]}, index=["priceToEarnings",
                                                        "priceToBook"])
    exito, _, _, salida = bajar_uno(
        "SPY", tmp_path,
        obtener=lambda t: FondoFalso(tabla({"AAPL": 7.0}), {}, canasta))
    assert exito
    assert salida == pytest.approx({"priceToBook": 4.1})


# ------------------------------------------------------------------ varios
def test_bajar_varios_separa_los_que_fallan(tmp_path):
    fondos = {"SPY": FondoFalso(tabla({"AAPL": 7.0}), {"technology": 0.3}),
              "QQQ": FondoFalso(tabla({"MSFT": 8.0}), {"technology": 0.5})}

    def obtener(tk):
        if tk not in fondos:
            raise RuntimeError("sin datos")
        return fondos[tk]

    ok, fallaron = bajar_varios(["SPY", "XXX", "QQQ"], tmp_path,
                                obtener=obtener, log=None)
    assert ok == ["SPY", "QQQ"]
    assert fallaron == ["XXX"]
    assert (tmp_path / "SPY.csv").exists()
    assert not (tmp_path / "XXX.csv").exists()


def test_sectores_de_todos_los_fondos_caen_en_un_archivo(tmp_path):
    fondos = {"SPY": FondoFalso(tabla({"AAPL": 7.0}), {"technology": 0.30}),
              "QQQ": FondoFalso(tabla({"MSFT": 8.0}), {"technology": 0.50})}
    bajar_varios(["SPY", "QQQ"], tmp_path, obtener=lambda t: fondos[t], log=None)

    filas = list(csv.DictReader((tmp_path / "_sectores.csv").open()))
    assert {f["fondo"] for f in filas} == {"SPY", "QQQ"}
    assert float(next(f["peso"] for f in filas
                      if f["fondo"] == "QQQ")) == pytest.approx(0.50)


def test_sin_nada_que_escribir_no_deja_archivos_vacios(tmp_path):
    escribir_auxiliares(tmp_path, {}, {})
    assert not (tmp_path / "_sectores.csv").exists()
    assert not (tmp_path / "_canasta.csv").exists()


def test_lo_bajado_lo_lee_el_lookthrough_sin_tocar_nada(tmp_path):
    # La prueba que importa: el directorio que produce el descargador es el
    # que consume la transparencia. Los auxiliares empiezan con '_' porque
    # load_holdings los salta; si el nombre cambiara, intentaría leer
    # _sectores.csv como si fuera un fondo y la corrida moriría ahí.
    from screener.lookthrough import effective_exposure, load_holdings

    fondos = {"SPY": FondoFalso(tabla({"AAPL": 0.07, "MSFT": 0.06}),
                                {"technology": 0.30}),
              "QQQ": FondoFalso(tabla({"AAPL": 0.09}), {"technology": 0.50})}
    bajar_varios(["SPY", "QQQ"], tmp_path, obtener=lambda t: fondos[t], log=None)

    tenencias, _, avisos = load_holdings(tmp_path)
    assert set(tenencias) == {"SPY", "QQQ"}
    assert avisos == []

    exposicion, cobertura, _ = effective_exposure(
        {"SPY": 0.5, "QQQ": 0.5}, tenencias)
    assert cobertura == pytest.approx(1.0)
    # AAPL entra por los dos fondos: 0.5*7% + 0.5*9%.
    assert exposicion["AAPL"] == pytest.approx(0.08)
