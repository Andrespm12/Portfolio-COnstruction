"""
Pruebas del tope sectorial y del núcleo indexado.

Las dos cosas salieron de la misma corrida real: una cartera Agresiva con ~35%
en la cadena de semiconductores que pasó su auditoría de bandas limpia, y cuyos
únicos ETFs de renta variable eran XBI, EWT y EWY — porque el ranking, pesado
en momentum, decidió no solo qué era bueno sino **qué estaba disponible**.

Lo que se verifica aquí es que el tope efectivamente ate, que no se pueda
duplicar partiendo el nombre del sector, y que lo que no tiene dato quede
declarado en vez de supuesto.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.cci_regulation import EXPOSICIONES_NUCLEO, SECTOR_CAPS
from screener.lookthrough import (normalize_sector, sector_map,
                                  stock_sectors_for)
from screener.optimizer import (audit_sectors, core_vehicles, optimize,
                                sector_exposures, select_basket)


@dataclass
class Fila:
    """Lo que select_basket / core_vehicles leen de un ScoredInstrument."""
    ticker: str
    score: float
    asset_type: str = "ETF"
    eligible: bool = True
    sector: str | None = None


def covarianza(tickers, vol=0.20, corr=0.3):
    n = len(tickers)
    m = np.full((n, n), corr * vol * vol)
    np.fill_diagonal(m, vol * vol)
    return pd.DataFrame(m, index=tickers, columns=tickers)


# ------------------------------------------------------------ normalización
@pytest.mark.parametrize("crudo,esperado", [
    ("technology", "Technology"),
    ("Technology", "Technology"),
    ("information technology", "Technology"),
    ("financial_services", "Financial Services"),
    ("Financial Services", "Financial Services"),
    ("financials", "Financial Services"),
    ("healthcare", "Health Care"),
    ("Health Care", "Health Care"),
    ("consumer_cyclical", "Consumer Discretionary"),
    ("Consumer Discretionary", "Consumer Discretionary"),
    ("basic_materials", "Materials"),
    ("  real   estate  ", "Real Estate"),
])
def test_las_dos_escrituras_caen_en_el_mismo_cubo(crudo, esperado):
    assert normalize_sector(crudo) == esperado


def test_un_sector_desconocido_no_se_traduce():
    # Meterlo a la fuerza en un cubo conocido lo sometería a un tope que no le
    # toca. Se limpia y se deja aparte.
    assert normalize_sector("Blockchain  Infrastructure") == "Blockchain Infrastructure"


def test_vacio_es_vacio():
    assert normalize_sector("") == ""
    assert normalize_sector(None) == ""


def test_el_mapa_une_la_accion_con_el_fondo_que_la_contiene():
    # El defecto que esto evita: con 'technology' y 'Technology' separados, un
    # tope del 22% permite 44% en la misma industria.
    mapa, _, _ = sector_map(["SPY", "AAPL"],
                            {"SPY": {"technology": 0.6, "financials": 0.4}},
                            {"AAPL": "Technology"})
    assert set(mapa) == {"Technology", "Financial Services"}
    assert mapa["Technology"] == pytest.approx({"SPY": 0.6, "AAPL": 1.0})


# ------------------------------------------------------------------ mapa
def test_una_accion_aporta_todo_su_peso_a_un_solo_sector():
    mapa, cobertura, _ = sector_map(["AAPL"], {}, {"AAPL": "Technology"})
    assert mapa == {"Technology": {"AAPL": 1.0}}
    assert cobertura["AAPL"] == 1.0


def test_un_fondo_aporta_su_desglose():
    mapa, _, _ = sector_map(
        ["XLK"], {"XLK": {"technology": 0.9, "industrials": 0.1}}, {})
    assert mapa["Technology"]["XLK"] == pytest.approx(0.9)
    assert mapa["Industrials"]["XLK"] == pytest.approx(0.1)


def test_el_desglose_se_normaliza_a_uno():
    # Los emisores publican en fracción o en por ciento. Sin normalizar, un
    # fondo publicado en por ciento aportaría 100 veces su peso al tope.
    mapa, _, _ = sector_map(["XLK"], {"XLK": {"technology": 90.0,
                                              "industrials": 10.0}}, {})
    assert mapa["Technology"]["XLK"] == pytest.approx(0.9)


def test_lo_que_no_tiene_sector_queda_fuera_y_se_declara():
    mapa, cobertura, notas = sector_map(["AAPL", "GLD"], {}, {"AAPL": "Technology"})
    assert "GLD" not in {t for fila in mapa.values() for t in fila}
    assert cobertura["GLD"] == 0.0
    assert any("GLD" in n and "sin restringir" in n for n in notas)


def test_el_fondo_manda_sobre_el_sector_de_accion():
    # Si un ticker aparece en los dos, el desglose del fondo es el completo.
    mapa, _, _ = sector_map(["XLK"], {"XLK": {"technology": 1.0}},
                            {"XLK": "Financial Services"})
    assert set(mapa) == {"Technology"}


# ------------------------------------------------------------- restricción
def _problema(sectores_por_accion, cap):
    tickers = list(sectores_por_accion) + ["AGG", "BIL"]
    tipos = {**{t: "STOCK" for t in sectores_por_accion},
             "AGG": "ETF", "BIL": "ETF"}
    mu = pd.Series({t: (0.12 if t in sectores_por_accion else 0.02)
                    for t in tickers})
    mapa, _, _ = sector_map(tickers, {}, sectores_por_accion)
    return optimize(mu, covarianza(tickers), tipos, "Agresivo",
                    min_position=None, sector_weights=mapa, sector_cap=cap)


def test_el_tope_ata_de_verdad():
    # Cinco tecnológicas con el retorno más alto: sin tope el optimizador las
    # compra todas hasta el techo de renta variable.
    acciones = {t: "Technology" for t in ("AAPL", "MSFT", "NVDA", "AMD", "MU")}
    acciones.update({"JNJ": "Health Care", "XOM": "Energy",
                     "JPM": "Financial Services"})

    libre = _problema(acciones, None)
    atado = _problema(acciones, 0.25)

    assert libre.feasible and atado.feasible
    assert sector_exposures(libre.weights, sector_map(
        list(libre.weights.index), {}, acciones)[0])["Technology"] > 0.25
    assert atado.sector_exposure["Technology"] <= 0.25 + 1e-6


def test_sin_mapa_sectorial_la_corrida_lo_dice():
    acciones = {"AAPL": "Technology", "JNJ": "Health Care"}
    tickers = list(acciones) + ["AGG", "BIL"]
    tipos = {"AAPL": "STOCK", "JNJ": "STOCK", "AGG": "ETF", "BIL": "ETF"}
    mu = pd.Series({t: 0.08 for t in tickers})
    alloc = optimize(mu, covarianza(tickers), tipos, "Agresivo",
                     min_position=None)
    assert any("SIN restringir" in n for n in alloc.notes)
    assert alloc.sector_exposure == {}


def test_la_nota_declara_que_el_numero_es_de_la_mesa():
    # Un tope inventado por nosotros no puede leerse como límite regulatorio.
    alloc = _problema({"AAPL": "Technology", "JNJ": "Health Care",
                       "XOM": "Energy"}, 0.30)
    assert any("Procedimiento" in n and "Comité" in n for n in alloc.notes)


def test_lo_no_cubierto_se_reporta_como_hueco():
    tickers = ["AAPL", "MSFT", "GLD", "AGG", "BIL"]
    tipos = {"AAPL": "STOCK", "MSFT": "STOCK", "GLD": "ETF",
             "AGG": "ETF", "BIL": "ETF"}
    mu = pd.Series({t: 0.06 for t in tickers})
    mapa, _, _ = sector_map(tickers, {}, {"AAPL": "Technology",
                                          "MSFT": "Technology"})
    alloc = optimize(mu, covarianza(tickers), tipos, "Agresivo",
                     min_position=None, sector_weights=mapa, sector_cap=0.25)
    assert any("10 de" in n or "de 5 instrumentos" in n for n in alloc.notes)


def test_un_tope_imposible_explica_por_que_no_hay_cartera():
    # Dos sectores, tope del 10% cada uno: entre los dos no llegan al 100% que
    # hay que invertir, así que no existe solución. Sin el tope sí la hay, y esa
    # es exactamente la diferencia que el diagnóstico tiene que nombrar en vez
    # de devolver un "infeasible" pelado.
    tickers = ["AAPL", "MSFT", "NVDA", "TLT", "AGG", "BIL"]
    tipos = {t: ("STOCK" if t in ("AAPL", "MSFT", "NVDA") else "ETF")
             for t in tickers}
    sectores = {"AAPL": "Technology", "MSFT": "Technology",
                "NVDA": "Technology", "TLT": "Government",
                "AGG": "Government", "BIL": "Government"}
    mu = pd.Series({t: 0.06 for t in tickers})
    mapa, _, _ = sector_map(tickers, {}, sectores)

    libre = optimize(mu, covarianza(tickers), tipos, "Agresivo",
                     min_position=None, sector_weights=mapa, sector_cap=None)
    assert libre.feasible, "sin el tope el problema sí tiene solución"

    alloc = optimize(mu, covarianza(tickers), tipos, "Agresivo",
                     min_position=None, sector_weights=mapa, sector_cap=0.10)
    assert not alloc.feasible
    assert any("tope sectorial" in b for b in alloc.breaches), alloc.breaches


def test_el_mapa_no_arrastra_nombres_fuera_del_problema():
    # Un mapa armado para un universo más amplio no puede meter al optimizador
    # tickers que no están en su covarianza.
    tickers = ["AAPL", "AGG", "BIL"]
    tipos = {"AAPL": "STOCK", "AGG": "ETF", "BIL": "ETF"}
    mu = pd.Series({t: 0.05 for t in tickers})
    mapa = {"Technology": {"AAPL": 1.0, "FUERA": 1.0}}
    alloc = optimize(mu, covarianza(tickers), tipos, "Agresivo",
                     min_position=None, sector_weights=mapa, sector_cap=0.25)
    assert alloc.feasible
    assert "FUERA" not in alloc.weights.index


# ------------------------------------------------------------- auditoría
def test_la_auditoria_sectorial_puede_fallar():
    mapa = {"Technology": {"AAPL": 1.0, "MSFT": 1.0}}
    pesos = pd.Series({"AAPL": 0.20, "MSFT": 0.20})
    assert audit_sectors(pesos, mapa, 0.25)
    assert not audit_sectors(pesos, mapa, 0.45)


def test_sin_tope_no_hay_incumplimiento_que_reportar():
    mapa = {"Technology": {"AAPL": 1.0}}
    assert audit_sectors(pd.Series({"AAPL": 0.9}), mapa, None) == []


# ------------------------------------------------------------------ núcleo
def test_el_nucleo_elige_el_mejor_vehiculo_por_exposicion():
    scored = [Fila("VOO", 80.0), Fila("SPY", 95.0), Fila("IVV", 90.0),
              Fila("IWM", 70.0), Fila("EFA", 60.0), Fila("IEMG", 55.0),
              Fila("QQQ", 50.0)]
    vehiculos, notas = core_vehicles(scored)
    assert vehiculos["EEUU amplio"] == "SPY"
    assert any("SPY" in n and "IVV" in n for n in notas)


def test_un_vehiculo_no_elegible_no_gana_su_cupo():
    scored = [Fila("SPY", 99.0, eligible=False), Fila("IVV", 60.0)]
    vehiculos, _ = core_vehicles(scored)
    assert vehiculos["EEUU amplio"] == "IVV"


def test_una_exposicion_sin_vehiculo_se_reporta_no_se_sustituye():
    scored = [Fila("SPY", 90.0)]
    vehiculos, notas = core_vehicles(scored)
    assert set(vehiculos) == {"EEUU amplio"}
    assert any("Emergentes" in n and "vehículo" in n for n in notas)
    # Y sobre todo: no se coló un fondo de otra línea para tapar el hueco.
    assert set(vehiculos.values()) == {"SPY"}


def test_el_nucleo_entra_a_la_cesta_aunque_puntue_bajo():
    # El caso real: SPY salió #149 en un ranking pesado en momentum y nunca
    # llegó al optimizador. Un ranking decide qué corre bien; no puede decidir
    # también qué está disponible.
    scored = ([Fila(f"HOT{i}", 99.0 - i, asset_type="STOCK") for i in range(25)]
              + [Fila("SPY", 30.0), Fila("IWM", 28.0), Fila("EFA", 27.0),
                 Fila("IEMG", 26.0), Fila("QQQ", 25.0),
                 Fila("AGG", 20.0), Fila("BIL", 19.0), Fila("TLT", 18.0)])
    cesta, notas = select_basket(scored, "Agresivo", top_n=25, min_per_class=1)

    assert "SPY" in cesta and "IWM" in cesta and "IEMG" in cesta
    assert any("por construcción" in n for n in notas)


def test_se_puede_apagar_el_nucleo():
    scored = ([Fila(f"HOT{i}", 99.0 - i, asset_type="STOCK") for i in range(25)]
              + [Fila("SPY", 30.0), Fila("AGG", 20.0)])
    cesta, _ = select_basket(scored, "Agresivo", top_n=25, min_per_class=0,
                             include_core=False)
    assert "SPY" not in cesta


def test_el_nucleo_no_duplica_lo_que_ya_estaba_en_el_top():
    scored = [Fila("SPY", 99.0), Fila("AGG", 50.0), Fila("BIL", 40.0)]
    cesta, _ = select_basket(scored, "Agresivo", top_n=25, min_per_class=1)
    assert cesta.count("SPY") == 1


def test_las_exposiciones_nucleo_son_solo_beta_amplia():
    # Un ETF de valor o de dividendo es una apuesta, no beta de mercado: tiene
    # que ganarse la entrada por el ranking como cualquier otra.
    todos = {t for v in EXPOSICIONES_NUCLEO.values() for t in v}
    assert not todos & {"SCHD", "VTV", "VLUE", "IWD", "VYM", "XLK", "XBI",
                        "EWY", "EWT", "SOXX", "SMH"}


# ------------------------------------------------- sector de las acciones
def test_no_se_baja_lo_que_ya_se_tiene():
    def explota(_):
        raise AssertionError("no debió llamar a la red")

    sectores, notas = stock_sectors_for(
        ["AAPL", "MSFT"], {"AAPL": "Technology", "MSFT": "Technology"},
        fetch=explota)
    assert sectores == {"AAPL": "Technology", "MSFT": "Technology"}
    assert notas == []


def test_se_baja_solo_lo_que_falta():
    pedidos = []

    def falso(tickers):
        pedidos.append(list(tickers))
        return {}, {t: "Technology" for t in tickers}

    sectores, notas = stock_sectors_for(["AAPL", "MSFT"], {"AAPL": "Health Care"},
                                        fetch=falso)
    assert pedidos == [["MSFT"]], "no debe volver a pedir lo que ya estaba"
    assert sectores == {"AAPL": "Health Care", "MSFT": "Technology"}
    assert any("1 nombre" in n for n in notas)


def test_si_la_bajada_falla_la_corrida_sigue_y_lo_dice():
    # El defecto que esto evita: una corrida real trae CON_NOMBRES_Y_SECTORES
    # apagado, así que ninguna acción tiene sector. Si eso pasara callado, el
    # tope vería solo los ETFs y dejaría pasar la concentración en acciones
    # individuales que existe para frenar.
    def explota(_):
        raise RuntimeError("Yahoo dijo que no")

    sectores, notas = stock_sectors_for(["AAPL"], {}, fetch=explota)
    assert sectores == {}
    assert any("fuera del tope sectorial" in n for n in notas)


def test_lo_que_yahoo_no_cubre_se_nombra():
    def parcial(tickers):
        return {}, {"AAPL": "Technology"}

    sectores, notas = stock_sectors_for(["AAPL", "RARO"], {}, fetch=parcial)
    assert sectores == {"AAPL": "Technology"}
    assert any("RARO" in n for n in notas)


def test_una_cartera_sin_sectores_de_acciones_no_finge_estar_topada():
    # Solo los fondos traen sector: la parte en acciones individuales queda
    # fuera del tope y la nota tiene que decirlo con nombre y apellido.
    tickers = ["AAPL", "MSFT", "NVDA", "SPY", "AGG", "BIL"]
    tipos = {t: ("ETF" if t in ("SPY", "AGG", "BIL") else "STOCK")
             for t in tickers}
    mu = pd.Series({t: 0.07 for t in tickers})
    mapa, _, _ = sector_map(tickers, {"SPY": {"technology": 1.0}}, {})
    alloc = optimize(mu, covarianza(tickers), tipos, "Agresivo",
                     min_position=None, sector_weights=mapa, sector_cap=0.25)
    aviso = [n for n in alloc.notes if "sin sector conocido" in n]
    assert aviso, alloc.notes
    assert all(t in aviso[0] for t in ("AAPL", "MSFT", "NVDA"))


def test_hay_un_tope_declarado_para_cada_estrategia():
    from screener.cci_regulation import REGULACIONES
    assert set(SECTOR_CAPS) == set(REGULACIONES)
    # Y son monótonos: un mandato más agresivo tolera más concentración.
    orden = ["Conservador_Defensivo", "Conservador", "Moderado", "Agresivo"]
    valores = [SECTOR_CAPS[e] for e in orden]
    assert valores == sorted(valores)
