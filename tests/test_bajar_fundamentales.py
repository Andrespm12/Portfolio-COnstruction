"""
El guion de descarga, corrido entero contra un EDGAR falso.

Sustituye ``fetch_json``, que es la única línea que sale a la red. Todo lo
demás — mapa de tickers, extracción, escritura, incremental, cobertura — corre
de verdad, porque es donde están los errores que llegarían a la máquina de
quien lo use.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import bajar_fundamentales as guion  # noqa: E402
import screener.edgar as edgar  # noqa: E402

TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL"},
           "1": {"cik_str": 789019, "ticker": "MSFT"},
           "2": {"cik_str": 1045810, "ticker": "NVDA"}}


def facts(cik, ingresos, activos, *, etiqueta_ingresos="Revenues"):
    return {"cik": cik, "facts": {"us-gaap": {
        etiqueta_ingresos: {"units": {"USD": [
            {"val": ingresos, "start": "2024-01-01", "end": "2024-12-31",
             "filed": "2025-02-01", "form": "10-K", "accn": "a", "fy": 2024,
             "fp": "FY"},
            # El mismo período, corregido más tarde: el point-in-time existe
            # por esto.
            {"val": ingresos * 1.1, "start": "2024-01-01", "end": "2024-12-31",
             "filed": "2026-02-01", "form": "10-K", "accn": "b", "fy": 2025,
             "fp": "FY"}]}},
        "Assets": {"units": {"USD": [
            {"val": activos, "end": "2024-12-31", "filed": "2025-02-01",
             "form": "10-K", "accn": "a", "fy": 2024, "fp": "FY"}]}}}}}


CUERPOS = {
    "0000320193": facts(320193, 100.0, 1000.0),
    "0000789019": facts(789019, 200.0, 2000.0),
    # NVDA usa la etiqueta post-ASC 606; el mapeo tiene que resolverlo solo.
    "0001045810": facts(
        1045810, 300.0, 3000.0,
        etiqueta_ingresos="RevenueFromContractWithCustomerExcludingAssessedTax"),
}


@pytest.fixture
def edgar_falso(monkeypatch):
    """Sustituye la red y cuenta las peticiones."""
    pedidos: list[str] = []

    def fetch(url, *, contacto=None, limitador=None):
        pedidos.append(url)
        if url == edgar.SEC_TICKERS:
            return TICKERS
        cik = url.rsplit("CIK", 1)[-1].removesuffix(".json")
        if cik not in CUERPOS:
            raise RuntimeError(f"404 para {cik}")
        return CUERPOS[cik]

    monkeypatch.setattr(edgar, "fetch_json", fetch)
    # Sin espera entre peticiones: la cortesía con la SEC se prueba aparte.
    # Limitador lee SEC_MAX_RPS al construir, así que basta con bajarlo.
    monkeypatch.setattr(edgar, "SEC_MAX_RPS", 0)
    return pedidos


def correr(tmp_path, *extra):
    return guion.main(["--contacto", "Pruebas x@y.com",
                       "--salida", str(tmp_path), *extra])


# ------------------------------------------------------------ corrida base
def test_baja_y_escribe_un_csv_por_nombre(tmp_path, edgar_falso, capsys):
    assert correr(tmp_path, "AAPL", "MSFT") == 0
    assert (tmp_path / "AAPL.csv").exists()
    assert (tmp_path / "MSFT.csv").exists()

    salida = capsys.readouterr().out
    assert "COBERTURA POR MÉTRICA" in salida


def test_el_historico_viene_desde_la_primera_corrida(tmp_path, edgar_falso):
    # Es la diferencia entre EDGAR y los precios: no hay que acumular nada.
    # Una sola llamada trae todos los períodos que la empresa ha reportado.
    correr(tmp_path, "AAPL")
    df = edgar.leer_hechos(tmp_path)
    assert len(df) >= 3, "las dos versiones de ingresos más el saldo"
    assert set(df["fin"]) == {"2024-12-31"}
    assert sorted(df[df["metrica"] == "ingresos"]["filed"]) == \
        ["2025-02-01", "2026-02-01"]


def test_el_point_in_time_sobrevive_al_viaje_por_disco(tmp_path, edgar_falso):
    correr(tmp_path, "AAPL")
    df = edgar.leer_hechos(tmp_path)
    entonces = edgar.as_of(df, "2025-06-30")
    ingresos = entonces[entonces["metrica"] == "ingresos"]["valor"].tolist()
    assert ingresos == [100.0], "en 2025 nadie conocía la corrección de 2026"


def test_se_registra_de_que_etiqueta_salio_cada_numero(tmp_path, edgar_falso):
    # Sin esto, un número raro es un misterio en una hoja de Excel.
    correr(tmp_path, "AAPL", "NVDA")
    filas = list(csv.DictReader((tmp_path / "_etiquetas.csv").open()))
    por_ticker = {(f["ticker"], f["metrica"]): f["etiqueta"] for f in filas}
    assert por_ticker[("AAPL", "ingresos")] == "Revenues"
    assert por_ticker[("NVDA", "ingresos")] == \
        "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_la_cobertura_queda_escrita(tmp_path, edgar_falso):
    correr(tmp_path, "AAPL", "MSFT")
    cob = pd.read_csv(tmp_path / "_cobertura.csv").set_index("metrica")
    assert cob.loc["ingresos", "cobertura"] == 1.0
    assert cob.loc["capex", "cobertura"] == 0.0


def test_los_restatements_se_reportan_en_su_archivo(tmp_path, edgar_falso, capsys):
    correr(tmp_path, "AAPL")
    assert (tmp_path / "_restatements.csv").exists()
    rest = pd.read_csv(tmp_path / "_restatements.csv")
    assert rest["cambio"].tolist() == pytest.approx([0.10])
    assert "Restatements" in capsys.readouterr().out


# ------------------------------------------------------------ incremental
def test_la_segunda_corrida_no_vuelve_a_pedir_lo_que_ya_esta(tmp_path,
                                                             edgar_falso):
    correr(tmp_path, "AAPL")
    antes = len(edgar_falso)
    correr(tmp_path, "AAPL")
    assert len(edgar_falso) == antes, "no debió salir a la red otra vez"


def test_forzar_vuelve_a_bajar(tmp_path, edgar_falso):
    correr(tmp_path, "AAPL")
    antes = len(edgar_falso)
    correr(tmp_path, "AAPL", "--forzar")
    assert len(edgar_falso) > antes


def test_el_mapa_de_tickers_se_pide_una_sola_vez(tmp_path, edgar_falso):
    correr(tmp_path, "AAPL")
    correr(tmp_path, "MSFT")
    assert edgar_falso.count(edgar.SEC_TICKERS) == 1


def test_agregar_un_concepto_avisa_de_los_ceros_falsos(tmp_path, edgar_falso,
                                                       monkeypatch, capsys):
    # El caso literal que pasó: tres nombres bajados con una lista de conceptos
    # anterior, después se agregan dos conceptos, y la corrida siguiente los
    # SALTA por existir el archivo. Su cobertura para las métricas nuevas sale
    # en cero sin ser cero, que es peor que un hueco: el hueco invita a mirar,
    # el cero falso invita a concluir.
    correr(tmp_path, "AAPL")
    capsys.readouterr()

    nuevo = edgar.Concepto("metrica_nueva", ("EtiquetaNueva",))
    monkeypatch.setattr(edgar, "CONCEPTOS", edgar.CONCEPTOS + (nuevo,))

    correr(tmp_path, "AAPL")
    salida = capsys.readouterr().out
    assert "lista de conceptos" in salida
    assert "metrica_nueva" in salida
    assert "--forzar" in salida


def test_sin_conceptos_nuevos_no_hay_aviso(tmp_path, edgar_falso, capsys):
    correr(tmp_path, "AAPL")
    capsys.readouterr()
    correr(tmp_path, "AAPL")
    assert "lista de conceptos" not in capsys.readouterr().out


def test_un_almacen_recien_creado_no_esta_desactualizado(tmp_path):
    assert edgar.conceptos_desactualizados(tmp_path) == []


def test_forzar_actualiza_el_manifiesto(tmp_path, edgar_falso, monkeypatch):
    correr(tmp_path, "AAPL")
    nuevo = edgar.Concepto("metrica_nueva", ("EtiquetaNueva",))
    monkeypatch.setattr(edgar, "CONCEPTOS", edgar.CONCEPTOS + (nuevo,))
    assert "metrica_nueva" in edgar.conceptos_desactualizados(tmp_path)

    correr(tmp_path, "AAPL", "--forzar")
    assert edgar.conceptos_desactualizados(tmp_path) == []


# ------------------------------------------------------------ fallos
def test_un_emisor_sin_cik_no_tumba_la_corrida(tmp_path, edgar_falso, capsys):
    assert correr(tmp_path, "AAPL", "NOEXISTE") == 0
    assert (tmp_path / "AAPL.csv").exists()
    assert "sin CIK" in capsys.readouterr().out


def test_un_404_no_tumba_la_corrida(tmp_path, edgar_falso, monkeypatch, capsys):
    monkeypatch.setitem(guion.__dict__, "DESTINO", str(tmp_path))
    mapa = dict(TICKERS)
    mapa["3"] = {"cik_str": 999999, "ticker": "ROTO"}
    monkeypatch.setattr(edgar, "fetch_json",
                        lambda url, **kw: mapa if url == edgar.SEC_TICKERS
                        else CUERPOS.get(url.rsplit("CIK", 1)[-1].removesuffix(".json"))
                        or (_ for _ in ()).throw(RuntimeError("404")))
    assert correr(tmp_path, "AAPL", "ROTO") == 0
    assert (tmp_path / "AAPL.csv").exists()
    assert not (tmp_path / "ROTO.csv").exists()
    assert "FALLO" in capsys.readouterr().out


def test_un_almacen_vacio_no_revienta_el_reporte(tmp_path, edgar_falso, capsys):
    assert correr(tmp_path, "NOEXISTE") == 0
    assert "No hay nada en el almacén" in capsys.readouterr().out


# ------------------------------------------------------------ universo
def test_el_universo_por_defecto_sale_del_repo():
    nombres = guion.universo("sp500", [])
    assert len(nombres) > 100
    assert all(n == n.upper() for n in nombres)
    # Yahoo usa guion donde la SEC usa punto; el mapa de la SEC trae guion.
    assert not any("." in n for n in nombres)


def test_los_tickers_explicitos_mandan_sobre_el_universo():
    assert guion.universo("sp500", ["aapl", " msft "]) == ["AAPL", "MSFT"]


def test_el_limite_recorta_para_probar(tmp_path, edgar_falso, capsys):
    correr(tmp_path, "--universo", "sp500", "--limite", "2")
    assert "Universo: 2 nombre(s)" in capsys.readouterr().out


def test_la_sec_exige_contacto():
    with pytest.raises(SystemExit):
        guion.main(["AAPL"])          # sin --contacto
