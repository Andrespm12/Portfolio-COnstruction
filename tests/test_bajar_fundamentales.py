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
# La lista real trae más de diez mil emisores y el guion descarta un mapa
# demasiado corto, así que el falso también tiene que ser plausible: con tres
# nombres nunca se cachearía y la prueba del incremental mediría otra cosa.
TICKERS |= {str(i): {"cik_str": 500000 + i, "ticker": f"T{i:05d}"}
            for i in range(3, edgar.MIN_EMISORES + 3)}

# La segunda lista oficial, con su forma propia. Trae un nombre que a la
# primera le falta: es el caso de AVB, BK, EQR, IPG y MMC.
TICKERS_EXCHANGE = {
    "fields": ["cik", "name", "ticker", "exchange"],
    "data": [[fila["cik_str"], "X", fila["ticker"], "NYSE"]
             for fila in TICKERS.values()]
            + [[915912, "AvalonBay Communities", "AVB", "NYSE"]]}


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
    "0000915912": facts(915912, 400.0, 4000.0),
}


@pytest.fixture
def edgar_falso(monkeypatch):
    """Sustituye la red y cuenta las peticiones."""
    pedidos: list[str] = []

    def fetch(url, *, contacto=None, limitador=None):
        pedidos.append(url)
        if url == edgar.SEC_TICKERS:
            return TICKERS
        if url == edgar.SEC_TICKERS_EXCHANGE:
            return TICKERS_EXCHANGE
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


def test_remapear_rehace_el_mapa_sin_rebajar_los_fundamentales(tmp_path,
                                                               edgar_falso):
    # Rebajar 280 companyfacts para arreglar el mapa serían gigabytes por dos
    # peticiones de trabajo.
    correr(tmp_path, "AAPL")
    antes = [u for u in edgar_falso if "companyfacts" in u]

    edgar_falso.clear()
    correr(tmp_path, "AAPL", "--remapear")
    assert edgar.SEC_TICKERS in edgar_falso
    assert [u for u in edgar_falso if "companyfacts" in u] == [], \
        "AAPL.csv ya estaba: --remapear no debe tocar los fundamentales"
    assert antes, "la primera corrida sí bajó AAPL"


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


def test_los_fallos_quedan_registrados_con_su_motivo(tmp_path, edgar_falso):
    # Ocho nombres fallaron dos corridas seguidas en una descarga real y el
    # motivo solo estaba en la consola. "sin CIK" y "404" llevan a sitios
    # distintos, así que la diferencia tiene que sobrevivir a cerrar la pestaña.
    correr(tmp_path, "AAPL", "NOEXISTE")
    fallos = pd.read_csv(tmp_path / "_fallos.csv").set_index("ticker")
    assert fallos.loc["NOEXISTE", "motivo"] == "sin CIK"
    assert "company_tickers" in fallos.loc["NOEXISTE", "detalle"]


def test_un_nombre_que_solo_esta_en_la_segunda_lista_se_baja(tmp_path,
                                                             edgar_falso):
    # AVB es el caso literal: registrante vigente ausente de la primera lista.
    # Con una sola fuente falla "sin CIK" en cada corrida, para siempre.
    assert correr(tmp_path, "AVB") == 0
    assert (tmp_path / "AVB.csv").exists()


def test_el_fallo_distingue_emisor_retirado_de_mapa_incompleto(tmp_path,
                                                               edgar_falso):
    # Las dos causas llevan a sitios opuestos: una se arregla en el universo,
    # la otra rebajando el mapa. El CSV tiene que decir cuál es.
    correr(tmp_path, "AAPL", "NOEXISTE")
    fallos = pd.read_csv(tmp_path / "_fallos.csv").set_index("ticker")
    detalle = fallos.loc["NOEXISTE", "detalle"]
    assert "ya no cotice" in detalle
    assert "revisa el universo" in detalle
    assert str(edgar.MIN_EMISORES) in detalle or "company_tickers=" in detalle


def test_con_el_mapa_corto_el_fallo_no_culpa_al_emisor(tmp_path, monkeypatch,
                                                       capsys):
    # Si el mapa vino a medias, "sin CIK" no prueba nada sobre el emisor.
    monkeypatch.setattr(edgar, "SEC_MAX_RPS", 0)
    monkeypatch.setattr(edgar, "fetch_json",
                        lambda url, **kw: TICKERS_EXCHANGE
                        if url == edgar.SEC_TICKERS_EXCHANGE
                        else {"0": {"cik_str": 320193, "ticker": "AAPL"}}
                        if url == edgar.SEC_TICKERS
                        else CUERPOS[url.rsplit("CIK", 1)[-1].removesuffix(".json")])
    monkeypatch.setattr(edgar, "MIN_EMISORES", 10 ** 9)
    monkeypatch.setattr(guion, "MIN_EMISORES", 10 ** 9)

    correr(tmp_path, "AAPL", "NOEXISTE")
    fallos = pd.read_csv(tmp_path / "_fallos.csv").set_index("ticker")
    detalle = fallos.loc["NOEXISTE", "detalle"]
    assert "incompleto" in detalle
    assert "ya no cotice" not in detalle
    assert "El mapa está incompleto" in capsys.readouterr().out


def test_sin_fallos_no_se_escribe_el_archivo(tmp_path, edgar_falso):
    correr(tmp_path, "AAPL")
    assert not (tmp_path / "_fallos.csv").exists()


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
