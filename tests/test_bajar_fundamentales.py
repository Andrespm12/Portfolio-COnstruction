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

# La tercera lista oficial, en texto plano. No trae AVB: es el caso real, en el
# que las tres vienen completas y el nombre no esta en ninguna.
TICKER_TXT = "\n".join(f"{f['ticker'].lower()}\t{f['cik_str']}"
                       for f in TICKERS.values())


def facts(cik, ingresos, activos, *, etiqueta_ingresos="Revenues"):
    return {"cik": cik, "entityName": "PRUEBA INC", "facts": {"us-gaap": {
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

    def fetch_txt(url, *, contacto=None, limitador=None):
        pedidos.append(url)
        assert url == edgar.SEC_TICKER_TXT, f"URL inesperada: {url}"
        return TICKER_TXT

    monkeypatch.setattr(edgar, "fetch_json", fetch)
    # Hay DOS líneas que salen a la red desde que ticker.txt no es JSON. Sin
    # sustituir las dos, la prueba marca en verde porque el fallo de red se
    # degrada en silencio, que es exactamente lo que no queremos probar.
    monkeypatch.setattr(edgar, "fetch_text", fetch_txt)
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


def test_se_consultan_las_tres_listas_oficiales(tmp_path, edgar_falso):
    # Un fallo de red en cualquiera de las tres se degrada en silencio — es lo
    # correcto en producción y una trampa en las pruebas, porque una lista que
    # dejara de consultarse marcaría en verde igual.
    correr(tmp_path, "AAPL")
    for url in (edgar.SEC_TICKERS, edgar.SEC_TICKERS_EXCHANGE,
                edgar.SEC_TICKER_TXT):
        assert url in edgar_falso, f"nadie pidió {url}"


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


def test_el_fallo_dice_donde_mirar_en_vez_de_concluir(tmp_path, edgar_falso):
    # Con las listas sanas, faltar no prueba nada: la SEC las publica sin
    # garantizar su alcance. El CSV manda a EDGAR, no a una conclusión.
    correr(tmp_path, "AAPL", "NOEXISTE")
    fallos = pd.read_csv(tmp_path / "_fallos.csv").set_index("ticker")
    detalle = fallos.loc["NOEXISTE", "detalle"]
    assert "NO prueba" in detalle
    assert "cik-lookup" in detalle
    assert "company_tickers=" in detalle


def test_un_cik_a_mano_desbloquea_el_nombre_de_punta_a_punta(tmp_path,
                                                             edgar_falso):
    # El camino completo para los ocho que ninguna lista trae: una línea
    # verificada a mano y el nombre baja como cualquier otro.
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nAVB,915912,verificado en EDGAR\n",
        encoding="utf-8")
    assert correr(tmp_path, "AVB") == 0
    assert (tmp_path / "AVB.csv").exists()
    assert pd.read_csv(tmp_path / "_fallos.csv").empty


def test_queda_registrado_que_nombre_tiene_la_sec_para_cada_cik(tmp_path,
                                                                edgar_falso,
                                                                capsys):
    # Un CIK equivocado no da error: da los estados de otra empresa con nuestro
    # ticker encima. La única defensa es ver el nombre.
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nAVB,915912,verificado\n", encoding="utf-8")
    correr(tmp_path, "AAPL", "AVB")

    emisores = pd.read_csv(tmp_path / "_emisores.csv").set_index("ticker")
    assert emisores.loc["AVB", "cik"] == 915912
    assert emisores.loc["AVB", "fuente"] == "a mano"
    assert emisores.loc["AAPL", "fuente"] == "SEC"
    assert emisores.loc["AVB", "entidad"] == "PRUEBA INC"
    # Y se imprime, porque un CSV que nadie abre no verifica nada.
    salida = capsys.readouterr().out
    assert "verifica que el nombre sea el que esperas" in salida
    assert "PRUEBA INC" in salida


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


def test_se_deja_el_formulario_de_overrides_ya_con_los_nombres(tmp_path,
                                                               edgar_falso,
                                                               capsys):
    # Un archivo que hay que crear desde cero se pospone; uno que ya está
    # escrito con las filas pendientes se llena.
    correr(tmp_path, "AAPL", "NOEXISTE")
    filas = list(csv.DictReader((tmp_path / "_ciks_manuales.csv").open()))
    assert [f["ticker"] for f in filas] == ["NOEXISTE"]
    assert filas[0]["cik"] == "", "el CIK lo pone una persona, no el guion"
    assert "llénalo y vuelve a correr" in capsys.readouterr().out


def test_el_formulario_no_pisa_lo_que_ya_escribiste(tmp_path, edgar_falso):
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nAVB,915912,verificado\n", encoding="utf-8")
    correr(tmp_path, "NOEXISTE")
    assert "915912" in (tmp_path / "_ciks_manuales.csv").read_text()


def test_una_corrida_limpia_no_deja_los_fallos_de_la_anterior(tmp_path,
                                                              edgar_falso,
                                                              capsys):
    # Pasó de verdad: la corrida que resolvió los ocho nombres dejó el
    # _fallos.csv viejo en disco, con el diagnóstico anterior, al lado de un
    # _cobertura.csv al 100%. Dos archivos contradiciéndose y ninguna forma de
    # saber cuál era el de hoy.
    correr(tmp_path, "NOEXISTE")
    assert pd.read_csv(tmp_path / "_fallos.csv").shape[0] == 1

    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nNOEXISTE,320193,resuelto a mano\n",
        encoding="utf-8")
    capsys.readouterr()
    correr(tmp_path, "NOEXISTE")

    fallos = pd.read_csv(tmp_path / "_fallos.csv")
    assert fallos.empty, "el fallo de ayer no puede sobrevivir al arreglo"
    assert list(fallos.columns) == ["ticker", "motivo", "detalle"]
    assert "no falló ninguno" in capsys.readouterr().out


def test_sin_intentar_nada_no_se_toca_el_archivo_de_fallos(tmp_path,
                                                           edgar_falso):
    # Una corrida que se salta todo no sabe nada de nadie: borrar el archivo
    # ahí sería tirar el diagnóstico sin haberlo repetido.
    correr(tmp_path, "NOEXISTE")
    correr(tmp_path, "AAPL")
    (tmp_path / "AAPL.csv").touch()
    antes = (tmp_path / "_fallos.csv").read_text()
    correr(tmp_path, "AAPL")
    assert (tmp_path / "_fallos.csv").read_text() == antes


def test_sin_fallos_no_se_escribe_el_archivo(tmp_path, edgar_falso):
    correr(tmp_path, "AAPL")
    assert pd.read_csv(tmp_path / "_fallos.csv").empty


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
