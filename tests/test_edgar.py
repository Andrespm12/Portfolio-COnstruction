"""
Pruebas de screener/edgar.py.

Ninguna toca la red: se sustituye ``fetch``, que es exactamente la línea que
llama a la SEC. Lo que se verifica es lo que decide si el número significa algo
— el orden de preferencia de etiquetas, que los restatements sobrevivan, y que
``as_of`` devuelva lo que se sabía ese día y no lo corregido después.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.edgar import (CONCEPTOS, CONCEPTO_POR_CLAVE, MIN_EMISORES,
                            SEC_TICKER_TXT, SEC_TICKERS,
                            SEC_TICKERS_EXCHANGE, Limitador, as_of,
                            company_facts, coverage_report, detalle_sin_cik,
                            escribir_hechos, extract_facts, fuentes_del_mapa,
                            fuentes_path, historia_por_ticker, leer_hechos,
                            leer_overrides, load_ticker_map, parse_ticker_map,
                            parse_ticker_map_exchange, parse_ticker_txt,
                            restatements, ultimo_anual, user_agent)


def hecho(val, end, filed, start=None, form="10-K", accn="0000-1", fy=2024,
          fp="FY"):
    d = {"val": val, "end": end, "filed": filed, "form": form, "accn": accn,
         "fy": fy, "fp": fp}
    if start:
        d["start"] = start
    return d


def companyfacts(cik=320193, **conceptos):
    """Un companyfacts con la forma real: taxonomía -> etiqueta -> units."""
    return {"cik": cik, "entityName": "PRUEBA INC",
            "facts": {"us-gaap": conceptos}}


def serie(unidad="USD", *hechos):
    return {"units": {unidad: list(hechos)}}


# ------------------------------------------------------------ User-Agent
def test_la_sec_exige_un_correo():
    # No es burocracia: sin User-Agent identificable bloquean la IP.
    with pytest.raises(ValueError, match="correo"):
        user_agent("CCI Puesto de Bolsa")
    assert user_agent(" CCI x@y.com ") == "CCI x@y.com"


# ------------------------------------------------------------ ticker->CIK
def test_el_cik_se_rellena_a_diez_digitos():
    # Sin los ceros a la izquierda companyfacts responde 404 y no dice por qué.
    mapa = parse_ticker_map({"0": {"cik_str": 320193, "ticker": "aapl"},
                             "1": {"cik_str": 789019, "ticker": "MSFT"}})
    assert mapa == {"AAPL": "0000320193", "MSFT": "0000789019"}


def test_una_fila_rota_no_tumba_el_mapa():
    mapa = parse_ticker_map({"0": {"cik_str": "x", "ticker": "MAL"},
                             "1": {"ticker": "SIN_CIK"},
                             "2": {"cik_str": 320193, "ticker": "AAPL"}})
    assert mapa == {"AAPL": "0000320193"}


def test_la_segunda_lista_tiene_otra_forma():
    # {"fields": [...], "data": [[...]]} en vez de un objeto por fila.
    mapa = parse_ticker_map_exchange({
        "fields": ["cik", "name", "ticker", "exchange"],
        "data": [[320193, "Apple Inc.", "aapl", "Nasdaq"],
                 [1045810, "NVIDIA", "NVDA", "Nasdaq"]]})
    assert mapa == {"AAPL": "0000320193", "NVDA": "0001045810"}


def test_la_segunda_lista_con_otras_columnas_no_revienta():
    assert parse_ticker_map_exchange({"fields": ["a", "b"], "data": [[1, 2]]}) == {}
    assert parse_ticker_map_exchange({}) == {}


def lista_grande(n=MIN_EMISORES, desde=1000):
    """Una primera lista de tamaño plausible, para que el caché se acepte."""
    return {str(i): {"cik_str": desde + i, "ticker": f"T{i:05d}"}
            for i in range(n)}


def lista_grande_exchange(n=MIN_EMISORES, desde=1000):
    return {"fields": ["cik", "name", "ticker", "exchange"],
            "data": [[desde + i, f"E{i}", f"T{i:05d}", "NYSE"] for i in range(n)]}


def lista_grande_txt(n=MIN_EMISORES, desde=1000):
    return "\n".join(f"t{i:05d}\t{desde + i}" for i in range(n))


def responder(primera, segunda=None, registro=None, texto=None):
    """Sustituye las dos líneas que salen a la red: la JSON y la de texto."""
    def falso(url, **kw):
        if registro is not None:
            registro.append(url)
        if url == SEC_TICKERS:
            return primera
        if url == SEC_TICKERS_EXCHANGE:
            if segunda is None:
                raise RuntimeError("404")
            return segunda
        raise AssertionError(f"URL inesperada: {url}")

    def falso_texto(url, **kw):
        if registro is not None:
            registro.append(url)
        if url == SEC_TICKER_TXT:
            if texto is None:
                raise RuntimeError("404")
            return texto
        raise AssertionError(f"URL inesperada: {url}")

    return {"fetch": falso, "fetch_texto": falso_texto}


def test_el_mapa_se_cachea_y_no_se_vuelve_a_pedir(tmp_path):
    llamadas: list[str] = []
    kw = dict(contacto="x@y.com", cache=tmp_path / "_tickers.json",
              **responder(lista_grande(), lista_grande_exchange(), llamadas,
                          lista_grande_txt()))
    a = load_ticker_map(**kw)
    b = load_ticker_map(**kw)
    assert a == b and len(a) == MIN_EMISORES
    assert llamadas == [SEC_TICKERS, SEC_TICKERS_EXCHANGE, SEC_TICKER_TXT], \
        "la segunda vez tiene que salir del disco"


def test_un_cache_sin_procedencia_se_rehace_una_vez(tmp_path):
    # El caché que ya está en la máquina de quien usa esto lo escribió el código
    # de una sola fuente: tiene tamaño de sobra y le faltan los nombres. Sin
    # esto, el arreglo no llegaría nunca a quien lo necesita.
    cache = tmp_path / "_tickers.json"
    viejo = {f"T{i:05d}": f"{i:010d}" for i in range(MIN_EMISORES)}
    cache.write_text(json.dumps(viejo), encoding="utf-8")
    assert not fuentes_del_mapa(cache)

    llamadas: list[str] = []
    segunda = lista_grande_exchange()
    segunda["data"].append([915912, "AvalonBay", "AVB", "NYSE"])
    kw = dict(contacto="x@y.com", cache=cache,
              **responder(lista_grande(), segunda, llamadas))

    assert "AVB" in load_ticker_map(**kw), "el caché viejo no debió sobrevivir"
    llamadas.clear()
    assert "AVB" in load_ticker_map(**kw)
    assert llamadas == [], "una vez, no en cada corrida"


def test_refrescar_vuelve_a_pedir_el_mapa(tmp_path):
    cache = tmp_path / "_tickers.json"
    kw = dict(contacto="x@y.com", cache=cache,
              **responder(lista_grande(), lista_grande_exchange()))
    load_ticker_map(**kw)

    llamadas: list[str] = []
    load_ticker_map(contacto="x@y.com", cache=cache, refrescar=True,
                    **responder(lista_grande(), lista_grande_exchange(),
                                    llamadas))
    assert llamadas == [SEC_TICKERS, SEC_TICKERS_EXCHANGE, SEC_TICKER_TXT]


def test_un_cache_a_medias_se_descarta_en_vez_de_creerse(tmp_path):
    # El error que no se ve: un mapa incompleto es JSON válido, se devuelve
    # igual en cada corrida, y los emisores que le faltan salen "sin CIK" para
    # siempre. Sin piso, la corrida número cien falla idéntico a la primera.
    cache = tmp_path / "_tickers.json"
    cache.write_text(json.dumps({"AAPL": "0000320193"}), encoding="utf-8")

    mapa = load_ticker_map(contacto="x@y.com", cache=cache,
                           **responder(lista_grande(),
                                           lista_grande_exchange()))
    assert len(mapa) == MIN_EMISORES, "el caché corto no debió sobrevivir"
    assert len(json.loads(cache.read_text())) == MIN_EMISORES


def test_un_cache_ilegible_no_tumba_la_corrida(tmp_path):
    cache = tmp_path / "_tickers.json"
    cache.write_text("{esto no es json", encoding="utf-8")
    mapa = load_ticker_map(contacto="x@y.com", cache=cache,
                           **responder(lista_grande(),
                                           lista_grande_exchange()))
    assert len(mapa) == MIN_EMISORES


def test_un_mapa_corto_no_se_escribe_al_cache(tmp_path):
    # Si la SEC responde a medias, guardarlo convierte un fallo de una corrida
    # en un fallo permanente.
    cache = tmp_path / "_tickers.json"
    load_ticker_map(contacto="x@y.com", cache=cache,
                    **responder({"0": {"cik_str": 320193, "ticker": "AAPL"}},
                                    None))
    assert not cache.exists()
    # Pero la procedencia sí: el mapa corto es justo el caso que hay que poder
    # explicar después.
    assert fuentes_del_mapa(cache)["total"] == 1


def test_la_segunda_lista_tapa_el_hueco_de_la_primera(tmp_path):
    # AVB, BK, EQR, IPG y MMC salieron "sin CIK" tres corridas seguidas siendo
    # registrantes vigentes. Depender de un solo archivo hace que su hueco sea
    # un hueco del modelo.
    primera = lista_grande()
    segunda = lista_grande_exchange()
    segunda["data"].append([915912, "AvalonBay Communities", "AVB", "NYSE"])

    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(primera, segunda))
    assert mapa["AVB"] == "0000915912"


def test_la_segunda_lista_no_pisa_a_la_primera(tmp_path):
    # Ante un desacuerdo manda company_tickers.json, que es la lista canónica.
    primera = lista_grande()
    primera["0"] = {"cik_str": 320193, "ticker": "AAPL"}
    segunda = lista_grande_exchange()
    segunda["data"].append([999999, "Impostora", "AAPL", "NYSE"])

    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(primera, segunda))
    assert mapa["AAPL"] == "0000320193"


def test_si_la_segunda_lista_falla_la_primera_alcanza(tmp_path):
    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(lista_grande(), None))
    assert len(mapa) == MIN_EMISORES


# --------------------------------------------- procedencia del mapa
def test_queda_escrito_cuanto_trajo_cada_lista(tmp_path):
    # Sin esto, "sin CIK" no se puede leer: no se sabe si falta el emisor o
    # falta el mapa. Y como el caché evita la red, tiene que estar en disco.
    cache = tmp_path / "_tickers.json"
    load_ticker_map(contacto="x@y.com", cache=cache,
                    **responder(lista_grande(),
                                lista_grande_exchange(n=MIN_EMISORES - 3),
                                texto=lista_grande_txt()))
    fuentes = fuentes_del_mapa(cache)
    assert fuentes["company_tickers"] == MIN_EMISORES
    assert fuentes["company_tickers_exchange"] == MIN_EMISORES - 3
    assert fuentes["total"] == MIN_EMISORES
    assert fuentes["ticker_txt"] == MIN_EMISORES
    assert fuentes["errores"] == {}


def test_la_procedencia_registra_el_fallo_de_la_segunda_lista(tmp_path):
    cache = tmp_path / "_tickers.json"
    load_ticker_map(contacto="x@y.com", cache=cache,
                    **responder(lista_grande(), None))
    errores = fuentes_del_mapa(cache)["errores"]
    assert "404" in errores["company_tickers_exchange"]
    assert "404" in errores["ticker_txt"]


def test_la_procedencia_sobrevive_a_la_corrida_cacheada(tmp_path):
    cache = tmp_path / "_tickers.json"
    load_ticker_map(contacto="x@y.com", cache=cache,
                    **responder(lista_grande(), lista_grande_exchange()))
    # Segunda corrida: sale del disco, no se cuenta nada, y aun así se sabe.
    load_ticker_map(contacto="x@y.com", cache=cache,
                    **responder({}, None))
    assert fuentes_del_mapa(cache)["company_tickers"] == MIN_EMISORES


# --------------------------------------------- tercera lista y overrides
def test_ticker_txt_se_lee_aunque_no_sea_json():
    assert parse_ticker_txt("aapl\t320193\nmsft\t789019\n") == \
        {"AAPL": "0000320193", "MSFT": "0000789019"}


def test_una_linea_rota_de_ticker_txt_no_tumba_el_archivo():
    assert parse_ticker_txt("basura\naapl\t320193\n\nx\ty\n") == \
        {"AAPL": "0000320193"}


def test_la_tercera_lista_tapa_lo_que_las_dos_primeras_no(tmp_path):
    # Las dos JSON vinieron completas (10.412 y 10.415) y aun así faltaban ocho
    # miembros del S&P 500. ticker.txt se genera aparte.
    texto = lista_grande_txt() + "\navb\t915912\n"
    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(lista_grande(), lista_grande_exchange(),
                                       texto=texto))
    assert mapa["AVB"] == "0000915912"


def test_si_la_tercera_lista_falla_las_otras_alcanzan(tmp_path):
    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(lista_grande(), lista_grande_exchange()))
    assert len(mapa) == MIN_EMISORES


def test_un_cik_a_mano_rescata_lo_que_ninguna_lista_trae(tmp_path):
    # La salida cuando las tres listas fallan: una línea que una persona
    # verificó en EDGAR. Es la única forma de no quedar atado a un archivo que
    # la SEC publica sin garantizar su alcance.
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nAVB,915912,verificado en EDGAR 2026-09\n",
        encoding="utf-8")
    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(lista_grande(), lista_grande_exchange(),
                                       texto=lista_grande_txt()))
    assert mapa["AVB"] == "0000915912"


def test_el_cik_a_mano_llega_tambien_con_el_mapa_cacheado(tmp_path):
    # Si solo entrara al bajar el mapa, agregar una línea al CSV no haría nada
    # hasta la próxima descarga y quien lo edita concluiría que no sirve.
    cache = tmp_path / "_tickers.json"
    kw = dict(contacto="x@y.com", cache=cache,
              **responder(lista_grande(), lista_grande_exchange(),
                          texto=lista_grande_txt()))
    load_ticker_map(**kw)
    assert "AVB" not in load_ticker_map(**kw)

    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\navb,915912,verificado a mano\n", encoding="utf-8")
    assert load_ticker_map(**kw)["AVB"] == "0000915912"


def test_un_cik_a_mano_no_contradice_a_la_sec_en_silencio(tmp_path):
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nT00000,999999,me equivoqué\n", encoding="utf-8")
    mapa = load_ticker_map(contacto="x@y.com", cache=tmp_path / "_tickers.json",
                           **responder(lista_grande(), lista_grande_exchange(),
                                       texto=lista_grande_txt()))
    assert mapa["T00000"] == "0000001000", "manda la lista oficial"


def test_los_overrides_no_se_guardan_en_el_cache_de_la_sec(tmp_path):
    # El caché es lo que dijo la SEC. Mezclarlos haría imposible saber después
    # de dónde salió un CIK.
    cache = tmp_path / "_tickers.json"
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\nAVB,915912,x\n", encoding="utf-8")
    load_ticker_map(contacto="x@y.com", cache=cache,
                    **responder(lista_grande(), lista_grande_exchange(),
                                texto=lista_grande_txt()))
    assert "AVB" not in json.loads(cache.read_text())


def test_un_csv_de_overrides_roto_no_tumba_la_corrida(tmp_path):
    (tmp_path / "_ciks_manuales.csv").write_text(
        "ticker,cik,por_que\n,,\nAVB,no-es-un-numero,x\nEQR,11170,y\n",
        encoding="utf-8")
    assert leer_overrides(tmp_path) == {"EQR": "0000011170"}


def test_sin_archivo_de_overrides_no_pasa_nada(tmp_path):
    assert leer_overrides(tmp_path) == {}
    assert leer_overrides(None) == {}


# --------------------------------------------- el diagnóstico no concluye
def test_el_diagnostico_no_declara_muerto_a_un_emisor_vivo():
    # Lo que pasó: las tres listas sanas, y AvalonBay, BNY Mellon, Equity
    # Residential, Interpublic y Marsh McLennan afuera. La SEC publica estos
    # archivos sin garantizar su alcance, así que faltar no prueba nada.
    texto = detalle_sin_cik({"company_tickers": 10412,
                             "company_tickers_exchange": 10415,
                             "ticker_txt": 10402})
    assert "NO prueba" in texto
    assert "cik-lookup" in texto, "tiene que decir dónde mirar"
    assert "_ciks_manuales.csv" in texto, "y qué hacer con la respuesta"
    assert "ya no cotice" not in texto


def test_el_diagnostico_si_acusa_al_mapa_cuando_vino_corto():
    texto = detalle_sin_cik({"company_tickers": 12,
                             "company_tickers_exchange": 0, "ticker_txt": 0})
    assert "incompleto" in texto


def test_sin_procedencia_no_se_inventa_nada(tmp_path):
    assert fuentes_del_mapa(tmp_path / "_tickers.json") == {}
    assert fuentes_del_mapa(None) == {}
    fuentes_path(tmp_path / "_tickers.json").write_text("roto")
    assert fuentes_del_mapa(tmp_path / "_tickers.json") == {}


# ------------------------------------------------------------ extracción
def test_se_extrae_lo_declarado_y_nada_mas():
    payload = companyfacts(
        Assets=serie("USD", hecho(1000, "2024-12-31", "2025-02-01")),
        # Etiqueta que no está en CONCEPTOS: no nos interesa.
        GoodwillImpairmentLoss=serie("USD", hecho(5, "2024-12-31", "2025-02-01")))
    hechos, elegidas = extract_facts(payload, "AAPL")

    assert {h.metrica for h in hechos} == {"activos"}
    assert elegidas == {"activos": "Assets"}


def test_gana_la_etiqueta_de_mayor_prioridad():
    # Un emisor puede reportar las dos. El orden de CONCEPTOS decide, y el
    # resultado dice cuál se usó para poder rastrearlo después.
    payload = companyfacts(
        Revenues=serie("USD", hecho(50, "2024-12-31", "2025-02-01",
                                    start="2024-01-01")),
        RevenueFromContractWithCustomerExcludingAssessedTax=serie(
            "USD", hecho(100, "2024-12-31", "2025-02-01", start="2024-01-01")))
    hechos, elegidas = extract_facts(payload, "AAPL")

    assert elegidas["ingresos"] == \
        "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert [h.valor for h in hechos] == [100.0]


def test_se_usa_la_alternativa_cuando_falta_la_principal():
    payload = companyfacts(
        SalesRevenueNet=serie("USD", hecho(70, "2020-12-31", "2021-02-01",
                                           start="2020-01-01")))
    _, elegidas = extract_facts(payload, "OLD")
    assert elegidas["ingresos"] == "SalesRevenueNet"


def test_una_moneda_distinta_no_entra():
    # Mezclar CAD con USD en un z-score transversal es comparar monedas, no
    # empresas. Se descarta y la métrica queda sin cobertura.
    payload = companyfacts(Assets=serie("CAD", hecho(1000, "2024-12-31",
                                                     "2025-02-01")))
    hechos, elegidas = extract_facts(payload, "CAN")
    assert hechos == [] and elegidas == {}


def test_solo_entran_formularios_periodicos():
    payload = companyfacts(Assets=serie(
        "USD",
        hecho(1000, "2024-12-31", "2025-02-01", form="10-K"),
        hecho(9999, "2024-12-31", "2025-01-15", form="8-K")))
    hechos, _ = extract_facts(payload, "AAPL")
    assert [h.valor for h in hechos] == [1000.0]


def test_un_valor_no_numerico_se_descarta_sin_romper():
    payload = companyfacts(Assets=serie(
        "USD",
        hecho("n/a", "2023-12-31", "2024-02-01"),
        hecho(1000, "2024-12-31", "2025-02-01")))
    hechos, _ = extract_facts(payload, "AAPL")
    assert [h.valor for h in hechos] == [1000.0]


def test_el_eps_va_en_su_propia_unidad():
    payload = companyfacts(EarningsPerShareDiluted=serie(
        "USD/shares", hecho(6.11, "2024-12-31", "2025-02-01",
                            start="2024-01-01")))
    hechos, _ = extract_facts(payload, "AAPL")
    assert hechos[0].metrica == "eps_diluido"
    assert hechos[0].unidad == "USD/shares"


# ------------------------------------------------- restatements y point-in-time
RESTATED = companyfacts(Assets=serie(
    "USD",
    # Lo que se reportó el 2024-02-01 para el cierre de 2023.
    hecho(1000, "2023-12-31", "2024-02-01", accn="0000-1"),
    # La corrección presentada un año después, con el comparativo del 10-K.
    hecho(1200, "2023-12-31", "2025-02-01", accn="0000-2"),
    hecho(1500, "2024-12-31", "2025-02-01", accn="0000-2")))


def test_las_dos_versiones_sobreviven_a_la_extraccion():
    # Quedarse solo con la última convertiría a EDGAR en otro proveedor de
    # datos restatados, que es justamente lo que no queremos.
    hechos, _ = extract_facts(RESTATED, "AAPL")
    de_2023 = sorted(h.valor for h in hechos if h.fin == "2023-12-31")
    assert de_2023 == [1000.0, 1200.0]


def _frame(payload, ticker="AAPL"):
    hechos, _ = extract_facts(payload, ticker)
    return pd.DataFrame([h.fila() for h in hechos],
                        columns=list(__import__("screener.edgar",
                                                fromlist=["COLUMNAS"]).COLUMNAS))


def test_as_of_devuelve_lo_que_se_sabia_ese_dia():
    df = _frame(RESTATED)
    entonces = as_of(df, "2024-06-30")
    assert list(entonces["valor"]) == [1000.0], \
        "en junio de 2024 nadie conocía la corrección de 2025"


def test_as_of_sin_fecha_devuelve_lo_ultimo_conocido():
    df = _frame(RESTATED)
    hoy = as_of(df).set_index("fin")["valor"].to_dict()
    assert hoy == {"2023-12-31": 1200.0, "2024-12-31": 1500.0}


def test_as_of_antes_del_primer_filing_no_devuelve_nada():
    assert as_of(_frame(RESTATED), "2020-01-01").empty


def test_as_of_no_rompe_con_un_almacen_vacio():
    vacio = pd.DataFrame(columns=["ticker", "metrica", "fin", "inicio",
                                  "filed", "accn", "valor"])
    assert as_of(vacio, "2024-01-01").empty


def test_los_restatements_se_reportan():
    tabla = restatements(_frame(RESTATED))
    assert len(tabla) == 1
    fila = tabla.iloc[0]
    assert fila["fin"] == "2023-12-31"
    assert fila["primero"] == 1000.0 and fila["ultimo"] == 1200.0
    assert fila["cambio"] == pytest.approx(0.20)


def test_un_cambio_pequeno_no_se_reporta_como_restatement():
    payload = companyfacts(Assets=serie(
        "USD",
        hecho(1000, "2023-12-31", "2024-02-01", accn="a"),
        hecho(1001, "2023-12-31", "2025-02-01", accn="b")))
    assert restatements(_frame(payload)).empty


# ------------------------------------------------------- anual: flujo vs saldo
ANUAL = companyfacts(
    Assets=serie("USD",
                 hecho(900, "2023-12-31", "2024-02-01"),
                 hecho(1000, "2024-12-31", "2025-02-01")),
    Revenues=serie("USD",
                   # Un trimestre: no es el período anual.
                   hecho(25, "2024-03-31", "2024-04-20", start="2024-01-01",
                         form="10-Q", fp="Q1"),
                   # El año completo.
                   hecho(100, "2024-12-31", "2025-02-01", start="2024-01-01")))


def test_un_saldo_toma_el_corte_mas_reciente():
    fila = ultimo_anual(_frame(ANUAL)).set_index("metrica").loc["activos"]
    assert fila["valor"] == 1000.0 and fila["fin"] == "2024-12-31"


def test_un_flujo_toma_el_ano_y_no_el_trimestre():
    # Confundirlos es la forma más rápida de calcular un múltiplo cuatro veces
    # más grande de lo que es.
    fila = ultimo_anual(_frame(ANUAL)).set_index("metrica").loc["ingresos"]
    assert fila["valor"] == 100.0


def test_sin_periodo_anual_la_metrica_no_aparece():
    solo_trimestres = companyfacts(Revenues=serie(
        "USD", hecho(25, "2024-03-31", "2024-04-20", start="2024-01-01",
                     form="10-Q")))
    assert ultimo_anual(_frame(solo_trimestres)).empty


def test_el_anual_tambien_respeta_la_fecha_de_corte():
    fila = ultimo_anual(_frame(ANUAL), "2024-06-30")
    assert list(fila[fila["metrica"] == "activos"]["valor"]) == [900.0]


# ------------------------------------------------------------ almacén
def test_lo_escrito_se_vuelve_a_leer_igual(tmp_path):
    hechos, _ = extract_facts(RESTATED, "AAPL")
    escribir_hechos(tmp_path, "AAPL", hechos)

    df = leer_hechos(tmp_path)
    assert len(df) == len(hechos)
    assert set(df["ticker"]) == {"AAPL"}
    assert as_of(df, "2024-06-30")["valor"].tolist() == [1000.0]


def test_los_auxiliares_no_se_leen_como_un_emisor(tmp_path):
    hechos, _ = extract_facts(ANUAL, "AAPL")
    escribir_hechos(tmp_path, "AAPL", hechos)
    (tmp_path / "_tickers.json").write_text('{"AAPL": "0000320193"}')
    (tmp_path / "_cobertura.csv").write_text("metrica,cobertura\nactivos,1.0\n")

    df = leer_hechos(tmp_path)
    assert set(df["ticker"]) == {"AAPL"}


def test_un_almacen_inexistente_no_es_un_error(tmp_path):
    assert leer_hechos(tmp_path / "no-existe").empty


def test_se_puede_leer_solo_algunos_tickers(tmp_path):
    for t in ("AAPL", "MSFT"):
        escribir_hechos(tmp_path, t, extract_facts(ANUAL, t)[0])
    assert set(leer_hechos(tmp_path, ["AAPL"])["ticker"]) == {"AAPL"}


# ------------------------------------------------------------ cobertura
def test_la_cobertura_cuenta_sobre_el_universo_pedido(tmp_path):
    # Dos nombres con datos, uno sin bajar: la cobertura es 2/3, no 2/2.
    for t in ("AAPL", "MSFT"):
        escribir_hechos(tmp_path, t, extract_facts(ANUAL, t)[0])

    cob = coverage_report(leer_hechos(tmp_path), ["AAPL", "MSFT", "NVDA"])
    activos = cob.set_index("metrica").loc["activos"]
    assert activos["cobertura"] == pytest.approx(2 / 3)
    assert activos["con_dato"] == 2 and activos["sin_dato"] == 1
    assert "NVDA" in activos["faltan"]


def test_la_cobertura_lista_todas_las_metricas_aunque_esten_en_cero(tmp_path):
    # Una métrica ausente tiene que salir en cero, no desaparecer del reporte.
    # Desaparecer se lee como "no aplica"; cero se lee como "no lo tenemos".
    escribir_hechos(tmp_path, "AAPL", extract_facts(ANUAL, "AAPL")[0])
    cob = coverage_report(leer_hechos(tmp_path), ["AAPL"])
    assert set(cob["metrica"]) == {c.clave for c in CONCEPTOS}
    assert cob.set_index("metrica").loc["capex", "cobertura"] == 0.0


def test_la_cobertura_dice_que_etiqueta_se_uso(tmp_path):
    escribir_hechos(tmp_path, "AAPL", extract_facts(ANUAL, "AAPL")[0])
    cob = coverage_report(leer_hechos(tmp_path), ["AAPL"])
    assert cob.set_index("metrica").loc["activos", "etiqueta_principal"] == "Assets"


def test_la_etiqueta_principal_es_la_mas_usada_no_la_primera_del_alfabeto(tmp_path):
    # El defecto real: la columna reportaba `Depreciation` — el ULTIMO recurso
    # de la lista de candidatos — solo porque ordena antes alfabeticamente que
    # `DepreciationDepletionAndAmortization`, que era la que traia los numeros.
    # Una columna de trazabilidad que apunta a la etiqueta equivocada invita a
    # buscar el problema donde no esta.
    def con(etiqueta):
        return companyfacts(Assets=serie("USD", hecho(1, "2024-12-31",
                                                      "2025-02-01")),
                            **{etiqueta: serie("USD", hecho(
                                7, "2024-12-31", "2025-02-01",
                                start="2024-01-01"))})

    # Dos emisores con la etiqueta preferida, uno con el ultimo recurso.
    for t in ("AAA", "BBB"):
        escribir_hechos(tmp_path, t, extract_facts(
            con("DepreciationDepletionAndAmortization"), t)[0])
    escribir_hechos(tmp_path, "CCC",
                    extract_facts(con("Depreciation"), "CCC")[0])

    cob = coverage_report(leer_hechos(tmp_path),
                          ["AAA", "BBB", "CCC"]).set_index("metrica")
    fila = cob.loc["depreciacion"]
    assert fila["etiqueta_principal"] == "DepreciationDepletionAndAmortization"
    assert fila["etiquetas_usadas"] == 2
    assert "DepreciationDepletionAndAmortization×2" in fila["etiquetas_detalle"]
    assert "Depreciation×1" in fila["etiquetas_detalle"]


def test_un_empate_se_rompe_por_el_orden_declarado(tmp_path):
    # Con un emisor cada una manda la prioridad de CONCEPTOS, que es la
    # preferencia real, y no otra vez el alfabeto.
    def con(etiqueta):
        return companyfacts(**{etiqueta: serie("USD", hecho(
            7, "2024-12-31", "2025-02-01", start="2024-01-01"))})

    escribir_hechos(tmp_path, "AAA",
                    extract_facts(con("Depreciation"), "AAA")[0])
    escribir_hechos(tmp_path, "BBB", extract_facts(
        con("DepreciationDepletionAndAmortization"), "BBB")[0])

    cob = coverage_report(leer_hechos(tmp_path), ["AAA", "BBB"]).set_index("metrica")
    assert cob.loc["depreciacion", "etiqueta_principal"] == \
        "DepreciationDepletionAndAmortization"


def test_la_cobertura_de_un_almacen_vacio_es_cero():
    cob = coverage_report(pd.DataFrame(columns=["ticker", "metrica", "fin",
                                                "inicio", "filed", "accn",
                                                "valor", "etiqueta"]),
                          ["AAPL"])
    assert (cob["cobertura"] == 0.0).all()


def test_la_historia_por_nombre_cuenta_versiones_y_periodos(tmp_path):
    escribir_hechos(tmp_path, "AAPL", extract_facts(RESTATED, "AAPL")[0])
    hist = historia_por_ticker(leer_hechos(tmp_path)).set_index("ticker")
    assert hist.loc["AAPL", "versiones"] == 3   # dos de 2023 + una de 2024
    assert hist.loc["AAPL", "periodos"] == 2    # 2023 y 2024


# ------------------------------------------------------------ cortesía
def test_el_limitador_espacia_las_peticiones():
    import time
    lim = Limitador(rps=50.0)
    t0 = time.monotonic()
    for _ in range(5):
        lim.esperar()
    assert time.monotonic() - t0 >= 0.06, "5 peticiones a 50/s son >= 80ms"


def test_se_puede_apagar_el_limitador_en_pruebas():
    lim = Limitador(rps=0)
    lim.esperar()          # no debe bloquear


def test_company_facts_arma_la_url_con_el_cik_relleno():
    vistas = []

    def falso(url, **kw):
        vistas.append(url)
        return {}

    company_facts("0000320193", contacto="x@y.com", fetch=falso)
    assert vistas == [
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"]


# ------------------------------------------------------------ el mapeo
def test_cada_concepto_declara_al_menos_una_etiqueta():
    assert all(c.etiquetas for c in CONCEPTOS)


def test_ninguna_etiqueta_se_reparte_entre_dos_metricas():
    # Una etiqueta en dos métricas haría que el mismo número entre dos veces al
    # compuesto con nombres distintos.
    vistas: dict[str, str] = {}
    for c in CONCEPTOS:
        for e in c.etiquetas:
            assert e not in vistas, f"{e} está en {vistas[e]} y en {c.clave}"
            vistas[e] = c.clave


def test_los_componentes_de_pasivos_no_son_sinonimos_del_total():
    # ABBV no reporta `Liabilities`: es opcional en US GAAP. La tentación es
    # meter LiabilitiesCurrent como candidato del total para tapar el hueco,
    # y eso etiquetaría el pasivo corriente como pasivo total en todo emisor
    # que reporte ambos — subestimando el apalancamiento en silencio.
    total = set(CONCEPTO_POR_CLAVE["pasivos"].etiquetas)
    for clave in ("pasivos_corrientes", "pasivos_no_corrientes"):
        assert not total & set(CONCEPTO_POR_CLAVE[clave].etiquetas)
        assert CONCEPTO_POR_CLAVE[clave].instantaneo


def test_un_emisor_sin_liabilities_igual_trae_sus_componentes():
    payload = companyfacts(
        LiabilitiesCurrent=serie("USD", hecho(300, "2024-12-31", "2025-02-01")),
        LiabilitiesNoncurrent=serie("USD", hecho(700, "2024-12-31",
                                                 "2025-02-01")))
    hechos, elegidas = extract_facts(payload, "ABBV")
    assert "pasivos" not in elegidas
    assert elegidas["pasivos_corrientes"] == "LiabilitiesCurrent"
    assert elegidas["pasivos_no_corrientes"] == "LiabilitiesNoncurrent"
    assert sorted(h.valor for h in hechos) == [300.0, 700.0]


def test_ninguna_partida_residual_hace_de_subtotal():
    # El error que una corrida real destapó: `OtherLiabilitiesNoncurrent` como
    # respaldo de `LiabilitiesNoncurrent`. Gana en 173 de 221 emisores porque
    # casi nadie presenta el subtotal, así que la mayoría habría quedado con el
    # renglón «otros» etiquetado como su pasivo no corriente completo.
    #
    # La regla general: una etiqueta que empieza con `Other` es una partida
    # residual dentro de un subtotal, nunca el subtotal.
    for c in CONCEPTOS:
        residuales = [e for e in c.etiquetas if e.startswith("Other")]
        assert not residuales, (
            f"{c.clave} usa {residuales} como si fuera un total; "
            "una partida «otros» mide una fracción, no el agregado")


def test_los_saldos_estan_marcados_como_instantaneos():
    # Si un saldo se marcara como flujo, ultimo_anual le exigiría un período de
    # un año y lo descartaría entero.
    for clave in ("activos", "pasivos", "patrimonio", "efectivo"):
        assert CONCEPTO_POR_CLAVE[clave].instantaneo, clave
    for clave in ("ingresos", "utilidad_neta", "flujo_operativo", "capex"):
        assert not CONCEPTO_POR_CLAVE[clave].instantaneo, clave
