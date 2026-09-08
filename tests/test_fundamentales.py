"""
Pruebas de screener/fundamentales.py.

Lo que se verifica es lo que decide si un ratio significa algo: que el signo
ordene bien a través del cero, que el fundamental y el precio sean del mismo
día, que un ejercicio no se mezcle con otro, y que una cohorte de doce nombres
no entre al modelo como si fuera una de trescientos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener import fundamentales as fx  # noqa: E402
from screener.edgar import COLUMNAS  # noqa: E402


def hecho(ticker, metrica, valor, fin, filed, inicio=None, etiqueta="X",
          forma="10-K", accn="a"):
    return {"ticker": ticker, "cik": "0000000001", "metrica": metrica,
            "etiqueta": etiqueta, "valor": float(valor), "unidad": "USD",
            "inicio": inicio or "", "fin": fin, "filed": filed, "forma": forma,
            "fy": 2024, "fp": "FY", "accn": accn}


def emisor(ticker, *, anio=2024, ingresos=1000.0, utilidad=100.0, eps=1.0,
           activos=2000.0, patrimonio=800.0, ebit=150.0, depreciacion=50.0,
           flujo=180.0, capex=40.0, efectivo=100.0, deuda_corto=50.0,
           deuda_largo=350.0, acciones=100.0, pasivos_corrientes=300.0,
           dividendos=-30.0, filed=None):
    """Un ejercicio completo, con todas las etiquetas que el modelo usa."""
    fin = f"{anio}-12-31"
    ini = f"{anio}-01-01"
    filed = filed or f"{anio + 1}-02-15"
    flujos = {"ingresos": ingresos, "utilidad_neta": utilidad,
              "eps_diluido": eps, "ebit": ebit, "depreciacion": depreciacion,
              "flujo_operativo": flujo, "capex": capex,
              "dividendos_pagados": dividendos}
    saldos = {"activos": activos, "patrimonio": patrimonio,
              "efectivo": efectivo, "deuda_corto_plazo": deuda_corto,
              "deuda_largo_plazo": deuda_largo,
              "acciones_en_circulacion": acciones,
              "pasivos_corrientes": pasivos_corrientes}
    filas = [hecho(ticker, m, v, fin, filed, inicio=ini)
             for m, v in flujos.items() if v is not None]
    filas += [hecho(ticker, m, v, fin, filed)
              for m, v in saldos.items() if v is not None]
    return filas


def marco(*grupos):
    return pd.DataFrame([f for g in grupos for f in g],
                        columns=list(COLUMNAS))


HOY = "2025-06-30"


# ------------------------------------------------------- rendimiento vs múltiplo
def test_el_que_pierde_dinero_queda_ultimo_y_no_primero():
    # La trampa del P/E, que es la razón entera de usar rendimientos. Con
    # múltiplos, PERD cotiza a -100x y en un ranking de "P/E bajo es mejor"
    # gana. Con rendimientos queda donde debe: debajo de todos.
    hechos = marco(emisor("CARA", eps=1.0), emisor("BARATA", eps=1.0),
                   emisor("PERD", eps=-1.0, utilidad=-100.0))
    tabla = fx.ratios(hechos, {"CARA": 100.0, "BARATA": 10.0, "PERD": 100.0},
                      HOY)

    ey = tabla["earnings_yield"]
    assert ey["BARATA"] > ey["CARA"] > ey["PERD"]
    assert ey["PERD"] < 0, "perder dinero es un rendimiento negativo"


def test_el_multiplo_de_una_perdida_sale_vacio_y_no_barato():
    hechos = marco(emisor("PERD", eps=-1.0, utilidad=-100.0))
    tabla = fx.ratios(hechos, {"PERD": 100.0}, HOY)
    assert not np.isfinite(fx.multiplos_legibles(tabla)["pe"]["PERD"])


def test_los_multiplos_legibles_son_el_inverso_exacto():
    hechos = marco(emisor("A", eps=5.0))
    tabla = fx.ratios(hechos, {"A": 100.0}, HOY)
    assert tabla["earnings_yield"]["A"] == pytest.approx(0.05)
    assert fx.multiplos_legibles(tabla)["pe"]["A"] == pytest.approx(20.0)


# ------------------------------------------------------------- point-in-time
def test_no_se_ve_lo_que_se_presento_despues():
    # La razón de ser del almacén. En marzo de 2025 nadie conocía la corrección
    # de 2026, y un ratio que la use está mirando el futuro.
    base = emisor("A", eps=1.0, utilidad=100.0)
    correccion = [hecho("A", "eps_diluido", 2.0, "2024-12-31", "2026-02-01",
                        inicio="2024-01-01", accn="b")]
    hechos = marco(base, correccion)

    entonces = fx.ratios(hechos, {"A": 100.0}, "2025-03-01")
    ahora = fx.ratios(hechos, {"A": 100.0}, "2026-06-30")
    assert entonces["earnings_yield"]["A"] == pytest.approx(0.01)
    assert ahora["earnings_yield"]["A"] == pytest.approx(0.02)


def test_el_precio_es_el_argumento_no_una_constante():
    # El otro lado del point-in-time: el mismo fundamental contra dos precios
    # da dos ratios, y eso es lo que permite reconstruir una fecha pasada.
    hechos = marco(emisor("A", eps=1.0))
    a = fx.ratios(hechos, {"A": 50.0}, HOY)["earnings_yield"]["A"]
    b = fx.ratios(hechos, {"A": 100.0}, HOY)["earnings_yield"]["A"]
    assert a == pytest.approx(2 * b)


def test_un_ejercicio_viejo_no_describe_a_la_empresa_de_hoy():
    hechos = marco(emisor("VIEJA", anio=2019))
    assert len(fx.ratios(hechos, {"VIEJA": 10.0}, HOY)) == 0
    # Sin tope sí aparece: la decisión es del tope, no del dato.
    suelto = fx.ratios(hechos, {"VIEJA": 10.0}, HOY, max_antiguedad_dias=None)
    assert np.isfinite(suelto["earnings_yield"]["VIEJA"])


# ------------------------------------------------- coherencia de período
def test_no_se_arma_un_ebitda_con_dos_anos_distintos():
    # El error que sale de un groupby().last() inocente: EBIT de 2024 con
    # depreciación de 2023 no es el EBITDA de ningún año.
    completo = emisor("A", anio=2023, ebit=100.0, depreciacion=40.0)
    parcial = [f for f in emisor("A", anio=2024, ebit=200.0)
               if f["metrica"] != "depreciacion"]
    tabla = fx.ratios(marco(completo, parcial), {"A": 100.0}, HOY)

    assert tabla["periodo"]["A"] == "2024-12-31"
    assert not np.isfinite(tabla["ebitda_ev"]["A"]), \
        "sin depreciación de 2024 no hay EBITDA de 2024"


def test_el_periodo_de_referencia_es_el_mas_reciente():
    hechos = marco(emisor("A", anio=2023, ingresos=900.0),
                   emisor("A", anio=2024, ingresos=1000.0))
    tabla = fx.ratios(hechos, {"A": 100.0}, HOY)
    assert tabla["periodo"]["A"] == "2024-12-31"
    assert tabla["crecimiento_ingresos"]["A"] == pytest.approx(1000 / 900 - 1)


def test_un_saldo_de_la_portada_entra_por_la_tolerancia():
    # EntityCommonStockSharesOutstanding lleva la fecha de la presentación, dos
    # meses después del cierre. Con una tolerancia estrecha se perdería justo el
    # conteo de acciones, que es lo que hace falta para una capitalización.
    filas = [f for f in emisor("A") if f["metrica"] != "acciones_en_circulacion"]
    filas.append(hecho("A", "acciones_en_circulacion", 100.0, "2025-02-10",
                       "2025-02-15"))
    tabla = fx.ratios(marco(filas), {"A": 100.0}, HOY)
    assert tabla["acciones_efectivas"]["A"] == pytest.approx(100.0)


# ------------------------------------------------------------- derivadas
def test_el_pasivo_se_deriva_por_identidad_cuando_no_lo_reportan():
    # Liabilities es un subtotal opcional: lo reportó el 70% del universo.
    # Activos menos patrimonio ES el pasivo, no una aproximación suya.
    tabla = fx.derivar(fx.panel(marco(emisor("A", activos=2000.0,
                                             patrimonio=800.0)), HOY))
    assert tabla["pasivos_der"]["A"] == pytest.approx(1200.0)
    assert tabla["pasivos_no_corrientes_der"]["A"] == pytest.approx(900.0)


def test_lo_reportado_manda_sobre_lo_derivado():
    filas = emisor("A", activos=2000.0, patrimonio=800.0)
    filas.append(hecho("A", "pasivos", 1150.0, "2024-12-31", "2025-02-15"))
    tabla = fx.derivar(fx.panel(marco(filas), HOY))
    assert tabla["pasivos_der"]["A"] == pytest.approx(1150.0)


def test_la_deuda_total_no_se_arma_con_una_sola_pata():
    # Sumar solo la larga y llamarla deuda total subestima el apalancamiento
    # justo en los emisores que más se financian a corto.
    filas = [f for f in emisor("A") if f["metrica"] != "deuda_corto_plazo"]
    tabla = fx.derivar(fx.panel(marco(filas), HOY))
    assert not np.isfinite(tabla["deuda_total"]["A"])


def test_la_deuda_neta_puede_ser_negativa():
    tabla = fx.derivar(fx.panel(marco(emisor("A", efectivo=1000.0,
                                             deuda_corto=10.0,
                                             deuda_largo=20.0)), HOY))
    assert tabla["deuda_neta"]["A"] == pytest.approx(-970.0)


# --------------------------------------------------------------- acciones
def test_el_conteo_sale_del_eps_y_no_de_una_sola_clase():
    # CommonStockSharesOutstanding es POR CLASE. En una empresa con acciones A
    # y B trae una sola y la capitalización sale a la mitad.
    hechos = marco(emisor("DOS", utilidad=200.0, eps=1.0, acciones=100.0))
    tabla = fx.ratios(hechos, {"DOS": 10.0}, HOY)
    assert tabla["acciones_efectivas"]["DOS"] == pytest.approx(200.0)
    assert tabla["acciones_fuente"]["DOS"] == "eps"
    assert tabla["capitalizacion"]["DOS"] == pytest.approx(2000.0)
    assert tabla["acciones_discrepancia"]["DOS"] == pytest.approx(1.0)


def test_sin_eps_queda_el_conteo_reportado():
    filas = [f for f in emisor("A", acciones=100.0)
             if f["metrica"] != "eps_diluido"]
    tabla = fx.ratios(marco(filas), {"A": 10.0}, HOY)
    assert tabla["acciones_efectivas"]["A"] == pytest.approx(100.0)
    assert tabla["acciones_fuente"]["A"] == "reportadas"


def test_un_eps_de_cero_no_produce_una_capitalizacion_infinita():
    hechos = marco(emisor("A", eps=0.0, acciones=100.0))
    tabla = fx.ratios(hechos, {"A": 10.0}, HOY)
    assert tabla["acciones_efectivas"]["A"] == pytest.approx(100.0)


# ------------------------------------------------------- guardas de signo
def test_sin_patrimonio_positivo_no_hay_roe():
    # Utilidad negativa sobre patrimonio negativo da un ROE positivo por doble
    # signo, y leería como excelente.
    hechos = marco(emisor("Q", utilidad=-50.0, eps=-0.5, patrimonio=-200.0))
    tabla = fx.ratios(hechos, {"Q": 10.0}, HOY)
    assert not np.isfinite(tabla["roe"]["Q"])
    # Y el book_yield sí se calcula, negativo: como rendimiento ordena bien.
    assert tabla["book_yield"]["Q"] < 0


def test_un_ebitda_negativo_da_rendimiento_negativo_no_vacio():
    # Aquí sí se calcula: EBITDA negativo sobre EV positivo ordena bien, que es
    # justo lo contrario del EV/EBITDA.
    hechos = marco(emisor("A", ebit=-200.0, depreciacion=50.0))
    tabla = fx.ratios(hechos, {"A": 10.0}, HOY)
    assert tabla["ebitda_ev"]["A"] < 0


def test_el_payout_toma_el_valor_absoluto_del_flujo():
    # Los dividendos salen del estado de flujo de caja, donde una salida va con
    # signo negativo. Sin el abs, el payout sale negativo en todo el universo.
    hechos = marco(emisor("A", dividendos=-30.0, utilidad=100.0))
    assert fx.ratios(hechos, {"A": 10.0}, HOY)["payout"]["A"] == \
        pytest.approx(0.30)


def test_sin_precio_no_hay_ratios_de_valuacion():
    hechos = marco(emisor("A"))
    tabla = fx.ratios(hechos, {}, HOY)
    assert not np.isfinite(tabla["earnings_yield"]["A"])
    # Pero los de calidad no dependen del precio y sí se calculan.
    assert np.isfinite(tabla["margen_neto"]["A"])


# ----------------------------------------------------------------- cohorte
def test_un_ratio_que_cubre_a_pocos_no_entra_al_modelo():
    # scoring.py renormaliza el bloque sobre las métricas presentes, así que un
    # ratio medido sobre 4 emisores de 40 entra al promedio con el mismo peso
    # que uno medido sobre 40: el z de una anécdota discutiendo de igual a igual
    # con el z de una distribución.
    completos = [emisor(f"T{i:03d}") for i in range(40)]
    # A 36 de ellos les falta el EPS, así que earnings_yield cubre 4 de 40.
    recortados = [[f for f in g if f["metrica"] != "eps_diluido"]
                  for g in completos[4:]]
    hechos = marco(*completos[:4], *recortados)
    precios = {f"T{i:03d}": 100.0 for i in range(40)}
    tabla = fx.ratios(hechos, precios, HOY)

    emitidas = {m for v in fx.metricas_para_scoring(tabla).values() for m in v}
    assert "earnings_yield" not in emitidas, "4 de 40 no es una distribución"
    assert "sales_yield" in emitidas, "ese sí lo tienen todos"


def test_el_umbral_es_relativo_y_no_una_constante():
    # Un mínimo fijo de 30 nombres funcionaba en el universo grande y rompía el
    # chico: en una lista de 12, ningún ratio llegaría nunca y el bloque
    # fundamental no existiría por una constante pensada para otro tamaño.
    assert fx.cohorte_minima(300) == 150
    assert fx.cohorte_minima(12) == 6
    assert fx.cohorte_minima(4) == 3, "el piso lo pone el z-score, no la fracción"

    hechos = marco(*[emisor(f"T{i:03d}") for i in range(12)])
    precios = {f"T{i:03d}": 100.0 for i in range(12)}
    tabla = fx.ratios(hechos, precios, HOY)
    assert set(fx.metricas_para_scoring(tabla)["T000"]) == \
        set(fx.RATIOS_PUNTUADOS)


def test_la_cobertura_se_mide_contra_el_universo_pedido():
    hechos = marco(emisor("A"), emisor("B"))
    tabla = fx.ratios(hechos, {"A": 10.0, "B": 10.0}, HOY)
    cob = fx.cohortes(tabla, ["A", "B", "ETF"]).set_index("ratio")
    assert cob.loc["earnings_yield", "cobertura"] == pytest.approx(2 / 3)
    assert cob.loc["earnings_yield", "sin_dato"] == 1
    assert set(cob["familia"]) == {"valuacion", "calidad"}


def test_la_calidad_se_calcula_pero_todavia_no_puntua():
    # Ponerla a puntuar exige decidir su peso, y un peso es una decisión del
    # Comité, no un detalle de implementación.
    hechos = marco(*[emisor(f"T{i:03d}") for i in range(40)])
    precios = {f"T{i:03d}": 100.0 for i in range(40)}
    tabla = fx.ratios(hechos, precios, HOY)

    emitidas = set(fx.metricas_para_scoring(tabla)["T000"])
    assert "roe" not in emitidas and np.isfinite(tabla["roe"]["T000"])
    assert "earnings_yield" in emitidas


# ------------------------------------------------------------- enganche
def payload(*tickers, precio=100.0):
    return {"instruments": [
        {"ticker": t, "asset_type": "STOCK",
         "snapshot": {"last": {"price": precio}}} for t in tickers]}


def test_adjuntar_usa_el_precio_del_mismo_payload():
    # Es la garantía estructural de la regla del módulo: fundamental y precio
    # salen del mismo sitio, así que no hay forma de casar el balance de un año
    # con la cotización de otro por descuido.
    tickers = [f"T{i:03d}" for i in range(35)]
    hechos = marco(*[emisor(t, eps=5.0) for t in tickers])
    md = fx.adjuntar(payload(*tickers, precio=100.0), hechos, HOY)

    inst = md["instruments"][0]
    assert inst["fundamentals"]["metricas"]["earnings_yield"] == \
        pytest.approx(0.05)
    assert inst["fundamentals"]["periodo"] == "2024-12-31"
    assert md["fundamentals_meta"]["con_ratios"] == 35


def test_un_etf_no_tiene_estados_financieros_y_eso_no_es_un_hueco():
    tickers = [f"T{i:03d}" for i in range(35)]
    hechos = marco(*[emisor(t) for t in tickers])
    md = fx.adjuntar(payload(*tickers, "SPY"), hechos, HOY)

    spy = next(i for i in md["instruments"] if i["ticker"] == "SPY")
    assert "fundamentals" not in spy
    assert fx.metricas_del_instrumento(spy) == {}


def test_sin_hechos_el_modelo_corre_igual_que_antes():
    md = fx.adjuntar(payload("A", "B"), marco(), HOY)
    assert all("fundamentals" not in i for i in md["instruments"])
    assert md["fundamentals_meta"]["con_ratios"] == 0


def test_el_meta_dice_sobre_que_se_calculo():
    tickers = [f"T{i:03d}" for i in range(35)]
    hechos = marco(*[emisor(t) for t in tickers])
    meta = fx.adjuntar(payload(*tickers), hechos, HOY)["fundamentals_meta"]

    assert meta["as_of"] == HOY
    assert set(meta["puntuados"]) == set(fx.RATIOS_PUNTUADOS)
    cob = {f["ratio"]: f for f in meta["cobertura"]}
    assert cob["earnings_yield"]["puntuable"] is True
    assert cob["roe"]["puntuable"] is False


# ------------------------------------------------- el modelo, de punta a punta
def test_los_ratios_llegan_al_score_y_mueven_el_bloque():
    # La prueba que importa: que el número calculado aquí termine dentro del
    # z-score del bloque de valuación, no en una columna decorativa.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from screener.run_screen import run
    from screener.yahoo_adapter import build_market_data
    from test_yahoo_adapter import make_yf_frame

    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "LLY", "AMZN",
               "META", "MU", "GLD", "TLT"]
    md = build_market_data(make_yf_frame(tickers), tickers, benchmark="SPY")
    acciones = [i["ticker"] for i in md["instruments"]
                if i.get("asset_type") == "STOCK"]
    # Un EPS distinto por nombre: si todos fueran iguales el z-score saldría
    # degenerado y la prueba pasaría sin medir nada.
    hechos = marco(*[emisor(t, eps=float(n + 1), utilidad=100.0 * (n + 1))
                     for n, t in enumerate(acciones)])

    antes, _ = run(md, {}, standalone=True, target_position_usd=250_000.0)
    cobertura_antes = {r.ticker: r.block_coverage.get("valuation_carry", 0.0)
                       for r in antes}

    fx.adjuntar(md, hechos, HOY)
    despues, _ = run(md, {}, standalone=True, target_position_usd=250_000.0)

    con_fund = [r for r in despues if "earnings_yield" in r.raw_metrics]
    assert len(con_fund) >= 3, "ningún nombre recibió el bloque fundamental"
    for fila in con_fund:
        assert np.isfinite(fila.metric_z["earnings_yield"])
        assert fila.block_coverage["valuation_carry"] > \
            cobertura_antes[fila.ticker], "el bloque no se enteró"

    etf = next(r for r in despues if r.ticker == "SPY")
    assert "earnings_yield" not in etf.raw_metrics
    assert etf.block_coverage.get("valuation_carry", 0.0) > 0, \
        "el ETF conserva su bloque con las métricas de mercado"


# ------------------------------------------------- la hoja de cobertura
def test_la_cobertura_no_declara_100_por_ciento_sin_mirar():
    # El verde falso que llegó al comité: la hoja Cobertura de una corrida real
    # declaró 100% en las cinco métricas de EDGAR, incluidos los ETF, que no
    # tienen estados financieros. Se calculaba tratando "no está en la lista de
    # snapshot" como "sale de precios, luego está siempre".
    from screener.yahoo_adapter import coverage_report

    tickers = [f"T{i:03d}" for i in range(35)]
    md = payload(*tickers, "SPY")
    fx.adjuntar(md, marco(*[emisor(t) for t in tickers]), HOY)

    cob = coverage_report(md).set_index("metric")
    ey = cob.loc["Earnings yield (EPS/price)"]
    assert ey["coverage"] == pytest.approx(35 / 36), \
        "el ETF no tiene fundamentales y no puede contar como cubierto"
    assert "EDGAR" in ey["source"], ey["source"]
    assert cob.loc["12M-1M total return", "coverage"] == 1.0


def test_sin_almacen_la_cobertura_lo_dice_en_vez_de_mentir():
    from screener.yahoo_adapter import coverage_report

    cob = coverage_report(payload("A", "B", "C")).set_index("metric")
    fila = cob.loc["Free cash flow yield"]
    assert fila["coverage"] == 0.0
    assert "UNAVAILABLE" in fila["source"]


def test_toda_metrica_declara_de_donde_sale():
    # La defensa estructural: una métrica nueva sin fuente declarada volvería a
    # contarse como "sale de precios, luego está siempre".
    import screener.config as config

    fuentes = {m.source for b in config.FACTOR_MODEL for m in b.metrics}
    assert fuentes <= {"price", "snapshot", "edgar"}, fuentes
    edgar = {m.key for b in config.FACTOR_MODEL for m in b.metrics
             if m.source == "edgar"}
    assert edgar == set(fx.RATIOS_PUNTUADOS), \
        "lo que puntúa desde EDGAR y lo que se declara como EDGAR se separaron"
