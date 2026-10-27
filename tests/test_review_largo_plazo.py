"""
Lo que pidió el review de las carteras de largo plazo, más allá de los arreglos
bloqueantes: estrés histórico, sensibilidad a los parámetros sin calibrar,
reproducibilidad, tasa libre de riesgo con fecha, una sola lista de pendientes
del Comité, vigencia de las views y rotación contra la cartera actual.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.black_litterman import (  # noqa: E402
    DEFAULT_PARAMS, HORIZONTE_VIEWS_MESES, aplicar_vigencia,
)
from screener.cci_regulation import (  # noqa: E402
    DRAWDOWN_TOLERADO, MODELO_ASIGNACION, REGULACIONES, SATELITE_MAX,
    SECTOR_CAPS,
)
from screener.comite import pendientes  # noqa: E402
from screener.corrida import guardar_corrida, huella_motor, leer_corrida  # noqa: E402
from screener.estres import (  # noqa: E402
    ESCENARIOS, PROXY_POR_CLASE, beta_del_sleeve, estresar, pesos_modelo,
    proxies, retornos_escenarios, tabla_estres,
)
from screener.optimizer import (  # noqa: E402
    RISK_AVERSION, implied_equilibrium, leer_cartera_actual, optimize,
    policy_weights, posterior, shrunk_covariance, split_core_satellite,
)
from screener.sensibilidad import escalar_ic, rotacion, tabla_sensibilidad  # noqa: E402
from screener.yahoo_adapter import fetch_risk_free  # noqa: E402


# ------------------------------------------------------------------ estrés
def _precios_de_ventanas(retornos: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Precios que rinden exactamente ``retornos[escenario][ticker]`` en cada ventana."""
    idx = pd.bdate_range("2007-09-03", "2022-12-30")
    tickers = sorted({t for r in retornos.values() for t in r})
    precios = pd.DataFrame(100.0, index=idx, columns=tickers)
    for escenario, por_ticker in retornos.items():
        inicio, fin, _ = ESCENARIOS[escenario]
        for t, r in por_ticker.items():
            precios.loc[pd.Timestamp(fin):, t] *= 1.0 + r
    return precios


VENTANAS = {
    "Crisis 2008": {"SPY": -0.55, "IEF": 0.15, "LQD": -0.05, "AGG": 0.04,
                    "HYG": -0.30, "BIL": 0.02, "GLD": 0.20},
    "COVID 2020": {"SPY": -0.34, "IEF": 0.06, "LQD": -0.13, "AGG": -0.02,
                   "HYG": -0.20, "BIL": 0.00, "GLD": -0.03},
    "Inflación 2022": {"SPY": -0.25, "IEF": -0.15, "LQD": -0.22, "AGG": -0.16,
                       "HYG": -0.15, "BIL": 0.01, "GLD": -0.08},
}


def test_los_proxies_cubren_todas_las_clases_y_las_tres_ventanas():
    assert {"SPY", "IEF", "LQD", "BIL"} <= set(proxies())
    assert set(ESCENARIOS) == {"Crisis 2008", "COVID 2020", "Inflación 2022"}
    for estrategia in MODELO_ASIGNACION:
        assert set(pesos_modelo(estrategia)) <= set(PROXY_POR_CLASE)
        assert sum(pesos_modelo(estrategia).values()) == pytest.approx(1.0)


def test_el_retorno_de_cada_ventana_es_el_de_punta_a_punta():
    ret, notas = retornos_escenarios(_precios_de_ventanas(VENTANAS))
    assert not notas
    for escenario, por_ticker in VENTANAS.items():
        for t, r in por_ticker.items():
            assert ret.at[escenario, t] == pytest.approx(r, abs=1e-9)


def test_una_ventana_sin_datos_no_se_inventa():
    precios = _precios_de_ventanas(VENTANAS)
    precios.loc[:"2010-01-01", "HYG"] = np.nan        # HYG "nació" en 2010
    ret, notas = retornos_escenarios(precios)
    assert np.isnan(ret.at["Crisis 2008", "HYG"])
    assert any("HYG" in n and "2008" in n for n in notas)
    res, _ = estresar({"RentaFija_NoIG": 0.1, "Equity": 0.9}, ret)
    assert np.isnan(res["Crisis 2008"]), "sin dato no se rellena con cero"
    assert np.isfinite(res["COVID 2020"])


def test_la_cartera_se_estresa_por_clase_con_la_beta_del_sleeve():
    ret, _ = retornos_escenarios(_precios_de_ventanas(VENTANAS))
    pesos = {"Equity": 0.5, "RentaFija_Soberana_IG": 0.5}
    res, _ = estresar(pesos, ret, beta_rv=1.4)
    assert res["Crisis 2008"] == pytest.approx(0.5 * 1.4 * -0.55 + 0.5 * 0.15)
    assert res["Inflación 2022"] == pytest.approx(0.5 * 1.4 * -0.25 + 0.5 * -0.15)


def test_2022_es_el_que_mas_le_duele_al_conservador_frente_a_2008():
    # La razón por la que el review pidió estrés: en 2008 los bonos del
    # Tesoro compensaron; en 2022 cayeron con las acciones. Con estos
    # retornos, el Modelo Defensivo pierde menos en 2008 que el Agresivo por
    # mucho, pero en 2022 la distancia se acorta porque su renta fija no
    # protege.
    ret, _ = retornos_escenarios(_precios_de_ventanas(VENTANAS))
    cd, _ = estresar(pesos_modelo("Conservador_Defensivo"), ret)
    ag, _ = estresar(pesos_modelo("Agresivo"), ret)
    brecha_2008 = cd["Crisis 2008"] - ag["Crisis 2008"]
    brecha_2022 = cd["Inflación 2022"] - ag["Inflación 2022"]
    assert brecha_2022 < brecha_2008


def test_la_tabla_marca_lo_que_pasa_el_drawdown_tolerado():
    ret, _ = retornos_escenarios(_precios_de_ventanas(VENTANAS))
    carteras = {f"Modelo {e}": (e, pesos_modelo(e), 1.0) for e in REGULACIONES}
    carteras["Todo acciones"] = ("Conservador_Defensivo", {"Equity": 1.0}, 1.0)
    tabla, notas = tabla_estres(carteras, ret)
    fila = tabla.set_index("cartera").loc["Todo acciones"]
    assert fila["peor"] == pytest.approx(-0.55)
    assert fila["excede_tolerancia"]
    assert fila["dd_tolerado"] == DRAWDOWN_TOLERADO["Conservador_Defensivo"]
    assert any("Todo acciones" in n for n in notas)


def test_la_beta_del_sleeve_se_mide_y_si_no_se_puede_se_dice():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, 400)
    ret = pd.DataFrame({"SPY": m, "AAA": 1.5 * m + rng.normal(0, 0.002, 400),
                        "AGG": rng.normal(0, 0.002, 400)})
    clases = {"SPY": "ETF_RentaVariable", "AAA": "Equity", "AGG": "ETF_RentaFija"}
    beta, nota = beta_del_sleeve({"AAA": 0.4, "AGG": 0.6}, clases, ret)
    assert beta == pytest.approx(1.5, abs=0.05)
    beta, nota = beta_del_sleeve({"AAA": 0.4}, clases, ret.drop(columns="SPY"))
    assert beta == 1.0 and "supuesta" in nota


# ------------------------------------------------------------- sensibilidad
@pytest.fixture(scope="module")
def mercado():
    rng = np.random.default_rng(21)
    n = 600
    f_eq = rng.normal(0.0005, 0.011, n)
    f_bond = rng.normal(0.0001, 0.004, n)
    datos = {
        "AAA": 1.3 * f_eq + rng.normal(0, 0.015, n),
        "BBB": 1.1 * f_eq + rng.normal(0, 0.013, n),
        "SPY": 1.0 * f_eq + rng.normal(0, 0.001, n),
        "EFA": 0.9 * f_eq + rng.normal(0, 0.004, n),
        "GOVT": 1.5 * f_bond + rng.normal(0, 0.0008, n),
        "LQD": 1.4 * f_bond + rng.normal(0, 0.0009, n),
        "BIL": 0.02 * f_bond + 0.00005,
    }
    ret = pd.DataFrame(datos)
    tipos = {t: ("STOCK" if t in ("AAA", "BBB") else "ETF") for t in datos}
    caps = {t: 1e11 for t in datos}
    return ret, shrunk_covariance(ret), tipos, caps


def _base(mercado, views, estrategia="Moderado", **kw):
    _ret, cov, tipos, caps = mercado
    nucleo, satelite, _ = split_core_satellite(tipos)
    ancla, _ = policy_weights(tipos, estrategia, caps=caps, core=nucleo)
    pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
    er, cov_post = posterior(pi, cov, views)
    alloc = optimize(er, cov_post, tipos, estrategia, views=views,
                     anchor=ancla, prior=pi, min_position=None,
                     satellite=satelite, **kw)
    return pi, cov, tipos, ancla, satelite, alloc


VIEWS = [{"tipo": "absoluto", "activo": "AAA", "Q": 0.05, "_q_bruto": 0.09,
          "conviccion": 0.7},
         {"tipo": "relativo", "activo_long": "BBB", "activo_short": "SPY",
          "Q": 0.02, "_q_bruto": 0.02, "conviccion": 0.5}]


def test_escalar_el_ic_parte_del_valor_previo_al_recorte():
    medio = escalar_ic(VIEWS, 0.5)
    # 0.09 * 0.5 = 0.045: la view recortada en 5% baja menos de lo que bajaría
    # escalando la Q ya recortada (2.5%).
    assert medio[0]["Q"] == pytest.approx(0.045)
    assert escalar_ic(VIEWS, 1.5)[0]["Q"] == pytest.approx(DEFAULT_PARAMS.max_abs_q)
    assert escalar_ic(VIEWS, 1.5)[1]["Q"] == pytest.approx(0.03)
    neg = escalar_ic([{"tipo": "absoluto", "activo": "X", "Q": -0.02}], 2.0)
    assert neg[0]["Q"] == pytest.approx(-0.04)


def test_la_rotacion_es_la_mitad_de_la_suma_de_diferencias():
    a = pd.Series({"X": 0.6, "Y": 0.4})
    b = pd.Series({"X": 0.5, "Z": 0.5})
    assert rotacion(a, b) == pytest.approx(0.5)
    assert rotacion(a, a) == 0.0


def test_la_tabla_mueve_un_parametro_a_la_vez(mercado):
    pi, cov, tipos, ancla, satelite, base = _base(mercado, VIEWS)
    tabla, _ = tabla_sensibilidad(pi, cov, VIEWS, tipos, "Moderado", base=base,
                                  anchor=ancla, min_position=None,
                                  satellite=satelite)
    assert list(tabla["parametro"]) == ["base", "IC", "IC", "τ", "τ", "λ", "λ"]
    assert tabla.loc[0, "rotacion_vs_base"] == 0.0
    assert (tabla["rotacion_vs_base"].iloc[1:] >= 0).all()
    # Más IC o menos λ dejan a las views mover más: más satélite.
    fila = tabla.set_index(["parametro", "valor"])["satelite"]
    assert fila[("IC", "x1.5")] >= fila[("IC", "x0.5")] - 1e-6


def test_una_variante_que_rota_mucho_se_avisa(mercado):
    pi, cov, tipos, ancla, satelite, base = _base(mercado, VIEWS)
    _, notas = tabla_sensibilidad(pi, cov, VIEWS, tipos, "Moderado", base=base,
                                  anchor=ancla, min_position=None,
                                  satellite=satelite, factores_lambda=(0.01,))
    assert any("λ" in n and "rota" in n for n in notas), notas


# ---------------------------------------------------------- rotación / actual
def test_la_cartera_actual_se_lee_de_un_csv_y_se_normaliza():
    tmp = Path(tempfile.mkdtemp()) / "actual.csv"
    tmp.write_text("Ticker,Peso\nspy,40\nAGG,40\nXYZ,30\n", encoding="utf-8")
    pesos, notas = leer_cartera_actual(tmp)
    assert pesos.sum() == pytest.approx(1.0)
    assert set(pesos.index) == {"SPY", "AGG", "XYZ"}
    assert any("porcentaje" in n for n in notas)
    assert any("normalizaron" in n for n in notas)


def test_la_rotacion_se_reporta_y_el_tope_se_respeta(mercado):
    _ret, cov, tipos, caps = mercado
    pi, cov, tipos, ancla, satelite, libre = _base(mercado, VIEWS)
    # La cartera de hoy: el ancla de otro mandato, más una posición fuera de
    # la cesta que habría que vender entera.
    otra, _ = policy_weights(tipos, "Agresivo", caps=caps,
                             core=split_core_satellite(tipos)[0])
    actual = (otra * 0.95).to_dict() | {"XYZ": 0.05}
    _, _, _, _, _, sin_tope = _base(mercado, VIEWS, current=actual)
    assert sin_tope.turnover is not None and sin_tope.turnover > 0.05
    assert any("XYZ" in n for n in sin_tope.notes)
    # Pasar de Agresivo a Moderado obliga a vender ~12% de ETFs de renta
    # variable (banda de 50%) más el 5% fuera de la cesta: la rotación mínima
    # ronda 17%. Un tope de 20% es alcanzable y aprieta, porque sin tope la
    # cartera rota más.
    tope = 0.20
    assert sin_tope.turnover > tope + 0.01
    _, _, _, _, _, con_tope = _base(mercado, VIEWS, current=actual,
                                    turnover_cap=tope)
    assert con_tope.feasible, con_tope.breaches
    assert con_tope.turnover <= tope + 1e-4
    assert libre.turnover is None


def test_un_tope_de_rotacion_imposible_dice_por_que(mercado):
    _ret, cov, tipos, caps = mercado
    actual = {"XYZ": 1.0}      # todo fuera de la cesta: rotar 100% es obligatorio
    _, _, _, _, _, alloc = _base(mercado, VIEWS, current=actual,
                                 turnover_cap=0.20)
    assert not alloc.feasible
    assert any("rotación" in b for b in alloc.breaches), alloc.breaches


# --------------------------------------------------------------- vigencia
def test_las_propuestas_llevan_su_vigencia():
    from screener.black_litterman import views_payload
    from screener.profiles import MODERADO

    payload = views_payload(VIEWS, strategy="Moderado", profile=MODERADO, meta={})
    assert payload["horizonte_meses"] == HORIZONTE_VIEWS_MESES
    hasta = date.fromisoformat(payload["vigente_hasta"])
    assert 360 <= (hasta - date.today()).days <= 370


def test_la_conviccion_decae_hasta_vencer():
    payload = {"generado": "2026-01-01T00:00:00Z", "horizonte_meses": 12,
               "views": [{"tipo": "absoluto", "activo": "A", "Q": 0.03,
                          "conviccion": 0.8},
                         {"tipo": "absoluto", "activo": "B", "Q": 0.02,
                          "conviccion": 0.3}]}
    hoy, notas = aplicar_vigencia(payload, hoy=date(2026, 1, 1))
    assert [v["conviccion"] for v in hoy] == [0.8, 0.3] and not notas

    medio, notas = aplicar_vigencia(payload, hoy=date(2026, 7, 2))
    assert medio[0]["conviccion"] == pytest.approx(0.4, abs=0.01)
    assert [v["activo"] for v in medio] == ["A"], "B cae bajo el mínimo"
    assert notas and "descartada" in notas[0]

    vencido, _ = aplicar_vigencia(payload, hoy=date(2027, 2, 1))
    assert vencido == []

    sin_fecha, notas = aplicar_vigencia({"views": payload["views"]})
    assert len(sin_fecha) == 2 and "sin fecha" in notas[0].lower()


# ----------------------------------------------------------- tasa libre
def test_la_tasa_de_mercado_viaja_con_su_fecha():
    idx = pd.to_datetime(["2026-10-22", "2026-10-23"])
    frame = pd.DataFrame({("Close", "^IRX"): [3.95, 3.90]}, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    tasa, fuente = fetch_risk_free(0.0425, download=lambda: frame)
    assert tasa == pytest.approx(0.039)
    assert "2026-10-23" in fuente and "^IRX" in fuente


def test_si_no_se_puede_bajar_se_usa_la_manual_y_se_dice():
    def falla():
        raise ConnectionError("sin red")
    tasa, fuente = fetch_risk_free(0.0425, download=falla)
    assert tasa == 0.0425
    assert "manual" in fuente and "ConnectionError" in fuente


def test_un_valor_absurdo_no_se_acepta():
    frame = pd.DataFrame({"Close": [425.0]}, index=pd.to_datetime(["2026-10-23"]))
    tasa, fuente = fetch_risk_free(0.0425, download=lambda: frame)
    assert tasa == 0.0425 and "manual" in fuente


# ------------------------------------------------------- pendientes Comité
def test_los_pendientes_salen_de_las_constantes_vivas():
    filas = pendientes()
    por_parametro = {f["parametro"]: f for f in filas}
    satelite = por_parametro["Techo del satélite (SATELITE_MAX)"]["valor"]
    assert f"A {SATELITE_MAX['Agresivo']:.1%}" in satelite
    tope = por_parametro["Tope sectorial, fracción del sleeve de RV (SECTOR_CAPS)"]
    assert f"{SECTOR_CAPS['Moderado']:.0%}" in tope["valor"]
    assert {f["origen"] for f in filas} <= {"mesa", "derivado", "supuesto", "asesoría"}
    temas = {f["tema"] for f in filas}
    assert {"Cartera", "Riesgo", "Screener", "Views", "Jurisdicción"} <= temas
    assert any("Retención" in f["parametro"] for f in filas)


# ------------------------------------------------------ reproducibilidad
def test_la_corrida_se_guarda_y_se_lee_igual():
    base = Path(tempfile.mkdtemp())
    md = {"instruments": [{"ticker": "AAA", "history": {"close": [1.0, np.float64(2.0)]}}],
          "risk_free_rate": np.float64(0.04)}
    params = pd.DataFrame([("Perfil", "Moderado"), ("IC", 0.08)],
                          columns=["Parámetro", "Valor"])
    carpeta = guardar_corrida(base, market_data=md, parametros=params,
                              views=VIEWS, cartera=pd.Series({"AAA": 0.6, "B": 0.0}),
                              retornos=pd.DataFrame({"AAA": [0.01, -0.02]},
                                                    index=pd.to_datetime(["2026-01-01", "2026-01-02"])),
                              tablas={"vacia": pd.DataFrame()})
    leida = leer_corrida(carpeta)
    assert leida["market_data"]["instruments"][0]["history"]["close"] == [1.0, 2.0]
    assert leida["parametros"]["Perfil"] == "Moderado"
    assert all(not k.startswith("_") for v in leida["views"] for k in v)
    assert list(leida["cartera"].index) == ["AAA"]
    assert not (carpeta / "vacia.csv").exists()
    # Nunca se pisa una corrida guardada.
    from datetime import datetime
    mismo_segundo = datetime.strptime(carpeta.name[:15], "%Y%m%d-%H%M%S")
    otra = guardar_corrida(base, market_data=md, parametros=params,
                           cuando=mismo_segundo)
    assert otra != carpeta and otra.name.startswith(carpeta.name)


def test_la_huella_del_motor_es_la_que_lleva_el_notebook():
    nb = (ROOT / "notebooks" / "screener_colab.ipynb").read_text(encoding="utf-8")
    embebida = re.search(r'ENGINE_SHA256 = \\"([0-9a-f]{64})\\"', nb)
    assert embebida, "el notebook no declara su huella"
    assert embebida.group(1) == huella_motor(), (
        "el script y el notebook registrarían motores distintos para el mismo "
        "código: reconstruye el notebook")
