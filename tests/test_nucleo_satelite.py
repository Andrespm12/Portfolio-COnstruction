"""
Núcleo y satélite, y la renta fija que ningún universo puede quitar.

Salen del review de las carteras de largo plazo. Una corrida Agresiva con
UNIVERSO = "Solo acciones" no tenía bonos ni efectivo, ningún mandato podía
invertir el 100% bajo su techo de renta variable, y el solver devolvió
infactible. Y el ancla repartía la línea de acciones del Modelo por
capitalización entre acciones individuales y ETFs, así que el "neutral" eran
sobre todo acciones elegidas por el ranking: una sola acción podía desplazar el
núcleo estratégico.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.cci_regulation import (  # noqa: E402
    ASSET_CLASS, CLASES_DEFENSIVAS, EXPOSICIONES_NUCLEO, REGULACIONES,
    SATELITE_MAX, TRACKING_ERROR_OBJETIVO, UNIVERSO_NUCLEO,
    VOL_ACTIVA_SATELITE, classify_for_bands,
)
from screener.optimizer import (  # noqa: E402
    RISK_AVERSION, implied_equilibrium, optimize, policy_weights, posterior,
    risk_profile_table, select_basket, shrunk_covariance, split_core_satellite,
)
from screener.yahoo_adapter import con_nucleo  # noqa: E402


@pytest.fixture(scope="module")
def mercado():
    rng = np.random.default_rng(11)
    n = 600
    f_eq = rng.normal(0.0005, 0.011, n)
    f_bond = rng.normal(0.0001, 0.004, n)
    datos = {
        "AAA": 1.3 * f_eq + rng.normal(0, 0.015, n),
        "BBB": 1.1 * f_eq + rng.normal(0, 0.013, n),
        "CCC": 0.8 * f_eq + rng.normal(0, 0.012, n),
        "XLE": 0.7 * f_eq + rng.normal(0, 0.010, n),   # ETF sectorial: satélite
        "GLD": rng.normal(0.0002, 0.009, n),            # oro: satélite
        "SPY": 1.0 * f_eq + rng.normal(0, 0.001, n),    # núcleo
        "EFA": 0.9 * f_eq + rng.normal(0, 0.004, n),    # núcleo
        "GOVT": 1.5 * f_bond + rng.normal(0, 0.0008, n),
        "LQD": 1.4 * f_bond + rng.normal(0, 0.0009, n),
        "AGG": 1.0 * f_bond + rng.normal(0, 0.0006, n),
        "BIL": 0.02 * f_bond + 0.00005,
    }
    ret = pd.DataFrame(datos)
    tipos = {t: ("STOCK" if t in ("AAA", "BBB", "CCC") else "ETF") for t in datos}
    caps = {"AAA": 3e12, "BBB": 2e12, "CCC": 8e11, "XLE": 3e10, "GLD": 9e10,
            "SPY": 6e11, "EFA": 6e10, "GOVT": 3e10, "LQD": 3e10, "AGG": 1.2e11,
            "BIL": 4e10}
    return ret, shrunk_covariance(ret), tipos, caps


NUCLEO_RV = ("SPY", "EFA")


# ----------------------------------------------------------- tope del satélite
def test_el_tope_del_satelite_sale_del_presupuesto_de_tracking_error():
    for estrategia, (_lo, te_max) in TRACKING_ERROR_OBJETIVO.items():
        assert SATELITE_MAX[estrategia] == pytest.approx(te_max / VOL_ACTIVA_SATELITE)


def test_el_tope_crece_con_el_perfil_y_cae_en_el_rango_del_review():
    orden = ["Conservador_Defensivo", "Conservador", "Moderado", "Agresivo"]
    topes = [SATELITE_MAX[e] for e in orden]
    assert topes == sorted(topes) and len(set(topes)) == 4
    # El review sugirió ~0-10% en Conservador Defensivo y ~25-35% en Agresivo.
    assert 0.0 < SATELITE_MAX["Conservador_Defensivo"] <= 0.10
    assert 0.25 <= SATELITE_MAX["Agresivo"] <= 0.35


# -------------------------------------------------------- universo con núcleo
def test_el_universo_nucleo_trae_todas_las_clases_defensivas():
    defensivos = {t for t, c in ASSET_CLASS.items() if c in CLASES_DEFENSIVAS}
    assert defensivos <= set(UNIVERSO_NUCLEO)
    for vehiculos in EXPOSICIONES_NUCLEO.values():
        assert set(vehiculos) <= set(UNIVERSO_NUCLEO)
    assert "GLD" not in UNIVERSO_NUCLEO, "el oro es una apuesta, no núcleo"


def test_solo_acciones_recibe_renta_fija_y_efectivo():
    tickers, agregados = con_nucleo(["AAPL", "MSFT", "SPY"])
    clases = {classify_for_bands(t, "ETF") for t in agregados}
    assert {"RentaFija_Soberana_IG", "RentaFija_Corporativa_IG",
            "Efectivo_MM"} <= clases
    assert "SPY" not in agregados, "lo que ya estaba no se cuenta como agregado"
    assert len(tickers) == len(set(tickers))
    assert {"AAPL", "MSFT"} <= set(tickers)


# ------------------------------------------------------------- la separación
def test_la_separacion_pone_cada_cosa_en_su_lado(mercado):
    _ret, _cov, tipos, _caps = mercado
    nucleo, satelite, notas = split_core_satellite(tipos, NUCLEO_RV)
    assert nucleo == {"SPY", "EFA", "GOVT", "LQD", "AGG", "BIL"}
    assert satelite == {"AAA", "BBB", "CCC", "XLE", "GLD"}
    assert not notas


def test_sin_vehiculo_nucleo_de_renta_variable_no_se_separa(mercado):
    _ret, _cov, tipos, _caps = mercado
    nucleo, satelite, notas = split_core_satellite(tipos, ())
    assert {"AAA", "SPY", "XLE"} <= nucleo
    assert satelite == {"GLD"}
    assert notas and "no se separa" in notas[0]


def test_el_ancla_vive_en_el_nucleo(mercado):
    _ret, _cov, tipos, caps = mercado
    nucleo, satelite, _ = split_core_satellite(tipos, NUCLEO_RV)
    for estrategia in REGULACIONES:
        ancla, _ = policy_weights(tipos, estrategia, caps=caps, core=nucleo)
        assert ancla.sum() == pytest.approx(1.0)
        assert ancla[sorted(satelite)].abs().max() == 0.0
        # La línea de acciones del Modelo queda entera en los ETFs núcleo.
        assert ancla[list(NUCLEO_RV)].sum() > 0.0

    # Sin núcleo, el reparto por capitalización le da la renta variable
    # neutral sobre todo a las acciones: el defecto que corrige el núcleo.
    sin_nucleo, _ = policy_weights(tipos, "Moderado", caps=caps)
    assert sin_nucleo[["AAA", "BBB", "CCC"]].sum() > sin_nucleo[list(NUCLEO_RV)].sum()


# ------------------------------------------------------------ el techo pesa
def _resolver(mercado, estrategia, views, **kw):
    _ret, cov, tipos, caps = mercado
    nucleo, satelite, _ = split_core_satellite(tipos, NUCLEO_RV)
    ancla, _ = policy_weights(tipos, estrategia, caps=caps, core=nucleo)
    pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
    er, cov_post = posterior(pi, cov, views)
    return optimize(er, cov_post, tipos, estrategia, views=views,
                    min_position=None, anchor=ancla, prior=pi,
                    satellite=satelite, risk_budget=None, **kw)


# Muy por encima del equilibrio (π de AAA ≈ 6%), para que las views quieran
# más satélite del que el techo permite.
VIEWS_FUERTES = [
    {"tipo": "absoluto", "activo": t, "Q": 0.25, "conviccion": 0.85}
    for t in ("AAA", "BBB", "CCC")
]


@pytest.mark.parametrize("estrategia", list(REGULACIONES))
def test_las_views_no_pasan_el_techo_del_satelite(mercado, estrategia):
    alloc = _resolver(mercado, estrategia, VIEWS_FUERTES,
                      risk_aversion=0.05)
    assert alloc.feasible, alloc.breaches
    assert alloc.satellite_exposure <= SATELITE_MAX[estrategia] + 1e-4
    assert not any("Satélite en" in b for b in alloc.breaches)


def test_el_techo_es_el_que_contiene_las_views(mercado):
    # Control: el mismo problema sin techo pone más en el satélite. Si no, el
    # test de arriba pasaría aunque el techo no existiera.
    con = _resolver(mercado, "Agresivo", VIEWS_FUERTES, risk_aversion=0.05)
    sin = _resolver(mercado, "Agresivo", VIEWS_FUERTES, risk_aversion=0.05,
                    satellite_cap=None)
    assert sin.satellite_exposure > con.satellite_exposure + 0.01


def test_sin_views_la_cartera_es_el_nucleo(mercado):
    for estrategia in REGULACIONES:
        alloc = _resolver(mercado, estrategia, [])
        assert alloc.feasible
        # Ruido del solver, no posiciones: el ancla del satélite es cero.
        assert alloc.satellite_exposure < 1e-3


def test_la_tabla_por_perfil_reporta_el_satelite(mercado):
    ret, cov, tipos, caps = mercado
    nucleo, _sat, _ = split_core_satellite(tipos, NUCLEO_RV)
    tabla, _ = risk_profile_table(cov, tipos, caps, VIEWS_FUERTES, returns=ret,
                                  min_position=None, core=nucleo)
    assert tabla["retorno_esperado"].notna().all(), list(tabla["motivo"])
    assert (tabla["satelite"] <= tabla["satelite_max"] + 1e-4).all()


def test_solo_acciones_mas_nucleo_resuelve_los_cuatro_mandatos(mercado):
    # La corrida del review, arreglada: acciones + lo que con_nucleo agrega.
    ret, cov, tipos, caps = mercado
    nucleo, _sat, _ = split_core_satellite(tipos, NUCLEO_RV)
    tabla, _ = risk_profile_table(cov, tipos, caps, [], returns=ret,
                                  min_position=None, core=nucleo)
    assert tabla["retorno_esperado"].notna().all(), list(tabla["motivo"])


# ------------------------------------------------- renta fija por liquidez
def test_la_renta_fija_de_la_cesta_se_elige_por_liquidez():
    def fila(t, score, adv, tipo="ETF"):
        return SimpleNamespace(ticker=t, asset_type=tipo, eligible=True,
                               score_0_100=score, diagnostics={"adv_usd": adv})

    # Ordenadas por score, como llegan del screener: los bonos de mejor
    # momentum primero, los grandes y baratos al final.
    scored = [
        fila("AAA", 99, 5e9, "STOCK"),
        fila("TLT", 60, 2e9), fila("TIP", 55, 1e8), fila("MBB", 50, 9e7),
        fila("SHY", 20, 3e9), fila("IEF", 15, 4e9), fila("GOVT", 10, 5e9),
    ]
    cesta, _ = select_basket(scored, "Moderado", top_n=1, min_per_class=3,
                             include_core=False)
    soberanos = [t for t in cesta if t in {"TLT", "TIP", "MBB", "SHY", "IEF", "GOVT"}]
    assert set(soberanos) == {"GOVT", "IEF", "SHY"}, soberanos
