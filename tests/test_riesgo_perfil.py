"""
Pruebas de la coherencia con las views y del riesgo esperado por mandato.

Las dos salieron de la misma corrida real. Una cartera con una view que decía
"MU le gana a LRCX" llevaba MU 5.84% y LRCX 5.85% — posicionada, por poco, en
contra de su propia view. Y las cuatro estrategias optimizaban la misma función
con lambda 2.5, así que nada obligaba a la Agresiva a asumir más riesgo que la
Moderada: solo tenía bandas más anchas, y una banda es un techo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.cci_regulation import (RISK_AVERSION_BY_STRATEGY, RISK_TARGETS,
                                     REGULACIONES, risk_aversion_for)
from screener.optimizer import (RISK_AVERSION, drawdown_metrics,
                                implied_equilibrium, optimize, policy_weights,
                                coherence_notes, posterior,
                                relative_view_pairs,
                                risk_profile_table, shrunk_covariance,
                                view_coherence_breaches)


def serie(n, vol, drift=0.0004, seed=0):
    return np.random.default_rng(seed).normal(drift, vol, n)


@pytest.fixture(scope="module")
def mercado():
    """Universo chico pero con las clases que las bandas necesitan."""
    rng = np.random.default_rng(3)
    n = 600
    f_eq = rng.normal(0.0005, 0.011, n)
    f_bond = rng.normal(0.0001, 0.004, n)
    datos = {
        "AAA": 1.0 * f_eq + rng.normal(0, 0.010, n),
        "BBB": 1.0 * f_eq + rng.normal(0, 0.011, n),   # casi clon de AAA
        "CCC": 0.6 * f_eq + rng.normal(0, 0.008, n),
        "SPY": 0.9 * f_eq + rng.normal(0, 0.002, n),
        "AGG": 1.0 * f_bond + rng.normal(0, 0.0008, n),
        "TLT": 2.2 * f_bond + rng.normal(0, 0.0010, n),
        "LQD": 1.4 * f_bond + rng.normal(0, 0.0009, n),
        "BIL": 0.02 * f_bond + 0.00005,
    }
    ret = pd.DataFrame(datos)
    tipos = {"AAA": "STOCK", "BBB": "STOCK", "CCC": "STOCK",
             "SPY": "ETF", "AGG": "ETF", "TLT": "ETF", "LQD": "ETF",
             "BIL": "ETF"}
    caps = {"AAA": 1e12, "BBB": 8e11, "CCC": 5e11, "SPY": 5e11,
            "AGG": 1e11, "TLT": 5e10, "LQD": 3e10, "BIL": 3e10}
    return ret, shrunk_covariance(ret), tipos, caps


def resolver(mercado, estrategia, views, **kw):
    ret, cov, tipos, caps = mercado
    ancla, _ = policy_weights(tipos, estrategia, caps=caps)
    pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
    er, cov_post = posterior(pi, cov, views)
    return optimize(er, cov_post, tipos, estrategia, views=views,
                    min_position=None, anchor=ancla, prior=pi, **kw)


# ---------------------------------------------------- coherencia con views
def test_se_leen_las_dos_patas_de_una_view_relativa():
    vistas = [{"tipo": "relativa", "activo_long": "MU", "activo_short": "LRCX",
               "Q": 0.02}]
    assert relative_view_pairs(vistas) == [("MU", "LRCX")]


def test_una_view_absoluta_no_genera_par():
    assert relative_view_pairs([{"tipo": "absoluta", "activo": "JNJ",
                                 "Q": 0.03}]) == []


def test_un_q_negativo_invierte_la_direccion():
    # "A menos B = -2%" dice que gana B, no A.
    vistas = [{"tipo": "relativa", "activo_long": "A", "activo_short": "B",
               "Q": -0.02}]
    assert relative_view_pairs(vistas) == [("B", "A")]


def test_una_pata_fuera_del_problema_no_restringe_nada():
    # posterior() descarta las views que nombran algo fuera de la covarianza;
    # restringir por ellas ataría la cartera a una view que no se aplicó.
    vistas = [{"tipo": "relativa", "activo_long": "MU", "activo_short": "FUERA",
               "Q": 0.02}]
    assert relative_view_pairs(vistas, ["MU", "AAPL"]) == []


def test_la_auditoria_detecta_una_cartera_al_reves_de_su_view():
    # El caso literal de la corrida: MU 5.84% contra LRCX 5.85%.
    pesos = pd.Series({"MU": 0.0584, "LRCX": 0.0585})
    fuera = view_coherence_breaches(pesos, [("MU", "LRCX")])
    assert fuera and "al revés" in fuera[0]


def test_no_hay_incumplimiento_si_la_larga_pesa_mas():
    pesos = pd.Series({"MU": 0.08, "LRCX": 0.02})
    assert view_coherence_breaches(pesos, [("MU", "LRCX")]) == []


def test_pesos_iguales_cumplen():
    # Igualdad no es contradicción: en un libro solo-largo puede ser lo mejor
    # que se puede hacer, y forzar una diferencia sería inventar una apuesta.
    pesos = pd.Series({"MU": 0.05, "LRCX": 0.05})
    assert view_coherence_breaches(pesos, [("MU", "LRCX")]) == []


def test_la_restriccion_impide_que_el_libro_contradiga_su_view(mercado):
    # BBB es casi un clon de AAA. Sin la restricción el optimizador puede
    # preferir BBB pese a una view que dice que gana AAA.
    vistas = [{"tipo": "relativa", "activo_long": "AAA", "activo_short": "BBB",
               "Q": 0.01, "conviccion": 0.6}]
    suelto = resolver(mercado, "Moderado", vistas, enforce_view_coherence=False)
    atado = resolver(mercado, "Moderado", vistas)

    assert atado.feasible
    assert atado.weights["AAA"] >= atado.weights["BBB"] - 1e-6
    assert view_coherence_breaches(atado.weights, [("AAA", "BBB")]) == []
    assert not [b for b in atado.breaches if "al revés" in b]
    # Y la corrida dice que la restricción está puesta.
    assert any("Coherencia" in n for n in atado.notes)
    assert suelto.feasible


def test_la_restriccion_no_obliga_a_tener_la_posicion(mercado):
    # w_largo >= w_corto se cumple con 0 >= 0. Un piso con margen forzaría al
    # libro a comprar un nombre que el optimizador no quiere, que es una
    # apuesta que nadie aprobó.
    vistas = [{"tipo": "relativa", "activo_long": "CCC", "activo_short": "AAA",
               "Q": 0.001, "conviccion": 0.1}]
    alloc = resolver(mercado, "Conservador_Defensivo", vistas)
    assert alloc.feasible
    assert alloc.weights["CCC"] >= alloc.weights["AAA"] - 1e-6


def test_se_puede_apagar(mercado):
    vistas = [{"tipo": "relativa", "activo_long": "AAA", "activo_short": "BBB",
               "Q": 0.01, "conviccion": 0.6}]
    alloc = resolver(mercado, "Moderado", vistas, enforce_view_coherence=False)
    assert not any("Coherencia" in n for n in alloc.notes)


# ------------------------------------------------------- lambda por mandato
def test_cada_mandato_tiene_su_propia_aversion():
    assert set(RISK_AVERSION_BY_STRATEGY) == set(REGULACIONES)
    orden = ["Conservador_Defensivo", "Conservador", "Moderado", "Agresivo"]
    valores = [risk_aversion_for(e) for e in orden]
    assert valores == sorted(valores, reverse=True), (
        "un mandato más agresivo tiene que ser menos averso al riesgo")


def test_moderado_conserva_el_valor_del_documento_de_cci():
    assert risk_aversion_for("Moderado") == 2.5


def test_la_corrida_dice_que_lambda_es_del_mandato(mercado):
    alloc = resolver(mercado, "Agresivo", [])
    assert any("λ=1.5" in n for n in alloc.notes), alloc.notes


def test_un_lambda_menor_asume_mas_riesgo(mercado):
    # Con todo lo demás igual — misma cesta, mismas bandas, mismo ancla — bajar
    # la aversión tiene que subir la volatilidad. Si no, el parámetro no hace
    # nada y las cuatro carteras son la misma.
    ret, cov, tipos, caps = mercado
    ancla, _ = policy_weights(tipos, "Moderado", caps=caps)
    pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
    er, cov_post = posterior(pi, cov, [])
    averso = optimize(er, cov_post, tipos, "Moderado", risk_aversion=8.0,
                      min_position=None, risk_budget=None)
    audaz = optimize(er, cov_post, tipos, "Moderado", risk_aversion=1.0,
                     min_position=None, risk_budget=None)
    assert audaz.volatility > averso.volatility


# ------------------------------------------------------- presupuesto de riesgo
def test_el_rango_de_riesgo_crece_con_el_perfil():
    orden = ["Conservador_Defensivo", "Conservador", "Moderado", "Agresivo"]
    pisos = [RISK_TARGETS[e][0] for e in orden]
    techos = [RISK_TARGETS[e][1] for e in orden]
    assert pisos == sorted(pisos) and techos == sorted(techos)
    assert all(p < t for p, t in zip(pisos, techos))


def test_el_rango_no_es_un_incumplimiento_del_procedimiento(mercado):
    # breaches significa "se violó el Procedimiento de Inversión". El rango de
    # volatilidad no está en ese documento y ni siquiera lo aprobó el Comité:
    # mezclarlos haría que una señal de cumplimiento se dispare por un número
    # que nos inventamos nosotros.
    alloc = resolver(mercado, "Agresivo", [])
    for hallazgo in alloc.risk_findings:
        assert hallazgo not in alloc.breaches


def test_sin_views_la_cartera_sigue_siendo_el_ancla(mercado):
    # La segunda pasada NO se dispara sin views. Esa propiedad — sin nada que
    # decir, el modelo devuelve la asignación estratégica del mandato — es lo
    # que hace del ancla un neutral de verdad, y no se sacrifica por el piso.
    ret, cov, tipos, caps = mercado
    for estrategia in REGULACIONES:
        ancla, _ = policy_weights(tipos, estrategia, caps=caps)
        pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
        er, cov_post = posterior(pi, cov, [])
        alloc = optimize(er, cov_post, tipos, estrategia, min_position=None,
                         views=[])
        if not alloc.feasible:
            continue
        deriva = float((alloc.weights - ancla.reindex(alloc.weights.index)
                        .fillna(0.0)).abs().max())
        assert deriva < 0.02, f"{estrategia}: deriva {deriva:.4f}"


def test_un_libro_bajo_su_piso_se_reporta(mercado):
    ret, cov, tipos, caps = mercado
    ancla, _ = policy_weights(tipos, "Moderado", caps=caps)
    pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
    er, cov_post = posterior(pi, cov, [])
    # Un piso imposible para esta cesta: tiene que decirlo, no romperse.
    alloc = optimize(er, cov_post, tipos, "Moderado", min_position=None,
                     risk_budget=(0.90, 0.99), views=[])
    assert alloc.feasible
    assert any("DEBAJO del piso" in r for r in alloc.risk_findings)


def test_un_techo_inalcanzable_no_deja_la_cartera_vacia(mercado):
    # Un techo que la cesta no puede cumplir es información, no razón para
    # devolver cero cartera.
    ret, cov, tipos, caps = mercado
    ancla, _ = policy_weights(tipos, "Agresivo", caps=caps)
    pi = implied_equilibrium(ancla, cov, risk_aversion=RISK_AVERSION)
    er, cov_post = posterior(pi, cov, [])
    alloc = optimize(er, cov_post, tipos, "Agresivo", min_position=None,
                     risk_budget=(0.0, 0.0001), views=[])
    assert alloc.feasible, "el techo no puede vaciar la cartera"
    assert any("ENCIMA del techo" in r for r in alloc.risk_findings)


# ------------------------------------------------------------- drawdown
def test_la_caida_se_mide_sobre_la_serie_de_la_cartera(mercado):
    ret, cov, tipos, caps = mercado
    m = drawdown_metrics({"AGG": 0.5, "TLT": 0.5}, ret)
    assert m["max_drawdown"] <= 0.0
    assert "peor_12m" in m


def test_sin_historia_no_se_inventa_una_caida():
    assert drawdown_metrics({"AAA": 1.0}, None) == {}
    assert drawdown_metrics({"AAA": 1.0}, pd.DataFrame()) == {}


def test_una_cartera_mas_volatil_cae_mas(mercado):
    ret, cov, tipos, caps = mercado
    tranquila = drawdown_metrics({"BIL": 1.0}, ret)["max_drawdown"]
    movida = drawdown_metrics({"AAA": 1.0}, ret)["max_drawdown"]
    assert movida < tranquila


# --------------------------------------------------------- tabla por perfil
def test_la_tabla_resuelve_los_cuatro_mandatos(mercado):
    ret, cov, tipos, caps = mercado
    tabla, _ = risk_profile_table(cov, tipos, caps, [], returns=ret,
                                  min_position=None)
    assert list(tabla["estrategia"]) == list(REGULACIONES)
    assert set(["retorno_esperado", "volatilidad", "max_drawdown",
                "caida_1a_95", "lambda"]) <= set(tabla.columns)


def test_la_tabla_usa_el_lambda_de_cada_mandato(mercado):
    ret, cov, tipos, caps = mercado
    tabla, _ = risk_profile_table(cov, tipos, caps, [], returns=ret,
                                  min_position=None)
    assert list(tabla["lambda"]) == [risk_aversion_for(e) for e in REGULACIONES]


def test_el_riesgo_crece_con_el_perfil(mercado):
    ret, cov, tipos, caps = mercado
    vistas = [{"tipo": "absoluta", "activo": "AAA", "Q": 0.03,
               "conviccion": 0.7}]
    tabla, notas = risk_profile_table(cov, tipos, caps, vistas, returns=ret,
                                      min_position=None)
    vols = list(tabla["volatilidad"])
    assert vols == sorted(vols), f"{notas or vols}"


def test_una_inversion_del_orden_se_reporta():
    # La comprobación tiene que poder disparar, y probarla a través de cuatro
    # optimizaciones no la ejerce en el caso que importa: el ordenamiento de
    # las anclas del Modelo de Asignación domina y casi siempre sale bien.
    tabla = pd.DataFrame([
        {"estrategia": "Moderado", "retorno_esperado": 0.06,
         "volatilidad": 0.14, "vol_min_objetivo": 0.065, "vol_max_objetivo": 0.15},
        {"estrategia": "Agresivo", "retorno_esperado": 0.04,
         "volatilidad": 0.09, "vol_min_objetivo": 0.10, "vol_max_objetivo": 0.24},
    ])
    notas = coherence_notes(tabla)
    assert any("El riesgo NO crece" in n for n in notas), notas
    assert any("retorno esperado NO crece" in n for n in notas), notas
    assert any("fuera de su rango objetivo" in n for n in notas), notas


def test_una_tabla_coherente_no_genera_avisos():
    tabla = pd.DataFrame([
        {"estrategia": "Moderado", "retorno_esperado": 0.05,
         "volatilidad": 0.10, "vol_min_objetivo": 0.065, "vol_max_objetivo": 0.15},
        {"estrategia": "Agresivo", "retorno_esperado": 0.08,
         "volatilidad": 0.18, "vol_min_objetivo": 0.10, "vol_max_objetivo": 0.24},
    ])
    assert coherence_notes(tabla) == []


def test_una_estrategia_infactible_no_rompe_la_comprobacion():
    tabla = pd.DataFrame([
        {"estrategia": "Moderado", "retorno_esperado": float("nan"),
         "volatilidad": float("nan"), "vol_min_objetivo": 0.065,
         "vol_max_objetivo": 0.15},
        {"estrategia": "Agresivo", "retorno_esperado": 0.08,
         "volatilidad": 0.18, "vol_min_objetivo": 0.10, "vol_max_objetivo": 0.24},
    ])
    assert coherence_notes(tabla) == []


def test_el_equilibrio_no_usa_el_lambda_del_cliente(mercado):
    # pi = delta*Sigma*w describe el equilibrio del MERCADO. Meterle el lambda
    # del mandato escala pi y hace que la Agresiva salga con menos retorno
    # esperado que la Moderada — un artefacto de escala que se lee como que el
    # mandato agresivo es peor.
    ret, cov, tipos, caps = mercado
    vistas = [{"tipo": "absoluta", "activo": "AAA", "Q": 0.03,
               "conviccion": 0.7}]
    tabla, _ = risk_profile_table(cov, tipos, caps, vistas, returns=ret,
                                  min_position=None)
    retornos = list(tabla["retorno_esperado"])
    assert retornos == sorted(retornos), (
        f"el retorno esperado tiene que crecer con el perfil: {retornos}")
