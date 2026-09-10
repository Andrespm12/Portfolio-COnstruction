"""
Las bandas de riesgo y retorno tienen que seguir saliendo de sus supuestos.

Estas constantes no son números escogidos: son el resultado de aplicar
``SUPUESTOS_CLASE`` al ``MODELO_ASIGNACION`` de cada mandato. Un número editado
sin tocar el supuesto que lo produjo deja el modelo diciendo dos cosas
distintas, y nadie se entera — que es exactamente cómo el techo del mandato
Defensivo terminó siendo **inferior** a la volatilidad que su propia política
produce en el tope de renta variable.

Así que estas pruebas rederivan y comparan. Si el Comité cambia un supuesto,
fallan y hay que actualizar las bandas; si alguien cambia una banda a mano,
también fallan. Las dos cosas son correctas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.cci_regulation import (  # noqa: E402
    DRAWDOWN_TOLERADO, MODELO_ASIGNACION, REGULACIONES, RETORNO_ESPERADO,
    RISK_TARGETS, SECTOR_CAPS, SUPUESTOS_CLASE, TRACKING_ERROR_OBJETIVO,
    drawdown_tolerado_for, retorno_esperado_for, riesgo_de_asignacion,
    tracking_error_objetivo_for,
)

ESTRATEGIAS = tuple(MODELO_ASIGNACION)
RV = "Acciones y ETFs indexados"


def en_el_techo(estrategia: str) -> dict[str, float]:
    """La política con la renta variable en su tope regulatorio."""
    base = MODELO_ASIGNACION[estrategia]
    tope = REGULACIONES[estrategia]["max_equity_total"]
    otras = {k: v for k, v in base.items() if k != RV}
    escala = (1.0 - tope) / sum(otras.values())
    return {**{k: v * escala for k, v in otras.items()}, RV: tope}


# --------------------------------------------------- las bandas se derivan
@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_el_piso_es_el_ochenta_por_ciento_del_neutral(estrategia):
    vol, _ = riesgo_de_asignacion(MODELO_ASIGNACION[estrategia])
    piso, _ = RISK_TARGETS[estrategia]
    assert piso == pytest.approx(0.80 * vol, abs=0.006), \
        f"{estrategia}: piso {piso:.1%} contra 80% de {vol:.2%}"


@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_el_techo_alcanza_la_volatilidad_del_tope_de_renta_variable(estrategia):
    # El error que esto existe para no repetir: el techo Defensivo era 7,0% y
    # su propia política en el tope de renta variable produce 7,1%. Un libro
    # perfectamente conforme incumplía su propio techo.
    vol_techo, _ = riesgo_de_asignacion(en_el_techo(estrategia))
    _, techo = RISK_TARGETS[estrategia]
    assert techo >= vol_techo, (
        f"{estrategia}: el techo {techo:.1%} es INFERIOR a la volatilidad "
        f"{vol_techo:.2%} que el mandato produce estando conforme")


@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_el_techo_es_alcanzable(estrategia):
    # El otro extremo: un techo que el mandato no puede tocar ni forzándolo no
    # restringe nada, y el techo es justamente lo que el solver impone.
    vol_techo, _ = riesgo_de_asignacion(en_el_techo(estrategia))
    _, techo = RISK_TARGETS[estrategia]
    assert techo <= vol_techo * 1.15, (
        f"{estrategia}: el techo {techo:.1%} está {techo / vol_techo - 1:.0%} "
        f"por encima de lo máximo alcanzable ({vol_techo:.2%}); no ata nada")


@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_el_retorno_esperado_sale_de_los_mismos_supuestos(estrategia):
    _, ret = riesgo_de_asignacion(MODELO_ASIGNACION[estrategia])
    assert RETORNO_ESPERADO[estrategia] == pytest.approx(ret, abs=0.002)


@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_el_drawdown_es_un_multiplo_de_la_volatilidad_del_techo(estrategia):
    vol_techo, _ = riesgo_de_asignacion(en_el_techo(estrategia))
    esperado = {"Conservador_Defensivo": 2.5, "Conservador": 2.5,
                "Moderado": 2.5, "Agresivo": 3.0}[estrategia]
    implicito = abs(DRAWDOWN_TOLERADO[estrategia]) / vol_techo
    assert implicito == pytest.approx(esperado, abs=0.35), (
        f"{estrategia}: {implicito:.2f} sigma, se esperaban {esperado}")


def test_los_agresivos_llevan_mas_sigmas_que_los_conservadores():
    # Las colas de la renta variable son más gordas que la normal: el mismo
    # múltiplo en los cuatro subestima justo donde el cliente lo va a sentir.
    def sigmas(e):
        vol, _ = riesgo_de_asignacion(en_el_techo(e))
        return abs(DRAWDOWN_TOLERADO[e]) / vol

    assert sigmas("Agresivo") > sigmas("Conservador_Defensivo")


# --------------------------------------------------------- las escaleras
def test_todas_las_escaleras_van_en_el_mismo_sentido():
    orden = ["Conservador_Defensivo", "Conservador", "Moderado", "Agresivo"]

    def creciente(f, nombre):
        vals = [f(e) for e in orden]
        assert all(a < b for a, b in zip(vals, vals[1:])), f"{nombre}: {vals}"

    creciente(lambda e: RISK_TARGETS[e][0], "piso de volatilidad")
    creciente(lambda e: RISK_TARGETS[e][1], "techo de volatilidad")
    creciente(lambda e: RETORNO_ESPERADO[e], "retorno esperado")
    creciente(lambda e: -DRAWDOWN_TOLERADO[e], "drawdown tolerado")
    creciente(lambda e: SECTOR_CAPS[e], "tope sectorial")
    creciente(lambda e: TRACKING_ERROR_OBJETIVO[e][0], "piso de TE")
    creciente(lambda e: TRACKING_ERROR_OBJETIVO[e][1], "techo de TE")


@pytest.mark.parametrize("estrategia", ESTRATEGIAS)
def test_la_banda_contiene_a_su_propia_politica(estrategia):
    vol, _ = riesgo_de_asignacion(MODELO_ASIGNACION[estrategia])
    piso, techo = RISK_TARGETS[estrategia]
    assert piso <= vol <= techo, (
        f"{estrategia}: la asignación neutral produce {vol:.2%}, fuera de "
        f"[{piso:.1%}, {techo:.1%}]")


def test_el_sharpe_de_los_cuatro_esta_en_la_misma_recta():
    # Si un mandato paga mucho peor por su riesgo que los otros, la escalera no
    # es una escalera: es un mandato mal construido.
    efectivo = SUPUESTOS_CLASE["Efectivo / Money Market"][1]
    sharpes = []
    for e in ESTRATEGIAS:
        vol, ret = riesgo_de_asignacion(MODELO_ASIGNACION[e])
        sharpes.append((ret - efectivo) / vol)
    assert max(sharpes) - min(sharpes) < 0.05, dict(zip(ESTRATEGIAS, sharpes))


# ------------------------------------------------------- tope sectorial
def test_el_tope_sectorial_es_fraccion_del_sleeve_no_del_libro():
    # Traducido a libro sobre la política neutral, el tope nuevo es MÁS
    # estricto que el viejo en los cuatro mandatos — y sobre todo en los
    # conservadores, que es donde el viejo no ataba nada.
    viejo = {"Conservador_Defensivo": 0.15, "Conservador": 0.18,
             "Moderado": 0.22, "Agresivo": 0.25}
    for e in ESTRATEGIAS:
        rv = MODELO_ASIGNACION[e][RV]
        nuevo_en_libro = SECTOR_CAPS[e] * rv
        assert nuevo_en_libro < viejo[e], (
            f"{e}: {nuevo_en_libro:.1%} del libro no aprieta contra "
            f"{viejo[e]:.0%}")


def test_ningun_tope_queda_por_debajo_del_indice():
    # Tecnología ronda un tercio del S&P 500. Un tope por debajo de eso
    # obligaría a estar estructuralmente corto del índice en su mayor sector,
    # que es una apuesta activa, no una regla de concentración.
    assert min(SECTOR_CAPS.values()) >= 0.28


# --------------------------------------------------------------- accesores
def test_los_accesores_devuelven_none_para_una_estrategia_desconocida():
    assert retorno_esperado_for("Inventada") is None
    assert drawdown_tolerado_for("Inventada") is None
    assert tracking_error_objetivo_for("Inventada") is None
    assert retorno_esperado_for("Moderado") == pytest.approx(0.065)
    assert drawdown_tolerado_for("Agresivo") == pytest.approx(-0.40)
    assert tracking_error_objetivo_for("Moderado") == (0.025, 0.035)


def test_una_asignacion_vacia_no_revienta():
    assert riesgo_de_asignacion({}) == (0.0, 0.0)
    assert riesgo_de_asignacion({"Clase Inventada": 1.0}) == (0.0, 0.0)
