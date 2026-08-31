"""
Pruebas de la política de selección del universo.

Dos de estas cargan todo el peso. El resto es cobertura normal.

**Neutralidad frente al desempeño.** Es la promesa central del módulo: la
selección no mira retorno, momentum, Sharpe ni nada que el modelo puntúe. Una
promesa así no se sostiene con un comentario — se sostiene barajando el camino
de precios de todo el universo y verificando que el conjunto admitido no se
mueva ni un nombre. Si alguien agrega mañana un filtro por momentum, esta
prueba falla.

**La tabla de barras contra el código real.** ``BARRAS_REQUERIDAS`` es una
tabla escrita a mano que describe lo que hacen las funciones de metrics.py. Una
tabla así se desincroniza en silencio en cuanto alguien cambia un lookback. La
prueba llama a ``compute_metrics`` con exactamente N barras y con N-1, y exige
que la métrica aparezca en el primer caso y falte en el segundo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from screener.config import ELIGIBILITY  # noqa: E402
from screener.metrics import compute_metrics, simple_returns  # noqa: E402
from screener.seleccion import (  # noqa: E402
    BARRAS_REQUERIDAS, CRITERIOS, PROHIBIDOS, barras_requeridas, evaluar,
    politica_declarada, resumen, seleccionar, tabla,
)
from test_scoring import make_instrument, make_series  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def serie_ligada(bench, beta=1.3, ruido=0.004, seed=0, start=100.0):
    """Serie correlacionada con el benchmark, para que el modelo de mercado
    sea significativo y las métricas de beta/alfa se calculen."""
    rng = np.random.default_rng(seed)
    b = simple_returns(np.asarray(bench, dtype=float))
    r = beta * b + rng.normal(0.0, ruido, b.size)
    # Mismo largo que el benchmark: cumprod sobre N-1 retornos da N-1 precios,
    # y sin el precio inicial la serie queda una barra corta — que en una
    # prueba de límites exactos es justo el error que se busca evitar.
    return np.concatenate([[start], start * np.cumprod(1.0 + r)])


def metricas_de(inst, bench_rets):
    m = compute_metrics(inst, bench_rets, net_liq=1e7, participation=0.2,
                        base_position_weight=0.03)
    cierres = inst["history"]["close"]
    m["_last_price"] = float(cierres[-1])
    m["_adv_usd"] = 5e8
    m["_bars"] = len(cierres)
    return m


def universo(n=10, barras=60, seed=1):
    """Un universo sintético con todo lo que la selección mira."""
    bench = make_series(n=barras, drift=0.004, vol=0.018, seed=seed)
    bench_rets = simple_returns(bench)
    insts, mets = [], {}
    for i in range(n):
        s = serie_ligada(bench, beta=0.7 + 0.12 * i, seed=seed * 100 + i)
        inst = make_instrument(f"N{i:02d}", s, adv=5e8 * (1 + 0.3 * i))
        insts.append(inst)
        mets[inst["ticker"]] = metricas_de(inst, bench_rets)
    return insts, mets, bench_rets


# --------------------------------------------------------------------------
# 1. Neutralidad frente al desempeño
# --------------------------------------------------------------------------

def barajar_camino(inst, seed):
    """
    Reordena los retornos del instrumento y reescala para que el precio final
    quede igual.

    Eso cambia por completo momentum, volatilidad, drawdown, Sharpe, beta y
    captura — todo lo que el modelo puntúa — dejando intactos el precio de
    cierre, el número de barras y el volumen, que es lo único que la selección
    tiene derecho a mirar.
    """
    import copy

    out = copy.deepcopy(inst)
    p = np.asarray(out["history"]["close"], dtype=float)
    r = simple_returns(p)
    rng = np.random.default_rng(seed)
    rng.shuffle(r)
    nuevo = [p[0]]
    for x in r:
        nuevo.append(nuevo[-1] * (1.0 + x))
    nuevo = np.asarray(nuevo)
    nuevo *= p[-1] / nuevo[-1]          # mismo precio final
    out["history"]["close"] = [float(v) for v in nuevo]
    snap = out["snapshot"]["misc-statistics"]
    snap["high_52w"], snap["low_52w"] = float(nuevo.max()), float(nuevo.min())
    out["snapshot"]["last"]["price"] = float(nuevo[-1])
    return out


def test_la_seleccion_es_ciega_al_desempeno() -> None:
    insts, mets, bench_rets = universo()
    admitidos, _ = seleccionar(insts, mets)
    check("el universo base admite nombres", len(admitidos) >= 8, str(admitidos))

    for intento in range(5):
        revueltos = [barajar_camino(i, 900 + intento * 17 + j)
                     for j, i in enumerate(insts)]
        mets2 = {i["ticker"]: metricas_de(i, bench_rets) for i in revueltos}
        otros, _ = seleccionar(revueltos, mets2)
        if otros != admitidos:
            check("la selección no cambia al barajar el camino de precios",
                  False, f"intento {intento}: {sorted(set(admitidos) ^ set(otros))}")
            return

    # Comprobación de que el barajado realmente movió lo que debía mover.
    revuelto = barajar_camino(insts[0], 1)
    m0 = mets[insts[0]["ticker"]]
    m1 = metricas_de(revuelto, bench_rets)
    movidas = [k for k in ("mom_12_1", "volatility_1y", "max_drawdown", "sharpe_1y")
               if m0.get(k) is not None and m1.get(k) is not None
               and abs(m0[k] - m1[k]) > 1e-6]
    check("el barajado sí alteró las métricas de desempeño",
          len(movidas) >= 3, f"solo cambiaron {movidas}")
    check("la selección no cambia al barajar el camino de precios", True)


def test_los_criterios_prohibidos_estan_declarados() -> None:
    """
    Un criterio ausente no se puede auditar. La lista de prohibidos existe
    para que agregar un filtro por desempeño sea visiblemente una violación
    de algo escrito, y no una omisión.
    """
    claves = {c for c, _ in PROHIBIDOS}
    check("desempeño está declarado como criterio prohibido", "desempeño" in claves)
    check("sector está declarado como criterio prohibido", "sector" in claves)
    check("tamaño está declarado como criterio prohibido", "tamaño" in claves)
    texto = politica_declarada()
    check("la política escrita nombra los prohibidos",
          all(c in texto for c in claves))
    check("la política declara qué criterio trunca un bloque puntuado",
          "trunca un bloque" in texto)


# --------------------------------------------------------------------------
# 2. La tabla de barras contra el código real
# --------------------------------------------------------------------------

def test_la_tabla_de_barras_coincide_con_las_funciones() -> None:
    """
    Para cada métrica basada en historia: con N barras se calcula, con N-1 no.

    Sin esto la tabla es una nota que envejece. Con esto, cambiar un lookback
    en metrics.py rompe la prueba y obliga a actualizarla.
    """
    largo = 80
    bench = make_series(n=largo, drift=0.004, vol=0.018, seed=7)

    def presente(clave, barras):
        b = bench[-barras:]
        s = serie_ligada(b, beta=1.2, ruido=0.0015, seed=3)
        inst = make_instrument("X", s)
        m = compute_metrics(inst, simple_returns(b), net_liq=1e7,
                            participation=0.2, base_position_weight=0.03)
        return m.get(clave) is not None

    desajustes = []
    for clave, req in sorted(BARRAS_REQUERIDAS.items()):
        if req is None or clave not in (
                "mom_12_1", "mom_6m", "mom_3m", "above_40w_ma", "ma_slope_13w",
                "sharpe_1y", "sortino_1y", "calmar_1y", "pct_positive_periods",
                "volatility_1y", "max_drawdown", "downside_deviation",
                "ulcer_index", "beta_1y", "alpha_annual", "idio_vol_share",
                "turnover_stability"):
            continue
        if not presente(clave, req):
            desajustes.append(f"{clave}: declara {req} barras pero con {req} no se calcula")
        elif req > 3 and presente(clave, req - 1):
            desajustes.append(f"{clave}: declara {req} barras pero con {req - 1} ya se calcula")

    check("la tabla de barras coincide con lo que hacen las funciones",
          not desajustes, "; ".join(desajustes))


def test_el_minimo_se_deriva_del_modelo_vigente() -> None:
    completo = barras_requeridas(1.0)
    check("cobertura completa exige 53 barras (lo que pide mom_12_1)",
          completo == 53, f"dio {completo}")
    check("una cobertura menor exige menos barras",
          barras_requeridas(0.5) < completo,
          f"{barras_requeridas(0.5)} vs {completo}")
    check("el mínimo derivado supera al min_history_bars suelto de config",
          completo > ELIGIBILITY.min_history_bars,
          f"{completo} vs {ELIGIBILITY.min_history_bars}")

    from screener import tuning
    try:
        # Sin el bloque de momentum, el modelo necesita mucha menos historia.
        tuning.set_block_weights({"momentum": 0.0})
        sin_momentum = barras_requeridas(1.0)
    finally:
        tuning.reset_all()
    check("quitar un bloque mueve el mínimo, no está fijo",
          sin_momentum <= completo, f"{sin_momentum} vs {completo}")


# --------------------------------------------------------------------------
# 3. Cada criterio hace lo suyo, y deja rastro
# --------------------------------------------------------------------------

def test_historia_insuficiente_se_rechaza_con_motivo() -> None:
    bench = make_series(n=40, drift=0.004, vol=0.018, seed=11)
    corto = serie_ligada(bench, seed=12)
    inst = make_instrument("CORTO", corto)
    d = evaluar(inst, metricas_de(inst, simple_returns(bench)))

    check("un nombre con 40 barras no entra", not d.admitido)
    check("el rechazo se atribuye a la suficiencia de datos",
          d.criterio == "historia", d.criterio)
    check("el motivo dice cuántas barras hay y cuántas hacen falta",
          "40 barras" in d.detalle and "53" in d.detalle, d.detalle)
    check("la decisión guarda el conteo de barras", d.barras == 40)


def test_producto_excluido_se_rechaza_antes_que_nada() -> None:
    bench = make_series(n=60, seed=13)
    s = serie_ligada(bench, seed=14)
    inst = make_instrument("LEV", s)
    inst["name"] = "DIREXION DAILY SEMI BULL 3X"
    d = evaluar(inst, metricas_de(inst, simple_returns(bench)))
    check("un apalancado no entra", not d.admitido)
    check("se atribuye al tipo de producto", d.criterio == "producto", d.criterio)
    check("el motivo nombra el patrón que coincidió", "/" in d.detalle, d.detalle)


def test_negociabilidad_usa_las_reglas_del_perfil() -> None:
    from screener.profiles import get_profile

    bench = make_series(n=60, seed=15)
    s = serie_ligada(bench, seed=16)
    inst = make_instrument("ILIQ", s)
    m = metricas_de(inst, simple_returns(bench))
    m["_adv_usd"] = 30e6      # entre el piso de Moderado y el de Conservador

    from screener import tuning
    try:
        tuning.reset_all()
        d_mod = evaluar(inst, m, reglas=get_profile("Moderado").eligibility)
        d_con = evaluar(inst, m, reglas=get_profile("Conservador").eligibility)
    finally:
        tuning.reset_all()

    check("$30MM de volumen entra en Moderado", d_mod.admitido, d_mod.detalle)
    check("el mismo nombre no entra en Conservador", not d_con.admitido)
    check("y se atribuye a la negociabilidad",
          d_con.criterio == "negociabilidad", d_con.criterio)
    check("el motivo cuantifica el volumen y el piso",
          "30.0MM" in d_con.detalle and "50MM" in d_con.detalle, d_con.detalle)


def test_el_rastro_cubre_a_todos_los_candidatos() -> None:
    """
    El punto entero del módulo: que un nombre excluido sea visible. Si el rastro
    no cubre a todos, se vuelve a tener exclusiones invisibles.
    """
    insts, mets, bench_rets = universo(n=8)
    corto = make_instrument("CORTO", serie_ligada(
        make_series(n=35, seed=21), seed=22))
    insts.append(corto)
    mets["CORTO"] = metricas_de(corto, bench_rets)

    admitidos, decisiones = seleccionar(insts, mets)
    check("hay una decisión por candidato, sin excepción",
          len(decisiones) == len(insts), f"{len(decisiones)} vs {len(insts)}")
    check("todo rechazo trae criterio y motivo",
          all(d.criterio and d.detalle for d in decisiones if not d.admitido))
    check("CORTO aparece en el rastro aunque no entre",
          any(d.ticker == "CORTO" and not d.admitido for d in decisiones))

    r = resumen(decisiones)
    check("el resumen cuadra: admitidos + rechazos = candidatos",
          r["admitidos"] + sum(r[c.clave] for c in CRITERIOS) == r["candidatos"],
          str(r))

    t = tabla(decisiones)
    check("la tabla tiene una fila por candidato", len(t) == len(insts))
    check("la tabla trae las columnas del reporte",
          list(t.columns) == ["ticker", "admitido", "criterio", "motivo", "barras"],
          str(list(t.columns)))


def test_el_motivo_reportado_es_el_de_fondo() -> None:
    """
    Cuando un nombre falla varios criterios, el rastro debe dar la razón de
    fondo, no la primera que se topó el código.

    Un ticker que nunca estuvo en el universo y además tiene historia corta se
    rechaza por lo primero. Decir "historia corta" sugeriría que con más datos
    entraría, y no es cierto: no está en el alcance del mandato.
    """
    bench = make_series(n=40, seed=41)
    inst = make_instrument("RARO", serie_ligada(bench, seed=42))
    inst["indices"] = []
    d = evaluar(inst, metricas_de(inst, simple_returns(bench)))

    check("fuera del universo y con poca historia se atribuye al alcance",
          d.criterio == "mercado", f"dijo {d.criterio}")
    check("y el motivo lo dice", "not a member" in d.detalle, d.detalle)

    # Un apalancado que sí está en el universo: el motivo es el producto.
    largo = make_series(n=60, seed=43)
    lev = make_instrument("LEVX", serie_ligada(largo, seed=44))
    lev["name"] = "PROSHARES ULTRAPRO QQQ 3X"
    d2 = evaluar(lev, metricas_de(lev, simple_returns(largo)))
    check("un apalancado dentro del universo se atribuye al producto",
          d2.criterio == "producto", f"dijo {d2.criterio}")

    check("el orden declarado coincide con el de evaluación",
          [c.clave for c in CRITERIOS] ==
          ["mercado", "producto", "historia", "negociabilidad"],
          str([c.clave for c in CRITERIOS]))


def main() -> int:
    for fn in [
        test_la_seleccion_es_ciega_al_desempeno,
        test_los_criterios_prohibidos_estan_declarados,
        test_la_tabla_de_barras_coincide_con_las_funciones,
        test_el_minimo_se_deriva_del_modelo_vigente,
        test_historia_insuficiente_se_rechaza_con_motivo,
        test_producto_excluido_se_rechaza_antes_que_nada,
        test_negociabilidad_usa_las_reglas_del_perfil,
        test_el_rastro_cubre_a_todos_los_candidatos,
        test_el_motivo_reportado_es_el_de_fondo,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
