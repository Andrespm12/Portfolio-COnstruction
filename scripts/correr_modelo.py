#!/usr/bin/env python3
"""
Corre el modelo completo — screening, views, diagnósticos y cartera — en un
solo comando, sin notebook.

Hace exactamente lo mismo que ``notebooks/screener_colab.ipynb`` y en el mismo
orden, contra el mismo paquete ``screener/``. No hay una segunda copia de la
lógica: si los dos alguna vez dieran resultados distintos, sería un bug, no una
diferencia de diseño.

Uso
---
    python3 scripts/correr_modelo.py                      # todo por defecto
    python3 scripts/correr_modelo.py --perfil Agresivo
    python3 scripts/correr_modelo.py --universo etfs --top-n 30
    python3 scripts/correr_modelo.py --tickers SPY,QQQ,TLT,GLD,AAPL,MSFT
    python3 scripts/correr_modelo.py --ancla mercado      # comparar anclas
    python3 scripts/correr_modelo.py --sin-pares-automaticos
    python3 scripts/correr_modelo.py --salida ./corridas/2026-08

Tiene que correr desde el repo: importa el paquete ``screener/`` que está un
nivel arriba de este archivo. El notebook, en cambio, lleva el motor embebido
y no necesita nada al lado.

Requisitos
----------
    pip install pandas numpy yfinance openpyxl cvxpy scikit-learn

Necesita salida a internet para Yahoo Finance. Todo lo demás corre local.

Qué produce
-----------
En el directorio de salida (por defecto, el actual):

    screening.xlsx                              9 hojas, se explica solo
    {Estrategia}_screener_propuestas_{fecha}.json   entrada para el BL de CCI

Las nueve hojas: Ranking, Bloques, Perfiles, Views BL, Cartera, Cesta,
Universo (qué entró al ranking y qué se rechazó, con el motivo), Cobertura y
Parametros.

El JSON va a ``propuestas/``, nunca a ``aprobadas/``: esa carpeta es solo para
views que un gestor ya revisó y firmó, y el propio ``write_views`` se niega a
escribir ahí.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ==========================================================================
# PARÁMETROS
# ==========================================================================
# Los mismos nombres que en el notebook, para que se puedan mapear a ojo.
# Cualquiera se puede sobreescribir por línea de comandos; ver --help.

# --- Universo y ventana ---------------------------------------------------
UNIVERSO = "completo"          # completo | acciones | etfs | ndx | djia | lista
TICKERS_PERSONALIZADOS = ""    # separados por coma; solo si UNIVERSO = "lista"
BENCHMARK = "SPY"
PERIODO = "2y"                 # 1y | 2y | 5y
TASA_LIBRE_RIESGO = 0.0425

# --- Perfil de riesgo -----------------------------------------------------
# Cambia pesos de bloque, umbrales, gates, dimensionamiento y liquidez mínima,
# todo a la vez. No es una etiqueta sobre el mismo ranking.
PERFIL = "Moderado"            # Conservador Defensivo | Conservador | Moderado | Agresivo
TAMANO_POSICION_USD = 500_000  # supuesto de dimensionamiento, NO un dato de cuenta

# --- Datos opcionales (lentos) --------------------------------------------
CON_VOL_IMPLICITA = False      # ~2 requests extra por ticker
CON_NOMBRES_Y_SECTORES = False  # imprescindible si usas lista personalizada

# --- Views ----------------------------------------------------------------
ESTRATEGIA_CCI = "Moderado"    # Conservador_Defensivo | Conservador | Moderado | Agresivo
IC_SUPUESTO = 0.08             # supuesto declarado, no calibrado contra backtest
MAX_VIEWS = 8

#: Equivale a la columna activo_referencia del Google Sheet: empareja una
#: acción con el ETF contra el que debe medirse. Con referencia produce una
#: view RELATIVA; sin ella, ABSOLUTA.
REFERENCIAS = {
    "AAPL": "QQQ", "MSFT": "QQQ", "NVDA": "QQQ", "AVGO": "SMH",
    "JPM": "XLF", "BAC": "XLF", "LLY": "XLV", "UNH": "XLV",
    "XOM": "XLE", "CVX": "XLE",
}

#: Buscar contraparte para los nombres que REFERENCIAS no cubre. El par solo se
#: acepta si el spread es más tranquilo que la pata suelta; si no, la view queda
#: absoluta. Ver find_peer() en black_litterman.py.
PARES_AUTOMATICOS = True

# --- Cartera --------------------------------------------------------------
TOP_N_CARTERA = 25             # menos nombres = covarianza mejor estimada
ANCLA = "politica"             # politica | mercado  (ver el documento maestro)
POSICION_MINIMA = 0.01         # posición mínima ejecutable; 0 la desactiva

# --- Salida ---------------------------------------------------------------
#: Directorio con los CSV de tenencias de los emisores, uno por ETF
#: (SPY.csv, IVV.csv...). Sin él, la sección de transparencia dice que
#: no puede ver nada en vez de estimar.
TENENCIAS_DIR = "tenencias"

EXPORTAR_JSON_PARA_BL = True

_GRUPOS = {
    "completo": ("SP500", "NDX", "DJIA", "ETF"),
    "acciones": ("SP500", "NDX", "DJIA"),
    "etfs": ("ETF",),
    "ndx": ("NDX",),
    "djia": ("DJIA",),
}

_PERFILES_VALIDOS = ("Conservador Defensivo", "Conservador", "Moderado", "Agresivo")
_ESTRATEGIAS_VALIDAS = ("Conservador_Defensivo", "Conservador", "Moderado", "Agresivo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Corre el modelo completo: screening, views, diagnósticos y cartera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Uso\n---\n")[1].split("Requisitos")[0],
    )
    p.add_argument("--perfil", default=PERFIL, choices=_PERFILES_VALIDOS,
                   help="Perfil de riesgo del screening.")
    p.add_argument("--estrategia", default=ESTRATEGIA_CCI, choices=_ESTRATEGIAS_VALIDAS,
                   help="Estrategia CCI destino de las views y la cartera.")
    p.add_argument("--universo", default=UNIVERSO, choices=sorted(_GRUPOS) + ["lista"],
                   help="Qué instrumentos entran a competir.")
    p.add_argument("--tickers", default=TICKERS_PERSONALIZADOS,
                   help="Lista separada por coma. Implica --universo lista.")
    p.add_argument("--benchmark", default=BENCHMARK)
    p.add_argument("--periodo", default=PERIODO, choices=("1y", "2y", "5y"))
    p.add_argument("--rf", type=float, default=TASA_LIBRE_RIESGO,
                   help="Tasa libre de riesgo anual.")
    p.add_argument("--posicion-usd", type=float, default=TAMANO_POSICION_USD,
                   help="Posición asumida para el cálculo de días de liquidación.")
    p.add_argument("--ic", type=float, default=IC_SUPUESTO,
                   help="Coeficiente de información supuesto para traducir el ranking a Q.")
    p.add_argument("--max-views", type=int, default=MAX_VIEWS)
    p.add_argument("--sin-pares-automaticos", dest="pares_automaticos",
                   action="store_false", default=PARES_AUTOMATICOS,
                   help="Solo emparejar lo declarado en REFERENCIAS; el resto "
                        "queda como view absoluta.")
    p.add_argument("--top-n", type=int, default=TOP_N_CARTERA,
                   help="Nombres del ranking que entran a la optimización.")
    p.add_argument("--ancla", default=ANCLA, choices=("politica", "mercado"),
                   help="Cartera neutral de la que parten las views.")
    p.add_argument("--posicion-minima", type=float, default=POSICION_MINIMA,
                   help="Posición mínima ejecutable como fracción del libro "
                        "(0.01 = 1%%). 0 la desactiva.")
    p.add_argument("--con-vol-implicita", action="store_true", default=CON_VOL_IMPLICITA)
    p.add_argument("--con-nombres", action="store_true", default=CON_NOMBRES_Y_SECTORES,
                   help="Baja nombres largos y sectores. Necesario con --tickers.")
    p.add_argument("--sin-json", action="store_true",
                   help="Solo el Excel, sin el JSON de propuestas para el BL.")
    p.add_argument("--tenencias", default=TENENCIAS_DIR,
                   help="Directorio con los CSV de tenencias por ETF, "
                        "para el reporte de transparencia.")
    p.add_argument("--salida", default=".", help="Directorio donde escribir los archivos.")
    args = p.parse_args(argv)
    if args.tickers.strip():
        args.universo = "lista"
    return args


def titulo(texto: str) -> None:
    print(f"\n{'=' * 72}\n{texto}\n{'=' * 72}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import pandas as pd

    from screener.black_litterman import (
        DRIVE_PROPOSALS_DIR, ViewParams, build_basket, build_views,
        default_views_filename, public_view, write_views,
    )
    from screener.cci_regulation import CLASE_EQUITY, REGULACIONES, classify_for_bands
    from screener.diagnostics import run_diagnostics
    from screener.lookthrough import load_holdings, report
    from screener.optimizer import (
        ALLOW_LEVERAGE, gross_budget, allocation_table,
        implied_equilibrium, market_weights,
        optimize, policy_weights, posterior, select_basket, shrunk_covariance,
    )
    from screener.profiles import PROFILES, get_profile, profile_for_strategy
    from screener.report import console_summary
    from screener.run_screen import run_standalone
    from screener.seleccion import CRITERIOS, barras_requeridas, politica_declarada
    from screener.seleccion import tabla as tabla_seleccion
    from screener.tuning import reset_all
    from screener.yahoo_adapter import (
        coverage_report, daily_returns, default_universe, fetch_market_caps,
        fetch_market_data,
    )

    salida = Path(args.salida).expanduser().resolve()
    salida.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 1. universo
    titulo("1 · UNIVERSO Y PERFIL")

    if args.universo == "lista":
        tickers = [t.strip().upper().replace(".", "-")
                   for t in args.tickers.split(",") if t.strip()]
        if not tickers:
            raise SystemExit("Elegiste lista personalizada pero no pusiste tickers.")
        if args.benchmark.upper() not in tickers:
            tickers.append(args.benchmark.upper())
        if not args.con_nombres:
            print("AVISO: sin nombres largos, un ETF apalancado o de covered-call\n"
                  "       en tu lista pasaría el filtro de producto. Considera --con-nombres.")
    else:
        tickers = default_universe(_GRUPOS[args.universo], benchmark=args.benchmark)

    print(f"{len(tickers)} tickers  |  benchmark {args.benchmark}  |  "
          f"{args.periodo} de historia diaria")

    perfil = get_profile(args.perfil)
    print()
    print(perfil.describe())

    perfil_cci = profile_for_strategy(args.estrategia)
    if perfil_cci.key != perfil.key:
        print(f"\nAVISO: corres el screen con perfil {perfil.label} pero exportas para "
              f"{args.estrategia}, que corresponde a {perfil_cci.label}.")
        print("       Alinea ambos o las views llevarán umbrales y gates de otro mandato.")

    # ---------------------------------------------------------------- 2. datos
    titulo("2 · DESCARGA")
    t0 = time.time()
    market_data, frame_diario = fetch_market_data(
        tickers,
        benchmark=args.benchmark,
        risk_free_rate=args.rf,
        period=args.periodo,
        with_metadata=args.con_nombres,
        with_iv=args.con_vol_implicita,
        progress=True,
        with_frame=True,          # el optimizador necesita retornos diarios
    )
    print(f"\n{len(market_data['instruments'])} instrumentos utilizables "
          f"en {time.time() - t0:.0f}s")

    dropped = market_data.get("dropped", [])
    if dropped:
        print(f"\n{len(dropped)} descartados antes de puntuar:")
        for t, r in dropped[:15]:
            print(f"  {t:8s} {r}")
        if len(dropped) > 15:
            print(f"  ... y {len(dropped) - 15} más")

    # ---------------------------------------------------------------- 3. cobertura
    titulo("3 · COBERTURA DE MÉTRICAS")
    cov = coverage_report(market_data)
    faltantes = cov[cov["coverage"] < 1.0]
    if faltantes.empty:
        print("Cobertura completa en todas las métricas.")
    else:
        print("Métricas por debajo de cobertura total:\n")
        for _, r in faltantes.iterrows():
            print(f"  {r['coverage']:6.1%}  {r['metric']:34s} ({r['block']}) — {r['source']}")

    # ---------------------------------------------------------------- 4. screening
    titulo("4 · SCREENING")
    scored, meta = run_standalone(
        market_data, profile=args.perfil,
        position_usd=args.posicion_usd, rf=args.rf,
    )
    print(console_summary(scored, meta))

    # ---------------------------------------------------------- 4b. universo
    titulo("4b · POLÍTICA DE SELECCIÓN DEL UNIVERSO")
    print(politica_declarada())
    print()
    _sel = meta["seleccion_resumen"]
    print(f"Candidatos evaluados por la política: {_sel['candidatos']}")
    print(f"  admitidos al ranking:              {_sel['admitidos']}")
    for _c in CRITERIOS:
        if _sel.get(_c.clave):
            print(f"  rechazados por {_c.titulo.lower():24s} {_sel[_c.clave]}")
    universo_df = tabla_seleccion(meta["seleccion"])
    _fuera = universo_df[universo_df["admitido"] == "no"]
    if not _fuera.empty:
        print("\nRechazados, con motivo:")
        for _r in _fuera.head(20).itertuples():
            print(f"  {_r.ticker:8s} [{_r.criterio}] {_r.motivo[:70]}")
        if len(_fuera) > 20:
            print(f"  ... y {len(_fuera) - 20} más (la hoja Universo del Excel los trae todos)")

    import screener.config as _cfg
    modelo = _cfg.FACTOR_MODEL           # ya con el perfil aplicado
    bloques = [b.key for b in modelo]
    etiquetas = {b.key: b.label for b in modelo}

    tabla = pd.DataFrame([{
        "rank": i, "ticker": r.ticker, "tipo": r.asset_type,
        "reco": r.recommendation, "score": r.score_0_100, "z": r.composite_z,
        "peso_ind": r.indicative_weight,
        "ret_1a": r.diagnostics.get("return_1y"),
        "vol": r.diagnostics.get("volatility"),
        "max_dd": r.diagnostics.get("max_drawdown"),
        "beta": r.diagnostics.get("beta"),
        "sharpe": r.raw_metrics.get("sharpe_1y"),
        # Sin libro no hay correlación contra el libro; se muestra alfa.
        "alpha": r.diagnostics.get("alpha_annual"),
        "gates": ", ".join(r.gates_triggered),
    } for i, r in enumerate(scored, 1)])

    mapa = pd.DataFrame(
        [{"ticker": r.ticker,
          **{etiquetas[k]: r.block_scores.get(k) for k in bloques}}
         for r in scored[:30]]
    ).set_index("ticker")

    print("\nTop 15:")
    print(f"  {'#':>3} {'ticker':8s} {'reco':14s} {'score':>6s} {'z':>7s} "
          f"{'ret 1A':>8s} {'vol':>7s} {'sharpe':>7s}")
    for row in tabla.head(15).itertuples():
        ret = f"{row.ret_1a:+.1%}" if pd.notna(row.ret_1a) else "—"
        vol = f"{row.vol:.1%}" if pd.notna(row.vol) else "—"
        shp = f"{row.sharpe:.2f}" if pd.notna(row.sharpe) else "—"
        print(f"  {row.rank:>3} {row.ticker:8s} {row.reco:14s} "
              f"{row.score:6.1f} {row.z:+7.2f} {ret:>8s} {vol:>7s} {shp:>7s}")

    # ---------------------------------------------------------------- 5. perfiles
    titulo("5 · COMPARACIÓN ENTRE PERFILES")
    recos, excluidos = {}, {}
    try:
        for k, p in PROFILES.items():
            s, m = run_standalone(market_data, profile=k,
                                  position_usd=args.posicion_usd, rf=args.rf)
            recos[p.label] = {r.ticker: r.recommendation for r in s}
            excluidos[p.label] = dict(m.get("excluded", []))
    finally:
        # Deja el modelo como lo espera el resto del script.
        reset_all()
        scored, meta = run_standalone(market_data, profile=args.perfil,
                                      position_usd=args.posicion_usd, rf=args.rf)

    NO_ELEGIBLE = "NO ELEGIBLE"
    tickers_scored = [r.ticker for r in scored]

    # Cada perfil filtra por liquidez distinto, así que no todos puntúan el
    # mismo conjunto. Indexar a ciegas revienta con KeyError en cuanto un
    # perfil excluya algo que otro sí aceptó.
    comparacion = pd.DataFrame({
        label: pd.Series({t: r.get(t, NO_ELEGIBLE) for t in tickers_scored})
        for label, r in recos.items()
    })
    comparacion.insert(0, "score_" + args.perfil.split()[0].lower(),
                       pd.Series({r.ticker: r.score_0_100 for r in scored}))

    etiquetas_perfiles = [p.label for p in PROFILES.values()]
    ow = comparacion[etiquetas_perfiles].eq("OVERWEIGHT").sum(axis=1)
    print(f"Overweight en TODOS los perfiles: "
          f"{list(comparacion.index[ow == len(etiquetas_perfiles)]) or 'ninguno'}")
    print(f"Overweight solo en Agresivo:      "
          f"{list(comparacion.index[(ow == 1) & comparacion['Agresivo'].eq('OVERWEIGHT')]) or 'ninguno'}")

    for label in etiquetas_perfiles:
        fuera = [t for t in tickers_scored if recos[label].get(t) is None]
        if fuera:
            print(f"\n{label} no considera {len(fuera)} de estos nombres:")
            for t in fuera[:8]:
                razon = (excluidos[label].get(t) or ["fuera del universo"])[0]
                print(f"  {t:8s} {razon}")

    # ---------------------------------------------------------------- 6. views
    titulo("6 · VIEWS BLACK-LITTERMAN")
    params = ViewParams(information_coefficient=args.ic, max_views=args.max_views,
                        auto_pair=args.pares_automaticos)

    # La cesta se calcula aquí, antes de las views, porque el pool de pares
    # tiene que ser exactamente el universo de la covarianza: posterior()
    # descarta en silencio cualquier view que nombre un ticker fuera de él, así
    # que un par contra un nombre que no llega a la cesta no debilita la view,
    # la borra — y encima gasta uno de los cupos de max_views.
    cartera_tickers = select_basket(scored, args.estrategia,
                                    top_n=args.top_n, min_per_class=3)
    views = build_views(scored, market_data, strategy=args.estrategia,
                        reference_map=REFERENCIAS,
                        pair_pool=cartera_tickers, params=params)
    cesta = build_basket(scored, strategy=args.estrategia, reference_map=REFERENCIAS)

    print(f"{len(views)} views para {args.estrategia} "
          f"(perfil {perfil_cci.label}, IC {args.ic})\n")
    for v in views:
        quien = (v["activo"] if v["tipo"] == "absoluto"
                 else f"{v['activo_long']} / {v['activo_short']}")
        marca = {"declarado": " (REFERENCIAS)", "automatico": " (par automático)"}
        print(f"  {v['tipo']:9s} {quien:18s} Q {v['Q']:+.2%}   "
              f"convicción {v['conviccion']:.2f}"
              f"{marca.get(v.get('_pairing', ''), '')}")

    autom = [v for v in views if v.get("_pairing") == "automatico"]
    if autom:
        print(f"\n{len(autom)} par(es) los eligió el modelo, no REFERENCIAS. "
              f"Cada uno pasó el filtro de cobertura (el spread es al menos "
              f"{ViewParams().min_hedge_benefit:.0%} menos volátil que la pata "
              f"suelta); el motivo va escrito en la justificación de la view.")
    elif args.pares_automaticos:
        print("\nNingún par automático: ningún candidato de la cesta cubría lo "
              "suficiente. Las views quedan absolutas, que es el resultado "
              "correcto cuando no hay con qué cubrir.")

    # public_view quita la columna interna _q_bruto, que solo usa el
    # diagnóstico de abajo y no viaja al archivo de CCI.
    cesta_df = pd.DataFrame(cesta)

    # ---------------------------------------------------------------- 7. diagnósticos
    titulo("7 · DIAGNÓSTICOS DEL MODELO")
    print(run_diagnostics(scored, views, params))

    # ---------------------------------------------------------------- 8. cartera
    titulo("8 · CARTERA")
    tipos_todos = {r.ticker: r.asset_type for r in scored}

    # La cesta no puede ser solo el top-N por score: el ranking premia momentum
    # y riesgo-retorno, donde la renta variable domina, y con una cesta 100%
    # equity el techo del mandato queda por debajo del libro invertido — el
    # solver responde 'infactible' y la cartera sale vacía.
    clases_cesta = sorted({classify_for_bands(t, tipos_todos.get(t, "ETF"))
                           for t in cartera_tickers})
    print(f"Cesta: {len(cartera_tickers)} nombres en {len(clases_cesta)} clases")
    print(f"  {', '.join(clases_cesta)}\n")

    retornos = daily_returns(frame_diario, cartera_tickers)
    covarianza = shrunk_covariance(retornos)

    capitalizaciones = fetch_market_caps(list(covarianza.columns))
    tipos_cesta = {t: tipos_todos.get(t, "ETF") for t in covarianza.columns}
    # Presupuesto bruto en vigor. Con el apalancamiento apagado es 1.0, y el
    # ancla se construye sobre ese mismo total para que "sin views el
    # optimizador devuelve el ancla" siga siendo cierto.
    presupuesto = gross_budget(args.estrategia)

    if args.ancla == "politica":
        pesos_ancla, notas_ancla = policy_weights(
            tipos_cesta, args.estrategia, caps=capitalizaciones, total=presupuesto)
        for n in notas_ancla:
            print(f"  {n}")
    else:
        pesos_ancla, sin_cap = market_weights(capitalizaciones,
                                              list(covarianza.columns))
        notas_ancla = []
        if sin_cap:
            print(f"Sin capitalización, excluidos del equilibrio: {sin_cap}")

    print(f"\nAncla ({args.ancla}) por clase de activo:")
    cl_ancla = pd.Series({t: classify_for_bands(t, tipos_cesta[t])
                          for t in pesos_ancla.index})
    for clase, peso in pesos_ancla.groupby(cl_ancla).sum().sort_values(
            ascending=False).items():
        if peso > 0.0001:
            print(f"  {peso:7.2%}  {clase}")

    pi = implied_equilibrium(pesos_ancla, covarianza)
    er_posterior, cov_posterior = posterior(pi, covarianza, views)
    clases = {t: classify_for_bands(t, tipos_todos.get(t, "ETF"))
              for t in covarianza.columns}
    cartera = optimize(er_posterior, cov_posterior, tipos_todos, args.estrategia,
                       min_position=args.posicion_minima or None)

    print(f"\n{args.estrategia}  |  estado: {cartera.status}")
    print(f"Exposición bruta   {cartera.gross_exposure:.1%}")
    print(f"Retorno esperado   {cartera.expected_return:+.2%} anual")
    print(f"Volatilidad        {cartera.volatility:.1%} anual")
    print(f"Posiciones         {int((cartera.weights > 0).sum())}")

    print("\nPor clase de activo")
    for clase, peso in cartera.by_class.items():
        if peso > 0.0001:
            print(f"  {peso:7.2%}  {clase}")

    if cartera.breaches:
        print("\nAUDITORÍA — INCUMPLIMIENTOS:")
        for b in cartera.breaches:
            print(f"  {b}")
    else:
        print("\nAuditoría de bandas: sin incumplimientos.")
    for n in cartera.notes:
        print(f"NOTA: {n}")

    cartera_df = allocation_table(cartera, classes=clases)
    if cartera_df.empty:
        # Una hoja vacía no dice nada. El motivo viaja con el resultado.
        cartera_df = pd.DataFrame({
            "ticker": ["SIN CARTERA"],
            "nombre": [f"La optimización no encontró solución ({cartera.status})"],
            "clase_activo": [" | ".join(cartera.breaches) or "sin detalle"],
            "peso": [0.0],
        })
        print("\nNO HAY CARTERA. Motivo:")
        for b in cartera.breaches:
            print(f"  {b}")
    else:
        print("\nPesos:")
        for row in cartera_df.itertuples():
            if row.peso > 0.0001:
                print(f"  {row.peso:7.2%}  {row.ticker}")

    # ------------------------------------------------------- 8b. transparencia
    #
    # La hoja de Cartera de arriba lista instrumentos. Esto lista lo que de
    # verdad tienes: un libro con fondos adentro carga emisores que nadie eligió
    # uno por uno, y el tope del Procedimiento está escrito sobre el instrumento
    # cuando su intención es sobre el emisor.
    titulo("8b · TRANSPARENCIA (LOOK-THROUGH)")
    tenencias, sectores_lt, notas_lt = load_holdings(args.tenencias)
    for n in notas_lt:
        print(f"  {n}")
    if notas_lt:
        print()
    acciones_cesta = [t for t in cartera.weights.index
                      if classify_for_bands(t, tipos_todos.get(t, "ETF")) == CLASE_EQUITY]
    print(report(cartera.weights[cartera.weights > 0].to_dict(),
                 tenencias, sectores_lt,
                 cap=REGULACIONES[args.estrategia]["max_equity_individual"],
                 only=acciones_cesta))

    # ---------------------------------------------------------------- 9. export
    titulo("9 · ARCHIVOS")
    archivo_excel = salida / "screening.xlsx"
    archivo_views = salida / DRIVE_PROPOSALS_DIR / default_views_filename(args.estrategia)

    parametros = pd.DataFrame([
        ("Generado (UTC)", pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M")),
        ("Perfil", perfil.label),
        ("Perfil — resumen", perfil.summary),
        ("Estrategia CCI destino", args.estrategia),
        ("Universo", args.universo),
        ("Nombres puntuados", len(scored)),
        ("Benchmark", args.benchmark),
        ("Historia", args.periodo),
        ("Tasa libre de riesgo", f"{args.rf:.2%}"),
        ("Posición asumida (liquidez)", f"${args.posicion_usd:,.0f}"),
        ("Fuente de datos", market_data["data_source"]),
        ("Portafolio", "ninguno — screen independiente"),
        ("IC supuesto (views)", args.ic),
        ("Nota sobre el IC", "supuesto declarado, no calibrado contra backtest"),
        ("Pares automáticos", "sí" if args.pares_automaticos else "no"),
        ("Views con par declarado",
         sum(1 for v in views if v.get("_pairing") == "declarado")),
        ("Views con par automático",
         sum(1 for v in views if v.get("_pairing") == "automatico")),
        ("Cobertura mínima exigida al par",
         f"{ViewParams().min_hedge_benefit:.0%} menos volatilidad que la pata"),
        ("Ancla del equilibrio", args.ancla),
        ("Universo — candidatos", meta["seleccion_resumen"]["candidatos"]),
        ("Universo — admitidos", meta["seleccion_resumen"]["admitidos"]),
        ("Universo — barras mínimas", barras_requeridas()),
        ("Posición mínima",
         f"{args.posicion_minima:.2%}" if args.posicion_minima else "sin mínimo"),
        ("Nota sobre el ancla",
         " | ".join(notas_ancla) or "capitalización de mercado / AUM"),
        ("Estado de la optimización", cartera.status),
        ("Exposición bruta", f"{cartera.gross_exposure:.2%}"),
        ("Apalancamiento", "desactivado por política de mesa"
         if not ALLOW_LEVERAGE else "activo"),
        ("Presupuesto bruto en vigor", f"{presupuesto:.0%}"),
        ("Apalancamiento que permite el mandato",
         f"{REGULACIONES[args.estrategia]['leverage_max']:.0%}"),
        ("Auditoría de bandas",
         "sin incumplimientos" if not cartera.breaches
         else " | ".join(cartera.breaches)),
        ("Umbral Overweight", f"z >= {perfil.bands.overweight_z:+.2f}"),
        ("Umbral Underweight", f"z <= {perfil.bands.underweight_z:+.2f}"),
        ("Piso absoluto para Overweight",
         f"momentum 12M-1M >= {perfil.gates.min_momentum_for_overweight}, "
         f"Sharpe >= {perfil.gates.min_sharpe_for_overweight}"),
        ("Techo de volatilidad para OW",
         f"{perfil.gates.max_volatility_for_overweight:.0%}"),
        ("Beta máxima", f"{perfil.gates.beta_limit:.2f}"),
        ("Peso máximo por posición", f"{perfil.sizing.max_weight:.1%}"),
        ("Volumen diario mínimo", f"${perfil.eligibility.min_adv_usd / 1e6:,.0f}MM"),
    ] + [(f"Peso — {b.label}", f"{b.weight:.0%}") for b in modelo],
        columns=["Parámetro", "Valor"])

    # Las views en formato legible: una fila por view, con las dos formas
    # (absoluta y relativa) resueltas a columnas explícitas.
    views_excel = pd.DataFrame([{
        "tipo": v["tipo"],
        "activo": v.get("activo", ""),
        "long": v.get("activo_long", ""),
        "short": v.get("activo_short", ""),
        "Q": v["Q"],
        "conviccion": v["conviccion"],
        "justificacion": v["justificacion"],
    } for v in views])

    with pd.ExcelWriter(archivo_excel, engine="openpyxl") as xl:
        tabla.to_excel(xl, sheet_name="Ranking", index=False)
        mapa.to_excel(xl, sheet_name="Bloques")
        comparacion.to_excel(xl, sheet_name="Perfiles")
        views_excel.to_excel(xl, sheet_name="Views BL", index=False)
        cartera_df.to_excel(xl, sheet_name="Cartera", index=False)
        cesta_df.to_excel(xl, sheet_name="Cesta", index=False)
        universo_df.to_excel(xl, sheet_name="Universo", index=False)
        cov.to_excel(xl, sheet_name="Cobertura", index=False)
        parametros.to_excel(xl, sheet_name="Parametros", index=False)

        for hoja in xl.book.worksheets:
            hoja.freeze_panes = "A2"
            for col in hoja.columns:
                ancho = max((len(str(c.value)) for c in col if c.value), default=8)
                hoja.column_dimensions[col[0].column_letter].width = min(46, ancho + 3)

    print(f"{archivo_excel}  —  {len(scored)} nombres, 9 hojas")

    if not args.sin_json:
        write_views(views, archivo_views, strategy=args.estrategia,
                    profile=perfil_cci, meta=meta, params=params)
        print(f"{archivo_views}  —  {len(views)} propuestas")

    return 0


if __name__ == "__main__":
    sys.exit(main())
