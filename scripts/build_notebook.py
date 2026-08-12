"""
Builds notebooks/screener_colab.ipynb from the repo source.

The notebook is a build artifact, never hand-edited. The engine it runs is the
package in ``screener/``, embedded as a gzipped tarball so the notebook is
self-contained: it needs no clone, no repo access and no credentials, and it
keeps working if the repository is later made private or moved.

Regenerate after any change to ``screener/``:

    python3 scripts/build_notebook.py

``tests/test_notebook.py`` fails if the checked-in notebook does not match what
this script produces, so drift between the notebook's engine and the repo's is
caught rather than discovered in a trading decision.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "screener_colab.ipynb"

#: Package modules embedded in the notebook. Ordered for readability of the
#: printed manifest only -- import order is resolved by Python.
MODULES = (
    "__init__.py", "config.py", "universe.py", "metrics.py", "portfolio.py",
    "scoring.py", "report.py", "run_screen.py", "yahoo_adapter.py", "tuning.py",
)


def build_payload() -> tuple[str, str]:
    """Return ``(base64_tarball, sha256)`` of the screener package."""
    buffer = io.BytesIO()
    # mtime=0 and sorted members keep the archive byte-identical across runs,
    # so a rebuild with no source change produces no diff.
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name in MODULES:
            path = ROOT / "screener" / name
            data = path.read_bytes()
            info = tarfile.TarInfo(f"screener/{name}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))

    raw = buffer.getvalue()
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    return base64.b64encode(packed).decode("ascii"), hashlib.sha256(raw).hexdigest()


def wrap(blob: str, width: int = 76) -> list[str]:
    return [blob[i:i + width] for i in range(0, len(blob), width)]


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines: str, form: bool = False) -> dict:
    meta: dict = {}
    if form:
        meta["cellView"] = "form"
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": meta, "outputs": [], "source": list(lines),
    }


def build_cells() -> list[dict]:
    blob, digest = build_payload()
    portfolio = json.loads((ROOT / "data" / "portfolio_ibkr.json").read_text())
    positions = [p for p in portfolio.get("positions", []) if p.get("asset_class") == "STK"]

    position_lines = ",\n".join(
        f'        {{"ticker": "{p["ticker"]}", "market_value": {p["market_value"]}, '
        f'"quantity": {p["quantity"]}, "asset_class": "STK"}}'
        for p in positions
    )

    cells: list[dict] = []

    # ---------------------------------------------------------------- intro
    cells.append(md(
        "# Screening cuantitativo de acciones y ETFs\n",
        "\n",
        "Corre el modelo completo del repo sobre datos que se bajan en vivo de "
        "Yahoo Finance. Menú → **Entorno de ejecución → Ejecutar todo**.\n",
        "\n",
        "El motor es idéntico al de `screener/` en el repo (va embebido más "
        "abajo, con su SHA256). Lo único que cambia frente a la corrida con "
        "IBKR es **de dónde salen los datos**.\n",
        "\n",
        "### Qué cambia al usar Yahoo en vez de IBKR\n",
        "\n",
        "| | IBKR | Yahoo |\n",
        "|---|---|---|\n",
        "| Precio, máx/mín 52s, volumen, dividendos | ✅ | ✅ |\n",
        "| Universo | 21 nombres del snapshot | ~600, o el que definas |\n",
        "| Datos | congelados en la captura | en vivo |\n",
        "| Vol implícita (`iv_hv_spread`) | ✅ | opcional, lento |\n",
        "| Percentil de IV a 52s (`iv_percentile`) | ✅ | **no existe** |\n",
        "\n",
        "Yahoo publica la cadena de opciones de hoy, no un histórico de "
        "volatilidad implícita, así que el percentil de IV no se puede "
        "reconstruir. Esa métrica se **omite**, no se rellena con cero: el "
        "motor renormaliza los pesos del bloque sobre las métricas que sí "
        "están. La celda de cobertura te muestra exactamente cuánto pesa esa "
        "ausencia antes de que mires un solo ranking.\n",
    ))

    # -------------------------------------------------------------- install
    cells.append(md("## 1 · Instalación y motor\n"))
    cells.append(code(
        "%pip install -q yfinance\n",
        "print('yfinance listo')\n",
    ))

    cells.append(code(
        "# El paquete screener/ del repo, embebido. Se extrae a /content.\n",
        "import base64, gzip, hashlib, io, sys, tarfile\n",
        "\n",
        f'ENGINE_SHA256 = "{digest}"\n',
        "ENGINE_B64 = (\n",
        *[f'    "{line}"\n' for line in wrap(blob)],
        ")\n",
        "\n",
        "raw = gzip.decompress(base64.b64decode(ENGINE_B64))\n",
        "digest = hashlib.sha256(raw).hexdigest()\n",
        "assert digest == ENGINE_SHA256, f'engine checksum mismatch: {digest}'\n",
        "\n",
        "with tarfile.open(fileobj=io.BytesIO(raw)) as tar:\n",
        "    try:\n",
        "        tar.extractall('.', filter='data')  # Python 3.12+\n",
        "    except TypeError:\n",
        "        tar.extractall('.')\n",
        "if '.' not in sys.path:\n",
        "    sys.path.insert(0, '.')\n",
        "\n",
        "import screener\n",
        "from screener.config import FACTOR_MODEL, block_weight_total\n",
        "\n",
        "print(f'motor verificado  sha256={digest[:16]}...')\n",
        "print(f'{len(FACTOR_MODEL)} bloques, pesos suman "
        "{block_weight_total():.2f}')\n",
        "for b in FACTOR_MODEL:\n",
        "    print(f'  {b.weight:5.0%}  {b.label}')\n",
    ))

    cells.append(code(
        "# Ayudas de presentacion. Mismo par divergente que la pagina HTML del\n",
        "# repo, validado para daltonismo: naranja = adverso, arena = neutro,\n",
        "# azul = favorable. Sin matplotlib, y eligiendo el color del texto por\n",
        "# luminancia — background_gradient de pandas deja texto negro sobre\n",
        "# azul oscuro, que es ilegible.\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n",
        "_NARANJA, _NEUTRO, _AZUL = (194, 65, 12), (232, 228, 222), (3, 105, 161)\n",
        "\n",
        "def _mezcla(a, b, t):\n",
        "    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))\n",
        "\n",
        "def escala(v, vmin=-2.0, vmax=2.0):\n",
        "    \"\"\"Estilo CSS para un valor, divergente alrededor del punto medio.\"\"\"\n",
        "    if v is None or (isinstance(v, float) and not np.isfinite(v)):\n",
        "        return ''\n",
        "    t = min(1.0, max(0.0, (float(v) - vmin) / (vmax - vmin)))\n",
        "    rgb = (_mezcla(_NARANJA, _NEUTRO, t * 2) if t < 0.5\n",
        "           else _mezcla(_NEUTRO, _AZUL, (t - 0.5) * 2))\n",
        "    luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]\n",
        "    return f\"background-color:rgb{rgb};color:{'#1C1917' if luma > 140 "
        "else '#FFFFFF'}\"\n",
    ))

    # ------------------------------------------------------------ parameters
    cells.append(md(
        "## 2 · Parámetros\n",
        "\n",
        "`Universo completo` son ~600 nombres (S&P + Nasdaq-100 + Dow + ETFs "
        "curados) y tarda 1-3 min en bajar.\n",
    ))
    cells.append(code(
        "# @markdown ### Universo y ventana\n",
        'UNIVERSO = "Completo (S&P + Nasdaq + Dow + ETFs)"  '
        '# @param ["Completo (S&P + Nasdaq + Dow + ETFs)", "Solo acciones '
        '(S&P + Nasdaq + Dow)", "Solo ETFs", "Solo Nasdaq-100", "Solo Dow 30", '
        '"Lista personalizada"]\n',
        'TICKERS_PERSONALIZADOS = ""  # @param {type:"string"}\n',
        '# @markdown Separados por coma. Solo aplica si elegiste "Lista '
        'personalizada".\n',
        "\n",
        'BENCHMARK = "SPY"  # @param {type:"string"}\n',
        'PERIODO = "2y"  # @param ["1y", "2y", "5y"]\n',
        "TASA_LIBRE_RIESGO = 0.0425  # @param {type:\"number\"}\n",
        "\n",
        "# @markdown ### Datos opcionales (lentos)\n",
        'CON_VOL_IMPLICITA = False  # @param {type:"boolean"}\n',
        "# @markdown Baja la cadena de opciones para `iv_hv_spread`. ~2 "
        "requests por ticker.\n",
        'CON_NOMBRES_Y_SECTORES = False  # @param {type:"boolean"}\n',
        "# @markdown Necesario si usas lista personalizada: sin el nombre "
        "largo, el filtro de productos apalancados/inversos no puede actuar.\n",
        "\n",
        "from screener.yahoo_adapter import default_universe\n",
        "\n",
        "_GRUPOS = {\n",
        '    "Completo (S&P + Nasdaq + Dow + ETFs)": ("SP500", "NDX", "DJIA", "ETF"),\n',
        '    "Solo acciones (S&P + Nasdaq + Dow)": ("SP500", "NDX", "DJIA"),\n',
        '    "Solo ETFs": ("ETF",),\n',
        '    "Solo Nasdaq-100": ("NDX",),\n',
        '    "Solo Dow 30": ("DJIA",),\n',
        "}\n",
        "\n",
        'if UNIVERSO == "Lista personalizada":\n',
        "    TICKERS = [t.strip().upper().replace('.', '-')\n",
        "               for t in TICKERS_PERSONALIZADOS.split(',') if t.strip()]\n",
        "    if not TICKERS:\n",
        "        raise ValueError('Elegiste lista personalizada pero no pusiste "
        "tickers.')\n",
        "    if BENCHMARK.upper() not in TICKERS:\n",
        "        TICKERS.append(BENCHMARK.upper())\n",
        "    if not CON_NOMBRES_Y_SECTORES:\n",
        "        print('AVISO: sin nombres largos, un ETF apalancado o de "
        "covered-call\\n'\n",
        "              '       en tu lista pasaria el filtro de producto. "
        "Considera\\n'\n",
        "              '       activar CON_NOMBRES_Y_SECTORES.')\n",
        "else:\n",
        "    TICKERS = default_universe(_GRUPOS[UNIVERSO], benchmark=BENCHMARK)\n",
        "\n",
        "print(f'{len(TICKERS)} tickers  |  benchmark {BENCHMARK}  |  "
        "{PERIODO} de historia diaria')\n",
    ))

    # ------------------------------------------------------------- portfolio
    cells.append(md(
        "## 3 · Tu portafolio\n",
        "\n",
        "Alimenta el bloque **Portfolio Fit** (13% del modelo): correlación "
        "contra el libro actual, beneficio marginal de diversificación y "
        "solapamiento con lo que ya tienes.\n",
        "\n",
        "Viene precargado con el snapshot de IBKR del repo. Edítalo con tus "
        "posiciones reales — solo renta variable; efectivo y bonos van en "
        "`net_liquidation` pero no se correlacionan.\n",
    ))
    cells.append(code(
        "PORTAFOLIO = {\n",
        f'    "net_liquidation": {portfolio.get("net_liquidation", 0.0)},\n',
        '    "positions": [\n',
        *[line + "\n" for line in position_lines.split("\n")],
        "    ],\n",
        "}\n",
        "\n",
        "# Para correr sin libro (screening puro), descomenta:\n",
        '# PORTAFOLIO = {"net_liquidation": 1_000_000.0, "positions": []}\n',
        "\n",
        "_eq = sum(p['market_value'] for p in PORTAFOLIO['positions'])\n",
        "print(f\"{len(PORTAFOLIO['positions'])} posiciones de renta variable\")\n",
        "print(f\"valor neto     ${PORTAFOLIO['net_liquidation']:,.0f}\")\n",
        "print(f'expuesto       ${_eq:,.0f}  "
        "({_eq / PORTAFOLIO[\"net_liquidation\"]:.1%} del NLV)')\n",
    ))

    # ------------------------------------------------------------- download
    cells.append(md("## 4 · Bajar datos\n"))
    cells.append(code(
        "import time\n",
        "from screener.yahoo_adapter import fetch_market_data\n",
        "\n",
        "_t0 = time.time()\n",
        "market_data = fetch_market_data(\n",
        "    TICKERS,\n",
        "    benchmark=BENCHMARK,\n",
        "    risk_free_rate=TASA_LIBRE_RIESGO,\n",
        "    period=PERIODO,\n",
        "    with_metadata=CON_NOMBRES_Y_SECTORES,\n",
        "    with_iv=CON_VOL_IMPLICITA,\n",
        "    progress=True,\n",
        ")\n",
        "\n",
        "print(f'\\n{len(market_data[\"instruments\"])} instrumentos utilizables "
        "en {time.time() - _t0:.0f}s')\n",
        "\n",
        "_dropped = market_data.get('dropped', [])\n",
        "if _dropped:\n",
        "    print(f'\\n{len(_dropped)} descartados antes de puntuar:')\n",
        "    for _t, _r in _dropped[:15]:\n",
        "        print(f'  {_t:8s} {_r}')\n",
        "    if len(_dropped) > 15:\n",
        "        print(f'  ... y {len(_dropped) - 15} mas')\n",
    ))

    # ------------------------------------------------------------- coverage
    cells.append(md(
        "## 5 · Cobertura de métricas\n",
        "\n",
        "Léela antes del ranking. Una métrica con cobertura baja se está "
        "estandarizando contra una sección transversal chica mientras el resto "
        "del universo se puntúa sin ella.\n",
    ))
    cells.append(code(
        "from screener.yahoo_adapter import coverage_report\n",
        "\n",
        "_cov = coverage_report(market_data)\n",
        "_faltantes = _cov[_cov['coverage'] < 1.0]\n",
        "\n",
        "if _faltantes.empty:\n",
        "    print('Cobertura completa en las 28 metricas.')\n",
        "else:\n",
        "    print('Metricas por debajo de cobertura total:\\n')\n",
        "    for _, _r in _faltantes.iterrows():\n",
        "        print(f\"  {_r['coverage']:6.1%}  {_r['metric']:34s} "
        "({_r['block']}) — {_r['source']}\")\n",
        "\n",
        "(_cov.style\n",
        "    .format({'coverage': '{:.0%}'})\n",
        "    .map(lambda v: escala(v, 0.0, 1.0), subset=['coverage'])\n",
        "    .hide(axis='index'))\n",
    ))

    # ------------------------------------------------------------------ run
    cells.append(md("## 6 · Correr el modelo\n"))
    cells.append(code(
        "from screener.run_screen import run\n",
        "from screener.report import console_summary\n",
        "\n",
        "scored, meta = run(market_data, PORTAFOLIO, rf=TASA_LIBRE_RIESGO)\n",
        "print(console_summary(scored, meta))\n",
    ))

    # -------------------------------------------------------------- results
    cells.append(md(
        "## 7 · Ranking\n",
        "\n",
        "`indicative_weight` es tamaño por volatilidad inversa escalado por "
        "convicción, con topes duros — un punto de partida para dimensionar, "
        "no una orden.\n",
    ))
    cells.append(code(
        "BLOQUES = [b.key for b in FACTOR_MODEL]\n",
        "\n",
        "tabla = pd.DataFrame([{\n",
        "    'rank': i,\n",
        "    'ticker': r.ticker,\n",
        "    'tipo': r.asset_type,\n",
        "    'reco': r.recommendation,\n",
        "    'score': r.score_0_100,\n",
        "    'z': r.composite_z,\n",
        "    'peso_ind': r.indicative_weight,\n",
        "    'ret_1a': r.diagnostics.get('return_1y'),\n",
        "    'vol': r.diagnostics.get('volatility'),\n",
        "    'max_dd': r.diagnostics.get('max_drawdown'),\n",
        "    'beta': r.diagnostics.get('beta'),\n",
        "    'sharpe': r.raw_metrics.get('sharpe_1y'),\n",
        "    'corr_libro': r.raw_metrics.get('corr_to_portfolio'),\n",
        "    'gates': ', '.join(r.gates_triggered),\n",
        "} for i, r in enumerate(scored, 1)])\n",
        "\n",
        "PORCENTAJES = ['peso_ind', 'ret_1a', 'vol', 'max_dd']\n",
        "\n",
        "def pintar_reco(v):\n",
        "    return {\n",
        "        'OVERWEIGHT': 'background-color:#0369A1;color:white;font-weight:600',\n",
        "        'UNDERWEIGHT': 'background-color:#C2410C;color:white;font-weight:600',\n",
        "    }.get(v, 'color:#57534E')\n",
        "\n",
        "(tabla.head(40).style\n",
        "    .format({c: '{:.1%}' for c in PORCENTAJES} |\n",
        "            {'score': '{:.1f}', 'z': '{:+.2f}', 'beta': '{:.2f}',\n",
        "             'sharpe': '{:.2f}', 'corr_libro': '{:+.2f}'}, na_rep='—')\n",
        "    .map(pintar_reco, subset=['reco'])\n",
        "    .map(lambda v: escala(v, 20, 80), subset=['score'])\n",
        "    .hide(axis='index'))\n",
    ))

    # -------------------------------------------------------------- heatmap
    cells.append(md(
        "## 8 · Mapa de factores\n",
        "\n",
        "Dónde gana o pierde cada nombre. Un score compuesto alto sostenido por "
        "un solo bloque es frágil de una forma que el ranking no te muestra.\n",
    ))
    cells.append(code(
        "ETIQUETAS = {b.key: b.label for b in FACTOR_MODEL}\n",
        "\n",
        "mapa = pd.DataFrame(\n",
        "    [{'ticker': r.ticker, **{ETIQUETAS[k]: r.block_scores.get(k)\n",
        "                             for k in BLOQUES}}\n",
        "     for r in scored[:30]]\n",
        ").set_index('ticker')\n",
        "\n",
        "(mapa.style\n",
        "    .format('{:+.2f}', na_rep='—')\n",
        "    .map(escala)\n",
        "    .set_caption('Score z por bloque — azul favorable, naranja adverso'))\n",
    ))

    # --------------------------------------------------------------- detail
    cells.append(md("## 9 · Detalle de un nombre\n"))
    cells.append(code(
        'TICKER = "NVDA"  # @param {type:"string"}\n',
        "\n",
        "from screener.config import all_metrics\n",
        "\n",
        "_r = next((r for r in scored if r.ticker == TICKER.upper()), None)\n",
        "if _r is None:\n",
        "    _excluidos = dict(meta.get('excluded', []))\n",
        "    if TICKER.upper() in _excluidos:\n",
        "        print(f'{TICKER.upper()} fue excluido por filtros duros:')\n",
        "        for _m in _excluidos[TICKER.upper()]:\n",
        "            print(f'  - {_m}')\n",
        "    else:\n",
        "        print(f'{TICKER.upper()} no esta en el universo corrido.')\n",
        "else:\n",
        "    print(f'{_r.ticker} — {_r.name}')\n",
        "    print(f'{_r.recommendation}   score {_r.score_0_100:.1f}/100   "
        "z {_r.composite_z:+.2f}   peso indicativo {_r.indicative_weight:.2%}')\n",
        "    if _r.pre_gate_recommendation != _r.recommendation:\n",
        "        print(f'\\nDegradado desde {_r.pre_gate_recommendation} por:')\n",
        "        for _g in _r.gates_triggered:\n",
        "            print(f'  - {_g}')\n",
        "    if _r.duplicates:\n",
        "        print(f\"\\nExposicion duplicada: {', '.join(_r.duplicates)}\")\n",
        "\n",
        "    print('\\nBloques')\n",
        "    for _b in FACTOR_MODEL:\n",
        "        _s = _r.block_scores.get(_b.key)\n",
        "        _c = _r.block_coverage.get(_b.key, 0.0)\n",
        "        _bar = '#' * int(max(0, min(4, (_s or 0) + 2)) * 5)\n",
        "        print(f'  {_b.label:34s} {_s:+.2f}  cob {_c:4.0%}  {_bar}'\n",
        "              if _s is not None else f'  {_b.label:34s}    —')\n",
        "\n",
        "    print('\\nMetricas crudas')\n",
        "    _defs = all_metrics()\n",
        "    for _k, _v in _r.raw_metrics.items():\n",
        "        if _v is None or _k not in _defs:\n",
        "            continue\n",
        "        print(f'  {_defs[_k].label:36s} {_v:12.4f}   "
        "z {_r.metric_z.get(_k, float(\"nan\")):+.2f}')\n",
    ))

    # ---------------------------------------------------------------- export
    cells.append(md("## 10 · Exportar\n"))
    cells.append(code(
        "from screener.report import write_csv, write_markdown\n",
        "\n",
        "write_csv(scored, 'screen_results.csv')\n",
        "write_markdown(scored, meta, 'screen_report.md')\n",
        "\n",
        "try:\n",
        "    from google.colab import files\n",
        "    files.download('screen_results.csv')\n",
        "    files.download('screen_report.md')\n",
        "except ImportError:\n",
        "    print('Fuera de Colab: archivos escritos en el directorio actual.')\n",
    ))

    # ---------------------------------------------------------------- tuning
    cells.append(md(
        "## 11 · Cambiar el modelo\n",
        "\n",
        "Los pesos de bloque y los gates son juicios, no verdades. Cámbialos y "
        "vuelve a correr la celda 6 en adelante — no hace falta reiniciar el "
        "entorno.\n",
        "\n",
        "`set_block_weights` acepta tamaños relativos y renormaliza. Un bloque "
        "en `0.0` se sigue calculando y mostrando, pero no aporta al compuesto: "
        "es la forma limpia de preguntar *¿qué dice el modelo sin momentum?*\n",
    ))
    cells.append(code(
        "from screener.tuning import (block_weights, current_block_weights,\n",
        "                             override, reset_all, set_block_weights)\n",
        "\n",
        "# --- Ejemplo A: subir riesgo, bajar momentum ------------------------\n",
        "# set_block_weights({'momentum': 0.10, 'risk': 0.25})\n",
        "\n",
        "# --- Ejemplo B: quitar el techo de volatilidad para overweight ------\n",
        "# override('GATES', max_volatility_for_overweight=None)\n",
        "\n",
        "# --- Ejemplo C: bajar el minimo de liquidez a 5MM -------------------\n",
        "# override('ELIGIBILITY', min_adv_usd=5_000_000)\n",
        "\n",
        "# --- Ejemplo D: barrido de sensibilidad, sin efectos permanentes ----\n",
        "# for _peso in (0.0, 0.11, 0.22, 0.44):\n",
        "#     with block_weights({'momentum': _peso}):\n",
        "#         _s, _ = run(market_data, PORTAFOLIO, rf=TASA_LIBRE_RIESGO)\n",
        "#         _top = ', '.join(r.ticker for r in _s[:5])\n",
        "#         print(f'momentum {_peso:.0%} -> {_top}')\n",
        "\n",
        "# reset_all()   # vuelve a lo declarado en config.py\n",
        "\n",
        "for _k, _w in current_block_weights().items():\n",
        "    print(f'  {_w:6.1%}  {_k}')\n",
    ))

    return cells


def build_notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": build_cells(),
    }


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
