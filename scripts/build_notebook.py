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
    "profiles.py", "black_litterman.py", "cci_regulation.py",
    "optimizer.py",
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
    cells: list[dict] = []

    # ---------------------------------------------------------------- intro
    cells.append(md(
        "# Screening cuantitativo de acciones y ETFs\n",
        "\n",
        "Corre el modelo completo del repo sobre datos que se bajan en vivo de "
        "Yahoo Finance. Menú → **Entorno de ejecución → Ejecutar todo**.\n",
        "\n",
        "## Independiente de tu portafolio\n",
        "\n",
        "Este notebook **no lee ninguna cuenta**. Cada nombre se puntúa por sus "
        "propios méritos: el bloque *Portfolio Fit* está removido del modelo, "
        "no puesto en cero.\n",
        "\n",
        "La distinción importa. Pasar un libro vacío no habría bastado: con "
        "cero posiciones, `existing_overlap` sigue devolviendo `0.0` para cada "
        "nombre — un número real, idéntico en todos — que el motor "
        "estandarizaría y contaría como bloque poblado. Una cuenta vacía "
        "seguiría influyendo en el compuesto. Quitar el bloque es la única "
        "forma de que el screen sea de verdad independiente.\n",
        "\n",
        "## Perfil de riesgo\n",
        "\n",
        "Eliges **Conservador Defensivo**, **Conservador**, **Moderado** o "
        "**Agresivo** en Parámetros, y "
        "eso reconfigura cuatro cosas a la vez — no es una etiqueta sobre el "
        "mismo ranking:\n",
        "\n",
        "1. **Pesos de los bloques** — qué premia el score compuesto.\n",
        "2. **Umbrales de recomendación** — cuánto score exige un Overweight y "
        "qué tan poco basta para un Underweight. Asimétricos a propósito.\n",
        "3. **Gates de riesgo** — los techos duros que solo pueden degradar "
        "una recomendación.\n",
        "4. **Dimensionamiento y elegibilidad** — volatilidad objetivo, tope "
        "por posición y liquidez mínima para siquiera entrar al ranking.\n",
        "\n",
        "## Qué cambia al usar Yahoo en vez de IBKR\n",
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
        "%pip install -q yfinance openpyxl cvxpy scikit-learn\n",
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
        "from screener.profiles import PROFILES\n",
        "\n",
        "# Deliberadamente NO se importa FACTOR_MODEL aqui. El perfil lo\n",
        "# reemplaza mas abajo, y un nombre enlazado ahora quedaria obsoleto:\n",
        "# seguiria apuntando al modelo de 7 bloques con Portfolio Fit incluido.\n",
        "print(f'motor verificado  sha256={digest[:16]}...')\n",
        "print(f'perfiles disponibles: "
        "{\", \".join(p.label for p in PROFILES.values())}')\n",
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
        "# @markdown ### Perfil de riesgo\n",
        'PERFIL = "Moderado"  # @param ["Conservador Defensivo", "Conservador", "Moderado", "Agresivo"]\n',
        "# @markdown Cambia pesos de bloque, umbrales de recomendación, gates "
        "de riesgo, dimensionamiento y liquidez mínima — todo a la vez.\n",
        "TAMANO_POSICION_USD = 500000  # @param {type:\"number\"}\n",
        "# @markdown Tamaño de posición que asume el bloque de liquidez para "
        "calcular `days_to_liquidate`. Es un supuesto de dimensionamiento, no "
        "un dato de tu cuenta.\n",
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
        "\n",
        "from screener.profiles import get_profile\n",
        "\n",
        "perfil = get_profile(PERFIL)\n",
        "print()\n",
        "print(perfil.describe())\n",
    ))

    # ------------------------------------------------------------- download
    cells.append(md("## 3 · Bajar datos\n"))
    cells.append(code(
        "import time\n",
        "from screener.yahoo_adapter import fetch_market_data\n",
        "\n",
        "_t0 = time.time()\n",
        "market_data, frame_diario = fetch_market_data(\n",
        "    TICKERS,\n",
        "    benchmark=BENCHMARK,\n",
        "    risk_free_rate=TASA_LIBRE_RIESGO,\n",
        "    period=PERIODO,\n",
        "    with_metadata=CON_NOMBRES_Y_SECTORES,\n",
        "    with_iv=CON_VOL_IMPLICITA,\n",
        "    progress=True,\n",
        "    with_frame=True,   # el optimizador necesita retornos diarios\n",
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
        "## 4 · Cobertura de métricas\n",
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
    cells.append(md("## 5 · Correr el modelo\n"))
    cells.append(code(
        "from screener.run_screen import run_standalone\n",
        "from screener.report import console_summary\n",
        "\n",
        "# Sin libro: ninguna cuenta se lee y el bloque Portfolio Fit no esta\n",
        "# en el modelo. El perfil reconfigura pesos, umbrales, gates,\n",
        "# dimensionamiento y elegibilidad de una sola vez.\n",
        "scored, meta = run_standalone(\n",
        "    market_data,\n",
        "    profile=PERFIL,\n",
        "    position_usd=TAMANO_POSICION_USD,\n",
        "    rf=TASA_LIBRE_RIESGO,\n",
        ")\n",
        "print(console_summary(scored, meta))\n",
    ))

    # -------------------------------------------------------------- results
    cells.append(md(
        "## 6 · Ranking\n",
        "\n",
        "`indicative_weight` es tamaño por volatilidad inversa escalado por "
        "convicción, con topes duros — un punto de partida para dimensionar, "
        "no una orden.\n",
    ))
    cells.append(code(
        "import screener.config as _cfg\n",
        "\n",
        "# El modelo VIGENTE, ya con el perfil aplicado: seis bloques, sin\n",
        "# Portfolio Fit. Se lee aqui y no al importar, por la misma razon.\n",
        "MODELO = _cfg.FACTOR_MODEL\n",
        "BLOQUES = [b.key for b in MODELO]\n",
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
        "    # Sin libro no hay correlacion contra el libro. Se muestra alfa\n",
        "    # anualizado en su lugar, no una columna vacia.\n",
        "    'alpha': r.diagnostics.get('alpha_annual'),\n",
        "    'gates': ', '.join(r.gates_triggered),\n",
        "} for i, r in enumerate(scored, 1)])\n",
        "\n",
        "PORCENTAJES = ['peso_ind', 'ret_1a', 'vol', 'max_dd', 'alpha']\n",
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
        "             'sharpe': '{:.2f}'}, na_rep='—')\n",
        "    .map(pintar_reco, subset=['reco'])\n",
        "    .map(lambda v: escala(v, 20, 80), subset=['score'])\n",
        "    .hide(axis='index'))\n",
    ))

    # -------------------------------------------------------------- heatmap
    cells.append(md(
        "## 7 · Mapa de factores\n",
        "\n",
        "Dónde gana o pierde cada nombre. Un score compuesto alto sostenido por "
        "un solo bloque es frágil de una forma que el ranking no te muestra.\n",
    ))
    cells.append(code(
        "ETIQUETAS = {b.key: b.label for b in MODELO}\n",
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
    cells.append(md("## 8 · Detalle de un nombre\n"))
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
        "    for _b in MODELO:\n",
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

    # ------------------------------------------------------------- compare
    cells.append(md(
        "## 9 · Comparar los perfiles\n",
        "\n",
        "El mismo universo, los mismos datos, cuatro configuraciones. Un nombre "
        "que aparece Overweight en todas es una señal robusta; uno que solo "
        "sobrevive en Agresivo te está diciendo que su score depende de que le "
        "perdones la volatilidad.\n",
        "\n",
        "**`n/e` no es un error.** Cada perfil tiene su propio piso de liquidez "
        "($100MM / $50MM / $20MM / $10MM de volumen diario), así que un nombre "
        "puede ser elegible para uno y no para otro. Cuando eso pasa, el más "
        "estricto lo marca como no elegible y te dice por qué.\n",
    ))
    cells.append(code(
        "from screener.profiles import PROFILES\n",
        "from screener.tuning import reset_all\n",
        "\n",
        "_recos, _excluidos = {}, {}\n",
        "try:\n",
        "    for _k, _p in PROFILES.items():\n",
        "        _s, _m = run_standalone(market_data, profile=_k,\n",
        "                                position_usd=TAMANO_POSICION_USD,\n",
        "                                rf=TASA_LIBRE_RIESGO)\n",
        "        _recos[_p.label] = {r.ticker: r.recommendation for r in _s}\n",
        "        _excluidos[_p.label] = dict(_m.get('excluded', []))\n",
        "finally:\n",
        "    # Deja el modelo como lo espera el resto del notebook.\n",
        "    reset_all()\n",
        "    scored, meta = run_standalone(market_data, profile=PERFIL,\n",
        "                                  position_usd=TAMANO_POSICION_USD,\n",
        "                                  rf=TASA_LIBRE_RIESGO)\n",
        "\n",
        "NO_ELEGIBLE = 'NO ELEGIBLE'\n",
        "_tickers = [r.ticker for r in scored]\n",
        "\n",
        "# Cada perfil filtra por liquidez distinto, asi que no todos puntuan\n",
        "# el mismo conjunto de nombres. Indexar a ciegas aqui reventaria con\n",
        "# KeyError en cuanto un perfil excluya algo que otro si acepto.\n",
        "comparacion = pd.DataFrame({\n",
        "    _label: pd.Series({t: _r.get(t, NO_ELEGIBLE) for t in _tickers})\n",
        "    for _label, _r in _recos.items()\n",
        "})\n",
        "comparacion.insert(0, 'score_' + PERFIL.lower(),\n",
        "                   pd.Series({r.ticker: r.score_0_100 for r in scored}))\n",
        "\n",
        "_ABREV = {'OVERWEIGHT': 'OW', 'MARKET WEIGHT': 'MW',\n",
        "          'UNDERWEIGHT': 'UW', NO_ELEGIBLE: 'n/e'}\n",
        "_TONO = {'OVERWEIGHT': 1.6, 'MARKET WEIGHT': 0.0, 'UNDERWEIGHT': -1.6}\n",
        "_PERFILES = [p.label for p in PROFILES.values()]\n",
        "\n",
        "def _estilo_reco(v):\n",
        "    # No elegible es una categoria aparte, no un punto de la escala.\n",
        "    if v == NO_ELEGIBLE:\n",
        "        return 'background-color:#F5F5F4;color:#A8A29E;font-style:italic'\n",
        "    return escala(_TONO.get(v, 0.0))\n",
        "\n",
        "_ow = comparacion[_PERFILES].eq('OVERWEIGHT').sum(axis=1)\n",
        "print(f'Overweight en TODOS los perfiles: "
        "{list(comparacion.index[_ow == len(_PERFILES)]) or \"ninguno\"}')\n",
        "print(f'Overweight solo en Agresivo:     '\n",
        "      f'{list(comparacion.index[(_ow == 1) & "
        "comparacion[\"Agresivo\"].eq(\"OVERWEIGHT\")]) or \"ninguno\"}')\n",
        "\n",
        "for _label in _PERFILES:\n",
        "    _fuera = [t for t in _tickers if _recos[_label].get(t) is None]\n",
        "    if _fuera:\n",
        "        print(f'\\n{_label} no considera {len(_fuera)} de estos nombres:')\n",
        "        for _t in _fuera[:8]:\n",
        "            _razon = (_excluidos[_label].get(_t) or ['fuera del universo'])[0]\n",
        "            print(f'  {_t:8s} {_razon}')\n",
        "\n",
        "(comparacion.head(30).style\n",
        "    .format({comparacion.columns[0]: '{:.1f}'})\n",
        "    .format(lambda v: _ABREV.get(v, v), subset=_PERFILES)\n",
        "    .map(_estilo_reco, subset=_PERFILES)\n",
        "    .map(lambda v: escala(v, 20, 80), subset=[comparacion.columns[0]])\n",
        "    .set_caption('Recomendación por perfil'))\n",
    ))

    # -------------------------------------------------------- black-litterman
    cells.append(md(
        "## 10 · Views para Black-Litterman (CCI)\n",
        "\n",
        "Los dos sistemas son complementarios y la frontera es nítida: **el "
        "screener decide sobre qué nombres hay una view y cuán fuerte es; "
        "Black-Litterman decide los pesos.**\n",
        "\n",
        "Esta celda exporta los insumos tácticos — `Q` y convicción — en el "
        "esquema exacto que ya consumen `flujo_aprobacion` y "
        "`black_litterman_core` de tu notebook de CCI. No exporta pesos: bajo "
        "Black-Litterman los pesos salen del optimizador sujeto al "
        "Procedimiento de Inversión, y mandar un segundo juego de pesos sin "
        "restricciones al lado invita justo la confusión que una revisión de "
        "riesgo model existe para evitar.\n",
        "\n",
        "### Cómo se traduce un ranking a un retorno esperado\n",
        "\n",
        "Un z-score transversal es un **ranking**, no un pronóstico. La "
        "conversión es explícita:\n",
        "\n",
        "$$Q_i = IC \\times z_i \\times \\sigma_i$$\n",
        "\n",
        "Escalado por riesgo (a igual ranking, el nombre más volátil merece "
        "mayor retorno esperado, que es lo que el optimizador media-varianza "
        "necesita para dimensionar bien) y centrado (un nombre en el medio de "
        "la sección transversal da exactamente cero).\n",
        "\n",
        "**El IC es un supuesto declarado, no una estimación.** Es la "
        "correlación asumida entre el ranking del screener y los retornos "
        "realizados. El 0.08 por defecto es deliberadamente modesto y produce "
        "views dentro de la banda ±5% de tu documento técnico. No está "
        "calibrado contra ningún backtest.\n",
        "\n",
        "La convicción es otra cosa: alimenta Ω y mide **confianza en la "
        "estimación** — cuántos de los seis bloques coinciden en signo, cuánta "
        "cobertura de datos hubo, si se activó un gate. Un nombre en z=+1.5 "
        "sostenido por un solo bloque no merece la misma Ω que uno donde los "
        "seis coinciden.\n",
    ))
    cells.append(code(
        "from screener.black_litterman import (ViewParams, build_basket,\n",
        "                                      build_views, write_views)\n",
        "from screener.profiles import CCI_STRATEGIES, profile_for_strategy\n",
        "\n",
        "# @markdown Estrategia de destino en el sistema BL de CCI.\n",
        'ESTRATEGIA_CCI = "Moderado"  # @param ["Conservador_Defensivo", '
        '"Conservador", "Moderado", "Agresivo"]\n',
        "IC_SUPUESTO = 0.08  # @param {type:\"number\"}\n",
        "MAX_VIEWS = 8  # @param {type:\"integer\"}\n",
        "\n",
        "# Equivale a la columna activo_referencia de tu Google Sheet: empareja\n",
        "# una accion con el ETF contra el que debe medirse. Un nombre con\n",
        "# referencia produce una view RELATIVA; el resto, ABSOLUTA.\n",
        "REFERENCIAS = {\n",
        "    'AAPL': 'QQQ', 'MSFT': 'QQQ', 'NVDA': 'QQQ', 'AVGO': 'SMH',\n",
        "    'JPM': 'XLF', 'BAC': 'XLF', 'LLY': 'XLV', 'UNH': 'XLV',\n",
        "    'XOM': 'XLE', 'CVX': 'XLE',\n",
        "}\n",
        "\n",
        "_perfil_cci = profile_for_strategy(ESTRATEGIA_CCI)\n",
        "if _perfil_cci.key != perfil.key:\n",
        "    print(f'AVISO: corriste el screen con perfil {perfil.label} pero vas '\n",
        "          f'a exportar para {ESTRATEGIA_CCI}, que corresponde a '\n",
        "          f'{_perfil_cci.label}.')\n",
        "    print('       Vuelve a la celda de Parametros y alinea ambos, o las '\n",
        "          'views\\n       llevaran umbrales y gates de otro mandato.')\n",
        "\n",
        "_params = ViewParams(information_coefficient=IC_SUPUESTO,\n",
        "                     max_views=MAX_VIEWS)\n",
        "\n",
        "views = build_views(scored, market_data, strategy=ESTRATEGIA_CCI,\n",
        "                    reference_map=REFERENCIAS, params=_params)\n",
        "cesta = build_basket(scored, strategy=ESTRATEGIA_CCI,\n",
        "                     reference_map=REFERENCIAS)\n",
        "\n",
        "print(f'{len(views)} views para {ESTRATEGIA_CCI} '\n",
        "      f'(perfil {_perfil_cci.label}, IC {IC_SUPUESTO})\\n')\n",
        "for _v in views:\n",
        "    _quien = (_v['activo'] if _v['tipo'] == 'absoluto'\n",
        "              else f\"{_v['activo_long']} / {_v['activo_short']}\")\n",
        "    print(f\"  {_v['tipo']:9s} {_quien:18s} Q {_v['Q']:+.2%}   \"\n",
        "          f\"convicción {_v['conviccion']:.2f}\")\n",
        "\n",
        "views_df = pd.DataFrame(views)\n",
        "cesta_df = pd.DataFrame(cesta)\n",
        "views_df\n",
    ))
    # ------------------------------------------------------------- optimizer
    cells.append(md(
        "## 11 · Cartera Black-Litterman\n",
        "\n",
        "Aquí no hay archivo de por medio: `views` es una variable de Python "
        "que la celda anterior dejó en memoria, y esta la consume directo.\n",
        "\n",
        "El equilibrio de mercado (π) sale de capitalización real vía "
        "optimización inversa, la covarianza usa contracción Ledoit-Wolf sobre "
        "retornos **diarios** — con ~52 barras semanales y más de 52 nombres la "
        "matriz sería singular — y la optimización respeta las bandas del "
        "Procedimiento de Inversión.\n",
        "\n",
        "### Tres arreglos frente al sistema original\n",
        "\n",
        "1. **Solver.** Tu código pedía ECOS, que no viene en Colab; tu corrida "
        "guardada murió ahí sin producir cartera. Este usa CLARABEL, que viene "
        "con CVXPY.\n",
        "2. **Apalancamiento.** `leverage_max` de 1.25 y 1.50 estaba declarado "
        "pero el optimizador fijaba `sum(w) == 1`. Ahora es un presupuesto real, "
        "con el buffer de 95% que dice tu documento.\n",
        "3. **La auditoría ahora puede fallar.** `auditar_bandas` escribía "
        "\"Auditoría OK\" sin comparar nada. Esta compara contra cada límite y "
        "reporta lo que se rompe.\n",
        "\n",
        "**Un aviso:** tus bandas no tienen clase para materias primas, y tu "
        "optimizador solo restringe las clases que aparecen en `bandas` — oro "
        "podía tomar el libro entero. Le puse un techo por perfil, pero **ese "
        "número lo inventé yo**, no sale de tu Procedimiento de Inversión. "
        "Confírmalo con Compliance antes de operar con esto.\n",
    ))
    cells.append(code(
        "# @markdown Cuántos nombres del ranking entran a la optimización.\n",
        "TOP_N_CARTERA = 25  # @param {type:\"integer\"}\n",
        "# @markdown Menos nombres = covarianza mejor estimada; más = más "
        "diversificación.\n",
        "\n",
        "from screener.optimizer import (implied_equilibrium, market_weights,\n",
        "                               optimize, posterior, shrunk_covariance,\n",
        "                               allocation_table, select_basket)\n",
        "from screener.cci_regulation import classify_for_bands\n",
        "from screener.yahoo_adapter import daily_returns, fetch_market_caps\n",
        "\n",
        "tipos_todos = {r.ticker: r.asset_type for r in scored}\n",
        "\n",
        "# La cesta no puede ser solo el top-N por score. El ranking premia\n",
        "# momentum y riesgo-retorno, donde la renta variable domina, y con una\n",
        "# cesta 100% equity el techo del mandato (60% en Moderado) queda por\n",
        "# debajo del libro invertido: el solver responde 'infactible' y la\n",
        "# cartera sale vacia. select_basket asegura representacion de cada\n",
        "# clase disponible en el universo.\n",
        "cartera_tickers = select_basket(scored, ESTRATEGIA_CCI,\n",
        "                                top_n=TOP_N_CARTERA, min_per_class=3)\n",
        "\n",
        "_clases_cesta = sorted({classify_for_bands(t, tipos_todos.get(t, 'ETF'))\n",
        "                        for t in cartera_tickers})\n",
        "print(f'Cesta: {len(cartera_tickers)} nombres en {len(_clases_cesta)} clases')\n",
        "print(f'  {\", \".join(_clases_cesta)}\\n')\n",
        "\n",
        "retornos = daily_returns(frame_diario, cartera_tickers)\n",
        "covarianza = shrunk_covariance(retornos)\n",
        "\n",
        "capitalizaciones = fetch_market_caps(list(covarianza.columns))\n",
        "pesos_mkt, sin_cap = market_weights(capitalizaciones,\n",
        "                                    list(covarianza.columns))\n",
        "if sin_cap:\n",
        "    print(f'Sin capitalizacion, excluidos del equilibrio: {sin_cap}')\n",
        "\n",
        "pi = implied_equilibrium(pesos_mkt, covarianza)\n",
        "er_posterior, cov_posterior = posterior(pi, covarianza, views)\n",
        "\n",
        "tipos = tipos_todos\n",
        "clases = {t: classify_for_bands(t, tipos.get(t, 'ETF'))\n",
        "          for t in covarianza.columns}\n",
        "\n",
        "cartera = optimize(er_posterior, cov_posterior, tipos, ESTRATEGIA_CCI)\n",
        "\n",
        "print(f'{ESTRATEGIA_CCI}  |  estado: {cartera.status}')\n",
        "print(f'Exposicion bruta   {cartera.gross_exposure:.1%}')\n",
        "print(f'Retorno esperado   {cartera.expected_return:+.2%} anual')\n",
        "print(f'Volatilidad        {cartera.volatility:.1%} anual')\n",
        "print(f'Posiciones         {int((cartera.weights > 0).sum())}')\n",
        "\n",
        "print('\\nPor clase de activo')\n",
        "for _clase, _peso in cartera.by_class.items():\n",
        "    if _peso > 0.0001:\n",
        "        print(f'  {_peso:7.2%}  {_clase}')\n",
        "\n",
        "if cartera.breaches:\n",
        "    print('\\nAUDITORIA — INCUMPLIMIENTOS:')\n",
        "    for _b in cartera.breaches:\n",
        "        print(f'  {_b}')\n",
        "else:\n",
        "    print('\\nAuditoria de bandas: sin incumplimientos.')\n",
        "for _n in cartera.notes:\n",
        "    print(f'NOTA: {_n}')\n",
        "\n",
        "cartera_df = allocation_table(cartera, classes=clases)\n",
        "\n",
        "if cartera_df.empty:\n",
        "    # Una hoja vacia no dice nada. El motivo viaja con el resultado.\n",
        "    cartera_df = pd.DataFrame({\n",
        "        'ticker': ['SIN CARTERA'],\n",
        "        'nombre': [f'La optimizacion no encontro solucion "
        "({cartera.status})'],\n",
        "        'clase_activo': [' | '.join(cartera.breaches) or 'sin detalle'],\n",
        "        'peso': [0.0],\n",
        "    })\n",
        "    print('\\nNO HAY CARTERA. Motivo:')\n",
        "    for _b in cartera.breaches:\n",
        "        print(f'  {_b}')\n",
        "\n",
        "(cartera_df.style\n",
        "    .format({'peso': '{:.2%}'})\n",
        "    .map(lambda v: escala(v, 0, 0.12), subset=['peso'])\n",
        "    .hide(axis='index')\n",
        "    .set_caption(f'Cartera optimizada — {ESTRATEGIA_CCI}'))\n",
    ))

    # ---------------------------------------------------------------- export
    cells.append(md(
        "## 12 · Descargar\n",
        "\n",
        "**El Excel es para ti.** Ocho hojas: ranking, scores por bloque, "
        "comparación de perfiles, las views con su justificación, la cartera "
        "optimizada, la cesta, la cobertura de métricas y los parámetros de la "
        "corrida.\n",
        "\n",
        "**El JSON es para tu sistema Black-Litterman**, no para leerlo. "
        "`black_litterman_core` hace `json.load()` y espera diccionarios de "
        "estructura heterogénea — una view absoluta trae `activo`, una relativa "
        "trae `activo_long` y `activo_short` — que en una tabla plana obligarían "
        "a celdas vacías. Y Excel coacciona tipos: una convicción de `0.85` "
        "puede volver como texto o mostrarse como 85%, y ese número entra "
        "directo en Ω. El nombre del archivo sigue la convención que tu propio "
        "`flujo_aprobacion` ya escribe en Drive.\n",
        "\n",
        "Si no vas a alimentar el modelo BL hoy, desmarca la casilla y bájate "
        "solo el Excel.\n",
        "\n",
        "### Dónde cae el archivo\n",
        "\n",
        "En `CCI_BlackLitterman/propuestas/`, **nunca** en `aprobadas/`. "
        "Esa carpeta guarda las views que ya revisaste y justificaste, y tu "
        "`flujo_aprobacion` escribe ahí un archivo de la misma forma. Un "
        "archivo sin aprobar cayendo en esa ruta reemplazaría una decisión "
        "firmada por salida de máquina, sin dejar rastro. `write_views` se "
        "niega a escribir bajo `aprobadas/` aunque se lo pidas.\n",
        "\n",
        "### Del lado de tu notebook BL\n",
        "\n",
        "En el repo está `snippets/cci_bl_cargar_propuestas.py`: una celda "
        "para pegar entre `generar_propuestas_views` y `flujo_aprobacion`. "
        "Lee el archivo más reciente, avisa si está viejo, y fusiona con las "
        "propuestas de tu propio motor resolviendo duplicados por convicción "
        "— un mismo activo propuesto por ambas fuentes serían dos filas casi "
        "idénticas de P, lo que estrecha Ω artificialmente y le da a esa "
        "apuesta un peso que ninguna de las dos fuentes justifica sola.\n",
        "\n",
        "El gestor sigue viendo cada view y decidiendo. Nada se aplica sin "
        "tu aprobación.\n",
    ))
    cells.append(code(
        "EXPORTAR_JSON_PARA_BL = True  # @param {type:\"boolean\"}\n",
        "# @markdown Desmárcalo si solo quieres el Excel.\n",
        "GUARDAR_EN_DRIVE = False  # @param {type:\"boolean\"}\n",
        "# @markdown Escribe las propuestas directo en "
        "`CCI_BlackLitterman/propuestas/` de tu Drive, para que el "
        "notebook BL las encuentre sin descargar ni subir nada.\n",
        "\n",
        "from pathlib import Path\n",
        "\n",
        "from screener.black_litterman import default_views_filename\n",
        "\n",
        "ARCHIVO_EXCEL = 'screening.xlsx'\n",
        "ARCHIVO_VIEWS = default_views_filename(ESTRATEGIA_CCI)\n",
        "\n",
        "parametros = pd.DataFrame([\n",
        "    ('Generado (UTC)', pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')),\n",
        "    ('Perfil', perfil.label),\n",
        "    ('Perfil — resumen', perfil.summary),\n",
        "    ('Estrategia CCI destino', ESTRATEGIA_CCI),\n",
        "    ('Universo', UNIVERSO),\n",
        "    ('Nombres puntuados', len(scored)),\n",
        "    ('Benchmark', BENCHMARK),\n",
        "    ('Historia', PERIODO),\n",
        "    ('Tasa libre de riesgo', f'{TASA_LIBRE_RIESGO:.2%}'),\n",
        "    ('Posición asumida (liquidez)', f'${TAMANO_POSICION_USD:,.0f}'),\n",
        "    ('Fuente de datos', market_data['data_source']),\n",
        "    ('Portafolio', 'ninguno — screen independiente'),\n",
        "    ('IC supuesto (views)', IC_SUPUESTO),\n",
        "    ('Nota sobre el IC', 'supuesto declarado, no calibrado contra backtest'),\n",
        "    ('Estado de la optimizacion', cartera.status),\n",
        "    ('Exposicion bruta', f'{cartera.gross_exposure:.2%}'),\n",
        "    ('Auditoria de bandas',\n",
        "     'sin incumplimientos' if not cartera.breaches\n",
        "     else ' | '.join(cartera.breaches)),\n",
        "    ('Umbral Overweight', f'z >= {perfil.bands.overweight_z:+.2f}'),\n",
        "    ('Umbral Underweight', f'z <= {perfil.bands.underweight_z:+.2f}'),\n",
        "    ('Techo de volatilidad para OW',\n",
        "     f'{perfil.gates.max_volatility_for_overweight:.0%}'),\n",
        "    ('Beta máxima', f'{perfil.gates.beta_limit:.2f}'),\n",
        "    ('Peso máximo por posición', f'{perfil.sizing.max_weight:.1%}'),\n",
        "    ('Volumen diario mínimo', f'${perfil.eligibility.min_adv_usd/1e6:,.0f}MM'),\n",
        "] + [(f'Peso — {b.label}', f'{b.weight:.0%}') for b in MODELO],\n",
        "    columns=['Parámetro', 'Valor'])\n",
        "\n",
        "# Las views en formato legible: una fila por view, con las dos formas\n",
        "# (absoluta y relativa) resueltas a columnas explicitas.\n",
        "views_excel = pd.DataFrame([{\n",
        "    'tipo': v['tipo'],\n",
        "    'activo': v.get('activo', ''),\n",
        "    'long': v.get('activo_long', ''),\n",
        "    'short': v.get('activo_short', ''),\n",
        "    'Q': v['Q'],\n",
        "    'conviccion': v['conviccion'],\n",
        "    'justificacion': v['justificacion'],\n",
        "} for v in views])\n",
        "\n",
        "with pd.ExcelWriter(ARCHIVO_EXCEL, engine='openpyxl') as _xl:\n",
        "    tabla.to_excel(_xl, sheet_name='Ranking', index=False)\n",
        "    mapa.to_excel(_xl, sheet_name='Bloques')\n",
        "    comparacion.to_excel(_xl, sheet_name='Perfiles')\n",
        "    views_excel.to_excel(_xl, sheet_name='Views BL', index=False)\n",
        "    cartera_df.to_excel(_xl, sheet_name='Cartera', index=False)\n",
        "    cesta_df.to_excel(_xl, sheet_name='Cesta', index=False)\n",
        "    _cov.to_excel(_xl, sheet_name='Cobertura', index=False)\n",
        "    parametros.to_excel(_xl, sheet_name='Parametros', index=False)\n",
        "\n",
        "    for _hoja in _xl.book.worksheets:\n",
        "        _hoja.freeze_panes = 'A2'\n",
        "        for _col in _hoja.columns:\n",
        "            _ancho = max((len(str(c.value)) for c in _col if c.value), default=8)\n",
        "            _hoja.column_dimensions[_col[0].column_letter].width = min(46, _ancho + 3)\n",
        "\n",
        "print(f'{ARCHIVO_EXCEL}  —  {len(scored)} nombres, 8 hojas')\n",
        "\n",
        "if EXPORTAR_JSON_PARA_BL:\n",
        "    write_views(views, ARCHIVO_VIEWS, strategy=ESTRATEGIA_CCI,\n",
        "                profile=_perfil_cci, meta=meta, params=_params)\n",
        "    print(f'{ARCHIVO_VIEWS}  —  {len(views)} propuestas')\n",
        "\n",
        "if EXPORTAR_JSON_PARA_BL and GUARDAR_EN_DRIVE:\n",
        "    from screener.black_litterman import DRIVE_PROPOSALS_DIR\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    _destino = (Path('/content/drive/MyDrive/CCI_BlackLitterman')\n",
        "                / DRIVE_PROPOSALS_DIR / ARCHIVO_VIEWS)\n",
        "    write_views(views, _destino, strategy=ESTRATEGIA_CCI,\n",
        "                profile=_perfil_cci, meta=meta, params=_params)\n",
        "    print(f'Guardado en Drive: {_destino}')\n",
        "\n",
        "try:\n",
        "    from google.colab import files\n",
        "    files.download(ARCHIVO_EXCEL)\n",
        "    if EXPORTAR_JSON_PARA_BL:\n",
        "        files.download(ARCHIVO_VIEWS)\n",
        "except ImportError:\n",
        "    print('Fuera de Colab: los archivos quedaron en el directorio actual.')\n",
    ))

    # ---------------------------------------------------------------- tuning
    cells.append(md(
        "## 13 · Ajuste fino del modelo\n",
        "\n",
        "Los tres perfiles ya cubren la mayoría de los casos. Esto es para "
        "cuando quieras algo que ningún perfil expresa — mueve los pesos y "
        "vuelve a correr desde la celda 5, sin reiniciar el entorno.\n",
        "\n",
        "**Ojo con el orden:** `run_standalone` vuelve a aplicar el perfil en "
        "cada llamada, así que sobrescribe lo que pongas aquí. Para que un "
        "ajuste manual sobreviva, usa `run(market_data, {}, standalone=True, "
        "target_position_usd=TAMANO_POSICION_USD)` en lugar de "
        "`run_standalone`.\n",
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
        "#         _s, _ = run(market_data, {}, standalone=True,\n",
        "#                     target_position_usd=TAMANO_POSICION_USD,\n",
        "#                     rf=TASA_LIBRE_RIESGO)\n",
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
