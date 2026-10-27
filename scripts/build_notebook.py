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
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "screener_colab.ipynb"

#: Package modules embedded in the notebook. Ordered for readability of the
#: printed manifest only -- import order is resolved by Python.
def modules() -> tuple[str, ...]:
    """
    Todo ``screener/*.py``, en orden fijo.

    Era una lista escrita a mano y eso es una trampa que ya se disparó: agregar
    un módulo al paquete no lo mete en el tarball, el cuaderno se construye sin
    quejarse, y el error aparece en Colab como un ``ModuleNotFoundError`` a
    mitad de la corrida. El directorio ya sabe qué hay adentro; preguntarle es
    más barato que acordarse.
    """
    return tuple(sorted(p.name for p in (ROOT / "screener").glob("*.py")))


MODULES = modules()


def build_payload() -> tuple[str, str]:
    """Return ``(base64_tarball, sha256)`` of the screener package."""
    # El empaquetado vive en el motor (screener.corrida) para que el script de
    # línea de comandos calcule la misma huella y la registre en su corrida.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from screener.corrida import empaquetar_motor

    raw = empaquetar_motor(ROOT / "screener")
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
        "| Universo | 21 nombres del snapshot | 447 candidatos, o el que definas |\n",
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
        "`Universo completo` son 447 candidatos (317 acciones del S&P + "
        "Nasdaq-100 + Dow, sin duplicar, y 130 ETFs curados) y tarda 1-3 min "
        "en bajar. De ahí, la política de selección decide cuáles se evalúan.\n",
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
        "RF_DE_MERCADO = True  # @param {type:\"boolean\"}\n",
        "# @markdown Bajar la letra del Tesoro a 13 semanas (`^IRX`) y anotar su "
        "fecha. Si falla, se usa la tasa de arriba y la hoja de Parámetros lo "
        "dice. Una tasa tecleada sin fecha no se puede auditar.\n",
        "_rf_fuente = f'manual {TASA_LIBRE_RIESGO:.2%}, sin fecha de mercado'\n",
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
        "# Renta fija, efectivo y los ETFs nucleo entran siempre. El universo\n",
        "# dice donde buscar ideas; no puede quitarle al mandato las clases que\n",
        "# exige su politica. Con 'Solo acciones' la cesta quedaba 100% renta\n",
        "# variable y ningun mandato tenia solucion.\n",
        "from screener.yahoo_adapter import con_nucleo\n",
        "TICKERS, _agregados_nucleo = con_nucleo(TICKERS)\n",
        "if _agregados_nucleo:\n",
        "    print(f'Nucleo agregado al universo: {len(_agregados_nucleo)} '\n",
        "          f'vehiculo(s) de renta fija, efectivo y renta variable indexada '\n",
        "          f'({\", \".join(_agregados_nucleo[:8])}'\n",
        "          f'{\" ...\" if len(_agregados_nucleo) > 8 else \"\"}).')\n",
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
        "from screener.yahoo_adapter import fetch_market_data, fetch_risk_free\n",
        "\n",
        "if RF_DE_MERCADO:\n",
        "    TASA_LIBRE_RIESGO, _rf_fuente = fetch_risk_free(TASA_LIBRE_RIESGO)\n",
        "print(f'Tasa libre de riesgo {TASA_LIBRE_RIESGO:.2%}  ({_rf_fuente})')\n",
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

    # ------------------------------------------------------ fundamentales
    cells.append(md(
        "## 3b · Fundamentales (SEC EDGAR, point-in-time)\n",
        "\n",
        "**Esta celda se abastece sola.** Baja de EDGAR lo que le falte al "
        "almacén y refresca lo vencido; no hay que correr otro cuaderno antes. "
        "Es incremental: un nombre con archivo en disco no se vuelve a pedir, "
        "así que solo la primera vez cuesta minutos. Las siguientes, "
        "segundos.\n",
        "\n",
        "Dos cosas la condicionan. **`CONTACTO_SEC` no es opcional para "
        "bajar**: la SEC exige un User-Agent con correo real y bloquea por IP "
        "a quien no se identifica. Sin él la celda no sale a la red y se "
        "conforma con lo que ya haya. Y **el almacén tiene que vivir en "
        "Drive**: el disco de Colab desaparece al reciclarse el runtime, y con "
        "el almacén fuera de Drive cada sesión volvería a bajar varios GB.\n",
        "\n",
        "Los ETF no entran: no tienen estados financieros, así que pedírselos "
        "a EDGAR no es un dato que falte sino un error de categoría.\n",
        "\n",
        "**Cada ratio casa un fundamental con el precio del mismo día.** El "
        "precio sale del `market_data` que acabas de bajar y el fundamental "
        "de lo que estaba presentado a esa fecha, así que no hay forma de "
        "casar el balance de un año con la cotización de otro por descuido.\n",
        "\n",
        "**Todos los ratios de valuación son rendimientos, no múltiplos**, y "
        "eso no es una preferencia de presentación. Un P/E se rompe en el "
        "cero: una empresa que gana un centavo por acción a \\$100 cotiza a "
        "10.000x, y si pierde un centavo cotiza a −10.000x — que en un "
        "ranking de «P/E bajo es mejor» queda **primero**, por delante de "
        "cualquier empresa sana. El rendimiento de utilidades ordena bien "
        "atravesando el cero. Los múltiplos de siempre se calculan igual, "
        "para leer, y salen vacíos donde el rendimiento no es positivo.\n",
        "\n",
        "Los ratios de **calidad** — ROE, márgenes, devengos, apalancamiento "
        "— se calculan y se reportan, pero todavía **no puntúan**: ponerlos a "
        "puntuar exige decidir su peso, y un peso es una decisión del Comité.\n",
    ))
    cells.append(code(
        'ALMACEN_FUNDAMENTALES = "/content/drive/MyDrive/fundamentales"  '
        '# @param {type:"string"}\n',
        '# @markdown Vacío = sin bloque fundamental.\n',
        'CONTACTO_SEC = "CCI Puesto de Bolsa tucorreo@dominio.com"  '
        '# @param {type:"string"}\n',
        "# @markdown Obligatorio para bajar. Sin correo real la SEC bloquea\n",
        "# @markdown por IP. Vacio = usar solo lo que ya este en el almacen.\n",
        "DESCARGAR_FUNDAMENTALES = True  # @param {type:\"boolean\"}\n",
        "REFRESCAR_DIAS = 30  # @param {type:\"integer\"}\n",
        "# @markdown Rebajar un nombre cuyo archivo tenga mas dias que esto.\n",
        "MAX_REFRESCOS = 40  # @param {type:\"integer\"}\n",
        "# @markdown Tope por corrida, del mas viejo al mas nuevo, para que un\n",
        "# @markdown vencimiento masivo no vuelva la corrida una descarga de GB.\n",
        'FECHA_FUNDAMENTALES = ""  # @param {type:"date"}\n',
        "EXIGIR_FUNDAMENTALES = True  # @param {type:\"boolean\"}\n",
        "# @markdown Detener la corrida si el bloque fundamental no carga. Sin\n",
        "# @markdown esto una corrida sin fundamentales se ve como una normal.\n",
        "# @markdown Reconstruir lo que se sabia ese dia. Vacio = todo lo\n",
        "# @markdown conocido hoy, que es lo que quiere una corrida normal.\n",
        "\n",
        "from pathlib import Path\n",
        "\n",
        "from screener import fundamentales as fx\n",
        "from screener.descarga import sincronizar\n",
        "from screener.edgar import detalle_sin_cik, leer_hechos\n",
        "from screener.yahoo_adapter import classify\n",
        "\n",
        "fund_meta = {}\n",
        "_ruta = Path(ALMACEN_FUNDAMENTALES) if ALMACEN_FUNDAMENTALES else None\n",
        "# Drive se monta AQUI, no en la seccion 12. Ese fue el error: el\n",
        "# almacen apunta a Drive por defecto, el montaje ocurria al exportar —\n",
        "# nueve secciones despues — y para cuando esta celda miraba, la ruta no\n",
        "# existia. La corrida salia con cobertura fundamental en cero sin que\n",
        "# nada fallara.\n",
        "_estado_fund = 'sin almacen'\n",
        "if _ruta is not None and str(_ruta).startswith('/content/drive') \\\n",
        "        and not Path('/content/drive/MyDrive').is_dir():\n",
        "    try:\n",
        "        from google.colab import drive as _drive\n",
        "        _drive.mount('/content/drive')\n",
        "    except Exception as _exc:\n",
        "        # Falla por cosas ajenas al codigo: un popup bloqueado, cookies\n",
        "        # de terceros, o cancelarlo. Degrada y lo dice, en vez de dejar\n",
        "        # el bloque vacio con apariencia de bug.\n",
        "        print(f'No se pudo montar Drive ({type(_exc).__name__}: {_exc}).')\n",
        "        print('El almacen de fundamentales no esta accesible. Monta '\n",
        "              'Drive a mano o apunta ALMACEN_FUNDAMENTALES al disco '\n",
        "              'local — pero ojo: el disco de Colab se recicla.')\n",
        "\n",
        "# Sin el directorio padre no se baja nada. Crearlo a ciegas sobre una\n",
        "# ruta de Drive sin montar deja el almacen en el disco efimero de Colab\n",
        "# con nombre de Drive: se pierde al reciclarse el runtime y la proxima\n",
        "# sesion vuelve a bajar varios GB creyendo que estaba guardado.\n",
        "_padre_ok = _ruta is not None and _ruta.parent.is_dir()\n",
        "\n",
        "if _ruta and DESCARGAR_FUNDAMENTALES and CONTACTO_SEC and _padre_ok:\n",
        "    if not str(_ruta).startswith('/content/drive'):\n",
        "        print(f'AVISO: {_ruta} no esta en Drive. El disco de Colab se '\n",
        "              'recicla, y la proxima sesion volveria a bajarlo todo.\\n')\n",
        "    # Un ETF no tiene estados financieros: pedirselos a EDGAR no es un\n",
        "    # dato que falta, es un error de categoria.\n",
        "    _emisores = [t for t in TICKERS if classify(t) != 'ETF']\n",
        "    _res = sincronizar(_ruta, _emisores, contacto=CONTACTO_SEC,\n",
        "                       refrescar_dias=REFRESCAR_DIAS,\n",
        "                       max_refrescos=MAX_REFRESCOS, progreso=print)\n",
        "    print(_res.resumen())\n",
        "    if _res.sin_cik:\n",
        "        print(f\"  Sin CIK: {', '.join(_res.sin_cik[:15])}\")\n",
        "        print(f'           {detalle_sin_cik(_res.fuentes)}')\n",
        "    if _res.pendientes:\n",
        "        print(f'  {len(_res.pendientes)} vencidos quedaron para la '\n",
        "              f'proxima corrida (tope {MAX_REFRESCOS}).')\n",
        "elif _ruta and DESCARGAR_FUNDAMENTALES and not _padre_ok:\n",
        "    _estado_fund = f'no se bajo: no existe {_ruta.parent}'\n",
        "    print(f'No existe {_ruta.parent}, asi que no se baja nada. Si es '\n",
        "          'una ruta de Drive, el montaje de arriba fallo o lo cancelaste.')\n",
        "elif _ruta and DESCARGAR_FUNDAMENTALES:\n",
        "    _estado_fund = 'no se bajo: falta CONTACTO_SEC'\n",
        "    print('Sin CONTACTO_SEC no se puede bajar de EDGAR: la SEC exige un '\n",
        "          'User-Agent con correo real y bloquea por IP a quien no se '\n",
        "          'identifica.\\nSe usara lo que ya este en el almacen.')\n",
        "\n",
        "_hechos = (leer_hechos(_ruta, TICKERS) if _ruta and _ruta.is_dir()\n",
        "           else None)\n",
        "\n",
        "if _hechos is None or _hechos.empty:\n",
        "    print('\\nSin almacen de fundamentales. El bloque de valuacion corre '\n",
        "          'solo con proxies de mercado, como antes de la fase 3.')\n",
        "    print(f'  Estado: {_estado_fund}')\n",
        "    if _ruta and not _ruta.is_dir():\n",
        "        print(f'  (no existe {_ruta})')\n",
        "    # El review: 'hoy una corrida sin fundamentales se ve como una\n",
        "    # normal'. Con el almacen configurado y vacio, se detiene aqui.\n",
        "    if EXIGIR_FUNDAMENTALES and ALMACEN_FUNDAMENTALES:\n",
        "        raise RuntimeError(\n",
        "            f'CORRIDA DETENIDA: el bloque fundamental no cargo '\n",
        "            f'({_estado_fund}). La valuacion correria solo con proxies de '\n",
        "            f'mercado. Arregla el almacen o pon EXIGIR_FUNDAMENTALES = '\n",
        "            f'False para correr sin fundamentales a sabiendas.')\n",
        "else:\n",
        "    fx.adjuntar(market_data, _hechos, FECHA_FUNDAMENTALES or None)\n",
        "    fund_meta = market_data.get('fundamentals_meta', {})\n",
        "    _estado_fund = (f\"{fund_meta['con_ratios']} nombres con ratios, \"\n",
        "                    f\"al {fund_meta['as_of'] or 'ultimo dato conocido'}\")\n",
        "    print(f\"{fund_meta['con_ratios']} de {len(TICKERS)} nombres con \"\n",
        "          f\"ratios, al {fund_meta['as_of'] or 'ultimo dato conocido'}. \"\n",
        "          f\"Cohorte minima por ratio: {fund_meta['cohorte_minima']}.\")\n",
        "    # Un cero sin explicacion se lee como 'no hay datos' cuando lo que\n",
        "    # hay es 'los datos son viejos', y son dos problemas distintos.\n",
        "    _viejos = fund_meta.get('obsoletos') or {}\n",
        "    if _viejos:\n",
        "        print(f'\\n{len(_viejos)} con el ultimo ejercicio vencido '\n",
        "              f'(>{fx.MAX_ANTIGUEDAD_DIAS} dias); no reciben ratios:')\n",
        "        for _t, _f in sorted(_viejos.items())[:12]:\n",
        "            print(f'  {_t:8s} ultimo cierre {_f}')\n",
        "\n",
        "_cob_fund = pd.DataFrame(fund_meta.get('cobertura', []))\n",
        "_tabla_fund = None\n",
        "if not _cob_fund.empty:\n",
        "    _tabla_fund = (_cob_fund[['ratio', 'familia', 'etiqueta', 'formula',\n",
        "                              'cobertura', 'con_dato', 'mediana',\n",
        "                              'puntuable']]\n",
        "                   .style\n",
        "                   .format({'cobertura': '{:.0%}',\n",
        "                            'mediana': '{:,.3f}'})\n",
        "                   .map(lambda v: escala(v, 0.0, 1.0),\n",
        "                        subset=['cobertura'])\n",
        "                   .hide(axis='index'))\n",
        "_tabla_fund\n",
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
        "from screener.seleccion import (CRITERIOS, politica_declarada,\n",
        "                               tabla as tabla_seleccion)\n",
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
        "\n",
        "# Politica de seleccion del universo: quien entro, quien no, y por que.\n",
        "print()\n",
        "print(politica_declarada())\n",
        "_sel = meta['seleccion_resumen']\n",
        "print(f\"\\nCandidatos: {_sel['candidatos']}  ->  admitidos: {_sel['admitidos']}\")\n",
        "for _c in CRITERIOS:\n",
        "    if _sel.get(_c.clave):\n",
        "        print(f'  rechazados por {_c.titulo.lower()}: {_sel[_c.clave]}')\n",
        "universo = tabla_seleccion(meta['seleccion'])\n",
        "_fuera = universo[universo['admitido'] == 'no']\n",
        "if not _fuera.empty:\n",
        "    print()\n",
        "    for _r in _fuera.head(25).itertuples():\n",
        "        print(f'  {_r.ticker:8s} [{_r.criterio}] {_r.motivo[:66]}')\n",
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
        "                                      build_views, public_view,\n",
        "                                      write_views)\n",
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
        "# Para lo que REFERENCIAS no cubre, el modelo busca contraparte entre los\n",
        "# nombres de la cesta. Solo acepta el par si el spread es mas tranquilo\n",
        "# que la pata suelta; si no, la view queda absoluta.\n",
        "PARES_AUTOMATICOS = True  # @param {type:\"boolean\"}\n",
        "\n",
        "# @markdown Cuántos nombres del ranking entran a la optimización.\n",
        "TOP_N_CARTERA = 25  # @param {type:\"integer\"}\n",
        "# @markdown Núcleo–satélite: el ancla vive en ETFs índice y renta fija, "
        "y las views solo pueden nacer en el satélite (acciones, ETFs no "
        "núcleo, materias primas), que tiene techo por mandato.\n",
        'NUCLEO_SATELITE = True  # @param {type:"boolean"}\n',
        "# @markdown Menos nombres = covarianza mejor estimada; más = más "
        "diversificación. Vive aquí y no en la celda de Cartera porque el pool "
        "de pares automáticos tiene que ser exactamente esta cesta.\n",
        "\n",
        "from screener.optimizer import select_basket, split_core_satellite\n",
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
        "                     max_views=MAX_VIEWS,\n",
        "                     auto_pair=PARES_AUTOMATICOS)\n",
        "\n",
        "# La cesta se arma antes que las views porque el pool de pares tiene que\n",
        "# ser el universo de la covarianza: posterior() descarta en silencio\n",
        "# cualquier view que nombre un ticker fuera de el, asi que un par contra\n",
        "# un nombre que no llega a la cesta no debilita la view, la borra.\n",
        "cartera_tickers, _notas_cesta = select_basket(\n",
        "    scored, ESTRATEGIA_CCI, top_n=TOP_N_CARTERA, min_per_class=3)\n",
        "for _n in _notas_cesta:\n",
        "    print(f'  {_n}')\n",
        "print()\n",
        "\n",
        "# Con nucleo-satelite una view solo puede nacer en el satelite. Una\n",
        "# senal de momentum de 6-12 meses sobre un ETF de bonos o sobre SPY\n",
        "# moveria el nucleo estrategico, que es justo lo que la separacion\n",
        "# existe para impedir. La contraparte de un par si puede ser nucleo.\n",
        "_sujetos = None\n",
        "if NUCLEO_SATELITE:\n",
        "    _, _sujetos, _ = split_core_satellite(\n",
        "        {r.ticker: r.asset_type for r in scored})\n",
        "\n",
        "views = build_views(scored, market_data, strategy=ESTRATEGIA_CCI,\n",
        "                    reference_map=REFERENCIAS,\n",
        "                    pair_pool=cartera_tickers, params=_params,\n",
        "                    subjects=_sujetos)\n",
        "cesta = build_basket(scored, strategy=ESTRATEGIA_CCI,\n",
        "                     reference_map=REFERENCIAS)\n",
        "\n",
        "print(f'{len(views)} views para {ESTRATEGIA_CCI} '\n",
        "      f'(perfil {_perfil_cci.label}, IC {IC_SUPUESTO})\\n')\n",
        "_marca = {'declarado': ' (REFERENCIAS)', 'automatico': ' (par automático)'}\n",
        "for _v in views:\n",
        "    _quien = (_v['activo'] if _v['tipo'] == 'absoluto'\n",
        "              else f\"{_v['activo_long']} / {_v['activo_short']}\")\n",
        "    print(f\"  {_v['tipo']:9s} {_quien:18s} Q {_v['Q']:+.2%}   \"\n",
        "          f\"convicción {_v['conviccion']:.2f}\"\n",
        "          f\"{_marca.get(_v.get('_pairing', ''), '')}\")\n",
        "\n",
        "_autom = [_v for _v in views if _v.get('_pairing') == 'automatico']\n",
        "if _autom:\n",
        "    print(f'\\n{len(_autom)} par(es) los eligió el modelo, no REFERENCIAS. '\n",
        "          f'Cada uno pasó el filtro de cobertura; el motivo va escrito '\n",
        "          f'en la justificación de la view.')\n",
        "elif PARES_AUTOMATICOS:\n",
        "    print('\\nNingún par automático: ningún candidato de la cesta cubría lo '\n",
        "          'suficiente. Las views quedan absolutas, que es el resultado '\n",
        "          'correcto cuando no hay con qué cubrir.')\n",
        "\n",
        "# public_view quita la columna interna _q_bruto, que solo usa el\n",
        "# diagnóstico de la celda siguiente y no viaja al archivo de CCI.\n",
        "views_df = pd.DataFrame([public_view(_v) for _v in views])\n",
        "cesta_df = pd.DataFrame(cesta)\n",
        "views_df\n",
    ))
    # ----------------------------------------------------------- diagnostics
    cells.append(md(
        "## 10b · Diagnóstico del modelo\n",
        "\n",
        "Dos mediciones sobre el modelo mismo, no sobre el mercado. Ninguna "
        "cambia una recomendación ni un peso: están para que sepas cuánto "
        "confiar en lo de arriba.\n",
        "\n",
        "**Correlación entre bloques.** El modelo declara seis bloques y le "
        "asigna un peso a cada uno, lo que equivale a decir que cada bloque "
        "aporta información que los otros no tienen. Si dos bloques van "
        "juntos al 0.90, sus pesos son una sola apuesta hecha dos veces y la "
        "cartera está menos diversificada de lo que promete la tabla de "
        "pesos. El número de *factores efectivos* resume eso: si dice 2 sobre "
        "6, tienes seis columnas midiendo dos cosas.\n",
        "\n",
        "**Saturación de views.** La `Q` se recorta en ±5% porque así lo "
        "calibra el documento técnico de CCI. El recorte es una baranda; si "
        "casi todas las views terminan pegadas a ella, la baranda pasó a ser "
        "la señal: nombres que el screener rankeó muy distinto llegan al "
        "optimizador con el mismo retorno esperado y ese pedazo del ranking "
        "se tira a la basura. Dos views en el tope es normal; seis es un "
        "problema de calibración, y se arregla bajando el IC, no subiendo el "
        "tope.\n",
    ))
    cells.append(code(
        "from screener.diagnostics import run_diagnostics\n",
        "\n",
        "print(run_diagnostics(scored, views, _params))\n",
    ))
    # ------------------------------------------- fund composition (sectors)
    cells.append(md(
        "## 10c · Composición de los fondos\n",
        "\n",
        "Baja el desglose sectorial de los ETFs **de la cesta**, que es lo que "
        "el tope sectorial de la celda siguiente necesita para mirar a través "
        "de los fondos.\n",
        "\n",
        "Tiene que correr antes del optimizador, no después. Un ETF sectorial y "
        "una acción de la misma industria son ambos «renta variable» para las "
        "bandas del Procedimiento, así que sin este desglose la única forma de "
        "limitar la concentración por industria no existe — y así fue como una "
        "cartera Agresiva real terminó con cerca del **35% en la cadena de "
        "semiconductores** y pasó su auditoría de bandas limpia. La auditoría "
        "estaba bien; la cartera seguía siendo un fondo sectorial.\n",
        "\n",
        "Lo que Yahoo no cubra queda declarado y **fuera del tope**: ese peso "
        "puede concentrarse sin que la restricción lo vea, y la corrida lo "
        "dice en vez de suponer un sector.\n",
    ))
    cells.append(code(
        "from screener.tenencias_yahoo import bajar_varios\n",
        "from pathlib import Path\n",
        "\n",
        "DIR_TENENCIAS = (Path('/content') if Path('/content').is_dir()\n",
        "                 else Path('.')) / 'tenencias'\n",
        "\n",
        "_tipos_basket = {r.ticker: r.asset_type for r in scored}\n",
        "_fondos_cesta = sorted(t for t in cartera_tickers\n",
        "                       if _tipos_basket.get(t, 'ETF') == 'ETF')\n",
        "_faltan = [t for t in _fondos_cesta\n",
        "           if not (DIR_TENENCIAS / f'{t}.csv').exists()]\n",
        "\n",
        "if not _faltan:\n",
        "    print(f'Composicion ya bajada para los {len(_fondos_cesta)} '\n",
        "          'fondo(s) de la cesta.')\n",
        "else:\n",
        "    print(f'Bajando composicion de {len(_faltan)} fondo(s) de la cesta:')\n",
        "    _ok, _fallaron = bajar_varios(_faltan, DIR_TENENCIAS)\n",
        "    if _fallaron:\n",
        "        print(f'\\nSin composicion en Yahoo: {\", \".join(_fallaron)}. '\n",
        "              'Quedan fuera del tope sectorial.')\n",
    ))

    # ------------------------------------------------------------- optimizer
    cells.append(md(
        "## 11 · Cartera Black-Litterman\n",
        "\n",
        "Aquí no hay archivo de por medio: `views` es una variable de Python "
        "que la celda anterior dejó en memoria, y esta la consume directo.\n",
        "\n",
        "La covarianza usa contracción Ledoit-Wolf sobre retornos **diarios** "
        "— con ~52 barras semanales y más de 52 nombres la matriz sería "
        "singular — y la optimización respeta las bandas del Procedimiento de "
        "Inversión.\n",
        "\n",
        "### El ancla: de dónde parte la cartera\n",
        "\n",
        "`π = λ · Σ · w` es una multiplicación: la `w` que le pases **es** la "
        "cartera neutral. Con ocho views sobre veintitantos activos, esa `w` "
        "decide como tres cuartas partes del resultado. Es la decisión más "
        "grande de toda la asignación, y por eso el parámetro `ANCLA` está "
        "arriba del todo.\n",
        "\n",
        "**`mercado`** era lo que hacía el sistema original: normalizar "
        "capitalización de acciones contra patrimonio de ETFs. Dos problemas. "
        "Primero, no son la misma unidad — la capitalización de una empresa es "
        "lo que vale la empresa; el patrimonio de un ETF es cuánta plata hay "
        "metida en ese envoltorio, y si el ETF es de renta variable está "
        "contando otra vez acciones que ya están en la cesta. Segundo, y peor: "
        "esa cuenta ancla cerca de 95% en renta variable. Ningún mandato de "
        "aquí permite eso. El resultado es que el optimizador se pasa el "
        "ejercicio empujando la cartera de vuelta contra el techo, y termina "
        "pegado exactamente en el límite — o sea, **la banda decide la "
        "asignación, no el modelo**.\n",
        "\n",
        "**`politica`** (lo que corre por defecto) parte del **Modelo de "
        "Asignación de Mercado Internacional** de tu Procedimiento de "
        "Inversión: los porcentajes deseados por clase de activo, no una "
        "lectura de las bandas. Las bandas siguen siendo techos que se "
        "verifican; el Modelo es el objetivo. Dentro de cada línea del Modelo "
        "el reparto es por capitalización, con la banda de cada clase y el tope "
        "por nombre aplicados. La propiedad que importa: **sin views, el "
        "optimizador te devuelve exactamente esta cartera**. Las views se "
        "desvían de ahí, que es como debe funcionar.\n",
        "\n",
        "| Clase | Cons. Def. | Conservador | Moderado | Agresivo |\n",
        "|---|---|---|---|---|\n",
        "| Renta fija gubernamental IG | 45% | 40% | 30% | 20% |\n",
        "| Renta fija corporativa | 25% | 20% | 15% | 10% |\n",
        "| Acciones y ETFs indexados | 20% | 30% | 50% | 65% |\n",
        "| Efectivo / money market | 10% | 10% | 5% | 5% |\n",
        "\n",
        "Dos cosas que conviene saber. **Materias primas no tienen línea en el "
        "Procedimiento**, así que el ancla no les asigna nada: el oro entra "
        "solo si una view lo empuja. Y si a alguna línea no le queda ninguna "
        "clase en la cesta, su porcentaje se reparte entre las demás al "
        "renormalizar, y la corrida lo dice.\n",
        "\n",
        "### Núcleo y satélite\n",
        "\n",
        "El screener es un motor **táctico** (señales de 3-12 meses) y estas "
        "carteras son de **largo plazo**. Por eso la cartera se arma en dos "
        "partes, y `NUCLEO_SATELITE` (encendido por defecto) las separa:\n",
        "\n",
        "- **Núcleo**: el Modelo de Asignación implementado con ETFs índice "
        "por clase — renta fija, efectivo y un vehículo por exposición de "
        "renta variable (EEUU amplio, pequeña capitalización, desarrollados, "
        "emergentes, Nasdaq). El ancla vive entera aquí. Se queda quieto por "
        "años.\n",
        "- **Satélite**: acciones individuales, ETFs que no son vehículo "
        "núcleo (sectoriales, país, factor) y materias primas. Anclan en 0 y "
        "solo entran si una view los empuja, con un **techo por mandato**.\n",
        "\n",
        "| | Cons. Def. | Conservador | Moderado | Agresivo |\n",
        "|---|---|---|---|---|\n",
        "| Techo del satélite | 7.5% | 10% | 17.5% | 30% |\n",
        "\n",
        "El techo no se escogió: es el tracking error máximo del mandato "
        "(1.5 / 2.0 / 3.5 / 6.0%) dividido entre 20% de volatilidad activa "
        "supuesta para una canasta concentrada de acciones. Un satélite lleno "
        "no puede, por sí solo, agotar el presupuesto de TE. Es número de la "
        "mesa, pendiente del Comité.\n",
        "\n",
        "Sin el núcleo, la línea de acciones del Modelo se repartía por "
        "capitalización entre todo lo que hubiera en la cesta — y la "
        "capitalización de NVDA o MSFT es varias veces el patrimonio de SPY — "
        "así que la cartera \"neutral\" eran sobre todo acciones elegidas "
        "por el ranking.\n",
        "\n",
        "### Tres arreglos frente al sistema original\n",
        "\n",
        "1. **Solver.** Tu código pedía ECOS, que no viene en Colab; tu corrida "
        "guardada murió ahí sin producir cartera. Este usa CLARABEL, que viene "
        "con CVXPY.\n",
        "2. **Apalancamiento.** `leverage_max` de 1.25 y 1.50 estaba declarado "
        "pero el optimizador fijaba `sum(w) == 1`, asi que nunca restringio nada. "
        "Ahora es un presupuesto real — pero **la mesa lo tiene apagado**: todas "
        "las carteras resuelven invertidas al 100%, sin importar lo que permita "
        "el mandato. El limite sigue en `REGULACIONES` porque es lo que dice el "
        "Procedimiento; la decision de no usarlo vive en `ALLOW_LEVERAGE`, en el "
        "optimizador.\n",
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
        "# @markdown Cartera neutral de la que parten las views.\n",
        'ANCLA = "politica"  # @param ["politica", "mercado"]\n',
        "\n",
        "# @markdown Cartera actual (opcional): ruta a un CSV `ticker,peso` que "
        "subes tú. No se lee ninguna cuenta.\n",
        'CARTERA_ACTUAL = ""  # @param {type:"string"}\n',
        "ROTACION_MAXIMA = 0.0  # @param {type:\"number\"}\n",
        "# @markdown Fracción máxima del libro a operar contra la cartera actual "
        "(0.15 = 15%). 0 = sin tope. Solo aplica con cartera actual.\n",
        "\n",
        "# @markdown Posición mínima ejecutable, como fracción del libro.\n",
        "POSICION_MINIMA = 0.01  # @param {type:\"number\"}\n",
        "# @markdown El optimizador no sabe qué vale la pena operar: si le "
        "conviene, devuelve un 0.16% que cuesta una boleta, una línea en cada "
        "reporte y una conciliación para siempre. Las posiciones bajo este "
        "piso se eliminan **re-optimizando sin ellas**, no recortándolas del "
        "resultado — así las bandas del mandato siguen cumpliéndose exactas. "
        "Pon 0 para desactivarlo.\n",
        "\n",
        "from screener.optimizer import (RISK_AVERSION, core_vehicles,\n",
        "                               implied_equilibrium, market_weights,\n",
        "                               optimize, policy_weights, posterior,\n",
        "                               risk_profile_table, shrunk_covariance,\n",
        "                               allocation_table, select_basket,\n",
        "                               gross_budget, split_core_satellite,\n",
        "                               leer_cartera_actual)\n",
        "from screener.cci_regulation import REGULACIONES\n",
        "from screener.cci_regulation import classify_for_bands\n",
        "from screener.yahoo_adapter import daily_returns, fetch_market_caps\n",
        "\n",
        "tipos_todos = {r.ticker: r.asset_type for r in scored}\n",
        "\n",
        "# cartera_tickers viene de la celda de Views, que la necesita antes\n",
        "# para acotar el pool de pares automaticos. Se recalcula aqui por si\n",
        "# cambiaste TOP_N_CARTERA y corriste solo esta celda.\n",
        "#\n",
        "# La cesta no puede ser solo el top-N por score. El ranking premia\n",
        "# momentum y riesgo-retorno, donde la renta variable domina, y con una\n",
        "# cesta 100% equity el techo del mandato (60% en Moderado) queda por\n",
        "# debajo del libro invertido: el solver responde 'infactible' y la\n",
        "# cartera sale vacia. select_basket asegura representacion de cada\n",
        "# clase disponible en el universo, y ademas mete las exposiciones\n",
        "# nucleo aunque no hayan puntuado alto.\n",
        "cartera_tickers, _ = select_basket(\n",
        "    scored, ESTRATEGIA_CCI, top_n=TOP_N_CARTERA, min_per_class=3)\n",
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
        "_tipos_cesta = {t: tipos_todos.get(t, 'ETF') for t in covarianza.columns}\n",
        "# Presupuesto bruto en vigor. Con el apalancamiento apagado es 1.0.\n",
        "_presupuesto = gross_budget(ESTRATEGIA_CCI)\n",
        "\n",
        "# Nucleo y satelite. El nucleo es el Modelo de Asignacion en ETFs\n",
        "# indice y renta fija; el satelite es lo unico que las views mueven.\n",
        "_nucleo, _ = core_vehicles(scored)\n",
        "nucleo_cesta = satelite_cesta = None\n",
        "if NUCLEO_SATELITE and ANCLA == 'politica':\n",
        "    nucleo_cesta, satelite_cesta, _notas_ns = split_core_satellite(\n",
        "        _tipos_cesta)\n",
        "    for _n in _notas_ns:\n",
        "        print(f'  {_n}')\n",
        "\n",
        "if ANCLA == 'politica':\n",
        "    pesos_ancla, _notas_ancla = policy_weights(\n",
        "        _tipos_cesta, ESTRATEGIA_CCI, caps=capitalizaciones,\n",
        "        total=_presupuesto, core=nucleo_cesta)\n",
        "    for _n in _notas_ancla:\n",
        "        print(f'  {_n}')\n",
        "else:\n",
        "    pesos_ancla, sin_cap = market_weights(capitalizaciones,\n",
        "                                          list(covarianza.columns))\n",
        "    if sin_cap:\n",
        "        print(f'Sin capitalizacion, excluidos del equilibrio: {sin_cap}')\n",
        "\n",
        "print(f'\\nAncla ({ANCLA}) por clase de activo:')\n",
        "_cl_ancla = pd.Series({t: classify_for_bands(t, _tipos_cesta[t])\n",
        "                       for t in pesos_ancla.index})\n",
        "for _clase, _peso in pesos_ancla.groupby(_cl_ancla).sum().sort_values(\n",
        "        ascending=False).items():\n",
        "    if _peso > 0.0001:\n",
        "        print(f'  {_peso:7.2%}  {_clase}')\n",
        "\n",
        "# El equilibrio usa el lambda del MERCADO. El del cliente entra en\n",
        "# optimize(): son dos cosas distintas y confundirlas hace que la\n",
        "# Agresiva salga con menos retorno esperado que la Moderada.\n",
        "pi = implied_equilibrium(pesos_ancla, covarianza,\n",
        "                         risk_aversion=RISK_AVERSION)\n",
        "er_posterior, cov_posterior = posterior(pi, covarianza, views)\n",
        "\n",
        "tipos = tipos_todos\n",
        "clases = {t: classify_for_bands(t, tipos.get(t, 'ETF'))\n",
        "          for t in covarianza.columns}\n",
        "\n",
        "# Tope sectorial mirando a traves de los fondos. El desglose sale de\n",
        "# tenencias/_sectores.csv, que baja la seccion 11b; sin el la\n",
        "# concentracion por industria queda sin restringir y la corrida lo dice.\n",
        "# Las bandas del Procedimiento son por clase de activo y no limitan\n",
        "# sector: asi fue como una cartera Agresiva real llego a ~35% en la\n",
        "# cadena de semiconductores y paso su auditoria limpia.\n",
        "from screener.lookthrough import (load_fund_sectors, sector_map,\n",
        "                                  stock_sectors_for)\n",
        "from screener.cci_regulation import SECTOR_CAPS\n",
        "\n",
        "_fondos_sec = load_fund_sectors(DIR_TENENCIAS / '_sectores.csv')\n",
        "# CON_NOMBRES_Y_SECTORES viene apagado (una peticion por ticker sobre\n",
        "# cientos de nombres), asi que sin esto ninguna accion traeria sector y\n",
        "# el tope solo veria los fondos. La cesta son decenas de nombres: se\n",
        "# baja solo para ella.\n",
        "_acciones_cesta = [t for t in covarianza.columns\n",
        "                   if t not in _fondos_sec\n",
        "                   and tipos_todos.get(t, 'ETF') != 'ETF']\n",
        "_sec_acciones, _notas_meta = stock_sectors_for(\n",
        "    _acciones_cesta,\n",
        "    {r.ticker: r.sector for r in scored if getattr(r, 'sector', None)})\n",
        "mapa_sectores, _cob_sec, _notas_sec = sector_map(\n",
        "    list(covarianza.columns), _fondos_sec, _sec_acciones)\n",
        "for _n in _notas_meta + _notas_sec:\n",
        "    print(f'  {_n}')\n",
        "\n",
        "# Para la hoja de parametros: que el libro diga que quedo sin restringir\n",
        "# y que vehiculo gano cada exposicion nucleo, no solo el resultado.\n",
        "_sin_sector = sorted(t for t, v in _cob_sec.items() if not v)\n",
        "\n",
        "# La cartera actual la sube el usuario; nunca se lee de una cuenta.\n",
        "cartera_actual = None\n",
        "if CARTERA_ACTUAL:\n",
        "    cartera_actual, _notas_actual = leer_cartera_actual(CARTERA_ACTUAL)\n",
        "    for _n in _notas_actual:\n",
        "        print(f'  {_n}')\n",
        "_tope_rotacion = (ROTACION_MAXIMA or None) if cartera_actual is not None else None\n",
        "\n",
        "cartera = optimize(er_posterior, cov_posterior, tipos, ESTRATEGIA_CCI,\n",
        "                   min_position=POSICION_MINIMA or None,\n",
        "                   sector_weights=mapa_sectores,\n",
        "                   views=views,\n",
        "                   anchor=pesos_ancla, prior=pi,\n",
        "                   satellite=satelite_cesta,\n",
        "                   current=cartera_actual, turnover_cap=_tope_rotacion)\n",
        "\n",
        "print(f'{ESTRATEGIA_CCI}  |  estado: {cartera.status}')\n",
        "print(f'Exposicion bruta   {cartera.gross_exposure:.1%}')\n",
        "print(f'Retorno esperado   {cartera.expected_return:+.2%} anual')\n",
        "print(f'Volatilidad        {cartera.volatility:.1%} anual')\n",
        "print(f'Posiciones         {int((cartera.weights > 0).sum())}')\n",
        "if cartera.satellite_cap is not None:\n",
        "    print(f'Satelite           {cartera.satellite_exposure:.1%} '\n",
        "          f'(techo {cartera.satellite_cap:.1%})')\n",
        "if cartera.turnover is not None:\n",
        "    print(f'Rotacion vs actual {cartera.turnover:.1%} del libro')\n",
        "\n",
        "print('\\nPor clase de activo')\n",
        "for _clase, _peso in cartera.by_class.items():\n",
        "    if _peso > 0.0001:\n",
        "        print(f'  {_peso:7.2%}  {_clase}')\n",
        "\n",
        "if cartera.sector_exposure:\n",
        "    _tope = SECTOR_CAPS.get(ESTRATEGIA_CCI)\n",
        "    # El tope es fraccion del SLEEVE de renta variable, no del libro.\n",
        "    # Imprimir el porcentaje del libro al lado del tope invita a leer\n",
        "    # dos numeros que no se comparan entre si.\n",
        "    _rv = cartera.equity_exposure\n",
        "    _et = (f'tope {_tope:.0%} del sleeve de RV ({_rv:.0%} del libro)'\n",
        "           if _tope is not None else 'sin tope')\n",
        "    print(f'\\nPor sector, a traves de los fondos ({_et})')\n",
        "    for _s, _v in cartera.sector_exposure.items():\n",
        "        if _v > 0.0001:\n",
        "            _cuota = (_v / _rv) if _rv > 1e-9 else float('nan')\n",
        "            print(f'  {_cuota:7.1%} del sleeve   {_v:7.2%} del libro   {_s}')\n",
        "\n",
        "if cartera.risk_findings:\n",
        "    print('\\nRIESGO vs. MANDATO (expectativa de la mesa, no del '\n",
        "          'Procedimiento):')\n",
        "    for _r in cartera.risk_findings:\n",
        "        print(f'  {_r}')\n",
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

    # ------------------------------------------------------ riesgo por perfil
    cells.append(md(
        "## 11c · Riesgo esperado por perfil\n",
        "\n",
        "Los cuatro mandatos resueltos con la **misma cesta y las mismas "
        "views**. Lo único que cambia entre filas es el mandato.\n",
        "\n",
        "### El problema que cierra\n",
        "\n",
        "Las cuatro estrategias optimizaban **la misma función**, con "
        "`λ = 2.5` para todas. La única diferencia entre una cartera Agresiva y "
        "una Conservadora era el ancho de sus bandas — y una banda es un techo: "
        "nada obligaba a la Agresiva a usarlo. Dos mandatos con distinto "
        "apetito de riesgo que maximizan la misma utilidad no son dos "
        "mandatos.\n",
        "\n",
        "Ahora cada uno lleva su propia aversión al riesgo: **8.0 / 5.0 / 2.5 / "
        "1.5**. Un λ alto compra tranquilidad, uno bajo compra retorno "
        "esperado, que es exactamente lo que el cliente firmó.\n",
        "\n",
        "### Cómo leer cada columna\n",
        "\n",
        "| Columna | Qué es | Cuánto creerle |\n",
        "|---|---|---|\n",
        "| `retorno_esperado` | `w'μ` con la posterior | Sale del modelo. "
        "Depende del IC supuesto, que **no está calibrado**. |\n",
        "| `volatilidad` | `√(w'Σw)` anual | Lo más sólido de la tabla: "
        "covarianza estimada con contracción sobre datos diarios. |\n",
        "| `max_drawdown` | Peor caída pico a valle **de estos pesos aplicados "
        "al pasado** | No es un backtest. Estos pesos no existían entonces y "
        "salieron de un modelo que vio ese mismo período. |\n",
        "| `peor_12m` | Peor retorno móvil de 12 meses, misma advertencia | "
        "Igual. Es historia de la cartera de hoy, no de la estrategia. |\n",
        "| `caida_1a_95` | `μ − 1.645σ` | Paramétrica y **normal**. Las colas "
        "reales son más gordas: en un mercado malo de verdad se queda corta. |\n",
        "\n",
        "### El techo y el piso no se tratan igual\n",
        "\n",
        "`w'Σw ≤ máx²` es convexa y el solver la impone. `w'Σw ≥ mín²` es "
        "convexa **al revés** y no se puede pedir. Así que el techo se aplica y "
        "el piso se audita: una cartera Agresiva por debajo de su piso sale "
        "reportada como incumplimiento, porque lo es — un cliente que firmó "
        "Agresivo no contrató una cartera Moderada.\n",
        "\n",
        "Los rangos de volatilidad son **de la mesa, no del Procedimiento**, "
        "que no habla de volatilidad. Pendientes del Comité.\n",
    ))
    cells.append(code(
        "riesgo_df, _notas_riesgo = risk_profile_table(\n",
        "    covarianza, _tipos_cesta, capitalizaciones, views,\n",
        "    returns=retornos, sector_weights=mapa_sectores,\n",
        "    min_position=POSICION_MINIMA or None, core=nucleo_cesta)\n",
        "\n",
        "for _n in _notas_riesgo:\n",
        "    print(f'AVISO: {_n}')\n",
        "if not _notas_riesgo:\n",
        "    print('Riesgo y retorno crecen con el perfil, como debe ser.')\n",
        "# Un mandato sin solución sale en la tabla con NaN y aquí con su motivo.\n",
        "for _, _f in riesgo_df[riesgo_df['motivo'] != ''].iterrows():\n",
        "    print(f\"SIN SOLUCIÓN {_f['estrategia']}: {_f['motivo']}\")\n",
        "\n",
        "_pct = ['retorno_esperado', 'volatilidad', 'vol_min_objetivo',\n",
        "        'vol_max_objetivo', 'max_drawdown', 'peor_12m', 'caida_1a_95',\n",
        "        'satelite', 'satelite_max']\n",
        "(riesgo_df[['estrategia', 'estado', 'lambda'] + _pct + ['posiciones']].style\n",
        "    .format({c: '{:.2%}' for c in _pct if c in riesgo_df})\n",
        "    .hide(axis='index')\n",
        "    .set_caption('Riesgo y retorno esperados por mandato'))\n",
    ))

    # ---------------------------------------------------------------- estrés
    cells.append(md(
        "## 11d · Estrés histórico\n",
        "\n",
        "`caida_1a_95` supone normalidad, y la normalidad es lo que falla en un "
        "mercado malo. Esto aplica los pesos de hoy, **por clase de activo**, a "
        "tres ventanas reales: 2008, marzo de 2020 y 2022 — el año en que "
        "acciones y bonos cayeron juntos, que es el escenario que más daña a un "
        "perfil conservador y el que una covarianza de dos años no ve.\n",
        "\n",
        "Cada clase se representa con un ETF índice que existía en las tres "
        "ventanas (SPY, IEF, AGG, LQD, HYG, BIL, GLD). La renta variable entra "
        "con la beta del sleeve contra SPY, para que un satélite de beta alta "
        "caiga más que el índice. Es un orden de magnitud, no un backtest: "
        "pesos de hoy sobre ventanas pasadas, sin rebalancear. Las filas "
        "`Modelo` son el Modelo de Asignación de cada mandato, sin views.\n",
    ))
    cells.append(code(
        "from screener.estres import (FIN_DESCARGA, INICIO_DESCARGA,\n",
        "                             beta_del_sleeve, escenarios_descritos,\n",
        "                             pesos_modelo, proxies, retornos_escenarios,\n",
        "                             tabla_estres)\n",
        "from screener.yahoo_adapter import fetch_history\n",
        "\n",
        "_beta_rv, _nota_beta = beta_del_sleeve(cartera.weights, clases, retornos)\n",
        "print(_nota_beta)\n",
        "try:\n",
        "    _precios_estres = fetch_history(proxies(), INICIO_DESCARGA, FIN_DESCARGA)\n",
        "    _ret_esc, _notas_esc = retornos_escenarios(_precios_estres)\n",
        "except Exception as _exc:\n",
        "    _ret_esc = None\n",
        "    _notas_esc = [f'No se pudo bajar la historia de los proxies '\n",
        "                  f'({type(_exc).__name__}: {_exc}).']\n",
        "for _n in _notas_esc:\n",
        "    print(f'  {_n}')\n",
        "\n",
        "estres_df = pd.DataFrame()\n",
        "if _ret_esc is not None:\n",
        "    _carteras_estres = {}\n",
        "    if cartera.feasible:\n",
        "        _carteras_estres[f'Cartera {ESTRATEGIA_CCI}'] = (\n",
        "            ESTRATEGIA_CCI, cartera.by_class.to_dict(), _beta_rv)\n",
        "    for _e in REGULACIONES:\n",
        "        _carteras_estres[f'Modelo {_e}'] = (_e, pesos_modelo(_e), 1.0)\n",
        "    estres_df, _notas_estres = tabla_estres(_carteras_estres, _ret_esc)\n",
        "    print()\n",
        "    for _l in escenarios_descritos():\n",
        "        print(_l)\n",
        "    for _n in _notas_estres:\n",
        "        print(f'AVISO: {_n}')\n",
        "\n",
        "_cols_estres = [c for c in estres_df.columns\n",
        "                if c not in ('cartera', 'estrategia', 'beta_rv',\n",
        "                             'excede_tolerancia')]\n",
        "(estres_df.style\n",
        "    .format({c: '{:.1%}' for c in _cols_estres} | {'beta_rv': '{:.2f}'})\n",
        "    .hide(axis='index')\n",
        "    .set_caption('Retorno en cada ventana de estrés')\n",
        " if not estres_df.empty else None)\n",
    ))

    # ---------------------------------------------------------- sensibilidad
    cells.append(md(
        "## 11e · Sensibilidad a los parámetros sin calibrar\n",
        "\n",
        "El IC (0,08), τ y el λ de cada mandato no están estimados contra "
        "datos. Esta tabla resuelve la misma cartera moviendo uno a la vez — "
        "IC ±50%, τ a la mitad y al doble, λ a la mitad y al doble — y mide la "
        "**rotación** contra la cartera de la corrida: la fracción del libro "
        "que habría que operar para pasar de una a otra. Si una rotación pasa "
        "de 10%, la asignación la decide el parámetro más que las views, y el "
        "Comité tiene que verlo.\n",
    ))
    cells.append(code(
        "from screener.sensibilidad import tabla_sensibilidad\n",
        "\n",
        "sensibilidad_df, _notas_sens = pd.DataFrame(), []\n",
        "if cartera.feasible and views:\n",
        "    sensibilidad_df, _notas_sens = tabla_sensibilidad(\n",
        "        pi, covarianza, views, tipos, ESTRATEGIA_CCI, base=cartera,\n",
        "        anchor=pesos_ancla, min_position=POSICION_MINIMA or None,\n",
        "        sector_weights=mapa_sectores, satellite=satelite_cesta,\n",
        "        current=cartera_actual, turnover_cap=_tope_rotacion)\n",
        "    for _n in _notas_sens:\n",
        "        print(f'AVISO: {_n}')\n",
        "    if not _notas_sens:\n",
        "        print('Ninguna variante rota más de 10% del libro: la cartera no '\n",
        "              'depende de un parámetro sin calibrar.')\n",
        "else:\n",
        "    print('Sin cartera o sin views: no hay nada que sensibilizar.')\n",
        "\n",
        "(sensibilidad_df.style\n",
        "    .format({'retorno_esperado': '{:.2%}', 'volatilidad': '{:.2%}',\n",
        "             'satelite': '{:.1%}', 'rotacion_vs_base': '{:.1%}'})\n",
        "    .hide(axis='index')\n",
        " if not sensibilidad_df.empty else None)\n",
    ))

    # --------------------------------------------------------- transparencia
    cells.append(md(
        "## 11b · Transparencia (mirar a través de los ETFs)\n",
        "\n",
        "La tabla de arriba no es la cartera. Un 20% en un ETF de mercado "
        "amplio son posiciones en cientos de empresas que nadie eligió una por "
        "una, y eso esconde tres cosas:\n",
        "\n",
        "1. **Exposición efectiva por emisor.** El tope del Procedimiento está "
        "escrito sobre el instrumento, pero su intención es sobre el emisor. "
        "Con solo acciones las dos cosas coinciden; con ETFs se separan, y un "
        "nombre puede pasar su límite sumando la posición directa y la que "
        "entra por los fondos.\n",
        "2. **Exposición sectorial real.** Un ETF sectorial encima de uno "
        "amplio no da \"exposición al sector\": da un **sobrepeso** sobre lo "
        "que el amplio ya traía.\n",
        "3. **Solape estructural.** Que dos ETFs sigan el mismo índice es un "
        "hecho verificable, no una correlación que puede fallar en un régimen "
        "raro.\n",
        "\n",
        "Esta celda baja la composición de los ETFs **de la cartera** desde "
        "Yahoo (`funds_data`) y corre el reporte. No hace falta subir nada ni "
        "contratar a ningún proveedor.\n",
        "\n",
        "**Lo que este reporte no hace: estimar.** Yahoo publica las mayores "
        "posiciones de cada fondo, no las 500. El peso que no detalla se anota "
        "como `_RESTO` y se reporta como tal. Sin esa fila, un 7% se "
        "convertiría en 17% al normalizar y el reporte acusaría un "
        "incumplimiento que no existe. Un fondo que Yahoo no cubra queda "
        "declarado **opaco**, no rellenado con supuestos.\n",
        "\n",
        "Para lo que sirve el tope: la exposición efectiva se compara contra "
        "`max_equity_individual` del perfil, pero **solo sobre las acciones de "
        "la cesta** — un emisor al que solo se llega por dentro de un ETF "
        "indexado no es una posición individual del libro.\n",
    ))
    cells.append(code(
        "# @markdown Baja la composición de los ETFs de la cartera y mira a "
        "través de ellos.\n",
        "CORRER_TRANSPARENCIA = True  # @param {type:\"boolean\"}\n",
        "\n",
        "from screener.lookthrough import (load_fund_sectors, load_holdings,\n",
        "                                  report, sector_exposure_direct)\n",
        "from screener.tenencias_yahoo import bajar_varios\n",
        "from screener.cci_regulation import CLASE_EQUITY\n",
        "\n",
        "# DIR_TENENCIAS viene de la seccion 10c, que ya bajo los fondos de la\n",
        "# cesta. Aqui solo falta lo que quedo en la cartera y no estaba.\n",
        "\n",
        "if not CORRER_TRANSPARENCIA:\n",
        "    print('Transparencia desactivada.')\n",
        "else:\n",
        "    _pesos_cartera = cartera.weights[cartera.weights > 0].to_dict()\n",
        "    # Solo los ETFs: una accion mira a traves de si misma, y pedirle su\n",
        "    # composicion a Yahoo es una llamada que siempre falla.\n",
        "    _fondos = sorted(t for t in _pesos_cartera\n",
        "                     if tipos_todos.get(t, 'ETF') == 'ETF')\n",
        "\n",
        "    if not _fondos:\n",
        "        print('La cartera no tiene ETFs: lo que ves es lo que hay.')\n",
        "    else:\n",
        "        _faltan = [t for t in _fondos\n",
        "                   if not (DIR_TENENCIAS / f'{t}.csv').exists()]\n",
        "        if _faltan:\n",
        "            print(f'Bajando composicion de {len(_faltan)} fondo(s):')\n",
        "            _ok, _fallaron = bajar_varios(_faltan, DIR_TENENCIAS)\n",
        "            if _fallaron:\n",
        "                print(f'\\nSin composicion en Yahoo: "
        "{\", \".join(_fallaron)}')\n",
        "                print('Quedan declarados como opacos en el reporte. '\n",
        "                      'Si te importan, baja el CSV del emisor y subelo '\n",
        "                      f'a {DIR_TENENCIAS}/TICKER.csv')\n",
        "            print()\n",
        "        else:\n",
        "            print('Composicion ya bajada; se reutiliza.\\n')\n",
        "\n",
        "        _tenencias, _sectores_lt, _notas_lt = load_holdings(DIR_TENENCIAS)\n",
        "        for _n in _notas_lt:\n",
        "            print(f'  {_n}')\n",
        "\n",
        "        _acciones = [t for t in _pesos_cartera\n",
        "                     if classify_for_bands(t, tipos_todos.get(t, 'ETF'))\n",
        "                     == CLASE_EQUITY]\n",
        "        print(report(_pesos_cartera, _tenencias, _sectores_lt,\n",
        "                     cap=REGULACIONES[ESTRATEGIA_CCI]"
        "['max_equity_individual'],\n",
        "                     only=_acciones))\n",
        "\n",
        "        # El desglose sectorial del emisor es el total del fondo, no una\n",
        "        # muestra de sus mayores posiciones: da un numero completo aunque\n",
        "        # las tenencias sean parciales. Cuando esta, manda sobre el\n",
        "        # derivado de las posiciones.\n",
        "        _fondos_sec = load_fund_sectors(DIR_TENENCIAS / '_sectores.csv')\n",
        "        if _fondos_sec:\n",
        "            _sec, _cob_sec, _notas_sec = sector_exposure_direct(\n",
        "                _pesos_cartera, _fondos_sec,\n",
        "                {r.ticker: r.sector for r in scored if r.sector})\n",
        "            print('\\n  Exposicion sectorial (desglose completo del "
        "emisor):')\n",
        "            for _n in _notas_sec:\n",
        "                print(f'    {_n}')\n",
        "            for _s, _v in _sec.items():\n",
        "                print(f'    {_v:>7.2%}  {_s}')\n",
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
        "    ('Generado (UTC)', pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M')),\n",
        "    ('Perfil', perfil.label),\n",
        "    ('Perfil — resumen', perfil.summary),\n",
        "    ('Estrategia CCI destino', ESTRATEGIA_CCI),\n",
        "    ('Universo', UNIVERSO),\n",
        "    ('Nombres puntuados', len(scored)),\n",
        "    ('Benchmark', BENCHMARK),\n",
        "    ('Historia', PERIODO),\n",
        "    ('Tasa libre de riesgo', f'{TASA_LIBRE_RIESGO:.2%}'),\n",
        "    ('Fuente de la tasa libre de riesgo', _rf_fuente),\n",
        "    ('Motor (SHA-256 del paquete embebido)', ENGINE_SHA256),\n",
        "    ('Cartera actual', CARTERA_ACTUAL or 'no se usó'),\n",
        "    ('Rotación contra la cartera actual',\n",
        "     f'{cartera.turnover:.2%}' if cartera.turnover is not None\n",
        "     else 'no aplica'),\n",
        "    ('Tope de rotación',\n",
        "     f'{_tope_rotacion:.0%}' if _tope_rotacion else 'sin tope'),\n",
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
        "    # Dos topes distintos. Verlos sin etiqueta en la misma hoja se lee\n",
        "    # como contradiccion: el del screener dimensiona una idea suelta, el\n",
        "    # del optimizador es el limite del Procedimiento sobre la cartera.\n",
        "    ('Peso máx. por posición — dimensionamiento del screener',\n",
        "     f'{perfil.sizing.max_weight:.1%}'),\n",
        "    ('Peso máx. por acción individual — Procedimiento (optimizador)',\n",
        "     f\"{REGULACIONES[ESTRATEGIA_CCI]['max_equity_individual']:.1%}\"),\n",
        "    ('Volumen diario mínimo', f'${perfil.eligibility.min_adv_usd/1e6:,.0f}MM'),\n",
        "    ('Ancla del equilibrio', ANCLA),\n",
        "    ('Núcleo indexado forzado en la cesta', 'sí'),\n",
        "    ('Núcleo agregado al universo',\n",
        "     ', '.join(_agregados_nucleo) or 'ya estaba completo'),\n",
        "    ('Núcleo–satélite', 'sí' if satelite_cesta is not None else 'no'),\n",
        "    ('Techo del satélite',\n",
        "     f'{cartera.satellite_cap:.1%}' if cartera.satellite_cap is not None\n",
        "     else 'sin techo'),\n",
        "    ('Satélite en la cartera',\n",
        "     f'{cartera.satellite_exposure:.2%}' if satelite_cesta is not None\n",
        "     else 'no aplica'),\n",
        "    ('Nota sobre el satélite',\n",
        "     'techo = TE máximo del mandato / 20% de volatilidad activa; '\n",
        "     'número de la mesa, pendiente del Comité'),\n",
        "    ('Núcleo — vehículo por exposición',\n",
        "     ' | '.join(f'{_e}: {_t}' for _e, _t in _nucleo.items())\n",
        "     or 'ninguno disponible'),\n",
        "    ('Tope sectorial (look-through)',\n",
        "     'sin desglose sectorial — SIN restringir' if not mapa_sectores\n",
        "     else ('sin tope' if SECTOR_CAPS.get(ESTRATEGIA_CCI) is None\n",
        "           else f'{SECTOR_CAPS[ESTRATEGIA_CCI]:.0%}')),\n",
        "    ('Nota sobre el tope sectorial',\n",
        "     'número de la mesa, NO del Procedimiento de Inversión; '\n",
        "     'pendiente de confirmación del Comité'),\n",
        "    # Una corrida salio con el bloque fundamental vacio y el libro no\n",
        "    # lo decia: habia que mirar la consola de una sesion ya cerrada.\n",
        "    ('Fundamentales (SEC EDGAR)', _estado_fund),\n",
        "    ('Sectores restringidos', len(mapa_sectores) or 'ninguno'),\n",
        "    ('Instrumentos sin sector conocido',\n",
        "     ' | '.join(_sin_sector) or 'ninguno'),\n",
        "] + [(f'Peso — {b.label}', f'{b.weight:.0%}') for b in MODELO],\n",
        "    columns=['Parámetro', 'Valor'])\n",
        "\n",
        "# La concentracion sectorial es ahora una restriccion, no solo un dato:\n",
        "# tiene que viajar en el libro que lee el comite, con el techo al lado.\n",
        "_tope_sec = SECTOR_CAPS.get(ESTRATEGIA_CCI)\n",
        "_rv_libro = cartera.equity_exposure\n",
        "_nan = float('nan')\n",
        "sectores_df = pd.DataFrame(\n",
        "    [{'sector': _s,\n",
        "      'exposicion_libro': _v,\n",
        "      'exposicion_sleeve': (_v / _rv_libro) if _rv_libro > 1e-9 else _nan,\n",
        "      'tope_sleeve': _tope_sec if _tope_sec is not None else _nan,\n",
        "      'holgura_sleeve': ((_tope_sec - _v / _rv_libro)\n",
        "                         if _tope_sec is not None and _rv_libro > 1e-9\n",
        "                         else _nan)}\n",
        "     for _s, _v in cartera.sector_exposure.items()]\n",
        "    or [{'sector': 'sin desglose sectorial', 'exposicion_libro': _nan,\n",
        "         'exposicion_sleeve': _nan, 'tope_sleeve': _nan,\n",
        "         'holgura_sleeve': _nan}])\n",
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
        "# Los ratios, con el periodo y la presentacion de los que salieron. Sin\n",
        "# esas dos columnas un P/E es un numero sin fecha, y un numero sin fecha\n",
        "# no se puede auditar contra el 10-K que lo produjo.\n",
        "fundamentales_df = pd.DataFrame()\n",
        "if fund_meta.get('con_ratios'):\n",
        "    fundamentales_df = pd.DataFrame([\n",
        "        {'ticker': _i.get('ticker'),\n",
        "         'periodo': (_i.get('fundamentals') or {}).get('periodo'),\n",
        "         'filed': (_i.get('fundamentals') or {}).get('filed'),\n",
        "         'acciones': (_i.get('fundamentals') or {}).get('acciones_fuente'),\n",
        "         **((_i.get('fundamentals') or {}).get('metricas') or {})}\n",
        "        for _i in market_data.get('instruments', [])\n",
        "        if _i.get('fundamentals')])\n",
        "\n",
        "# Todo lo que el Procedimiento no dice, en una hoja: el review encontro\n",
        "# esos numeros marcados 'de la mesa' cada uno en su celda.\n",
        "from screener.comite import pendientes\n",
        "pendientes_df = pd.DataFrame(pendientes())\n",
        "\n",
        "def _o_vacio(df, motivo):\n",
        "    return df if not df.empty else pd.DataFrame({'nota': [motivo]})\n",
        "\n",
        "with pd.ExcelWriter(ARCHIVO_EXCEL, engine='openpyxl') as _xl:\n",
        "    tabla.to_excel(_xl, sheet_name='Ranking', index=False)\n",
        "    mapa.to_excel(_xl, sheet_name='Bloques')\n",
        "    comparacion.to_excel(_xl, sheet_name='Perfiles')\n",
        "    views_excel.to_excel(_xl, sheet_name='Views BL', index=False)\n",
        "    cartera_df.to_excel(_xl, sheet_name='Cartera', index=False)\n",
        "    sectores_df.to_excel(_xl, sheet_name='Sectores', index=False)\n",
        "    riesgo_df.to_excel(_xl, sheet_name='Riesgo', index=False)\n",
        "    _o_vacio(estres_df, 'sin datos de estres').to_excel(\n",
        "        _xl, sheet_name='Estres', index=False)\n",
        "    _o_vacio(sensibilidad_df, 'sin cartera o sin views').to_excel(\n",
        "        _xl, sheet_name='Sensibilidad', index=False)\n",
        "    cesta_df.to_excel(_xl, sheet_name='Cesta', index=False)\n",
        "    universo.to_excel(_xl, sheet_name='Universo', index=False)\n",
        "    _cov.to_excel(_xl, sheet_name='Cobertura', index=False)\n",
        "    if not fundamentales_df.empty:\n",
        "        fundamentales_df.to_excel(_xl, sheet_name='Fundamentales',\n",
        "                                  index=False)\n",
        "    pendientes_df.to_excel(_xl, sheet_name='Pendientes_Comite', index=False)\n",
        "    parametros.to_excel(_xl, sheet_name='Parametros', index=False)\n",
        "\n",
        "    for _hoja in _xl.book.worksheets:\n",
        "        _hoja.freeze_panes = 'A2'\n",
        "        for _col in _hoja.columns:\n",
        "            _ancho = max((len(str(c.value)) for c in _col if c.value), default=8)\n",
        "            _hoja.column_dimensions[_col[0].column_letter].width = min(46, _ancho + 3)\n",
        "\n",
        "print(f'{ARCHIVO_EXCEL}  —  {len(scored)} nombres, '\n",
        "      f'{len(pd.ExcelFile(ARCHIVO_EXCEL).sheet_names)} hojas')\n",
        "\n",
        "# Reproducibilidad: el dato exacto, el motor y los parametros de esta\n",
        "# corrida. Sin esto una propuesta no se puede rastrear a lo que la produjo.\n",
        "from screener.corrida import guardar_corrida\n",
        "_carpeta_corrida = guardar_corrida(\n",
        "    'corridas', market_data=market_data, parametros=parametros,\n",
        "    views=views, cartera=cartera.weights, retornos=retornos,\n",
        "    tablas={'riesgo': riesgo_df, 'estres': estres_df,\n",
        "            'sensibilidad': sensibilidad_df})\n",
        "print(f'Corrida guardada para reproducirla: {_carpeta_corrida}')\n",
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
        "    import shutil\n",
        "    _corridas_drive = Path('/content/drive/MyDrive/CCI_BlackLitterman/corridas')\n",
        "    _corridas_drive.mkdir(parents=True, exist_ok=True)\n",
        "    shutil.copytree(_carpeta_corrida, _corridas_drive / _carpeta_corrida.name)\n",
        "    print(f'Corrida en Drive: {_corridas_drive / _carpeta_corrida.name}')\n",
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
