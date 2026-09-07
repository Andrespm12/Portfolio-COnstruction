#!/usr/bin/env python3
"""
Construye notebooks/fundamentales_colab.ipynb desde el repo.

    python3 scripts/build_notebook_fundamentales.py

Por qué un notebook aparte y no una sección del principal
---------------------------------------------------------
Bajar EDGAR es un trabajo de una vez por semana, no de cada corrida: una empresa
presenta cuatro veces al año. Meterlo en el notebook diario obligaría a esperar
la descarga completa cada vez que quieres un ranking, y la descarga son minutos,
no segundos.

Además la Fase 3 no existe todavía. El modelo de scoring **no consume** estos
datos aún, así que mezclarlos sugeriría una integración que no está hecha.

Comparte motor con el principal: el mismo tarball, la misma verificación
SHA256, el mismo paquete ``screener/``. Si los dos alguna vez dieran resultados
distintos sería un bug, no una diferencia de diseño.

El disco de Colab es efímero
----------------------------
Es la diferencia que manda sobre el diseño de este notebook. Bajar cientos de
``companyfacts`` y perderlos cuando el runtime se recicla sería trabajo tirado,
así que el almacén va a Drive por defecto y la descarga es reanudable: lo que ya
está en disco no se vuelve a pedir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_notebook import build_payload, code, md, wrap  # noqa: E402

OUT = ROOT / "notebooks" / "fundamentales_colab.ipynb"


def build_cells() -> list[dict]:
    blob, digest = build_payload()
    cells: list[dict] = []

    cells.append(md(
        "# Fundamentales desde SEC EDGAR\n",
        "\n",
        "Baja los estados financieros que la SEC publica, **con la fecha en "
        "que se presentaron**, y reporta qué porcentaje del universo tiene "
        "cada métrica de verdad.\n",
        "\n",
        "## Por qué EDGAR y no un proveedor\n",
        "\n",
        "El bloque `valuation_carry` pesa 10–12% del compuesto y su propio "
        "texto admite que corre con *proxies*: no hay P/E ni EV/EBITDA en "
        "ninguna parte del modelo. En una corrida real dos de sus tres "
        "métricas salieron `UNAVAILABLE from Yahoo`.\n",
        "\n",
        "Cualquier proveedor de múltiplos arregla eso. **Ninguno arregla el "
        "problema de abajo.** Un vendor te da el número de hoy, ya corregido, "
        "y un backtest alimentado con datos restatados está recibiendo "
        "información que nadie tenía entonces — el IC que salga de ahí está "
        "inflado por construcción.\n",
        "\n",
        "EDGAR no tiene ese problema porque no es un proveedor: **es el "
        "archivo**. Cada dato trae el `filed` de la presentación que lo "
        "trajo, y las versiones sucesivas conviven como entradas separadas. "
        "Filtrar `filed <= fecha` reconstruye lo que se sabía ese día por "
        "construcción, no por promesa de nadie.\n",
        "\n",
        "Gratis, sin llave, y es la fuente primaria de la que los vendors "
        "revenden.\n",
        "\n",
        "## El histórico viene desde la primera corrida\n",
        "\n",
        "A diferencia de los precios, aquí no hay que acumular nada. Una sola "
        "llamada devuelve **todo lo que la empresa ha reportado bajo XBRL**, "
        "que son unos diez años. No esperas: bajas y ya lo tienes.\n",
        "\n",
        "## Lo que este notebook NO hace\n",
        "\n",
        "No calcula ratios y **no toca el modelo de scoring**. Baja hechos, "
        "los mapea a conceptos declarados y reporta cobertura. Un P/E "
        "necesita casar un fundamental con un precio alineando las dos "
        "fechas, y esa es una decisión aparte que todavía no está tomada.\n",
        "\n",
        "El entregable es el **reporte de cobertura**: con él se decide si "
        "vale la pena construir el bloque fundamental, sobre número medido y "
        "no sobre esperanza.\n",
    ))

    # ------------------------------------------------------------- motor
    cells.append(md("## 1 · Motor\n"))
    cells.append(code(
        "# El paquete screener/ del repo, embebido. Mismo tarball y misma\n",
        "# verificacion que el notebook principal: un motor solo, no dos.\n",
        "import base64, gzip, hashlib, io, sys, tarfile\n",
        "\n",
        f'ENGINE_SHA256 = "{digest}"\n',
        "ENGINE_B64 = (\n",
        *[f'    "{linea}"\n' for linea in wrap(blob)],
        ")\n",
        "\n",
        "raw = gzip.decompress(base64.b64decode(ENGINE_B64))\n",
        "digest = hashlib.sha256(raw).hexdigest()\n",
        "assert digest == ENGINE_SHA256, f'motor alterado: {digest}'\n",
        "with tarfile.open(fileobj=io.BytesIO(raw)) as tar:\n",
        "    tar.extractall('/content')\n",
        "if '/content' not in sys.path:\n",
        "    sys.path.insert(0, '/content')\n",
        "\n",
        "from screener import edgar\n",
        "print(f'motor verificado  sha256={ENGINE_SHA256[:16]}...')\n",
        "print(f'{len(edgar.CONCEPTOS)} conceptos declarados')\n",
    ))

    # -------------------------------------------------------- parametros
    cells.append(md(
        "## 2 · Parámetros\n",
        "\n",
        "**`CONTACTO` no es opcional.** La SEC exige identificarse con un "
        "correo real y **bloquea por IP** a quien no lo hace. No es "
        "burocracia: es la condición de uso de un servicio gratuito.\n",
        "\n",
        "**Empieza con `LIMITE = 3`.** Si tres nombres bajan bien, el resto es "
        "lo mismo repetido. Lanzar 400 peticiones para descubrir que el "
        "`User-Agent` estaba mal es la forma cara de aprenderlo.\n",
        "\n",
        "**Guardar en Drive es lo que hace la descarga reanudable.** El disco "
        "de Colab desaparece cuando el runtime se recicla; con el almacén en "
        "Drive, volver a correr esto retoma donde quedó en vez de empezar de "
        "cero.\n",
    ))
    cells.append(code(
        "# @markdown ### Identificación ante la SEC (obligatoria)\n",
        'CONTACTO = "CCI Puesto de Bolsa tucorreo@dominio.com"  # @param {type:"string"}\n',
        "\n",
        "# @markdown ### Qué bajar\n",
        'UNIVERSO = "sp500"  # @param ["sp500", "acciones", "ndx", "djia", "lista"]\n',
        'TICKERS_PERSONALIZADOS = "AAPL,MSFT,NVDA"  # @param {type:"string"}\n',
        "# @markdown Cuántos nombres como máximo. 0 = todos. **Empieza en 3.**\n",
        "LIMITE = 3  # @param {type:\"integer\"}\n",
        "\n",
        "# @markdown ### Dónde guardar\n",
        "GUARDAR_EN_DRIVE = True  # @param {type:\"boolean\"}\n",
        "# @markdown Sin Drive el almacén se pierde al reciclarse el runtime y\n",
        "# @markdown la próxima corrida vuelve a bajar todo.\n",
        "REBAJAR_TODO = False  # @param {type:\"boolean\"}\n",
        "# @markdown Marca solo para refrescar lo que ya está en disco.\n",
        "\n",
        "from pathlib import Path\n",
        "\n",
        "edgar.user_agent(CONTACTO)   # falla aqui si falta el correo\n",
        "\n",
        "if GUARDAR_EN_DRIVE:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    DESTINO = Path('/content/drive/MyDrive/CCI_Fundamentales')\n",
        "else:\n",
        "    DESTINO = Path('/content/fundamentales')\n",
        "DESTINO.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "if UNIVERSO == 'lista':\n",
        "    TICKERS = [t.strip().upper() for t in TICKERS_PERSONALIZADOS.split(',')\n",
        "               if t.strip()]\n",
        "else:\n",
        "    from screener.universe import all_index_members\n",
        "    _grupos = {'sp500': ('SP500',), 'acciones': ('SP500', 'NDX', 'DJIA'),\n",
        "               'ndx': ('NDX',), 'djia': ('DJIA',)}\n",
        "    _miembros = all_index_members()\n",
        "    _fuera = set()\n",
        "    for _clave in _grupos[UNIVERSO]:\n",
        "        _fuera |= set(_miembros.get(_clave, frozenset()))\n",
        "    # Yahoo usa guion donde la SEC usa punto (BRK-B vs BRK.B); el mapa de\n",
        "    # la SEC trae guion.\n",
        "    TICKERS = sorted(t.replace('.', '-') for t in _fuera)\n",
        "\n",
        "if LIMITE:\n",
        "    TICKERS = TICKERS[:LIMITE]\n",
        "\n",
        "print(f'{len(TICKERS)} nombre(s) -> {DESTINO}')\n",
        "print(f'Contacto: {CONTACTO}')\n",
        "_ya = len(list(DESTINO.glob('[!_]*.csv')))\n",
        "if _ya:\n",
        "    print(f'Ya en el almacen: {_ya} nombre(s). '\n",
        "          + ('Se rebajan todos.' if REBAJAR_TODO else 'No se vuelven a pedir.'))\n",
        form=True,
    ))

    # ---------------------------------------------------------- descarga
    cells.append(md(
        "## 3 · Descarga\n",
        "\n",
        "Una petición por emisor, espaciadas para no pasar del tope de la SEC. "
        "Un `companyfacts` de una empresa grande pesa 10–15 MB, así que el "
        "universo completo son varios minutos y unos pocos GB de tráfico — "
        "de los que se guarda menos del 5%, que es lo que declara "
        "`CONCEPTOS`.\n",
        "\n",
        "Si el runtime se cae a mitad, vuelve a correr esta celda: lo que ya "
        "está en disco no se vuelve a pedir.\n",
        "\n",
        "Un emisor que falle no tumba la corrida. Los extranjeros que "
        "presentan 20-F suelen traer menos etiquetas, y algunos ninguna de "
        "las que conocemos; salen listados al final en vez de rellenarse.\n",
    ))
    cells.append(code(
        "import time\n",
        "\n",
        "limitador = edgar.Limitador()\n",
        "mapa = edgar.load_ticker_map(contacto=CONTACTO,\n",
        "                             cache=DESTINO / '_tickers.json',\n",
        "                             limitador=limitador)\n",
        "print(f'Mapa ticker->CIK: {len(mapa):,} emisores\\n')\n",
        "\n",
        "etiquetas, ok, sin_cik, fallaron, saltados = [], [], [], [], []\n",
        "_t0 = time.time()\n",
        "\n",
        "for _i, _tk in enumerate(TICKERS, 1):\n",
        "    if (DESTINO / f'{_tk}.csv').exists() and not REBAJAR_TODO:\n",
        "        saltados.append(_tk)\n",
        "        continue\n",
        "\n",
        "    _cik = mapa.get(_tk) or mapa.get(_tk.replace('-', '.'))\n",
        "    if not _cik:\n",
        "        sin_cik.append(_tk)\n",
        "        print(f'  [{_i:3d}/{len(TICKERS)}] {_tk:6s} sin CIK en la SEC')\n",
        "        continue\n",
        "\n",
        "    try:\n",
        "        _payload = edgar.company_facts(_cik, contacto=CONTACTO,\n",
        "                                       limitador=limitador)\n",
        "        _hechos, _elegidas = edgar.extract_facts(_payload, _tk)\n",
        "    except Exception as _exc:\n",
        "        fallaron.append(_tk)\n",
        "        print(f'  [{_i:3d}/{len(TICKERS)}] {_tk:6s} '\n",
        "              f'FALLO {type(_exc).__name__}: {_exc}')\n",
        "        continue\n",
        "\n",
        "    if not _hechos:\n",
        "        fallaron.append(_tk)\n",
        "        print(f'  [{_i:3d}/{len(TICKERS)}] {_tk:6s} '\n",
        "              'sin ninguna etiqueta conocida (¿emisor extranjero?)')\n",
        "        continue\n",
        "\n",
        "    edgar.escribir_hechos(DESTINO, _tk, _hechos)\n",
        "    ok.append(_tk)\n",
        "    for _m, _e in _elegidas.items():\n",
        "        etiquetas.append({'ticker': _tk, 'metrica': _m, 'etiqueta': _e})\n",
        "    print(f'  [{_i:3d}/{len(TICKERS)}] {_tk:6s} {len(_hechos):5d} hechos, '\n",
        "          f'{len(_elegidas)}/{len(edgar.CONCEPTOS)} metricas')\n",
        "\n",
        "print(f'\\n{len(ok)} bajados, {len(saltados)} ya estaban, '\n",
        "      f'{len(sin_cik)} sin CIK, {len(fallaron)} fallaron '\n",
        "      f'({time.time() - _t0:.0f}s)')\n",
        "if sin_cik:\n",
        "    print(f'  Sin CIK: {\", \".join(sin_cik[:20])}')\n",
        "if fallaron:\n",
        "    print(f'  Fallaron: {\", \".join(fallaron[:20])}')\n",
    ))

    # -------------------------------------------------------- cobertura
    cells.append(md(
        "## 4 · Cobertura — el entregable\n",
        "\n",
        "Ésta es la tabla que decide si vale la pena construir el bloque "
        "fundamental. Con 60% de cobertura, un z-score transversal compara a "
        "los que reportaron contra un hueco, y eso no es una medición.\n",
        "\n",
        "Una métrica ausente sale en **0%, no desaparece**. Desaparecer se lee "
        "como *no aplica*; cero se lee como *no lo tenemos*.\n",
        "\n",
        "Mira la columna `etiquetas_usadas`. Si dice 3, significa que tres "
        "etiquetas XBRL distintas trajeron la misma idea en el mismo "
        "universo — XBRL es un vocabulario, no un esquema, y sin la tabla de "
        "prioridad de `CONCEPTOS` una parte de tus emisores habría quedado "
        "sin ese dato.\n",
    ))
    cells.append(code(
        "import pandas as pd\n",
        "\n",
        "if etiquetas:\n",
        "    _modo = 'w' if REBAJAR_TODO or not (DESTINO / '_etiquetas.csv').exists() else 'a'\n",
        "    pd.DataFrame(etiquetas).to_csv(DESTINO / '_etiquetas.csv',\n",
        "                                   mode=_modo, index=False,\n",
        "                                   header=(_modo == 'w'))\n",
        "\n",
        "hechos = edgar.leer_hechos(DESTINO, TICKERS)\n",
        "if hechos.empty:\n",
        "    print('No hay nada en el almacen todavia.')\n",
        "else:\n",
        "    cobertura = edgar.coverage_report(hechos, TICKERS)\n",
        "    cobertura.to_csv(DESTINO / '_cobertura.csv', index=False)\n",
        "\n",
        "    historia = edgar.historia_por_ticker(hechos)\n",
        "    print(f'{len(historia)} nombre(s), {len(hechos):,} hechos, '\n",
        "          f'de {historia[\"desde\"].min()} a {historia[\"hasta\"].max()}')\n",
        "    print(f'Periodos distintos por nombre: mediana '\n",
        "          f'{historia[\"periodos\"].median():.0f}')\n",
        "\n",
        "\n",
        "def _semaforo(v):\n",
        "    \"\"\"Rojo bajo 60%, ambar hasta 85%, verde arriba.\n",
        "\n",
        "    A mano y no con background_gradient porque ese exige matplotlib, y\n",
        "    una dependencia mas es una forma mas de que la celda reviente en la\n",
        "    maquina de otro.\n",
        "    \"\"\"\n",
        "    if v is None or not isinstance(v, (int, float)):\n",
        "        return ''\n",
        "    if v < 0.60:\n",
        "        return 'background-color:#7F1D1D;color:#FFFFFF'\n",
        "    if v < 0.85:\n",
        "        return 'background-color:#78350F;color:#FFFFFF'\n",
        "    return 'background-color:#14532D;color:#FFFFFF'\n",
        "\n",
        "display(cobertura[['metrica', 'cobertura', 'con_dato', 'sin_dato',\n",
        "                   'etiquetas_usadas', 'etiqueta_principal']].style\n",
        "        .format({'cobertura': '{:.1%}'})\n",
        "        .map(_semaforo, subset=['cobertura'])\n",
        "        .hide(axis='index')\n",
        "        .set_caption('Cobertura por metrica'))\n",
    ))

    # ------------------------------------------------------ point-in-time
    cells.append(md(
        "## 5 · El point-in-time, visto\n",
        "\n",
        "La razón de ser de todo esto, en una tabla.\n",
        "\n",
        "Cada fila es un período que se reportó **más de una vez con cifras "
        "distintas**: la empresa presentó un número y después lo corrigió. Un "
        "proveedor te habría dado directamente el corregido, y un backtest "
        "alimentado con él estaría viendo algo que en su momento nadie vio.\n",
        "\n",
        "Si esta tabla sale vacía no es que el mecanismo falle: es que en el "
        "universo que bajaste nadie corrigió nada por encima del 2%. Con "
        "cientos de nombres y diez años, salen.\n",
    ))
    cells.append(code(
        "if not hechos.empty:\n",
        "    rest = edgar.restatements(hechos)\n",
        "    print(f'{len(rest)} periodo(s) reportados dos veces con cambio >= 2%')\n",
        "    if not rest.empty:\n",
        "        rest.to_csv(DESTINO / '_restatements.csv', index=False)\n",
        "        display(rest.head(15).style\n",
        "                .format({'cambio': '{:+.1%}', 'primero': '{:,.0f}',\n",
        "                         'ultimo': '{:,.0f}'})\n",
        "                .hide(axis='index')\n",
        "                .set_caption('Lo que un proveedor te habria dado ya corregido'))\n",
    ))

    cells.append(md(
        "### La misma pregunta, dos fechas\n",
        "\n",
        "`as_of(fecha)` devuelve la última versión de cada período presentada "
        "**en o antes** de ese día. Cambia `FECHA_CORTE` y mira cómo cambia "
        "el número: eso es exactamente lo que un backtest honesto necesita y "
        "lo que ningún vendor te puede dar.\n",
    ))
    cells.append(code(
        'FECHA_CORTE = "2024-06-30"  # @param {type:"date"}\n',
        'METRICA = "activos"  # @param ["ingresos", "utilidad_neta", "activos", "patrimonio", "ebit", "efectivo", "flujo_operativo", "capex", "eps_diluido"]\n',
        "\n",
        "if not hechos.empty:\n",
        "    _entonces = edgar.as_of(hechos, FECHA_CORTE, metricas=[METRICA])\n",
        "    _hoy = edgar.as_of(hechos, None, metricas=[METRICA])\n",
        "    print(f'{METRICA}: {len(_entonces)} hecho(s) conocibles al "
        "{FECHA_CORTE}, '\n",
        "          f'{len(_hoy)} conocidos hoy')\n",
        "    if not _entonces.empty:\n",
        "        display(_entonces[['ticker', 'fin', 'valor', 'filed', 'forma',\n",
        "                           'etiqueta']].head(20))\n",
    ))

    # ---------------------------------------------------------- descarga
    cells.append(md(
        "## 6 · Llevarte el almacén\n",
        "\n",
        "Si guardaste en Drive ya está a salvo y esta celda sobra. Si no, "
        "**bájalo antes de cerrar**: el disco de Colab se recicla y con él se "
        "va la descarga entera.\n",
    ))
    cells.append(code(
        "import shutil\n",
        "\n",
        "_zip = shutil.make_archive('/content/fundamentales', 'zip', DESTINO)\n",
        "print(f'{_zip}  ({Path(_zip).stat().st_size / 1e6:.1f} MB)')\n",
        "\n",
        "try:\n",
        "    from google.colab import files\n",
        "    files.download(_zip)\n",
        "except Exception as _exc:\n",
        "    print(f'Fuera de Colab, no hay descarga automatica: {_exc}')\n",
    ))

    cells.append(md(
        "---\n",
        "\n",
        "## Qué hacer con la tabla de cobertura\n",
        "\n",
        "Si `ingresos`, `patrimonio` y `activos` salen **por encima de 90%**, "
        "el bloque fundamental es viable y la Fase 3 tiene sentido.\n",
        "\n",
        "Si salen **cerca de 60%**, el z-score transversal estaría comparando "
        "a los que reportaron contra un hueco. Eso no se arregla rellenando "
        "con el promedio del sector — daría un número con apariencia de "
        "medición — sino ampliando el mapeo de `CONCEPTOS` o aceptando que el "
        "bloque solo aplica a un subconjunto declarado del universo.\n",
        "\n",
        "**Un aviso sobre lo que esto todavía no arregla:** calibrar el IC "
        "sobre el universo actual lo **sobreestima**, porque la lista de "
        "nombres es una foto estática con sesgo de supervivencia — las "
        "empresas que quebraron no están. EDGAR sí las tiene. Hasta que el "
        "universo se arregle, un IC medido será mejor que el 0.08 supuesto "
        "sin ser todavía el número bueno.\n",
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
    texto = json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"
    OUT.write_text(texto, encoding="utf-8")
    print(f"escribí {OUT}  ({len(texto) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
