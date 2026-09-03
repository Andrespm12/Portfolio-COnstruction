#!/usr/bin/env python3
"""
Arma modelo_cci.zip: el modelo para correr en una máquina, sin clonar el repo.

    python3 scripts/build_bundle.py                 # a ./output/modelo_cci.zip
    python3 scripts/build_bundle.py --salida /tmp

Qué lleva y qué no
------------------
Lleva el paquete ``screener/``, los dos programas que se corren a mano
(``correr_modelo.py`` y ``bajar_tenencias.py``), el ``requirements.txt`` y la
carpeta ``tenencias/`` con sus instrucciones. No lleva pruebas, notebook, web
ni documentación: quien recibe el zip lo quiere para correr, y cada archivo de
más es una pregunta de más.

Los dos programas viven en ``scripts/`` dentro del repo y en la raíz dentro del
zip, así que la línea que arma ``ROOT`` no puede ser la misma. Se reescribe
aquí, y ``tests/test_bundle.py`` verifica que el zip resultante importa y corre.

Para Colab no hace falta este zip: el notebook trae el motor embebido.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Programas que se corren a mano. En el repo cuelgan de ``scripts/``.
PROGRAMAS = ("correr_modelo.py", "bajar_tenencias.py")

#: La línea que hay que reescribir, y por qué. Dentro del repo el programa está
#: en ``scripts/`` y la raíz es el padre; dentro del zip está en la raíz.
_ROOT_REPO = re.compile(
    r"^ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\].*$", re.MULTILINE)
_ROOT_ZIP = "ROOT = Path(__file__).resolve().parent   # paquete autocontenido"

REQUIREMENTS = """pandas
numpy
yfinance
openpyxl
cvxpy
scikit-learn
"""

LEEME_TENENCIAS = """COMPOSICIÓN DE ETFs — para el reporte de transparencia (sección 8b)
==================================================================

UN SOLO COMANDO
---------------
    python3 bajar_tenencias.py

Lo baja todo de yfinance, que ya es dependencia del modelo. No hay que
contratar proveedor, ni bajar archivos de cada emisor, ni llenar filas a mano.

    python3 bajar_tenencias.py SPY IVV QQQ EFA     # solo algunos
    python3 bajar_tenencias.py --universo          # todos los del universo

QUÉ ESCRIBE
-----------
    SPY.csv          mayores posiciones + fila _RESTO
    _sectores.csv    desglose sectorial COMPLETO por fondo
    _canasta.csv     P/E, P/B y demás de la canasta, cuando Yahoo los trae

Yahoo publica las mayores posiciones, no las 500. Alcanza: un nombre que pueda
romper un tope del 15% o 20% está entre las primeras o no está. El peso no
detallado se escribe como fila _RESTO, que es lo que evita que la normalización
infle los pesos listados.

El desglose sectorial sí es completo — es el total del fondo, no una muestra —
así que la exposición sectorial sale bien aunque las tenencias sean parciales.

SI QUIERES AGREGAR UNO A MANO
-----------------------------
    ticker,weight
    AAPL,7.05
    MSFT,6.45
    _RESTO,86.50      <- obligatoria si no pusiste todas

Sin la fila _RESTO, 7.05 sobre un total de 13.50 se convierte en 52%, y el
reporte te acusaría de romper topes que no rompiste.

SI ESTA CARPETA ESTÁ VACÍA
--------------------------
El modelo corre igual. La sección 8b dice "cobertura 0%" y no muestra números,
en vez de estimar. Es a propósito.
"""

LEEME = """MODELO CCI — cómo correrlo
==========================

    pip install -r modelo/requirements.txt
    python3 modelo/bajar_tenencias.py --salida modelo/tenencias   # una vez
    python3 modelo/correr_modelo.py --tenencias modelo/tenencias

El primero baja la composición de los ETFs (necesita internet, y solo hace
falta repetirlo cuando quieras refrescarla). El segundo corre el modelo entero
y escribe el Excel y el JSON de views en el directorio actual; con --salida los
manda a otro lado.

Opciones que se usan
--------------------
    python3 modelo/correr_modelo.py --estrategia Agresivo
    python3 modelo/correr_modelo.py --universo lista --tickers SPY,QQQ,AAPL
    python3 modelo/correr_modelo.py --salida ./salida
    python3 modelo/correr_modelo.py --help      # todas

Este zip es para correr en tu máquina. Para Colab no hace falta: sube solo
notebooks/screener_colab.ipynb, que trae el motor adentro.
"""


def rewrite_root(source: str, nombre: str) -> str:
    """Apunta ``ROOT`` a la carpeta del propio programa, no a su padre."""
    nuevo, n = _ROOT_REPO.subn(_ROOT_ZIP, source, count=1)
    if n != 1:
        raise SystemExit(
            f"{nombre}: esperaba una línea 'ROOT = Path(__file__).resolve()."
            "parents[1]' para reescribir y encontré "
            f"{n}. Revisa el programa antes de armar el zip.")
    return nuevo


def build(destino: Path) -> Path:
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / "modelo_cci.zip"

    with zipfile.ZipFile(archivo, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / "screener").glob("*.py")):
            z.writestr(f"modelo/screener/{path.name}",
                       path.read_text(encoding="utf-8"))
        for nombre in PROGRAMAS:
            fuente = (ROOT / "scripts" / nombre).read_text(encoding="utf-8")
            z.writestr(f"modelo/{nombre}", rewrite_root(fuente, nombre))
        z.writestr("modelo/requirements.txt", REQUIREMENTS)
        z.writestr("modelo/tenencias/LEEME.txt", LEEME_TENENCIAS)
        z.writestr("LEEME.txt", LEEME)

    return archivo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--salida", default=str(ROOT / "output"),
                   help="Directorio donde dejar el zip.")
    args = p.parse_args(argv)

    archivo = build(Path(args.salida))
    kb = archivo.stat().st_size / 1024
    with zipfile.ZipFile(archivo) as z:
        n = len(z.namelist())
    print(f"escribí {archivo}  ({kb:.0f} KB, {n} archivos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
