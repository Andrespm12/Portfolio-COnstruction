"""
Pruebas de scripts/build_bundle.py.

El zip se armaba a mano, y a mano se olvida un archivo. Lo que aquí se
verifica es lo que rompería la corrida de quien lo recibe: que estén todos los
módulos, que los dos programas importen el paquete desde su nueva ubicación, y
que la carpeta de tenencias venga con instrucciones en vez de vacía.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_bundle import PROGRAMAS, build, rewrite_root  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("bundle"))


def nombres(bundle: Path) -> set[str]:
    with zipfile.ZipFile(bundle) as z:
        return set(z.namelist())


# ------------------------------------------------------------------ contenido
def test_lleva_el_paquete_completo(bundle):
    del_repo = {f"modelo/screener/{p.name}"
                for p in (ROOT / "screener").glob("*.py")}
    faltan = del_repo - nombres(bundle)
    assert not faltan, f"módulos que quedaron fuera del zip: {sorted(faltan)}"


def test_lleva_los_dos_programas(bundle):
    assert {f"modelo/{n}" for n in PROGRAMAS} <= nombres(bundle)


def test_lleva_requirements_y_los_dos_leeme(bundle):
    assert {"modelo/requirements.txt", "modelo/tenencias/LEEME.txt",
            "LEEME.txt"} <= nombres(bundle)


def test_no_lleva_pruebas_ni_notebook(bundle):
    sobra = [n for n in nombres(bundle)
             if "test" in n or n.endswith(".ipynb") or "/web/" in n]
    assert not sobra, f"el zip es para correr, no para leer: {sobra}"


def test_requirements_trae_el_solver_y_la_contraccion(bundle):
    # cvxpy resuelve la cartera y scikit-learn hace Ledoit-Wolf. Sin uno de los
    # dos el modelo llega hasta el ranking y muere en la optimización.
    with zipfile.ZipFile(bundle) as z:
        req = z.read("modelo/requirements.txt").decode()
    assert {"cvxpy", "scikit-learn", "yfinance", "openpyxl"} <= set(req.split())


def test_el_leeme_de_tenencias_explica_la_fila_resto(bundle):
    # Es lo único que un archivo escrito a mano puede tener mal y que produce
    # una acusación falsa de incumplimiento.
    with zipfile.ZipFile(bundle) as z:
        texto = z.read("modelo/tenencias/LEEME.txt").decode()
    assert "_RESTO" in texto


# ------------------------------------------------------------------ ROOT
def test_root_se_reescribe_a_la_carpeta_del_programa():
    fuente = ("import sys\n"
              "ROOT = Path(__file__).resolve().parents[1]\n"
              "sys.path.insert(0, str(ROOT))\n")
    salida = rewrite_root(fuente, "x.py")
    assert "parents[1]" not in salida
    assert "resolve().parent " in salida


def test_un_programa_sin_esa_linea_detiene_el_armado():
    # Si alguien cambia cómo correr_modelo.py arma su ROOT, el zip saldría con
    # un sys.path que apunta afuera y fallaría al importar screener en la
    # máquina de quien lo recibe. Mejor que no salga.
    with pytest.raises(SystemExit):
        rewrite_root("import sys\n", "x.py")


@pytest.mark.parametrize("programa", PROGRAMAS)
def test_los_programas_del_zip_importan_el_paquete(bundle, tmp_path, programa):
    destino = tmp_path / programa.replace(".py", "")
    with zipfile.ZipFile(bundle) as z:
        z.extractall(destino)

    # --help importa el módulo entero (y con él screener/) antes de leer
    # argumentos: si el sys.path del zip estuviera mal, esto es lo que falla.
    r = subprocess.run([sys.executable, str(destino / "modelo" / programa),
                        "--help"],
                       capture_output=True, text=True, cwd=tmp_path, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage" in r.stdout.lower()


def test_el_zip_no_depende_del_repo(bundle, tmp_path):
    # Se extrae en un directorio cualquiera y se importa desde ahí, con el repo
    # fuera del sys.path. Es la situación real de quien recibe el zip.
    destino = tmp_path / "suelto"
    with zipfile.ZipFile(bundle) as z:
        z.extractall(destino)

    codigo = ("import sys; sys.path.insert(0, '.');"
              "from screener.optimizer import policy_weights;"
              "from screener.tenencias_yahoo import bajar_varios;"
              "from screener.lookthrough import report;"
              "print('ok')")
    r = subprocess.run([sys.executable, "-c", codigo], capture_output=True,
                       text=True, cwd=destino / "modelo", timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout
