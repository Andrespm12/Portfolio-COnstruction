"""
Prueba de que las pruebas pueden fallar.

Suena tautológico y no lo es. Diez archivos de este directorio verifican con
``check(etiqueta, condicion)``, que cuenta e imprime pero no levanta nada. Bajo
``pytest`` eso significaba que una prueba podía imprimir ``FAIL`` en cada línea
y aparecer en verde — y así fue como un ``AttributeError`` real llegó a Colab
con la suite entera aprobada.

``conftest.py`` lo arregla. Este archivo verifica que el arreglo funciona, y
sobre todo que **siga** funcionando: si alguien quita el hook, esta prueba se
cae y no un silencio.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODULO = '''\
PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


def test_que_pasa():
    check("esta si se cumple", True)


def test_que_falla():
    check("esta no se cumple", False, "detalle del fallo")
'''


def _correr(tmp_path: Path) -> subprocess.CompletedProcess:
    (tmp_path / "conftest.py").write_text(
        (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"),
        encoding="utf-8")
    (tmp_path / "test_muestra.py").write_text(MODULO, encoding="utf-8")
    return subprocess.run([sys.executable, "-m", "pytest", str(tmp_path), "-q"],
                          capture_output=True, text=True, cwd=tmp_path,
                          timeout=180)


def test_un_check_fallido_reprueba_la_prueba(tmp_path):
    r = _correr(tmp_path)
    assert r.returncode != 0, "una verificación fallida tiene que dar exit != 0"
    assert "1 failed" in r.stdout, r.stdout[-2000:]


def test_un_check_cumplido_no_reprueba_nada(tmp_path):
    r = _correr(tmp_path)
    assert "1 passed" in r.stdout, r.stdout[-2000:]


def test_el_fallo_se_reporta_como_prueba_fallida_no_como_error(tmp_path):
    # Desde un fixture de teardown pytest lo reportaba como "1 passed, 1 error",
    # y una prueba que detectó un fallo no puede salir en la línea de los
    # aprobados: quien lee el resumen lo cuenta como bueno.
    r = _correr(tmp_path)
    assert "error" not in r.stdout.lower().split("=====")[-1], r.stdout[-2000:]


def test_el_detalle_del_fallo_llega_al_reporte(tmp_path):
    # De nada sirve saber que algo falló si no dice qué.
    r = _correr(tmp_path)
    assert "detalle del fallo" in r.stdout, r.stdout[-2000:]


def test_los_diez_archivos_siguen_teniendo_el_contador_que_el_hook_lee():
    # El hook se engancha a la variable FAILED del módulo. Un archivo que la
    # renombre queda sin red sin que nada avise.
    faltan = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        texto = path.read_text(encoding="utf-8")
        if "def check(" in texto and "FAILED" not in texto:
            faltan.append(path.name)
    assert not faltan, f"usan check() pero no exponen FAILED: {faltan}"
