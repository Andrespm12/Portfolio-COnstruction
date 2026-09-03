"""
Hace que las pruebas de este directorio puedan fallar bajo pytest.

El problema
-----------
Diez archivos de prueba se escribieron como programas sueltos: cada uno tiene
``check(etiqueta, condicion)``, que **cuenta e imprime** pero no levanta nada, y
un ``main()`` al final que devuelve 1 si algo falló. Corridos como
``python3 tests/test_optimizer.py`` funcionan perfecto.

Corridos con ``pytest tests/``, no. pytest recoge las funciones ``test_*``, las
ejecuta, ninguna hace ``assert``, y las reporta **todas en verde** aunque hayan
impreso ``FAIL`` en cada línea. Fue así como un ``AttributeError`` real llegó a
Colab con la suite entera en verde: la prueba sí detectó el fallo, lo imprimió,
y pytest dijo que había pasado.

Es exactamente el defecto que este proyecto lleva meses sacando del código —
``auditar_bandas`` escribía "Auditoría OK" sin comparar nada — reproducido en
las pruebas que deberían haberlo evitado. Una verificación que no puede fallar
no es una verificación.

La solución
-----------
Un fixture automático mira el contador ``FAILED`` del módulo antes y después de
cada prueba. Si subió, la prueba falla, y pytest muestra la salida capturada,
donde está la línea ``FAIL`` con su detalle.

Los dos modos de correr siguen sirviendo: ``main()`` no se toca y sigue
devolviendo el código de salida correcto.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """
    Convierte los ``check()`` fallidos del módulo en una prueba fallida.

    Va en la fase de llamada y no en un fixture de teardown a propósito: desde
    el teardown, pytest lo reporta como ``1 passed, 1 error``, y una prueba que
    detectó un fallo no puede aparecer en la línea de los aprobados.
    """
    module = pyfuncitem.module
    before = getattr(module, "FAILED", None)

    result = yield          # si la prueba levanta, la excepción pasa por aquí

    if before is not None and module.FAILED > before:
        nuevos = module.FAILED - before
        pytest.fail(
            f"{nuevos} verificación(es) de check() fallaron. El detalle está "
            "en la salida capturada, en las líneas que empiezan con FAIL.",
            pytrace=False,
        )
    return result
