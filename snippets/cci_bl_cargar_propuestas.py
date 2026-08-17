"""
Celda para pegar en el notebook Black-Litterman de CCI.

Lee las propuestas que generó el screener y las inyecta en el flujo que ya
existe, sin cambiar nada más del sistema.

Dónde ponerla
-------------
Entre el paso 4 (`generar_propuestas_views`) y el paso 5 (`flujo_aprobacion`)
de `run_cci_black_litterman`. Las propuestas del screener se suman a las que
genera el motor propio, y el gestor sigue aprobando, editando o rechazando cada
una igual que hoy.

Por qué NO carga desde `aprobadas/`
-----------------------------------
`aprobadas/` guarda lo que un gestor ya revisó y justificó. El screener produce
propuestas, no decisiones. Mezclarlas ahí borraría la distinción entre lo que
la máquina sugirió y lo que una persona firmó, que es justo el rastro que el
principio de Ejecución Supervisada existe para dejar.
"""

import glob
import json
import os
from typing import Dict, List

# BASE_DIR ya está definido en el notebook de CCI; este es el valor por defecto.
BASE_DIR = globals().get("BASE_DIR", "/content/drive/MyDrive/CCI_BlackLitterman")
CARPETA_PROPUESTAS = os.path.join(BASE_DIR, "propuestas")


def cargar_propuestas_screener(estrategia: str,
                               max_dias: int = 7) -> List[Dict]:
    """
    Devuelve las propuestas del screener para una estrategia.

    Toma el archivo más reciente y avisa si está viejo: un screening de hace
    dos semanas describe un mercado que ya no existe, y una view estancada es
    peor que ninguna porque parece actual.
    """
    import datetime

    patron = os.path.join(CARPETA_PROPUESTAS,
                          f"{estrategia}_screener_propuestas_*.json")
    archivos = sorted(glob.glob(patron))
    if not archivos:
        print(f"Sin propuestas del screener para {estrategia} en {CARPETA_PROPUESTAS}")
        return []

    ruta = archivos[-1]
    with open(ruta, encoding="utf-8") as fh:
        payload = json.load(fh)

    views = payload.get("views", [])
    calib = payload.get("calibracion", {})

    # Antigüedad, a partir de la fecha en el nombre del archivo.
    try:
        fecha_txt = os.path.basename(ruta).rsplit("_", 1)[-1].replace(".json", "")
        edad = (datetime.date.today()
                - datetime.date.fromisoformat(fecha_txt)).days
    except ValueError:
        edad = None

    print(f"Propuestas del screener: {os.path.basename(ruta)}")
    print(f"  {len(views)} views | perfil {payload.get('perfil_screener')} "
          f"| universo {payload.get('universo_puntuado')} nombres")
    print(f"  IC supuesto {calib.get('information_coefficient')} "
          f"— {calib.get('nota', '')}")
    if edad is not None and edad > max_dias:
        print(f"  AVISO: el archivo tiene {edad} días. Vuelve a correr el "
              f"screener antes de usar estas views.")

    # El origen viaja con cada view para que la bitácora distinga después qué
    # propuso la máquina y qué propuso el motor propio.
    for v in views:
        v.setdefault("origen", "screener")
    return views


def fusionar(propuestas_motor: List[Dict],
             propuestas_screener: List[Dict],
             max_total: int = 8) -> List[Dict]:
    """
    Une ambas fuentes y resuelve duplicados quedándose con la de mayor convicción.

    Un mismo activo propuesto por las dos fuentes no son dos views: en
    Black-Litterman serían dos filas de P casi idénticas, lo que estrecha Omega
    artificialmente y le da a esa apuesta un peso que ninguna de las dos fuentes
    justifica por sí sola.
    """
    def clave(v: Dict) -> tuple:
        if v.get("tipo") == "relativo":
            return ("rel", frozenset({v["activo_long"], v["activo_short"]}))
        return ("abs", v.get("activo"))

    mejor: Dict[tuple, Dict] = {}
    for v in list(propuestas_motor) + list(propuestas_screener):
        k = clave(v)
        if k not in mejor or v.get("conviccion", 0) > mejor[k].get("conviccion", 0):
            mejor[k] = v

    fusionadas = sorted(mejor.values(),
                        key=lambda v: v.get("conviccion", 0), reverse=True)

    n_motor = sum(1 for v in fusionadas if v.get("origen") != "screener")
    n_screener = len(fusionadas) - n_motor
    descartadas = len(propuestas_motor) + len(propuestas_screener) - len(fusionadas)
    print(f"Fusionadas: {len(fusionadas)} views "
          f"({n_motor} motor propio, {n_screener} screener); "
          f"{descartadas} duplicadas descartadas")

    return fusionadas[:max_total]


# --------------------------------------------------------------------------
# Uso dentro de run_cci_black_litterman, reemplazando el paso 4-5:
# --------------------------------------------------------------------------
#
#     propuestas = generar_propuestas_views(df_cesta, factores, macro_env,
#                                           flujos, est)
#     propuestas_scr = cargar_propuestas_screener(est)
#     propuestas = fusionar(propuestas, propuestas_scr,
#                           max_total=3 if macro_env['divergencia'] else 8)
#
#     views_aprobadas = flujo_aprobacion(propuestas, est, df_cesta,
#                                        macro_env, factores, flujos)
#
# El gestor sigue viendo cada view y decidiendo. Nada se aplica sin su
# aprobación explícita.
