"""
Ejecuta el notebook de fundamentales y lo compara con el repo.

Dos cosas distintas:

**Deriva.** El notebook lleva una copia del paquete ``screener``. Una copia que
se queda atrás en silencio es el riesgo entero de embarcar una, así que se
reconstruye desde el código actual y se falla si el archivo versionado difiere.

**Ejecución.** Cada celda corre en orden, con dos sustituciones: la llamada a
EDGAR y ``google.colab``, que no existe fuera de Colab. Todo lo demás —
desempacar el motor, el universo, la extracción, el almacén, la cobertura, el
point-in-time — corre de verdad contra el mismo código que correrá en Colab.

Lo que esto NO prueba: que ``data.sec.gov`` devuelva lo que el adaptador espera.
Esa llamada queda sin ejercitar hasta que el notebook corra con internet.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

NOTEBOOK = ROOT / "notebooks" / "fundamentales_colab.ipynb"

TICKERS_SEC = {"0": {"cik_str": 320193, "ticker": "AAPL"},
               "1": {"cik_str": 789019, "ticker": "MSFT"},
               "2": {"cik_str": 1045810, "ticker": "NVDA"}}


def _facts(cik, base, etiqueta="Revenues"):
    """Diez años de historia, con una corrección en el medio."""
    ingresos, activos = [], []
    for anio in range(2015, 2025):
        filed = f"{anio + 1}-02-15"
        ingresos.append({"val": base * (1.08 ** (anio - 2015)),
                         "start": f"{anio}-01-01", "end": f"{anio}-12-31",
                         "filed": filed, "form": "10-K", "accn": f"a{anio}",
                         "fy": anio, "fp": "FY"})
        activos.append({"val": base * 10 * (1.05 ** (anio - 2015)),
                        "end": f"{anio}-12-31", "filed": filed, "form": "10-K",
                        "accn": f"a{anio}", "fy": anio, "fp": "FY"})
    # El cierre de 2020, corregido en 2023: esto es lo que as_of tiene que ver.
    ingresos.append({"val": base * (1.08 ** 5) * 1.15, "start": "2020-01-01",
                     "end": "2020-12-31", "filed": "2023-02-15", "form": "10-K",
                     "accn": "corr", "fy": 2022, "fp": "FY"})
    return {"cik": cik, "facts": {"us-gaap": {
        etiqueta: {"units": {"USD": ingresos}},
        "Assets": {"units": {"USD": activos}},
        "StockholdersEquity": {"units": {"USD": activos}}}}}


CUERPOS = {
    "0000320193": _facts(320193, 100.0),
    "0000789019": _facts(789019, 200.0),
    "0001045810": _facts(
        1045810, 300.0,
        etiqueta="RevenueFromContractWithCustomerExcludingAssessedTax"),
}

#: Sustituye la red y google.colab antes de que corra ninguna celda.
PRELUDIO = """
import sys, types
import screener.edgar as _edgar

def _fetch(url, *, contacto=None, limitador=None):
    if url == _edgar.SEC_TICKERS:
        return _TICKERS_SEC
    cik = url.rsplit('CIK', 1)[-1].removesuffix('.json')
    if cik not in _CUERPOS:
        raise RuntimeError(f'404 para {cik}')
    return _CUERPOS[cik]

_edgar.fetch_json = _fetch
_edgar.SEC_MAX_RPS = 0          # sin espera artificial en la prueba

# google.colab no existe fuera de Colab. Se declara ausente en vez de
# simularse: la celda de Drive tiene que degradar sola, que es lo que hara
# cuando alguien corra esto en jupyter local.
sys.modules['google'] = types.ModuleType('google')
"""


def executable_cells(nb: dict) -> list[str]:
    fuera = []
    for celda in nb["cells"]:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])
        fuente = "\n".join(l for l in fuente.splitlines()
                           if not l.strip().startswith("%"))
        fuera.append(fuente)
    return fuera


# --------------------------------------------------------------------- deriva
def test_el_notebook_versionado_sale_de_la_fuente():
    import build_notebook_fundamentales as builder

    fresco = json.dumps(builder.build_notebook(), indent=1,
                        ensure_ascii=False) + "\n"
    assert fresco == NOTEBOOK.read_text(encoding="utf-8"), \
        "corre: python3 scripts/build_notebook_fundamentales.py"


def test_la_construccion_es_determinista():
    import build_notebook_fundamentales as builder
    a = json.dumps(builder.build_notebook(), ensure_ascii=False)
    b = json.dumps(builder.build_notebook(), ensure_ascii=False)
    assert a == b


def test_el_motor_embebido_coincide_con_el_repo():
    import base64
    import gzip
    import io
    import re
    import tarfile

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celda = next(c for c in nb["cells"] if c["cell_type"] == "code"
                 and "ENGINE_B64" in "".join(c["source"]))
    blob = "".join(re.findall(r'^\s*"([A-Za-z0-9+/=]+)"\s*$',
                              "".join(celda["source"]), re.M))
    crudo = gzip.decompress(base64.b64decode(blob))

    viejos = []
    with tarfile.open(fileobj=io.BytesIO(crudo)) as tar:
        for nombre in tar.getnames():
            if tar.extractfile(nombre).read() != (ROOT / nombre).read_bytes():
                viejos.append(nombre)
    assert not viejos, f"módulos desactualizados: {viejos}"


def test_comparte_motor_con_el_notebook_principal():
    # Dos motores distintos serían dos modelos distintos con el mismo nombre.
    import build_notebook as principal
    import build_notebook_fundamentales as fundamentales

    assert principal.build_payload()[1] == fundamentales.build_payload()[1]


def test_no_hay_salidas_ni_contadores_versionados():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    codigo = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert all(c["outputs"] == [] for c in codigo)
    assert all(c["execution_count"] is None for c in codigo)


# ------------------------------------------------------------------ ejecución
@pytest.fixture(scope="module")
def corrida():
    """Ejecuta el notebook entero una vez y devuelve su espacio de nombres."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas = executable_cells(nb)

    trabajo = tempfile.mkdtemp(prefix="nb-fund-")

    parcheadas = 0
    for i, fuente in enumerate(celdas):
        if "GUARDAR_EN_DRIVE" in fuente:
            celdas[i] = (fuente
                         .replace("GUARDAR_EN_DRIVE = True",
                                  "GUARDAR_EN_DRIVE = False")
                         .replace('UNIVERSO = "sp500"', 'UNIVERSO = "lista"')
                         .replace('TICKERS_PERSONALIZADOS = "AAPL,MSFT,NVDA"',
                                  'TICKERS_PERSONALIZADOS = "AAPL,MSFT,NVDA,NOEXISTE"')
                         .replace("LIMITE = 3", "LIMITE = 0")
                         # /content/fundamentales es absoluto y sobrevive entre
                         # corridas del contenedor: la segunda vez se saltaria
                         # todo y la prueba mediria nada.
                         .replace("Path('/content/fundamentales')",
                                  f"Path({trabajo!r})"))
            parcheadas += 1
    assert parcheadas == 1, "la celda de parámetros cambió de forma"

    # display() RENDERIZA de verdad. Un stub que ignora su argumento deja sin
    # ejercitar toda la cadena de estilo, y ahi se escondia un
    # background_gradient que exige matplotlib: la celda habria reventado en
    # Colab con la suite entera en verde.
    renderizados: list[str] = []

    def _display(*objetos, **kw):
        for obj in objetos:
            if hasattr(obj, "to_html"):
                renderizados.append(obj.to_html())

    ns: dict = {"__name__": "__main__", "display": _display,
                "_TICKERS_SEC": TICKERS_SEC, "_CUERPOS": CUERPOS}
    cwd = os.getcwd()
    fallos: list[tuple[int, str]] = []

    try:
        os.chdir(trabajo)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # El preludio corre después de la celda del motor, que es la que
            # pone screener/ en sys.path.
            for i, fuente in enumerate(celdas):
                try:
                    exec(compile(fuente, f"<celda {i}>", "exec"), ns)
                    if "ENGINE_B64" in fuente:
                        exec(compile(PRELUDIO, "<preludio>", "exec"), ns)
                except Exception as exc:      # noqa: BLE001 - se reporta
                    fallos.append((i, f"{type(exc).__name__}: {exc}"))
    finally:
        os.chdir(cwd)

    ns["_fallos"] = fallos
    ns["_renderizados"] = renderizados
    ns["_trabajo"] = Path(trabajo)
    return ns


def test_todas_las_celdas_corren(corrida):
    assert not corrida["_fallos"], "; ".join(
        f"celda {i}: {m}" for i, m in corrida["_fallos"])


def test_el_motor_se_verifica_por_sha256(corrida):
    assert "ENGINE_SHA256" in corrida


def test_baja_y_escribe_el_almacen(corrida):
    destino = corrida["DESTINO"]
    assert (destino / "AAPL.csv").exists()
    assert (destino / "MSFT.csv").exists()
    assert set(corrida["ok"]) == {"AAPL", "MSFT", "NVDA"}


def test_un_nombre_sin_cik_no_tumba_la_corrida(corrida):
    assert corrida["sin_cik"] == ["NOEXISTE"]


def test_el_historico_llega_completo_en_la_primera_pasada(corrida):
    # La diferencia con los precios: no hay que acumular nada.
    historia = corrida["historia"].set_index("ticker")
    assert historia.loc["AAPL", "desde"] == "2015-12-31"
    assert historia.loc["AAPL", "hasta"] == "2024-12-31"
    assert historia.loc["AAPL", "periodos"] >= 20


def test_la_cobertura_se_mide_contra_el_universo_pedido(corrida):
    # Cuatro nombres pedidos, tres bajados: la cobertura es 3/4 y no 3/3.
    # Medirla sobre lo que sí bajó daría 100% siempre y no diría nada.
    cob = corrida["cobertura"].set_index("metrica")
    assert (corrida["DESTINO"] / "_cobertura.csv").exists()
    assert cob.loc["ingresos", "cobertura"] == 0.75
    assert cob.loc["ingresos", "con_dato"] == 3
    assert cob.loc["ingresos", "sin_dato"] == 1
    assert "NOEXISTE" in cob.loc["ingresos", "faltan"]
    # Una métrica que nadie reportó sale en cero, no desaparece: desaparecer se
    # lee como "no aplica" y cero como "no lo tenemos".
    assert cob.loc["capex", "cobertura"] == 0.0
    assert set(cob.index) == {c.clave for c in corrida["edgar"].CONCEPTOS}


def test_se_registra_de_que_etiqueta_salio_cada_numero(corrida):
    import pandas as pd

    filas = pd.read_csv(corrida["DESTINO"] / "_etiquetas.csv")
    por = {(r.ticker, r.metrica): r.etiqueta for r in filas.itertuples()}
    assert por[("AAPL", "ingresos")] == "Revenues"
    assert por[("NVDA", "ingresos")] == \
        "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_el_notebook_detecta_los_restatements(corrida):
    rest = corrida["rest"]
    assert not rest.empty, "el fixture corrige el cierre de 2020 a propósito"
    assert set(rest["fin"]) == {"2020-12-31"}


def test_el_point_in_time_distingue_las_dos_fechas(corrida):
    edgar = corrida["edgar"]
    hechos = corrida["hechos"]

    antes = edgar.as_of(hechos, "2022-06-30", metricas=["ingresos"])
    despues = edgar.as_of(hechos, None, metricas=["ingresos"])

    v_antes = antes[(antes["ticker"] == "AAPL")
                    & (antes["fin"] == "2020-12-31")]["valor"].iloc[0]
    v_despues = despues[(despues["ticker"] == "AAPL")
                        & (despues["fin"] == "2020-12-31")]["valor"].iloc[0]
    assert v_despues > v_antes, (
        "la corrección de 2023 no puede verse desde 2022; si estos dos "
        "números fueran iguales el point-in-time no estaría funcionando")


def test_las_tablas_con_estilo_se_renderizan_de_verdad(corrida):
    # No basta con que la celda no levante: el estilo se evalua al renderizar,
    # asi que si nadie llama to_html() un fallo de estilo viaja hasta Colab.
    assert corrida["_renderizados"], "ninguna tabla con estilo llego a HTML"
    assert all("<table" in h for h in corrida["_renderizados"])


def test_el_almacen_se_puede_empaquetar(corrida):
    zip_ = corrida["_zip"]
    assert Path(zip_).exists() and Path(zip_).stat().st_size > 0


# ------------------------------------------------------------- sin cuenta
def test_el_notebook_no_lee_datos_de_cuenta():
    # Misma regla que el notebook principal: este trabajo es independiente de
    # cualquier cartera. Un fundamental no tiene por qué saber qué tienes.
    texto = NOTEBOOK.read_text(encoding="utf-8").lower()
    for prohibido in ("ibkr", "interactive brokers", "portfolio_ibkr",
                      "account_id", "flex_token"):
        assert prohibido not in texto, prohibido


def test_el_contacto_es_obligatorio_y_se_valida_temprano():
    # user_agent() se llama en la celda de parámetros, antes de la primera
    # petición: mejor fallar ahí que después de 400 llamadas rechazadas.
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    celdas = executable_cells(nb)
    params = next(c for c in celdas if "CONTACTO" in c)
    descarga = next(c for c in celdas if "company_facts" in c)
    assert "edgar.user_agent(CONTACTO)" in params
    assert celdas.index(params) < celdas.index(descarga)
