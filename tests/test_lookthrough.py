"""
Tests para la transparencia (look-through).

Lo que se prueba de verdad aquí no es que sume bien -- es que **no invente**.
Un módulo que rellena lo que no sabe produce un reporte con apariencia de
medición, que es peor que no tener reporte. Así que los casos centrales son:

* una posición sin archivo de tenencias se declara opaca y baja la cobertura,
* la cobertura viaja con todo número que se reporte,
* los archivos de los emisores se leen tal como se bajan, con su preámbulo y
  sus filas de efectivo, sin pedirle al usuario que los limpie,
* y un archivo ilegible se reporta en vez de desaparecer.

El tope por emisor se prueba en su caso interesante: el nombre que cumple
mirando la posición directa y se pasa mirando la efectiva.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.lookthrough import (  # noqa: E402
    effective_exposure, issuer_cap_breaches, load_holdings, parse_holdings_csv,
    report, sector_exposure, structural_overlap,
)

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Formatos reales de emisores
# --------------------------------------------------------------------------

ISHARES = """\
iShares Core S&P 500 ETF
Fund Holdings as of,"Sep 02, 2026"
Inception Date,"May 15, 2000"
Shares Outstanding,"1,000,000"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%)
AAPL,APPLE INC,Information Technology,Equity,"1,234,567.00",7.00
MSFT,MICROSOFT CORP,Information Technology,Equity,"1,000,000.00",6.50
NVDA,NVIDIA CORP,Information Technology,Equity,"980,000.00",6.20
JPM,JPMORGAN CHASE,Financials,Equity,"400,000.00",1.30
USD,USD CASH,Cash and/or Derivatives,Cash,"10,000.00",0.05
"""

SPDR = """\
Fund Name:,SPDR S&P 500 ETF Trust
Holdings:,"As of 02-Sep-2026"

Name,Ticker,Identifier,Weight,Sector
APPLE INC,AAPL,037833100,7.05%,Information Technology
MICROSOFT CORP,MSFT,594918104,6.45%,Information Technology
UNASSIGNED,-,-,0.02%,Unassigned
"""

SIMPLE = "ticker,weight\nAAA,60\nBBB,40\n"


def _dir_with(files: dict[str, str]) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="hold-"))
    for name, body in files.items():
        (tmp / name).write_text(body, encoding="utf-8")
    return tmp


def test_reads_issuer_files_as_downloaded() -> None:
    """Con su preámbulo, sus comas en los números y sus filas de efectivo."""
    tmp = _dir_with({"IVV.csv": ISHARES, "SPY.csv": SPDR, "MINI.csv": SIMPLE})
    holdings, sectores, notas = load_holdings(tmp)

    check("los tres archivos se leyeron", set(holdings) == {"IVV", "SPY", "MINI"},
          f"{sorted(holdings)} · notas {notas}")
    check("el preámbulo de iShares no estorbó", "AAPL" in holdings["IVV"])
    # 7.05 y 6.45 son lo único que queda tras descartar la fila sin ticker,
    # así que AAPL normaliza contra 13.50, no contra el total del archivo.
    check("el porcentaje con signo % de SPDR se leyó",
          abs(holdings["SPY"]["AAPL"] - 7.05 / 13.50) < 1e-6,
          str(holdings["SPY"]))
    check("la fila de efectivo se descartó", "USD" not in holdings["IVV"])
    check("la fila sin ticker se descartó", "-" not in holdings["SPY"])
    check("los pesos se normalizan a uno",
          all(abs(sum(v.values()) - 1.0) < 1e-9 for v in holdings.values()),
          str({k: sum(v.values()) for k, v in holdings.items()}))
    check("el sector viaja con la tenencia",
          sectores.get("AAPL") == "Information Technology", str(sectores))
    check("un CSV mínimo de dos columnas también sirve",
          abs(holdings["MINI"]["AAA"] - 0.6) < 1e-9)


PARCIAL = """\
ticker,weight
AAPL,7.05
MSFT,6.45
NVDA,6.20
_RESTO,80.30
"""

RECORTADO = "ticker,weight\nAAPL,7.05\nMSFT,6.45\nNVDA,6.20\n"


def test_a_partial_file_with_a_declared_remainder_stays_correct() -> None:
    """
    El caso que hace usable el módulo sin bajar 500 filas por fondo.

    Un archivo con las mayores posiciones y una fila _RESTO da la exposición
    correcta para los nombres listados. Sin esa fila, la normalización infla
    cada peso y el reporte acusaría de romper topes que nadie rompió -- que es
    exactamente el tipo de error que este módulo existe para no cometer.
    """
    tmp = _dir_with({"BIEN.csv": PARCIAL, "MAL.csv": RECORTADO})
    holdings, _, _ = load_holdings(tmp)

    check("con _RESTO declarado, el peso del nombre se conserva",
          abs(holdings["BIEN"]["AAPL"] - 0.0705) < 1e-9,
          str(holdings["BIEN"]))
    check("el resto queda como su propia entrada",
          abs(holdings["BIEN"]["_RESTO"] - 0.8030) < 1e-9)
    check("sin _RESTO, el mismo nombre queda inflado 5 veces",
          abs(holdings["MAL"]["AAPL"] - 7.05 / 19.70) < 1e-6,
          f"{holdings['MAL']['AAPL']:.4f}")

    # Y esa inflación es la que produciría una falsa acusación de incumplimiento.
    pesos = {"BIEN": 1.0}
    exp, _, _ = effective_exposure(pesos, holdings)
    check("con archivo parcial bien declarado, AAPL no rompe el tope de 15%",
          not issuer_cap_breaches(pesos, holdings, 0.15, only=["AAPL"]),
          str(exp))
    mal = issuer_cap_breaches({"MAL": 1.0}, holdings, 0.15, only=["AAPL"])
    check("con el archivo recortado sí lo rompería -- el falso positivo",
          len(mal) == 1, str(mal))


def test_the_remainder_is_never_treated_as_an_issuer() -> None:
    holdings = {"SPY": {"AAPL": 0.10, "_RESTO": 0.90}}
    pesos = {"SPY": 1.0}

    check("el resto no puede violar un tope, por grande que sea",
          not issuer_cap_breaches(pesos, holdings, 0.15),
          str(issuer_cap_breaches(pesos, holdings, 0.15)))

    sec = sector_exposure(effective_exposure(pesos, holdings)[0],
                          {"AAPL": "Tecnología"})
    check("y se etiqueta como resto, no como 'sin clasificar'",
          abs(sec.get("Resto no detallado", 0) - 0.90) < 1e-9, str(sec))

    texto = report(pesos, holdings, {"AAPL": "Tecnología"})
    check("el reporte avisa cuánto del libro quedó sin atribuir",
          "sin atribuir" in texto, texto)


def test_an_unreadable_file_is_reported_not_swallowed() -> None:
    tmp = _dir_with({"BUENO.csv": SIMPLE, "ROTO.csv": "esto no es un csv de tenencias\n"})
    holdings, _, notas = load_holdings(tmp)
    check("el archivo bueno se cargó", "BUENO" in holdings)
    check("el roto no entró", "ROTO" not in holdings)
    check("y se dijo que no se pudo leer",
          any("ROTO" in n for n in notas), str(notas))


def test_missing_directory_says_so() -> None:
    holdings, _, notas = load_holdings("/no/existe/tenencias")
    check("sin directorio no hay tenencias", holdings == {})
    check("y se reporta en vez de fallar",
          any("No existe" in n for n in notas), str(notas))


# --------------------------------------------------------------------------
# Lo que importa: no inventar
# --------------------------------------------------------------------------

def test_a_position_without_holdings_is_opaque_and_lowers_coverage() -> None:
    """
    El caso que hace útil o inútil todo el módulo.

    Un fondo sin archivo no puede contarse como si fuera transparente. Se
    declara, y la cobertura baja para que nadie lea los números sectoriales
    como si describieran el libro entero.
    """
    holdings = {"SPY": {"AAPL": 0.5, "MSFT": 0.5}}
    pesos = {"SPY": 0.60, "QQQ": 0.30, "LLY": 0.10}

    exp, cobertura, notas = effective_exposure(pesos, holdings)

    check("lo que sí se ve, se descompone",
          abs(exp["AAPL"] - 0.30) < 1e-9 and abs(exp["MSFT"] - 0.30) < 1e-9,
          str(exp))
    check("el fondo sin archivo queda como sí mismo",
          abs(exp["QQQ"] - 0.30) < 1e-9)
    check("la acción individual se mira a sí misma, y eso es exacto",
          abs(exp["LLY"] - 0.10) < 1e-9)
    check("la cobertura refleja solo lo que se pudo ver",
          abs(cobertura - 0.60) < 1e-9, f"{cobertura}")
    check("los opacos se nombran", any("QQQ" in n for n in notas), str(notas))
    check("la cobertura viaja en las notas",
          any("Cobertura" in n for n in notas), str(notas))
    check("la exposición suma el libro",
          abs(sum(exp.values()) - 1.0) < 1e-9, str(sum(exp.values())))


def test_the_report_refuses_to_pretend_with_no_data() -> None:
    texto = report({"SPY": 0.5, "LLY": 0.5}, {}, {})
    check("con cobertura cero el reporte lo dice y no lista nada",
          "0%" in texto and "Exposición sectorial" not in texto, texto)
    check("y explica cómo arreglarlo", "tenencias" in texto.lower(), texto)


# --------------------------------------------------------------------------
# El tope por emisor
# --------------------------------------------------------------------------

def test_effective_exposure_catches_what_the_direct_cap_misses() -> None:
    """
    El caso interesante: cumple mirando la posición directa, se pasa mirando
    la efectiva. Es toda la razón por la que este módulo existe.
    """
    holdings = {"SPY": {"AAPL": 0.10, "OTROS": 0.90},
                "QQQ": {"AAPL": 0.14, "OTROS": 0.86}}
    pesos = {"AAPL": 0.14, "SPY": 0.30, "QQQ": 0.20}   # directa 14% < 15%

    exp, _, _ = effective_exposure(pesos, holdings)
    efectiva = exp["AAPL"]
    check("la exposición efectiva supera a la directa",
          abs(efectiva - (0.14 + 0.03 + 0.028)) < 1e-9, f"{efectiva}")

    fuera = issuer_cap_breaches(pesos, holdings, cap=0.15, only=["AAPL"])
    check("y el tope de 15% se marca como excedido", len(fuera) == 1, str(fuera))
    check("el desglose separa directa de la que entra por fondos",
          abs(fuera[0]["directa"] - 0.14) < 1e-9
          and abs(fuera[0]["via_fondos"] - 0.058) < 1e-9, str(fuera[0]))

    limpio = issuer_cap_breaches({"AAPL": 0.14}, {}, cap=0.15, only=["AAPL"])
    check("sin fondos en el libro, directa y efectiva coinciden", not limpio)

    check("el tope no se aplica a los fondos mismos",
          not any(d["emisor"] in ("SPY", "QQQ")
                  for d in issuer_cap_breaches(pesos, holdings, 0.15,
                                               only=["AAPL"])))


# --------------------------------------------------------------------------
# Solape y sectores
# --------------------------------------------------------------------------

def test_overlap_is_a_fact_not_an_inference() -> None:
    holdings = {"SPY": {"A": 0.5, "B": 0.5},
                "IVV": {"A": 0.5, "B": 0.5},
                "XLE": {"C": 1.0}}
    check("dos fondos sobre el mismo índice comparten todo",
          abs(structural_overlap("SPY", "IVV", holdings) - 1.0) < 1e-9)
    check("fondos sin nada en común dan cero",
          abs(structural_overlap("SPY", "XLE", holdings)) < 1e-9)
    check("un nombre sin archivo se compara consigo mismo",
          abs(structural_overlap("LLY", "LLY", holdings) - 1.0) < 1e-9)
    check("y contra otro distinto da cero",
          abs(structural_overlap("LLY", "JPM", holdings)) < 1e-9)


def test_sector_exposure_declares_the_unknown() -> None:
    exp = {"AAPL": 0.4, "JPM": 0.3, "RARO": 0.3}
    sec = sector_exposure(exp, {"AAPL": "Tecnología", "JPM": "Financiero"})
    check("agrupa lo conocido", abs(sec["Tecnología"] - 0.4) < 1e-9)
    check("y no esconde lo que no sabe clasificar",
          abs(sec["Sin clasificar"] - 0.3) < 1e-9, str(sec))
    check("sale ordenado de mayor a menor",
          list(sec) == sorted(sec, key=lambda k: -sec[k]))


def main() -> int:
    for fn in [
        test_reads_issuer_files_as_downloaded,
        test_a_partial_file_with_a_declared_remainder_stays_correct,
        test_the_remainder_is_never_treated_as_an_issuer,
        test_an_unreadable_file_is_reported_not_swallowed,
        test_missing_directory_says_so,
        test_a_position_without_holdings_is_opaque_and_lowers_coverage,
        test_the_report_refuses_to_pretend_with_no_data,
        test_effective_exposure_catches_what_the_direct_cap_misses,
        test_overlap_is_a_fact_not_an_inference,
        test_sector_exposure_declares_the_unknown,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
