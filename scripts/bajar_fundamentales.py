#!/usr/bin/env python3
"""
Baja fundamentales de SEC EDGAR al almacén local.

Se corre **en tu máquina**: necesita salida a internet.

    python3 scripts/bajar_fundamentales.py --contacto "CCI tucorreo@dominio.com"
    python3 scripts/bajar_fundamentales.py --contacto "..." AAPL MSFT NVDA
    python3 scripts/bajar_fundamentales.py --contacto "..." --universo acciones
    python3 scripts/bajar_fundamentales.py --contacto "..." --forzar

El ``--contacto`` no es opcional: la SEC exige un User-Agent con correo real y
bloquea por IP a quien no se identifica.

Qué hace la primera vez y qué las siguientes
--------------------------------------------
La primera baja el ``companyfacts`` de cada nombre — una petición por empresa,
de 10 a 15 MB cada una en las grandes — y guarda solo la docena de conceptos de
:data:`screener.edgar.CONCEPTOS`. Son unos minutos para el universo de acciones.

Las siguientes solo bajan lo que no está: un archivo ya escrito se salta salvo
que pases ``--forzar``. Como cada empresa presenta cuatro veces al año, correrlo
semanalmente es más que suficiente.

Y lo importante: **el histórico viene desde la primera corrida.** Un
``companyfacts`` trae todo lo que la empresa ha reportado desde que existe XBRL,
no desde hoy. No hay que esperar a acumular nada.

Qué escribe
-----------
    datos/fundamentales/AAPL.csv     un hecho por fila, con su fecha de filing
    datos/fundamentales/_tickers.json  mapa ticker -> CIK, cacheado
    datos/fundamentales/_tickers_fuentes.json  cuántos emisores trajo cada lista
    datos/fundamentales/_cobertura.csv qué porcentaje del universo tiene qué
    datos/fundamentales/_etiquetas.csv qué etiqueta XBRL ganó en cada emisor
    datos/fundamentales/_emisores.csv  qué nombre tiene la SEC para cada CIK
    datos/fundamentales/_fallos.csv    qué falló en LA ÚLTIMA corrida

Y uno que escribes tú, no el guion:

    datos/fundamentales/_ciks_manuales.csv   ticker,cik,por_que

La SEC publica sus listas de tickers sin garantizar su alcance, así que un
emisor vigente puede faltar en las tres. Cuando pase, búscalo en
https://www.sec.gov/search-filings/cik-lookup y anota el CIK ahí. Se lee en cada
corrida y solo rellena huecos: nunca contradice a la SEC en silencio.

Y ``_etiquetas.csv`` es el que hay que leer antes de creerle a un número: dice
literalmente de qué etiqueta XBRL salió cada métrica en cada empresa.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.descarga import (MAX_REFRESCOS, REFRESCO_DIAS,  # noqa: E402
                               sincronizar)
from screener.edgar import (ARCHIVO_OVERRIDES, Limitador,  # noqa: E402
                            coverage_report, detalle_sin_cik,
                            historia_por_ticker, leer_hechos, restatements)

DESTINO = "datos/fundamentales"


def universo(nombre: str, tickers: list[str]) -> list[str]:
    if tickers:
        return [t.strip().upper() for t in tickers if t.strip()]

    from screener.universe import all_index_members

    miembros = all_index_members()
    grupos = {"acciones": ("SP500", "NDX", "DJIA"),
              "sp500": ("SP500",), "ndx": ("NDX",), "djia": ("DJIA",)}
    fuera: set[str] = set()
    for clave in grupos.get(nombre, ("SP500",)):
        fuera |= set(miembros.get(clave, frozenset()))
    # Yahoo usa guion donde la SEC usa punto (BRK-B vs BRK.B). El mapa de la
    # SEC trae la forma con guion, así que se normaliza a esa.
    return sorted(t.replace(".", "-") for t in fuera)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Baja fundamentales de SEC EDGAR (gratis, point-in-time).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("tickers", nargs="*", help="Nombres a bajar.")
    p.add_argument("--contacto", required=True,
                   help="User-Agent con correo real. Lo exige la SEC.")
    p.add_argument("--universo", default="acciones",
                   choices=("acciones", "sp500", "ndx", "djia"),
                   help="Grupo a bajar si no pasas tickers.")
    p.add_argument("--salida", default=DESTINO)
    p.add_argument("--forzar", action="store_true",
                   help="Rebajar lo que ya está en disco.")
    p.add_argument("--remapear", action="store_true",
                   help="Volver a pedir el mapa ticker->CIK sin rebajar los "
                        "fundamentales. Dos peticiones, no doscientas.")
    p.add_argument("--refrescar-dias", type=int, default=REFRESCO_DIAS,
                   dest="refrescar_dias",
                   help="Rebajar un nombre cuyo archivo tenga más días que "
                        "esto. 0 lo desactiva.")
    p.add_argument("--max-refrescos", type=int, default=MAX_REFRESCOS,
                   dest="max_refrescos",
                   help="Tope de refrescos por corrida, del más viejo al más "
                        "nuevo. 0 = sin tope.")
    p.add_argument("--limite", type=int, default=0,
                   help="Cortar después de N nombres. Para probar.")
    args = p.parse_args(argv)

    destino = Path(args.salida)
    destino.mkdir(parents=True, exist_ok=True)
    limitador = Limitador()

    tickers = universo(args.universo, args.tickers)
    if args.limite:
        tickers = tickers[:args.limite]

    print(f"Universo: {len(tickers)} nombre(s) -> {destino}/")
    res = sincronizar(destino, tickers, contacto=args.contacto,
                      forzar=args.forzar, remapear=args.remapear,
                      refrescar_dias=args.refrescar_dias,
                      max_refrescos=args.max_refrescos,
                      limitador=limitador, progreso=print)

    if res.conceptos_nuevos and not args.forzar:
        print(f"\nAVISO: el almacén se escribió con una lista de conceptos "
              f"anterior.\n"
              f"       Estos {len(res.conceptos_nuevos)} son posteriores y NO "
              f"están en los archivos ya bajados:\n"
              f"       {', '.join(res.conceptos_nuevos)}\n"
              f"       Su cobertura va a salir en cero sin ser cero. Corre con "
              f"--forzar\n"
              f"       para rebajar, o ignóralo si esas métricas no te "
              f"importan todavía.")

    print(f"\nMapa ticker->CIK: {res.mapa} emisores"
          + (" (" + ", ".join(
              f"{k}={res.fuentes[k]}" for k in ("company_tickers",
                                                "company_tickers_exchange",
                                                "ticker_txt")
              if k in res.fuentes) + ")" if "company_tickers" in res.fuentes
             else ""))
    if res.manuales:
        print(f"  + {len(res.manuales)} CIK puesto(s) a mano en "
              f"_ciks_manuales.csv: {', '.join(res.manuales)}")

    a_mano = [e for e in res.emisores if e["fuente"] == "a mano"]
    if a_mano:
        print("\nCIK puestos a mano — verifica que el nombre sea el que "
              "esperas:")
        for e in a_mano:
            print(f"  {e['ticker']:6s} {e['cik']}  {e['entidad']}")

    print(f"\n{res.resumen()}")
    if res.intento and not res.motivos:
        print("  _fallos.csv queda vacío: no falló ninguno.")
    if res.sin_cik:
        print(f"  Sin CIK: {', '.join(res.sin_cik[:15])}")
        print(f"           {detalle_sin_cik(res.fuentes)}")
        print(f"  Te dejé {destino / ARCHIVO_OVERRIDES} con esos nombres y el "
              "CIK en blanco: llénalo y vuelve a correr.")
    if res.fallaron:
        print(f"  Fallaron: {', '.join(res.fallaron[:15])}")
    if res.pendientes:
        print(f"  {len(res.pendientes)} vencidos quedaron para la próxima "
              f"corrida (tope {args.max_refrescos}).")

    # ---- Cobertura: el entregable que decide si seguimos ------------------
    hechos = leer_hechos(destino, tickers)
    if hechos.empty:
        print("\nNo hay nada en el almacén todavía.")
        return 0

    cobertura = coverage_report(hechos, tickers)
    cobertura.to_csv(destino / "_cobertura.csv", index=False)

    print("\n" + "=" * 72)
    print("COBERTURA POR MÉTRICA")
    print("=" * 72)
    print(cobertura[["metrica", "cobertura", "con_dato", "sin_dato",
                     "etiquetas_usadas", "etiqueta_principal"]].to_string(
        index=False, formatters={"cobertura": "{:.1%}".format}))

    hist = historia_por_ticker(hechos)
    print(f"\nHistoria: {len(hist)} nombre(s), "
          f"{int(hist['versiones'].sum()):,} hechos, "
          f"desde {hist['desde'].min()} hasta {hist['hasta'].max()}")
    print(f"Períodos distintos por nombre: mediana {hist['periodos'].median():.0f}")

    rest = restatements(hechos)
    print(f"\nRestatements con cambio >= 2%: {len(rest)}")
    if not rest.empty:
        print("Los diez mayores — cada uno es un dato que un proveedor te "
              "habría dado ya corregido:")
        print(rest.head(10)[["ticker", "metrica", "fin", "versiones",
                             "cambio"]].to_string(
            index=False, formatters={"cambio": "{:+.1%}".format}))
        rest.to_csv(destino / "_restatements.csv", index=False)

    print(f"\nArchivos en {destino}/: _cobertura.csv, _etiquetas.csv"
          + (", _restatements.csv" if not rest.empty else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
