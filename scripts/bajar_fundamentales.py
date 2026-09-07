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
    datos/fundamentales/_cobertura.csv qué porcentaje del universo tiene qué
    datos/fundamentales/_etiquetas.csv qué etiqueta XBRL ganó en cada emisor
    datos/fundamentales/_fallos.csv    qué nombre falló y por qué

Ese último archivo es el que hay que leer antes de creerle a un número: dice
literalmente de qué etiqueta salió cada métrica en cada empresa.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.edgar import (CONCEPTOS, Limitador,  # noqa: E402
                            company_facts, conceptos_desactualizados,
                            coverage_report, escribir_hechos,
                            escribir_manifiesto, extract_facts,
                            historia_por_ticker, leer_hechos,
                            load_ticker_map, restatements)

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
    p.add_argument("--limite", type=int, default=0,
                   help="Cortar después de N nombres. Para probar.")
    args = p.parse_args(argv)

    destino = Path(args.salida)
    destino.mkdir(parents=True, exist_ok=True)
    limitador = Limitador()

    tickers = universo(args.universo, args.tickers)
    if args.limite:
        tickers = tickers[:args.limite]

    nuevos = conceptos_desactualizados(destino)
    if nuevos and not args.forzar:
        print(f"AVISO: el almacén se escribió con una lista de conceptos "
              f"anterior.\n"
              f"       Estos {len(nuevos)} son posteriores y NO están en los "
              f"archivos ya bajados:\n"
              f"       {', '.join(nuevos)}\n"
              f"       Su cobertura va a salir en cero sin ser cero. Corre con "
              f"--forzar\n"
              f"       para rebajar, o ignóralo si esas métricas no te "
              f"importan todavía.\n")

    print(f"Universo: {len(tickers)} nombre(s) -> {destino}/")
    mapa = load_ticker_map(contacto=args.contacto,
                           cache=destino / "_tickers.json",
                           limitador=limitador)
    print(f"Mapa ticker->CIK: {len(mapa)} emisores\n")

    etiquetas: list[dict] = []
    ok, sin_cik, fallaron, saltados = [], [], [], []
    # El motivo de cada fallo, para no depender del scrollback de la consola.
    # Un nombre que falla dos corridas seguidas necesita diagnóstico, y
    # "sin CIK" y "404" llevan a sitios distintos.
    motivos: list[dict] = []

    for i, ticker in enumerate(tickers, 1):
        archivo = destino / f"{ticker}.csv"
        if archivo.exists() and not args.forzar:
            saltados.append(ticker)
            continue

        cik = mapa.get(ticker) or mapa.get(ticker.replace("-", "."))
        if not cik:
            sin_cik.append(ticker)
            motivos.append({"ticker": ticker, "motivo": "sin CIK",
                            "detalle": "no está en company_tickers.json"})
            print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} sin CIK en la SEC")
            continue

        try:
            payload = company_facts(cik, contacto=args.contacto,
                                    limitador=limitador)
            hechos, elegidas = extract_facts(payload, ticker)
        except Exception as exc:            # noqa: BLE001 - se reporta
            fallaron.append(ticker)
            motivos.append({"ticker": ticker, "motivo": type(exc).__name__,
                            "detalle": f"CIK {cik}: {exc}"})
            print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} "
                  f"FALLO {type(exc).__name__}: {exc}")
            continue

        if not hechos:
            fallaron.append(ticker)
            motivos.append({"ticker": ticker, "motivo": "sin etiquetas",
                            "detalle": f"CIK {cik}: companyfacts respondió, "
                                       "pero sin ninguna etiqueta de CONCEPTOS"})
            print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} "
                  "sin ninguna etiqueta conocida (¿emisor extranjero?)")
            continue

        escribir_hechos(destino, ticker, hechos)
        ok.append(ticker)
        for metrica, etiqueta in elegidas.items():
            etiquetas.append({"ticker": ticker, "metrica": metrica,
                              "etiqueta": etiqueta})
        print(f"  [{i:3d}/{len(tickers)}] {ticker:6s} "
              f"{len(hechos):5d} hechos, {len(elegidas)}/{len(CONCEPTOS)} métricas")

    if etiquetas:
        modo = "w" if args.forzar or not (destino / "_etiquetas.csv").exists() else "a"
        with (destino / "_etiquetas.csv").open(modo, newline="",
                                               encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["ticker", "metrica", "etiqueta"])
            if modo == "w":
                w.writeheader()
            w.writerows(etiquetas)

    if motivos:
        with (destino / "_fallos.csv").open("w", newline="",
                                            encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["ticker", "motivo", "detalle"])
            w.writeheader()
            w.writerows(motivos)

    if ok and not args.forzar and not nuevos:
        escribir_manifiesto(destino)
    elif args.forzar:
        escribir_manifiesto(destino)

    print(f"\n{len(ok)} bajados, {len(saltados)} ya estaban, "
          f"{len(sin_cik)} sin CIK, {len(fallaron)} fallaron.")
    if sin_cik:
        print(f"  Sin CIK: {', '.join(sin_cik[:15])}")
    if fallaron:
        print(f"  Fallaron: {', '.join(fallaron[:15])}")

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
