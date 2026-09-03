#!/usr/bin/env python3
"""
Baja la composición de los ETFs para el reporte de transparencia.

Se corre **en tu máquina**, no dentro del modelo: necesita salida a internet.

    python3 scripts/bajar_tenencias.py                    # los de la cartera típica
    python3 scripts/bajar_tenencias.py SPY IVV QQQ EFA    # solo estos
    python3 scripts/bajar_tenencias.py --universo         # los 130 ETFs del universo
    python3 scripts/bajar_tenencias.py --salida ./tenencias

De dónde salen los datos
------------------------
De ``yfinance``, que ya es dependencia del modelo. No hace falta contratar a
nadie ni bajar archivos a mano de cada emisor.

Yahoo publica las **mayores** posiciones de cada fondo, no las 500. Eso alcanza
para lo que el reporte necesita: un nombre que pueda romper un tope del 15% o
20% está en las primeras posiciones o no está. El peso no detallado se escribe
como una fila ``_RESTO``, que es la que evita que la normalización infle los
pesos listados y produzca una acusación falsa de incumplimiento.

Además baja el **desglose sectorial completo** por fondo, que es mejor que
derivarlo de las tenencias parciales: no es una muestra, es el total.

Qué produce
-----------
    tenencias/SPY.csv          mayores posiciones + _RESTO
    tenencias/_sectores.csv    desglose sectorial por fondo, completo
    tenencias/_canasta.csv     P/E, P/B y demás de la canasta, cuando los hay

Lo que no se pudo bajar simplemente no se escribe, y la sección de
transparencia lo reporta como cobertura faltante. Nunca se inventa contenido.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Cartera típica: los amplios y los de renta fija que más pesan. Con estos ya
#: se ve la mayor parte de un libro.
POR_DEFECTO = ("SPY", "IVV", "QQQ", "IWM", "EFA", "IEFA", "IEMG", "EEM",
               "AGG", "TLT", "IEF", "LQD", "HYG", "BIL", "GLD")


def _tabla_a_dict(df, col_peso_candidatas=("Holding Percent", "holdingPercent",
                                           "Weight", "weight")) -> dict[str, float]:
    """Extrae {ticker: peso} de la tabla de top_holdings de yfinance."""
    if df is None or len(df) == 0:
        return {}
    col = next((c for c in col_peso_candidatas if c in df.columns), None)
    if col is None:
        numericas = [c for c in df.columns if df[c].dtype.kind in "fc"]
        if not numericas:
            return {}
        col = numericas[0]

    out: dict[str, float] = {}
    for idx, fila in df.iterrows():
        tk = str(idx).strip().upper()
        if not tk or tk in {"-", "N/A", "NAN"}:
            continue
        try:
            w = float(fila[col])
        except (TypeError, ValueError):
            continue
        if w > 0:
            out[tk] = out.get(tk, 0.0) + w
    return out


def bajar_uno(ticker: str, destino: Path) -> tuple[bool, str, dict, dict]:
    """
    Devuelve (exito, detalle, sectores, canasta) para un ETF.

    Escribe ``TICKER.csv`` con las mayores posiciones y la fila ``_RESTO``.
    """
    import yfinance as yf

    try:
        fondo = yf.Ticker(ticker).funds_data
        tenencias = _tabla_a_dict(fondo.top_holdings)
        sectores = dict(fondo.sector_weightings or {})
    except Exception as exc:                    # noqa: BLE001 - se reporta
        return False, f"{type(exc).__name__}: {exc}", {}, {}

    canasta: dict[str, float] = {}
    try:
        eq = fondo.equity_holdings
        if eq is not None and len(eq):
            col = eq.columns[0]
            canasta = {str(i): float(eq.loc[i, col]) for i in eq.index
                       if str(eq.loc[i, col]).replace(".", "").replace("-", "").isdigit()
                       or isinstance(eq.loc[i, col], float)}
    except Exception:                           # noqa: BLE001 - opcional
        canasta = {}

    if not tenencias and not sectores:
        return False, "Yahoo no devolvió composición (¿es un ETF?)", {}, {}

    if tenencias:
        # Yahoo da pesos en fracción (0.07) o en por ciento (7.0) según el
        # campo. Se normaliza a por ciento para escribir el CSV.
        suma = sum(tenencias.values())
        escala = 100.0 if suma <= 1.5 else 1.0
        filas = [(tk, w * escala) for tk, w in
                 sorted(tenencias.items(), key=lambda kv: -kv[1])]
        detallado = sum(w for _, w in filas)
        resto = max(0.0, 100.0 - detallado)

        destino.mkdir(parents=True, exist_ok=True)
        with (destino / f"{ticker.upper()}.csv").open("w", newline="",
                                                      encoding="utf-8") as fh:
            escritor = csv.writer(fh)
            escritor.writerow(["ticker", "weight"])
            for tk, w in filas:
                escritor.writerow([tk, f"{w:.4f}"])
            if resto > 0.01:
                escritor.writerow(["_RESTO", f"{resto:.4f}"])
        detalle = f"{len(filas)} posiciones ({detallado:.1f}% detallado)"
    else:
        detalle = "sin tenencias, solo sectores"

    return True, detalle, sectores, canasta


def escribir_auxiliares(destino: Path, sectores: dict[str, dict],
                        canastas: dict[str, dict]) -> None:
    if sectores:
        with (destino / "_sectores.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fondo", "sector", "peso"])
            for fondo, mapa in sorted(sectores.items()):
                for sector, peso in sorted(mapa.items(), key=lambda kv: -kv[1]):
                    w.writerow([fondo, sector, f"{float(peso):.6f}"])
    if canastas:
        with (destino / "_canasta.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fondo", "metrica", "valor"])
            for fondo, mapa in sorted(canastas.items()):
                for k, v in mapa.items():
                    w.writerow([fondo, k, v])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Baja composición de ETFs (yfinance) para la transparencia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("tickers", nargs="*", help="ETFs a bajar.")
    p.add_argument("--universo", action="store_true",
                   help="Todos los ETFs del universo curado del modelo.")
    p.add_argument("--salida", default="tenencias", help="Directorio destino.")
    args = p.parse_args(argv)

    if args.universo:
        from screener.universe import ETF_UNIVERSE
        tickers = sorted(ETF_UNIVERSE)
    else:
        tickers = [t.upper() for t in (args.tickers or POR_DEFECTO)]

    destino = Path(args.salida)
    sectores: dict[str, dict] = {}
    canastas: dict[str, dict] = {}
    ok, fallaron = [], []

    print(f"Bajando composición de {len(tickers)} fondo(s) a {destino}/\n")
    for tk in tickers:
        exito, detalle, secs, canasta = bajar_uno(tk, destino)
        print(f"  {'OK   ' if exito else 'FALLO'}  {tk:6s} {detalle}")
        if exito:
            ok.append(tk)
            if secs:
                sectores[tk] = secs
            if canasta:
                canastas[tk] = canasta
        else:
            fallaron.append(tk)

    escribir_auxiliares(destino, sectores, canastas)

    print(f"\n{len(ok)} bajados, {len(fallaron)} sin datos.")
    if sectores:
        print(f"Desglose sectorial de {len(sectores)} fondo(s) en _sectores.csv")
    if canastas:
        print(f"Características de canasta de {len(canastas)} fondo(s) en _canasta.csv")
    if fallaron:
        print(f"\nSin composición en Yahoo: {', '.join(fallaron)}")
        print("Puede ser que no sean fondos, o que Yahoo no los cubra. "
              "Para esos, baja el CSV del emisor a mano o escribe las mayores "
              "posiciones con una fila _RESTO al final.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
