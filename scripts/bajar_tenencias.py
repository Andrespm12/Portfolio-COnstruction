#!/usr/bin/env python3
"""
Baja los archivos de tenencias de los ETFs, para el reporte de transparencia.

Se corre **en tu máquina**, no en el modelo: necesita salida a internet hacia
los sitios de los emisores.

    python3 scripts/bajar_tenencias.py                    # los de la cartera típica
    python3 scripts/bajar_tenencias.py SPY IVV QQQ EFA    # solo estos
    python3 scripts/bajar_tenencias.py --salida ./tenencias

Qué hace y qué no
-----------------
Intenta la descarga directa desde el emisor con los patrones de URL conocidos.
**Los emisores cambian esas rutas sin avisar**, así que cuando una falla, el
script imprime los pasos manuales exactos para ese fondo en vez de fallar en
silencio o -- peor -- escribir un archivo vacío que el modelo leería como si
fueran tenencias.

Un archivo que no se pudo bajar simplemente no existe, y la sección de
transparencia lo reporta como cobertura faltante. Nunca se inventa contenido.

Si un emisor se pone difícil
----------------------------
No hace falta el archivo completo. Copia a mano las 20 o 30 mayores posiciones
y cierra con una fila de resto:

    ticker,weight
    AAPL,7.05
    MSFT,6.45
    ...
    _RESTO,60.10

Esa última fila es obligatoria en un archivo parcial. Sin ella el modelo
normaliza sobre lo que le diste y cada peso queda inflado -- un 7% se
convertiría en 17% y el reporte acusaría incumplimientos que no existen.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Cartera típica del modelo: los amplios y los de renta fija que más pesan.
#: Con estos ya se ve la mayor parte de un libro; no hace falta bajar los 130.
POR_DEFECTO = ("SPY", "IVV", "QQQ", "IWM", "EFA", "IEFA", "IEMG", "EEM",
               "AGG", "TLT", "IEF", "LQD", "HYG", "BIL")

#: Emisor de cada ticker, para saber qué patrón de URL probar.
EMISOR = {
    "IVV": "ishares", "IEFA": "ishares", "IEMG": "ishares", "EFA": "ishares",
    "EEM": "ishares", "AGG": "ishares", "TLT": "ishares", "IEF": "ishares",
    "LQD": "ishares", "HYG": "ishares", "IWM": "ishares", "SHY": "ishares",
    "TIP": "ishares", "GOVT": "ishares", "MBB": "ishares", "IJH": "ishares",
    "IJR": "ishares", "IWB": "ishares",
    "SPY": "spdr", "XLK": "spdr", "XLF": "spdr", "XLV": "spdr", "XLE": "spdr",
    "XLI": "spdr", "XLY": "spdr", "XLP": "spdr", "XLU": "spdr", "XLB": "spdr",
    "XLRE": "spdr", "XLC": "spdr", "DIA": "spdr", "BIL": "spdr",
    "QQQ": "invesco",
}

#: Patrones de descarga. VERIFÍCALOS: los emisores los mueven cada tanto.
#: Cuando uno deje de servir, el script lo dice y te da la ruta manual.
_SPDR = ("https://www.ssga.com/us/en/institutional/library-content/products/"
         "fund-data/etfs/us/holdings-daily-us-en-{tk}.xlsx")
_INVESCO = ("https://www.invesco.com/us/financial-products/etfs/holdings/main/"
            "holdings/0?audienceType=Investor&action=download&ticker={tk}")

MANUAL = {
    "ishares": (
        "ishares.com -> busca {tk} -> pestaña Holdings -> "
        "'Detailed Holdings and Analytics' (CSV). "
        "La URL trae un id de producto distinto por fondo, por eso no se "
        "puede construir sola."),
    "spdr": (
        "ssga.com -> busca {tk} -> Holdings -> Daily. "
        "Si baja XLSX, ábrelo y guárdalo como CSV con el nombre {tk}.csv"),
    "invesco": (
        "invesco.com -> {tk} -> Portfolio Holdings -> descarga CSV"),
    None: (
        "Busca '{tk} etf holdings csv' en el sitio del emisor, o copia a mano "
        "las 20-30 mayores posiciones y cierra con una fila _RESTO."),
}


def url_para(ticker: str) -> str | None:
    emisor = EMISOR.get(ticker.upper())
    if emisor == "spdr":
        return _SPDR.format(tk=ticker.lower())
    if emisor == "invesco":
        return _INVESCO.format(tk=ticker.upper())
    return None                      # iShares necesita el id del producto


def bajar(ticker: str, destino: Path, timeout: int = 30) -> tuple[bool, str]:
    """Devuelve (exito, motivo). Nunca escribe un archivo vacío o de error."""
    url = url_para(ticker)
    if url is None:
        return False, "sin URL directa"

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; tenencias/1.0)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"

    # Una página de error HTML pesa poco y empieza con '<'. No la guardamos:
    # un archivo así se leería después como "no pude parsear" en vez de como
    # "no lo pude bajar", y el diagnóstico se perdería.
    if len(data) < 500 or data.lstrip()[:1] in (b"<", b"{"):
        return False, f"la respuesta no parece un archivo de tenencias ({len(data)} bytes)"

    ext = ".xlsx" if data[:2] == b"PK" else ".csv"
    destino.mkdir(parents=True, exist_ok=True)
    salida = destino / f"{ticker.upper()}{ext}"
    salida.write_bytes(data)
    nota = ("" if ext == ".csv" else
            "  <- es XLSX: ábrelo y guárdalo como CSV con el mismo nombre")
    return True, f"{len(data)//1024} KB{nota}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Baja tenencias de ETFs para el reporte de transparencia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("tickers", nargs="*", default=list(POR_DEFECTO),
                   help="ETFs a bajar. Por defecto, los de mayor peso típico.")
    p.add_argument("--salida", default="tenencias",
                   help="Directorio destino (por defecto: tenencias/).")
    args = p.parse_args(argv)

    destino = Path(args.salida)
    tickers = [t.upper() for t in (args.tickers or POR_DEFECTO)]
    ok, fallaron = [], []

    print(f"Bajando {len(tickers)} archivo(s) a {destino}/\n")
    for tk in tickers:
        exito, detalle = bajar(tk, destino)
        print(f"  {'OK  ' if exito else 'FALLO'}  {tk:6s} {detalle}")
        (ok if exito else fallaron).append(tk)

    print(f"\n{len(ok)} bajados, {len(fallaron)} pendientes.")
    if fallaron:
        print("\nEstos hay que bajarlos a mano. Guarda cada uno como "
              "TICKER.csv en ese mismo directorio:\n")
        for tk in fallaron:
            print(f"  {tk}: {MANUAL[EMISOR.get(tk)].format(tk=tk)}")
        print("\nY si alguno se resiste, un CSV parcial sirve igual:")
        print("    ticker,weight")
        print("    AAPL,7.05")
        print("    ...")
        print("    _RESTO,60.10      <- obligatoria si no pusiste todas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
