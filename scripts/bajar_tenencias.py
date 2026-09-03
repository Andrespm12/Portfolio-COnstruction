#!/usr/bin/env python3
"""
Baja la composición de los ETFs para el reporte de transparencia.

Se corre **en tu máquina**, no dentro del modelo: necesita salida a internet.

    python3 scripts/bajar_tenencias.py                    # los de la cartera típica
    python3 scripts/bajar_tenencias.py SPY IVV QQQ EFA    # solo estos
    python3 scripts/bajar_tenencias.py --universo         # los ETFs del universo
    python3 scripts/bajar_tenencias.py --salida ./tenencias

En Colab no hace falta: el notebook trae la sección de transparencia, que baja
lo mismo llamando al paquete directamente.

De dónde salen los datos
------------------------
De ``yfinance``, que ya es dependencia del modelo. No hace falta contratar a
nadie ni bajar archivos a mano de cada emisor. La lógica vive en
``screener/tenencias_yahoo.py``; este archivo solo es la línea de comandos.

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.tenencias_yahoo import POR_DEFECTO, bajar_varios  # noqa: E402


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
    print(f"Bajando composición de {len(tickers)} fondo(s) a {destino}/\n")
    ok, fallaron = bajar_varios(tickers, destino)

    print(f"\n{len(ok)} bajados, {len(fallaron)} sin datos.")
    if (destino / "_sectores.csv").exists():
        print("Desglose sectorial completo en _sectores.csv")
    if (destino / "_canasta.csv").exists():
        print("Características de canasta en _canasta.csv")
    if fallaron:
        print(f"\nSin composición en Yahoo: {', '.join(fallaron)}")
        print("Puede ser que no sean fondos, o que Yahoo no los cubra. "
              "Para esos, baja el CSV del emisor a mano o escribe las mayores "
              "posiciones con una fila _RESTO al final.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
