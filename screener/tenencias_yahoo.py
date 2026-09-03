"""
Baja la composición de fondos desde Yahoo, para el reporte de transparencia.

Vive dentro del paquete (y no en ``scripts/``) por una razón práctica: así
viaja embebido en el notebook y la sección de transparencia funciona en Colab
sin subir ningún archivo extra. ``scripts/bajar_tenencias.py`` es solo la
línea de comandos que llama aquí.

Qué se baja
-----------
``yfinance.Ticker(tk).funds_data`` trae tres cosas útiles:

``top_holdings``
    Las **mayores** posiciones, no las 500. Alcanza para lo que el reporte
    necesita: un nombre capaz de romper un tope del 15% o 20% está entre las
    primeras o no está. El peso no detallado se escribe como fila ``_RESTO``,
    que es lo que evita que la normalización infle los pesos listados y
    produzca una acusación falsa de incumplimiento.

``sector_weightings``
    El desglose sectorial **completo** del fondo. Es mejor que derivarlo de
    las tenencias parciales: no es una muestra, es el total.

``equity_holdings``
    Características de la canasta (P/E, P/B y demás). Opcional.

Lo que no se pudo bajar simplemente no se escribe, y la sección de
transparencia lo reporta como cobertura faltante. Nunca se inventa contenido.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable

#: Cartera típica: los amplios y los de renta fija que más pesan. Con estos ya
#: se ve la mayor parte de un libro.
POR_DEFECTO = ("SPY", "IVV", "QQQ", "IWM", "EFA", "IEFA", "IEMG", "EEM",
               "AGG", "TLT", "IEF", "LQD", "HYG", "BIL", "GLD")

#: Columnas donde Yahoo ha puesto el peso, según versión de yfinance.
COLUMNAS_PESO = ("Holding Percent", "holdingPercent", "Weight", "weight")

#: Tickers que Yahoo devuelve como relleno cuando no publica el nombre.
_TICKER_VACIO = {"", "-", "--", "N/A", "NA", "NAN", "NONE", "NULL", "."}


def _fondo_yahoo(ticker: str) -> Any:
    """Aislado para poder sustituirlo en las pruebas sin tocar la red."""
    import yfinance as yf

    return yf.Ticker(ticker).funds_data


def tabla_a_dict(df: Any,
                 columnas: Iterable[str] = COLUMNAS_PESO) -> dict[str, float]:
    """Extrae ``{ticker: peso}`` de la tabla ``top_holdings`` de yfinance."""
    if df is None or len(df) == 0:
        return {}
    col = next((c for c in columnas if c in df.columns), None)
    if col is None:
        numericas = [c for c in df.columns if df[c].dtype.kind in "fc"]
        if not numericas:
            return {}
        col = numericas[0]

    out: dict[str, float] = {}
    for idx, fila in df.iterrows():
        tk = str(idx).strip().upper()
        if tk in _TICKER_VACIO:
            continue
        try:
            peso = float(fila[col])
        except (TypeError, ValueError):
            continue
        if peso > 0:
            out[tk] = out.get(tk, 0.0) + peso
    return out


def a_por_ciento(tenencias: dict[str, float]) -> dict[str, float]:
    """
    Normaliza a por ciento.

    Yahoo devuelve fracción (0.07) o por ciento (7.0) según el campo y la
    versión. Se decide por la suma: quince posiciones que suman 0.6 son
    fracciones; que suman 60 son por cientos.
    """
    if not tenencias:
        return {}
    escala = 100.0 if sum(tenencias.values()) <= 1.5 else 1.0
    return {tk: peso * escala for tk, peso in tenencias.items()}


def filas_con_resto(tenencias: dict[str, float],
                    umbral: float = 0.01) -> list[tuple[str, float]]:
    """
    Ordena de mayor a menor y añade la fila ``_RESTO``.

    ``_RESTO`` es el peso del fondo que Yahoo no detalla. Sin esa fila, el
    reporte normaliza sobre lo listado y le atribuye a las mayores posiciones
    un peso que no tienen.
    """
    filas = sorted(tenencias.items(), key=lambda kv: -kv[1])
    resto = max(0.0, 100.0 - sum(peso for _, peso in filas))
    if resto > umbral:
        filas.append(("_RESTO", resto))
    return filas


def _canasta(fondo: Any) -> dict[str, float]:
    """Características de la canasta, si el fondo las publica."""
    try:
        eq = fondo.equity_holdings
    except Exception:                           # noqa: BLE001 - opcional
        return {}
    if eq is None or not len(eq):
        return {}
    col = eq.columns[0]
    out: dict[str, float] = {}
    for idx in eq.index:
        try:
            out[str(idx)] = float(eq.loc[idx, col])
        except (TypeError, ValueError):
            continue
    return out


def bajar_uno(ticker: str, destino: Path | str | None,
              *, obtener: Callable[[str], Any] = _fondo_yahoo,
              ) -> tuple[bool, str, dict[str, float], dict[str, float]]:
    """
    Baja un fondo. Devuelve ``(exito, detalle, sectores, canasta)``.

    Si ``destino`` no es ``None``, escribe ``TICKER.csv`` con las mayores
    posiciones y la fila ``_RESTO``. Los sectores y la canasta se devuelven
    para que quien llama los junte en los archivos auxiliares.
    """
    ticker = ticker.strip().upper()
    try:
        fondo = obtener(ticker)
        tenencias = a_por_ciento(tabla_a_dict(fondo.top_holdings))
        sectores = {str(k): float(v)
                    for k, v in dict(fondo.sector_weightings or {}).items()}
    except Exception as exc:                    # noqa: BLE001 - se reporta
        return False, f"{type(exc).__name__}: {exc}", {}, {}

    if not tenencias and not sectores:
        return False, "Yahoo no devolvió composición (¿es un ETF?)", {}, {}

    canasta = _canasta(fondo)

    if not tenencias:
        return True, "sin tenencias, solo sectores", sectores, canasta

    filas = filas_con_resto(tenencias)
    detallado = sum(peso for tk, peso in filas if tk != "_RESTO")

    if destino is not None:
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        with (destino / f"{ticker}.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
            escritor = csv.writer(fh)
            escritor.writerow(["ticker", "weight"])
            for tk, peso in filas:
                escritor.writerow([tk, f"{peso:.4f}"])

    n = sum(1 for tk, _ in filas if tk != "_RESTO")
    return True, f"{n} posiciones ({detallado:.1f}% detallado)", sectores, canasta


def escribir_auxiliares(destino: Path | str,
                        sectores: dict[str, dict[str, float]],
                        canastas: dict[str, dict[str, float]]) -> None:
    """Escribe ``_sectores.csv`` y ``_canasta.csv``."""
    destino = Path(destino)
    if sectores:
        destino.mkdir(parents=True, exist_ok=True)
        with (destino / "_sectores.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fondo", "sector", "peso"])
            for fondo, mapa in sorted(sectores.items()):
                for sector, peso in sorted(mapa.items(), key=lambda kv: -kv[1]):
                    w.writerow([fondo, sector, f"{float(peso):.6f}"])
    if canastas:
        destino.mkdir(parents=True, exist_ok=True)
        with (destino / "_canasta.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["fondo", "metrica", "valor"])
            for fondo, mapa in sorted(canastas.items()):
                for k, v in mapa.items():
                    w.writerow([fondo, k, v])


def bajar_varios(tickers: Iterable[str], destino: Path | str,
                 *, obtener: Callable[[str], Any] = _fondo_yahoo,
                 log: Callable[[str], None] | None = print,
                 ) -> tuple[list[str], list[str]]:
    """
    Baja varios fondos y escribe los auxiliares. Devuelve ``(ok, fallaron)``.

    Nunca levanta por un fondo que falle: el que no se pudo bajar se reporta y
    la transparencia lo cuenta como cobertura faltante.
    """
    destino = Path(destino)
    sectores: dict[str, dict[str, float]] = {}
    canastas: dict[str, dict[str, float]] = {}
    ok: list[str] = []
    fallaron: list[str] = []

    for tk in tickers:
        tk = tk.strip().upper()
        exito, detalle, secs, canasta = bajar_uno(tk, destino, obtener=obtener)
        if log:
            log(f"  {'OK   ' if exito else 'FALLO'}  {tk:6s} {detalle}")
        if exito:
            ok.append(tk)
            if secs:
                sectores[tk] = secs
            if canasta:
                canastas[tk] = canasta
        else:
            fallaron.append(tk)

    escribir_auxiliares(destino, sectores, canastas)
    return ok, fallaron
