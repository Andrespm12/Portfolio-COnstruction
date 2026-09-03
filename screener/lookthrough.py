"""
Transparencia: qué tienes de verdad, mirando a través de los ETFs.

El problema que resuelve
------------------------
La hoja de Cartera no es la cartera. Un libro con 20% en un ETF de mercado
amplio tiene posiciones en cientos de empresas que nadie eligió una por una, y
tres cosas que el gestor necesita saber quedan invisibles:

* **Exposición efectiva por emisor.** El tope del Procedimiento está escrito
  sobre el instrumento, pero su intención es sobre el emisor. Con solo acciones
  las dos cosas coinciden; con ETFs en el libro se separan, y un nombre puede
  pasar su límite entre la posición directa y la que entra por los fondos.
* **Exposición sectorial real.** Comprar un ETF sectorial encima de uno amplio
  no da "exposición al sector": da un *sobrepeso* sobre lo que el amplio ya
  traía. La tabla de pesos no lo muestra.
* **Solape estructural.** Hoy los duplicados se detectan por correlación de
  retornos, que es una inferencia estadística y puede fallar en un régimen raro.
  Dos ETFs sobre el mismo índice comparten tenencias: eso es un hecho.

Qué NO hace
-----------
No estima. Si no hay archivo de tenencias para un ETF, ese peso queda declarado
como **opaco** y todo lo que se reporte lo dice. Rellenar con supuestos sería
peor que no mirar: daría números con apariencia de medición.

Una acción individual mira a través de sí misma, que es lo correcto y no un
caso especial.

De dónde salen los datos
------------------------
Los emisores publican las tenencias a diario en CSV. Este módulo acepta esos
archivos tal como se bajan (iShares, SPDR, Invesco tienen encabezados
distintos, y el parser los detecta) o un CSV normalizado de dos columnas.

Sin archivos, todo aquí devuelve cobertura cero y lo reporta. El resto del
modelo funciona igual: la transparencia es un reporte, no una dependencia.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Nombres de columna que suelen traer los archivos de los emisores.
_COL_TICKER = ("ticker", "symbol", "holding ticker", "identifier", "isin")
_COL_PESO = ("weight (%)", "weight", "% of net assets", "percent of fund",
             "portfolio weight", "weighting", "% weight")
_COL_SECTOR = ("sector", "gics sector", "industry")
_COL_NOMBRE = ("name", "security name", "holding name", "description")

#: Filas que los emisores meten y que no son tenencias.
_NO_ES_TENENCIA = re.compile(
    r"^(cash|usd|efectivo|net other|other assets|futures?|margin|"
    r"receivable|payable|accrued|swap|dividend)", re.IGNORECASE)

#: Marcadores de "sin ticker". Los emisores los usan para lo que no cotiza
#: (efectivo, derivados, posiciones sin identificar) y sin filtrarlos todas
#: esas filas se agregan en un emisor fantasma llamado "-".
_TICKER_VACIO = {"-", "--", "---", "N/A", "NA", "NULL", "NONE", "."}


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace("﻿", "")


def _find_col(header: list[str], candidates: Iterable[str]) -> int | None:
    norm = [_norm(h) for h in header]
    for cand in candidates:
        if cand in norm:
            return norm.index(cand)
    for i, h in enumerate(norm):          # coincidencia parcial, como respaldo
        if any(c in h for c in candidates):
            return i
    return None


def _to_float(raw: str) -> float | None:
    """Los emisores escriben 1.23, '1.23%', '1,23' y '(1.23)'."""
    if raw is None:
        return None
    t = str(raw).strip().replace("%", "").replace(",", "").replace("$", "")
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    try:
        v = float(t)
    except ValueError:
        return None
    return v if v == v else None          # descarta NaN


def parse_holdings_csv(path: str | Path) -> tuple[dict[str, float], dict[str, str]]:
    """
    Tenencias de un archivo del emisor: ``{ticker: peso}`` y ``{ticker: sector}``.

    Los archivos traen líneas de preámbulo antes del encabezado real, así que
    se busca la primera fila que tenga a la vez algo parecido a un ticker y algo
    parecido a un peso. Los pesos se normalizan a sumar 1.0: vienen en por
    ciento, casi nunca suman exacto, y a veces excluyen el efectivo.
    """
    path = Path(path)
    filas = list(csv.reader(path.read_text(
        encoding="utf-8-sig", errors="replace").splitlines()))

    i_head = None
    for i, fila in enumerate(filas[:40]):
        if len(fila) < 2:
            continue
        if (_find_col(fila, _COL_TICKER) is not None
                and _find_col(fila, _COL_PESO) is not None):
            i_head = i
            break
    if i_head is None:
        raise ValueError(
            f"{path.name}: no encontré un encabezado con ticker y peso. "
            "Si es un formato raro, pásalo como CSV de dos columnas: "
            "ticker,peso")

    head = filas[i_head]
    c_tk = _find_col(head, _COL_TICKER)
    c_w = _find_col(head, _COL_PESO)
    c_sec = _find_col(head, _COL_SECTOR)

    pesos: dict[str, float] = {}
    sectores: dict[str, str] = {}
    for fila in filas[i_head + 1:]:
        if len(fila) <= max(c_tk, c_w):
            continue
        tk = (fila[c_tk] or "").strip().upper()
        w = _to_float(fila[c_w])
        if not tk or tk in _TICKER_VACIO or w is None or w <= 0:
            continue
        if _NO_ES_TENENCIA.match(tk) or _NO_ES_TENENCIA.match(fila[c_tk] or ""):
            continue
        pesos[tk] = pesos.get(tk, 0.0) + w
        if c_sec is not None and len(fila) > c_sec and fila[c_sec].strip():
            sectores.setdefault(tk, fila[c_sec].strip())

    total = sum(pesos.values())
    if total <= 0:
        raise ValueError(f"{path.name}: no se leyó ninguna tenencia con peso > 0.")
    return {k: v / total for k, v in pesos.items()}, sectores


def load_holdings(directory: str | Path) -> tuple[dict[str, dict[str, float]],
                                                  dict[str, str], list[str]]:
    """
    Carga todos los CSV de un directorio. El nombre del archivo es el ticker.

    ``SPY.csv`` son las tenencias de SPY. Devuelve las tenencias, el mapa de
    sectores acumulado, y las notas de lo que no se pudo leer -- un archivo
    ilegible se reporta, nunca se ignora en silencio.
    """
    directory = Path(directory)
    holdings: dict[str, dict[str, float]] = {}
    sectores: dict[str, str] = {}
    notas: list[str] = []

    if not directory.is_dir():
        return {}, {}, [f"No existe el directorio de tenencias: {directory}"]

    for archivo in sorted(directory.glob("*.csv")):
        etf = archivo.stem.strip().upper()
        try:
            pesos, secs = parse_holdings_csv(archivo)
        except Exception as exc:          # noqa: BLE001 - se reporta, no se traga
            notas.append(f"{archivo.name}: no se pudo leer ({exc})")
            continue
        holdings[etf] = pesos
        sectores.update(secs)
    return holdings, sectores, notas


# --------------------------------------------------------------------------
# Transparencia
# --------------------------------------------------------------------------

def effective_exposure(weights: Mapping[str, float],
                       holdings: Mapping[str, Mapping[str, float]],
                       ) -> tuple[dict[str, float], float, list[str]]:
    """
    Exposición por emisor mirando a través, la cobertura lograda y las notas.

    Un nombre sin archivo de tenencias mira a través de sí mismo. Para una
    acción eso es exacto. Para un ETF es una **ceguera**, y por eso la cobertura
    se devuelve y se reporta: si solo ves a través del 40% del libro, los
    números sectoriales de abajo describen ese 40%, no la cartera.
    """
    exposicion: dict[str, float] = {}
    visto = 0.0
    opacos: list[tuple[str, float]] = []

    for ticker, peso in weights.items():
        if peso is None or peso <= 0:
            continue
        tk = ticker.upper()
        canasta = holdings.get(tk)
        if canasta:
            visto += peso
            for sub, share in canasta.items():
                exposicion[sub] = exposicion.get(sub, 0.0) + peso * share
        else:
            exposicion[tk] = exposicion.get(tk, 0.0) + peso
            opacos.append((tk, peso))

    total = sum(w for w in weights.values() if w and w > 0)
    cobertura = visto / total if total > 0 else 0.0

    notas: list[str] = []
    if opacos:
        detalle = ", ".join(f"{t} {w:.1%}" for t, w in
                            sorted(opacos, key=lambda kv: -kv[1])[:8])
        notas.append(
            f"Sin tenencias para {len(opacos)} posición(es): {detalle}. "
            "Cada una se cuenta como exposición a sí misma. Para una acción eso "
            "es exacto; para un fondo es un punto ciego."
        )
    notas.append(f"Cobertura de transparencia: {cobertura:.0%} del libro.")
    return exposicion, cobertura, notas


def sector_exposure(exposicion: Mapping[str, float],
                    sectores: Mapping[str, str]) -> dict[str, float]:
    """Agrupa la exposición efectiva por sector. Lo desconocido se declara."""
    out: dict[str, float] = {}
    for emisor, peso in exposicion.items():
        s = sectores.get(emisor.upper()) or "Sin clasificar"
        out[s] = out.get(s, 0.0) + peso
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def structural_overlap(a: str, b: str,
                       holdings: Mapping[str, Mapping[str, float]]) -> float | None:
    """
    Fracción de canasta compartida entre dos posiciones, en [0, 1].

    Suma de los mínimos de los pesos comunes. Sustituye una inferencia por un
    hecho: dos ETFs sobre el mismo índice dan ~1.0 porque tienen los mismos
    papeles, no porque sus retornos hayan ido juntos el año pasado.
    """
    ca = holdings.get(a.upper()) or ({a.upper(): 1.0} if a else None)
    cb = holdings.get(b.upper()) or ({b.upper(): 1.0} if b else None)
    if not ca or not cb:
        return None
    return float(sum(min(ca.get(k, 0.0), cb.get(k, 0.0))
                     for k in set(ca) | set(cb)))


def issuer_cap_breaches(weights: Mapping[str, float],
                        holdings: Mapping[str, Mapping[str, float]],
                        cap: float,
                        *, only: Iterable[str] | None = None,
                        ) -> list[dict[str, Any]]:
    """
    Emisores cuya exposición **efectiva** pasa el tope por nombre.

    El límite del Procedimiento se aplica hoy sobre la posición directa. Su
    intención es sobre el emisor, y la diferencia solo aparece cuando hay fondos
    en el libro. Esto la mide.

    ``only`` restringe a los emisores a los que el tope aplica -- normalmente
    acciones individuales, no fondos.
    """
    exposicion, _, _ = effective_exposure(weights, holdings)
    permitidos = {t.upper() for t in only} if only is not None else None

    fuera = []
    for emisor, efectiva in exposicion.items():
        if permitidos is not None and emisor.upper() not in permitidos:
            continue
        if efectiva > cap + 1e-9:
            directo = float(weights.get(emisor, 0.0) or 0.0)
            fuera.append({
                "emisor": emisor,
                "efectiva": efectiva,
                "directa": directo,
                "via_fondos": efectiva - directo,
                "tope": cap,
            })
    return sorted(fuera, key=lambda d: -d["efectiva"])


def report(weights: Mapping[str, float],
           holdings: Mapping[str, Mapping[str, float]],
           sectores: Mapping[str, str],
           *, cap: float | None = None,
           only: Iterable[str] | None = None,
           top: int = 15) -> str:
    """Reporte de transparencia listo para imprimir en una corrida."""
    exposicion, cobertura, notas = effective_exposure(weights, holdings)
    lineas = ["TRANSPARENCIA (look-through)", ""]
    lineas += [f"  {n}" for n in notas]

    if cobertura <= 0:
        lineas.append("")
        lineas.append("  Sin archivos de tenencias no hay nada que mirar a través. "
                      "Bájalos de los emisores y ponlos en el directorio de "
                      "tenencias; el nombre del archivo es el ticker (SPY.csv).")
        return "\n".join(lineas)

    lineas += ["", f"  Exposición efectiva por emisor (top {top}):"]
    for emisor, w in sorted(exposicion.items(), key=lambda kv: -kv[1])[:top]:
        directo = float(weights.get(emisor, 0.0) or 0.0)
        via = w - directo
        marca = "" if via <= 1e-9 else f"  (directo {directo:.2%} + fondos {via:.2%})"
        lineas.append(f"    {w:>7.2%}  {emisor}{marca}")

    sec = sector_exposure(exposicion, sectores)
    if sec:
        lineas += ["", "  Exposición sectorial:"]
        for s, w in sec.items():
            lineas.append(f"    {w:>7.2%}  {s}")

    if cap is not None:
        fuera = issuer_cap_breaches(weights, holdings, cap, only=only)
        lineas += ["", f"  Tope por emisor ({cap:.0%}):"]
        if not fuera:
            lineas.append("    sin excesos sobre exposición efectiva.")
        for d in fuera:
            lineas.append(
                f"    {d['emisor']}: {d['efectiva']:.2%} efectiva "
                f"({d['directa']:.2%} directa + {d['via_fondos']:.2%} vía fondos) "
                f"excede {d['tope']:.0%}")
    return "\n".join(lineas)
