"""
Sincronizar el almacén de SEC EDGAR: bajar lo que falta y refrescar lo viejo.

Por qué está en el paquete y no en un guion
-------------------------------------------
Este lazo existía dos veces — en ``scripts/bajar_fundamentales.py`` y copiado
dentro del cuaderno de fundamentales — y las dos copias se separaron sin que
nadie lo notara: el cuaderno siguió escribiendo "no está en company_tickers.json"
meses después de que el guion dejara de decirlo, y siguió dejando el
``_fallos.csv`` viejo en disco después de que el guion aprendiera a reescribirlo.

Hacía falta una tercera copia para que la corrida del modelo se abasteciera
sola. Tres copias del mismo lazo es una fábrica de esa clase de error, así que
el lazo vive acá y los tres lo llaman.

Qué garantiza
-------------
**Incremental de verdad.** Un nombre con CSV en disco no se vuelve a pedir. La
primera sincronización del universo son minutos y unos GB; las siguientes, casi
nada. Es la diferencia entre poder llamarlo desde la corrida diaria y no poder.

**Un tope por corrida.** Los refrescos van con límite y por antigüedad, del más
viejo al más nuevo. Sin el tope, el primer día que venza el plazo la corrida
diaria se convertiría sin aviso en una descarga de varios gigas — y quien la
lanzó solo quería un ranking.

**Nunca tumba la corrida.** Un emisor que falla se anota y se sigue. El modelo
tiene que poder correr con el bloque fundamental a medias; lo que no puede es
correr a medias en silencio, y para eso está :class:`Resultado`.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .edgar import (CONCEPTOS, MIN_EMISORES, Limitador, company_facts,
                    conceptos_desactualizados, detalle_sin_cik, escribir_hechos,
                    escribir_manifiesto, extract_facts, fuentes_del_mapa,
                    leer_overrides, load_ticker_map, plantilla_overrides)

#: Cada empresa presenta cuatro veces al año, así que un archivo de un mes
#: todavía describe el mismo ejercicio salvo justo en temporada de resultados.
REFRESCO_DIAS = 30

#: Cuántos refrescos como máximo por corrida. El día que venza el plazo para
#: todo el universo, sin este tope la corrida diaria se volvería una descarga de
#: varios gigas sin avisar. Con él, el universo rota en unos días.
MAX_REFRESCOS = 40


@dataclass
class Resultado:
    """Qué hizo la sincronización. Lo que se imprime sale de acá."""

    bajados: list[str] = field(default_factory=list)
    refrescados: list[str] = field(default_factory=list)
    saltados: list[str] = field(default_factory=list)
    sin_cik: list[str] = field(default_factory=list)
    fallaron: list[str] = field(default_factory=list)
    motivos: list[dict] = field(default_factory=list)
    emisores: list[dict] = field(default_factory=list)
    #: Nombres que quedaron pendientes por el tope de refrescos. Sin esto, un
    #: almacén que nunca termina de ponerse al día se ve idéntico a uno al día.
    pendientes: list[str] = field(default_factory=list)
    mapa: int = 0
    manuales: list[str] = field(default_factory=list)
    fuentes: dict = field(default_factory=dict)
    conceptos_nuevos: list[str] = field(default_factory=list)
    segundos: float = 0.0

    @property
    def intento(self) -> int:
        return len(self.bajados) + len(self.refrescados) + len(self.sin_cik) \
            + len(self.fallaron)

    def resumen(self) -> str:
        partes = [f"{len(self.bajados)} nuevos"]
        if self.refrescados:
            partes.append(f"{len(self.refrescados)} refrescados")
        partes.append(f"{len(self.saltados)} ya estaban")
        if self.sin_cik:
            partes.append(f"{len(self.sin_cik)} sin CIK")
        if self.fallaron:
            partes.append(f"{len(self.fallaron)} fallaron")
        if self.pendientes:
            partes.append(f"{len(self.pendientes)} pendientes de refresco")
        return ", ".join(partes) + f" ({self.segundos:.0f}s)"


def _por_refrescar(destino: Path, tickers: Sequence[str], dias: int,
                   tope: int) -> tuple[set[str], list[str]]:
    """Los archivos vencidos, del más viejo al más nuevo, recortados al tope."""
    if dias <= 0:
        return set(), []

    limite = time.time() - dias * 86400
    vencidos = []
    for ticker in tickers:
        archivo = destino / f"{ticker}.csv"
        if archivo.exists() and archivo.stat().st_mtime < limite:
            vencidos.append((archivo.stat().st_mtime, ticker))
    vencidos.sort()
    elegidos = [t for _, t in vencidos[:tope]] if tope > 0 else \
        [t for _, t in vencidos]
    return set(elegidos), [t for _, t in vencidos[len(elegidos):]]


def sincronizar(destino: Path | str, tickers: Iterable[str], *, contacto: str,
                forzar: bool = False, remapear: bool = False,
                refrescar_dias: int = REFRESCO_DIAS,
                max_refrescos: int = MAX_REFRESCOS,
                limitador: Limitador | None = None,
                progreso: Callable[[str], None] | None = None) -> Resultado:
    """
    Deja el almacén con un CSV por nombre, bajando solo lo que hace falta.

    ``progreso`` recibe una línea por evento; ``None`` la calla, que es lo que
    quiere una corrida del modelo donde la descarga es un medio y no el fin.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    limitador = Limitador() if limitador is None else limitador
    decir = progreso or (lambda _: None)
    t0 = time.time()

    res = Resultado()
    res.conceptos_nuevos = conceptos_desactualizados(destino)

    cache_mapa = destino / "_tickers.json"
    mapa = load_ticker_map(contacto=contacto, cache=cache_mapa,
                           limitador=limitador, refrescar=remapear)
    res.mapa = len(mapa)
    res.fuentes = fuentes_del_mapa(cache_mapa)
    manuales = leer_overrides(destino)
    res.manuales = sorted(manuales)
    if len(mapa) < MIN_EMISORES:
        decir(f"AVISO: el mapa trae {len(mapa)} emisores, menos de "
              f"{MIN_EMISORES}. Está incompleto y lo que salga 'sin CIK' no "
              "prueba nada.")

    refrescar, res.pendientes = _por_refrescar(
        destino, tickers, 0 if forzar else refrescar_dias, max_refrescos)

    etiquetas: list[dict] = []
    # El aviso de "esto va a tardar" sale en la primera petición de verdad, no
    # antes. Calcularlo por adelantado contando los CSV que faltan lo hacía
    # aparecer en cada corrida por culpa de los nombres que no están en EDGAR y
    # nunca van a estar: prometía una descarga que no ocurría.
    anunciado = False
    candidatos = sum(1 for t in tickers
                     if forzar or t in refrescar
                     or not (destino / f"{t}.csv").exists())

    for i, ticker in enumerate(tickers, 1):
        archivo = destino / f"{ticker}.csv"
        existia = archivo.exists()
        if existia and not forzar and ticker not in refrescar:
            res.saltados.append(ticker)
            continue

        cik = mapa.get(ticker) or mapa.get(ticker.replace("-", "."))
        if not cik:
            res.sin_cik.append(ticker)
            res.motivos.append({"ticker": ticker, "motivo": "sin CIK",
                                "detalle": detalle_sin_cik(res.fuentes)})
            decir(f"  [{i:3d}/{len(tickers)}] {ticker:6s} sin CIK en la SEC")
            continue

        if not anunciado:
            anunciado = True
            decir(f"Bajando de EDGAR hasta {candidatos} emisor(es) — una "
                  "petición cada uno, de 10 a 15 MB en los grandes. Lo que ya "
                  "está en disco no se vuelve a pedir.")

        try:
            payload = company_facts(cik, contacto=contacto,
                                    limitador=limitador)
            hechos, elegidas = extract_facts(payload, ticker)
        except Exception as exc:            # noqa: BLE001 - se reporta
            res.fallaron.append(ticker)
            res.motivos.append({"ticker": ticker, "motivo": type(exc).__name__,
                                "detalle": f"CIK {cik}: {exc}"})
            decir(f"  [{i:3d}/{len(tickers)}] {ticker:6s} "
                  f"FALLO {type(exc).__name__}: {exc}")
            continue

        if not hechos:
            res.fallaron.append(ticker)
            res.motivos.append({"ticker": ticker, "motivo": "sin etiquetas",
                                "detalle": f"CIK {cik}: companyfacts respondió, "
                                           "pero sin ninguna etiqueta de "
                                           "CONCEPTOS"})
            decir(f"  [{i:3d}/{len(tickers)}] {ticker:6s} "
                  "sin ninguna etiqueta conocida (¿emisor extranjero?)")
            continue

        escribir_hechos(destino, ticker, hechos)
        (res.refrescados if existia else res.bajados).append(ticker)
        # El nombre que la SEC tiene para ese CIK. Es la comprobación de que el
        # CIK es el correcto: uno equivocado no da error, da los estados
        # financieros de otra empresa con nuestro ticker encima.
        res.emisores.append({"ticker": ticker, "cik": cik,
                             "entidad": payload.get("entityName", ""),
                             "fuente": "a mano" if ticker in manuales
                                       else "SEC"})
        for metrica, etiqueta in elegidas.items():
            etiquetas.append({"ticker": ticker, "metrica": metrica,
                              "etiqueta": etiqueta})
        decir(f"  [{i:3d}/{len(tickers)}] {ticker:6s} {len(hechos):5d} hechos, "
              f"{len(elegidas)}/{len(CONCEPTOS)} métricas")

    _escribir_auxiliares(destino, res, etiquetas, forzar)
    if res.bajados or res.refrescados or forzar:
        if forzar or not res.conceptos_nuevos:
            escribir_manifiesto(destino)
    if res.sin_cik:
        plantilla_overrides(destino, res.sin_cik)

    res.segundos = time.time() - t0
    return res


def _escribir_auxiliares(destino: Path, res: Resultado, etiquetas: list[dict],
                         forzar: bool) -> None:
    """Los CSV de trazabilidad. Ninguno es opcional para poder auditar."""
    if etiquetas:
        _anexar(destino / "_etiquetas.csv",
                ["ticker", "metrica", "etiqueta"], etiquetas, forzar)
    if res.emisores:
        _anexar(destino / "_emisores.csv",
                ["ticker", "cik", "entidad", "fuente"], res.emisores, forzar)

    # _fallos.csv describe la ÚLTIMA corrida, así que si esta no tuvo fallos hay
    # que vaciarlo. Dejarlo puesto es peor que no escribirlo nunca: la corrida
    # que arregló ocho nombres los dejó ahí, con el diagnóstico viejo, junto a
    # una cobertura del 100%. Una corrida que no intentó nada no lo toca, porque
    # no repitió el diagnóstico y borrarlo sería tirarlo sin saber.
    if res.intento:
        with (destino / "_fallos.csv").open("w", newline="",
                                            encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["ticker", "motivo", "detalle"])
            w.writeheader()
            w.writerows(res.motivos)


def _anexar(ruta: Path, campos: list[str], filas: list[dict],
            reescribir: bool) -> None:
    modo = "w" if reescribir or not ruta.exists() else "a"
    with ruta.open(modo, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        if modo == "w":
            w.writeheader()
        w.writerows(filas)
