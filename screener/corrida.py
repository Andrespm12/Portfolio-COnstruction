"""
Una carpeta por corrida con todo lo necesario para reproducirla.

El review encontró que los datos venían en vivo de Yahoo y no se guardaban:
una propuesta que llega al Comité no se podía rastrear al dato exacto, la
versión del motor ni los parámetros que la produjeron. Rehacer la corrida un
día después da otro resultado, y no hay forma de saber si la diferencia es el
mercado o el código.

Lo que se guarda:

* ``market_data.json.gz`` — exactamente lo que puntuó el screener: barras
  semanales, snapshot y fundamentales adjuntos.
* ``retornos_cesta.csv.gz`` — los retornos diarios de la cesta, de los que
  salen la covarianza y el equilibrio.
* ``parametros.json`` — cada parámetro de la corrida, la versión del motor
  (SHA-256 del paquete embebido) y la fecha de la tasa libre de riesgo.
* ``views.json`` y ``cartera.csv`` — lo que la corrida propuso.
* cualquier tabla extra (riesgo, estrés, sensibilidad) como CSV.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def empaquetar_motor(directorio: str | Path | None = None) -> bytes:
    """
    El paquete ``screener`` como tar determinista: mismo código, mismos bytes.

    Es lo que el notebook embebe, y de aquí sale su ``ENGINE_SHA256``. Vive en
    el motor y no en el script que arma el notebook para que el script de
    línea de comandos pueda calcular la misma huella: una corrida desde
    cualquiera de los dos registra con qué código se hizo, y dos corridas con
    la misma huella corrieron exactamente el mismo motor.
    """
    directorio = Path(directorio) if directorio else Path(__file__).resolve().parent
    buffer = io.BytesIO()
    # mtime=0 y miembros ordenados: sin cambio en el código no hay cambio en
    # los bytes, y una reconstrucción no produce diff.
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for ruta in sorted(directorio.glob("*.py"), key=lambda p: p.name):
            datos = ruta.read_bytes()
            info = tarfile.TarInfo(f"screener/{ruta.name}")
            info.size = len(datos)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(datos))
    return buffer.getvalue()


def huella_motor(directorio: str | Path | None = None) -> str:
    """SHA-256 de :func:`empaquetar_motor`: la versión del motor de una corrida."""
    return hashlib.sha256(empaquetar_motor(directorio)).hexdigest()


def _json_default(valor: Any) -> Any:
    """Lo que ``json`` no sabe escribir: tipos de numpy, fechas, NaN."""
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating,)):
        f = float(valor)
        return f if np.isfinite(f) else None
    if isinstance(valor, (np.ndarray, pd.Series)):
        return valor.tolist()
    if isinstance(valor, (pd.Timestamp, datetime)):
        return valor.isoformat()
    return str(valor)


def guardar_corrida(base: str | Path, *,
                    market_data: Mapping[str, Any],
                    parametros: Mapping[str, Any] | pd.DataFrame,
                    views: Sequence[Mapping[str, Any]] = (),
                    cartera: pd.Series | None = None,
                    retornos: pd.DataFrame | None = None,
                    tablas: Mapping[str, pd.DataFrame] | None = None,
                    cuando: datetime | None = None) -> Path:
    """
    Escribe la corrida en ``base/AAAAMMDD-HHMMSS/`` y devuelve esa carpeta.

    Nunca sobrescribe: si la carpeta del segundo ya existe, agrega un sufijo.
    Una corrida guardada es un registro, y un registro que se pisa no sirve.
    """
    cuando = cuando or datetime.now(timezone.utc)
    carpeta = Path(base) / cuando.strftime("%Y%m%d-%H%M%S")
    sufijo = 1
    while carpeta.exists():
        carpeta = Path(base) / f"{cuando.strftime('%Y%m%d-%H%M%S')}-{sufijo}"
        sufijo += 1
    carpeta.mkdir(parents=True)

    with gzip.open(carpeta / "market_data.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(market_data, fh, default=_json_default, ensure_ascii=False)

    if isinstance(parametros, pd.DataFrame):
        parametros = {str(k): v for k, v in parametros.itertuples(index=False)}
    (carpeta / "parametros.json").write_text(
        json.dumps({"guardado_utc": cuando.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    **parametros}, default=_json_default, ensure_ascii=False,
                   indent=2),
        encoding="utf-8")

    (carpeta / "views.json").write_text(
        json.dumps([{k: v for k, v in dict(x).items() if not str(k).startswith("_")}
                    for x in views], default=_json_default, ensure_ascii=False,
                   indent=2),
        encoding="utf-8")

    if cartera is not None:
        (cartera[cartera > 0].rename("peso").rename_axis("ticker")
         .to_csv(carpeta / "cartera.csv"))
    if retornos is not None:
        retornos.to_csv(carpeta / "retornos_cesta.csv.gz", compression="gzip")
    for nombre, tabla in (tablas or {}).items():
        if tabla is not None and not tabla.empty:
            tabla.to_csv(carpeta / f"{nombre}.csv", index=False)
    return carpeta


def leer_corrida(carpeta: str | Path) -> dict[str, Any]:
    """Lo guardado por :func:`guardar_corrida`, para reconstruir o auditar."""
    carpeta = Path(carpeta)
    with gzip.open(carpeta / "market_data.json.gz", "rt", encoding="utf-8") as fh:
        market_data = json.load(fh)
    out: dict[str, Any] = {
        "market_data": market_data,
        "parametros": json.loads((carpeta / "parametros.json").read_text(encoding="utf-8")),
        "views": json.loads((carpeta / "views.json").read_text(encoding="utf-8")),
    }
    if (carpeta / "cartera.csv").exists():
        out["cartera"] = pd.read_csv(carpeta / "cartera.csv", index_col=0)["peso"]
    if (carpeta / "retornos_cesta.csv.gz").exists():
        out["retornos"] = pd.read_csv(carpeta / "retornos_cesta.csv.gz",
                                      index_col=0, parse_dates=True)
    return out
