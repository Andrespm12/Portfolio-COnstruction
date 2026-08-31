"""
Auditoría geométrica del PPTX generado.

LibreOffice no abre PPTX en este entorno, así que en vez de mirar imágenes se
leen las coordenadas que efectivamente quedaron escritas en el XML y se
verifican contra la lámina. Cubre los defectos que sí se pueden comprobar sin
renderizar: elementos fuera de la lámina, choques con el pie, márgenes
insuficientes, solapamientos, y una estimación de desborde de texto.

El desborde es una estimación, no una medición: sin motor de tipografía se
aproxima el ancho de carácter por el tamaño de fuente. Se reporta solo cuando
el margen es amplio, para que las alertas signifiquen algo.
"""

import re
import sys
import zipfile
from dataclasses import dataclass

EMU = 914400.0
ANCHO, ALTO = 13.333, 7.5
PIE_Y = 6.98          # arriba de la franja dorada
MARGEN = 0.5

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


@dataclass
class Forma:
    lamina: int
    nombre: str
    x: float
    y: float
    w: float
    h: float
    texto: str
    pt: float
    es_texto: bool

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h


def leer(path):
    import defusedxml.ElementTree as ET

    z = zipfile.ZipFile(path)
    laminas = sorted(
        (n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
        key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1)),
    )
    formas = []
    for i, nombre in enumerate(laminas, 1):
        raiz = ET.fromstring(z.read(nombre))
        for sp in raiz.iter():
            tag = sp.tag.split("}")[-1]
            if tag not in ("sp", "graphicFrame", "pic"):
                continue
            xfrm = None
            for e in sp.iter():
                if e.tag.split("}")[-1] == "xfrm":
                    xfrm = e
                    break
            if xfrm is None:
                continue
            off = xfrm.find("a:off", NS)
            ext = xfrm.find("a:ext", NS)
            if off is None or ext is None:
                continue
            textos, tam = [], []
            for t in sp.iter("{%s}t" % NS["a"]):
                textos.append(t.text or "")
            for rPr in sp.iter("{%s}rPr" % NS["a"]):
                if rPr.get("sz"):
                    tam.append(int(rPr.get("sz")) / 100.0)
            nom = ""
            for nv in sp.iter():
                if nv.tag.split("}")[-1] == "cNvPr":
                    nom = nv.get("name", "")
                    break
            formas.append(Forma(
                lamina=i, nombre=nom or tag,
                x=int(off.get("x")) / EMU, y=int(off.get("y")) / EMU,
                w=int(ext.get("cx")) / EMU, h=int(ext.get("cy")) / EMU,
                texto="".join(textos), pt=max(tam) if tam else 0.0,
                es_texto=bool(textos),
            ))
    return len(laminas), formas


def desborde_estimado(f):
    """
    Estima si el texto cabe. Gill Sans es estrecha; se usa 0.46 em de ancho
    medio por carácter, que es conservador (sobreestima el ancho), y se deja
    10% de holgura como pide la guía para fuentes que el previsualizador no
    reproduce fiel.
    """
    if not f.es_texto or not f.pt or not f.texto.strip():
        return None
    ancho_car = f.pt * 0.46 / 72.0
    if ancho_car <= 0 or f.w <= 0:
        return None
    por_linea = max(1, int(f.w / ancho_car))
    lineas_necesarias = 0
    for parrafo in f.texto.split("\n"):
        lineas_necesarias += max(1, -(-len(parrafo) // por_linea))
    alto_linea = f.pt * 1.22 / 72.0
    necesario = lineas_necesarias * alto_linea
    return necesario / f.h if f.h > 0 else None


def main(path):
    n_lam, formas = leer(path)
    print(f"{n_lam} láminas · {len(formas)} formas con geometría\n")

    fallos, avisos = [], []

    for f in formas:
        et = f"L{f.lamina:02d} «{(f.texto[:42] or f.nombre)}»"

        if f.x < -0.01 or f.y < -0.01 or f.x2 > ANCHO + 0.01 or f.y2 > ALTO + 0.01:
            fallos.append(f"{et}: fuera de la lámina "
                          f"(x {f.x:.2f}–{f.x2:.2f}, y {f.y:.2f}–{f.y2:.2f})")
            continue

        # choque con el pie: solo el propio pie puede estar ahí abajo
        es_pie = f.y >= PIE_Y - 0.02
        if not es_pie and f.y2 > PIE_Y + 0.01:
            fallos.append(f"{et}: invade el pie (baja hasta y={f.y2:.2f}, pie en {PIE_Y})")

        if not es_pie:
            if f.x < MARGEN - 0.01:
                avisos.append(f"{et}: margen izquierdo {f.x:.2f}\" < {MARGEN}\"")
            if f.x2 > ANCHO - MARGEN + 0.01:
                avisos.append(f"{et}: margen derecho {ANCHO - f.x2:.2f}\" < {MARGEN}\"")

        r = desborde_estimado(f)
        if r and r > 1.35:
            fallos.append(f"{et}: texto no cabe (~{r:.0%} del alto disponible, "
                          f"{len(f.texto)} car. en {f.w:.2f}×{f.h:.2f}\" a {f.pt}pt)")
        elif r and r > 0.95:
            avisos.append(f"{et}: texto justo (~{r:.0%} del alto, "
                          f"{len(f.texto)} car. a {f.pt}pt)")

    # Solape entre tarjetas hermanas: dos paneles del mismo tamaño en la misma
    # lámina son una rejilla, y no deben tocarse. Solaparse ahí es un error de
    # aritmética, no una superposición intencional como texto sobre tarjeta.
    por_lamina = {}
    for f in formas:
        if not f.es_texto and f.w > 1.5 and f.h > 0.5:
            por_lamina.setdefault(f.lamina, []).append(f)
    for lam, grupo in por_lamina.items():
        for i in range(len(grupo)):
            for j in range(i + 1, len(grupo)):
                a, b = grupo[i], grupo[j]
                if abs(a.w - b.w) > 0.02 or abs(a.h - b.h) > 0.02:
                    continue
                sx = min(a.x2, b.x2) - max(a.x, b.x)
                sy = min(a.y2, b.y2) - max(a.y, b.y)
                if sx > 0.01 and sy > 0.01:
                    fallos.append(f"L{lam:02d}: dos paneles hermanos de "
                                  f"{a.w:.2f}×{a.h:.2f}\" se solapan "
                                  f"{sx:.2f}×{sy:.2f}\"")

    if fallos:
        print(f"FALLOS ({len(fallos)}):")
        for x in fallos:
            print("  ✗", x)
    else:
        print("Sin fallos: todo dentro de la lámina, nada invade el pie, "
              "ningún texto se estima desbordado.")

    if avisos:
        print(f"\nAvisos ({len(avisos)}):")
        for x in avisos[:40]:
            print("  ·", x)
        if len(avisos) > 40:
            print(f"  … y {len(avisos) - 40} más")

    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
