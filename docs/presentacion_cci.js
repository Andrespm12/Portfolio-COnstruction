// Presentación CCI — modelo de selección y construcción de cartera.
// Paleta y tipografía del manual de marca. Motivo visual: círculos, que es
// la construcción del propio isotipo.

const pptxgen = require("pptxgenjs");

const VERDE = "9EC229";
const ORO = "FBDE64";
const GRAFITO = "414042";
const GRIS = "D1D3D4";
const SUTIL = "6D6E70";
const FONDO = "F0F0F0";
const BLANCO = "FFFFFF";
const FUENTE = "Gill Sans MT";
const TAGLINE = "Capital para el logro  •  Crédito a tus sueños  •  Inversiones en tu futuro";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.author = "CCI Puesto de Bolsa";
pres.title = "Selección de activos y construcción de cartera";

let pagina = 0;

// ---------------------------------------------------------------- helpers

function isotipo(slide, x, y, tam, claro) {
  // Dos círculos superpuestos: el de atrás es la sombra (K20), el de
  // adelante K90 con las letras. Placeholder — sustituir por el archivo
  // oficial antes de presentar.
  slide.addShape(pres.ShapeType.ellipse, {
    x: x + tam * 0.22, y: y + tam * 0.06, w: tam, h: tam,
    fill: { color: claro ? "5A585A" : GRIS }, line: { color: claro ? "5A585A" : GRIS },
  });
  slide.addShape(pres.ShapeType.ellipse, {
    x: x, y: y, w: tam, h: tam,
    fill: { color: claro ? BLANCO : GRAFITO }, line: { color: claro ? BLANCO : GRAFITO },
  });
  slide.addText(
    [
      { text: "C", options: { color: claro ? GRAFITO : BLANCO, bold: true } },
      { text: "C", options: { color: VERDE, bold: true } },
      { text: "I", options: { color: ORO, bold: true } },
    ],
    {
      x: x, y: y + tam * 0.24, w: tam, h: tam * 0.5,
      fontFace: FUENTE, fontSize: Math.round(tam * 26), align: "center",
      isTextBox: true, margin: 0,
    }
  );
}

function marca(slide, claro) {
  isotipo(slide, 0.5, 0.28, 0.42, claro);
  slide.addText("PUESTO DE BOLSA, S.A.", {
    x: 1.12, y: 0.3, w: 3.0, h: 0.2,
    fontFace: FUENTE, fontSize: 9, bold: true, charSpacing: 0.6,
    color: claro ? BLANCO : GRAFITO, isTextBox: true, margin: 0,
  });
  slide.addText("MIEMBRO DE LA BVRD", {
    x: 1.12, y: 0.5, w: 3.0, h: 0.2,
    fontFace: FUENTE, fontSize: 7.5, charSpacing: 0.6,
    color: claro ? GRIS : SUTIL, isTextBox: true, margin: 0,
  });
}

function pie(slide) {
  pagina += 1;
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 6.98, w: 13.333, h: 0.05, fill: { color: ORO }, line: { color: ORO },
  });
  slide.addShape(pres.ShapeType.rect, {
    x: 0, y: 7.03, w: 13.333, h: 0.47, fill: { color: GRAFITO }, line: { color: GRAFITO },
  });
  slide.addText(String(pagina), {
    x: 0.5, y: 7.11, w: 0.8, h: 0.3,
    fontFace: FUENTE, fontSize: 10, color: BLANCO, isTextBox: true, margin: 0,
  });
  slide.addText(TAGLINE, {
    x: 6.0, y: 7.11, w: 6.83, h: 0.3, align: "right",
    fontFace: FUENTE, fontSize: 9, color: GRIS, isTextBox: true, margin: 0,
  });
  return slide;
}

function laminaContenido(titulo, bajada) {
  const s = pres.addSlide();
  s.background = { color: BLANCO };
  marca(s, false);
  s.addText(titulo, {
    x: 0.5, y: 0.95, w: 12.33, h: 0.55,
    fontFace: FUENTE, fontSize: 30, bold: true, color: VERDE,
    isTextBox: true, margin: 0,
  });
  if (bajada) {
    s.addText(bajada, {
      x: 0.5, y: 1.5, w: 12.33, h: 0.4,
      fontFace: FUENTE, fontSize: 14, color: SUTIL, isTextBox: true, margin: 0,
    });
  }
  pie(s);
  return s;
}

function laminaSeccion(numero, titulo, bajada) {
  const s = pres.addSlide();
  s.background = { color: GRAFITO };
  marca(s, true);
  s.addShape(pres.ShapeType.ellipse, {
    x: 0.9, y: 2.5, w: 1.5, h: 1.5, fill: { color: VERDE }, line: { color: VERDE },
  });
  s.addText(numero, {
    x: 0.9, y: 2.85, w: 1.5, h: 0.85,
    fontFace: FUENTE, fontSize: 44, bold: true, color: GRAFITO,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText(titulo, {
    x: 2.9, y: 2.5, w: 9.5, h: 0.9,
    fontFace: FUENTE, fontSize: 34, bold: true, color: BLANCO,
    isTextBox: true, margin: 0,
  });
  s.addText(bajada, {
    x: 2.9, y: 3.42, w: 9.3, h: 0.7,
    fontFace: FUENTE, fontSize: 15, color: GRIS, isTextBox: true, margin: 0,
  });
  pie(s);
  return s;
}

// Tarjeta con número en círculo verde
function tarjetaNumerada(s, n, x, y, w, h, titulo, cuerpo) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: x + 0.3, y: y + 0.28, w: 0.46, h: 0.46,
    fill: { color: VERDE }, line: { color: VERDE },
  });
  s.addText(String(n), {
    x: x + 0.3, y: y + 0.35, w: 0.46, h: 0.32,
    fontFace: FUENTE, fontSize: 17, bold: true, color: BLANCO,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText(titulo, {
    x: x + 0.92, y: y + 0.3, w: w - 1.2, h: 0.42,
    fontFace: FUENTE, fontSize: 16, bold: true, color: GRAFITO,
    isTextBox: true, margin: 0,
  });
  s.addText(cuerpo, {
    x: x + 0.3, y: y + 0.88, w: w - 0.6, h: h - 1.1,
    fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
  });
}

function destacado(s, x, y, w, h, texto, tono) {
  const fondo = tono === "oro" ? ORO : (tono === "verde" ? VERDE : GRAFITO);
  const color = tono === "oro" ? GRAFITO : BLANCO;
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06, fill: { color: fondo }, line: { color: fondo },
  });
  s.addText(texto, {
    x: x + 0.35, y: y + 0.12, w: w - 0.7, h: h - 0.24,
    fontFace: FUENTE, fontSize: 14, bold: true, color,
    valign: "middle", isTextBox: true, margin: 0,
  });
}

function cifra(s, x, y, w, numero, etiqueta, color) {
  s.addText(numero, {
    x, y, w, h: 0.85,
    fontFace: FUENTE, fontSize: 46, bold: true, color: color || VERDE,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText(etiqueta, {
    x, y: y + 0.85, w, h: 0.6,
    fontFace: FUENTE, fontSize: 11.5, color: SUTIL,
    align: "center", isTextBox: true, margin: 0,
  });
}

const ENC = { fill: GRAFITO, color: BLANCO, bold: true, fontFace: FUENTE, fontSize: 12 };
function celda(t, opts) {
  return Object.assign({ text: t, options: Object.assign({ fontFace: FUENTE, fontSize: 12, color: GRAFITO }, opts || {}) });
}

// ================================================================ PORTADA
{
  const s = pres.addSlide();
  s.background = { color: GRAFITO };
  isotipo(s, 0.9, 0.75, 0.85, true);
  s.addText("PUESTO DE BOLSA, S.A.", {
    x: 2.1, y: 0.86, w: 5, h: 0.28, fontFace: FUENTE, fontSize: 13, bold: true,
    charSpacing: 1, color: BLANCO, isTextBox: true, margin: 0,
  });
  s.addText("MIEMBRO DE LA BVRD", {
    x: 2.1, y: 1.16, w: 5, h: 0.25, fontFace: FUENTE, fontSize: 10,
    charSpacing: 1, color: GRIS, isTextBox: true, margin: 0,
  });

  s.addText("Selección de activos y\nconstrucción de cartera", {
    x: 0.9, y: 2.5, w: 9.6, h: 1.8,
    fontFace: FUENTE, fontSize: 40, bold: true, color: VERDE,
    lineSpacing: 46, isTextBox: true, margin: 0,
  });
  s.addText("Cómo funciona el modelo, qué encontramos al auditarlo y qué sigue pendiente", {
    x: 0.9, y: 4.35, w: 10.5, h: 0.5,
    fontFace: FUENTE, fontSize: 16, color: GRIS, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.rect, {
    x: 0.9, y: 5.15, w: 1.1, h: 0.045, fill: { color: ORO }, line: { color: ORO },
  });
  s.addText("Comité de Inversiones y Mesa Internacional", {
    x: 0.9, y: 5.45, w: 8, h: 0.3,
    fontFace: FUENTE, fontSize: 13, color: BLANCO, isTextBox: true, margin: 0,
  });
  s.addText("Agosto 2026", {
    x: 0.9, y: 5.78, w: 8, h: 0.3,
    fontFace: FUENTE, fontSize: 12, color: SUTIL, isTextBox: true, margin: 0,
  });
  // círculos decorativos, el motivo de la marca
  s.addShape(pres.ShapeType.ellipse, {
    x: 10.6, y: 1.9, w: 2.6, h: 2.6,
    fill: { color: VERDE, transparency: 82 }, line: { color: VERDE, transparency: 55 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.5, y: 3.4, w: 1.7, h: 1.7,
    fill: { color: ORO, transparency: 85 }, line: { color: ORO, transparency: 60 },
  });
  pie(s);
  s.addNotes("Objetivo de la sesión: que el comité entienda cómo decide el modelo, " +
    "qué controles tiene y qué supuestos pusimos nosotros y todavía hay que validar. " +
    "No vengo a pedir aprobación de operar; vengo a que lo puedan cuestionar.");
}

// ================================================================ AGENDA
{
  const s = laminaContenido("De qué vamos a hablar", "Cuatro bloques. El tercero es el más largo y es el que importa.");
  const cont = [
    ["Qué construimos", "El problema que resuelve y las siete etapas del proceso, de punta a punta."],
    ["Cómo puntúa el modelo", "Quién entra al universo, las 25 medidas, cómo se convierten en ranking y qué frenos tiene encima."],
    ["Black-Litterman", "Qué problema resuelve, cómo funciona por dentro y cómo conectamos las dos mitades."],
    ["Qué encontramos", "La auditoría del modelo, lo que cambiamos y lo que todavía necesita una decisión."],
  ];
  cont.forEach((c, i) => {
    const x = 0.5 + (i % 2) * 6.35;
    const y = 2.1 + Math.floor(i / 2) * 2.35;
    tarjetaNumerada(s, i + 1, x, y, 5.98, 2.05, c[0], c[1]);
  });
  s.addNotes("Avisar que el bloque 3 es el corazón. Si alguien tiene que salir antes, " +
    "que se quede al menos hasta terminar Black-Litterman.");
}

// ================================================================ PROBLEMA
{
  const s = laminaContenido("Elegir y ponderar son dos preguntas distintas",
    "Se mezclan todo el tiempo, y se contestan con herramientas diferentes.");

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.15, w: 5.98, h: 2.5, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("¿Qué compro?", {
    x: 0.9, y: 2.45, w: 5.2, h: 0.5,
    fontFace: FUENTE, fontSize: 22, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("Comparar cientos de instrumentos entre sí con criterios idénticos. " +
    "Un humano no puede hacerlo a mano sin inclinarse hacia lo que ya conoce.", {
    x: 0.9, y: 3.05, w: 5.2, h: 1.3,
    fontFace: FUENTE, fontSize: 13.5, color: SUTIL, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.85, y: 2.15, w: 5.98, h: 2.5, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("¿Cuánto pongo en cada cosa?", {
    x: 7.25, y: 2.45, w: 5.2, h: 0.5,
    fontFace: FUENTE, fontSize: 22, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("Correlaciones, cuánto riesgo aporta cada posición al total y los límites del " +
    "mandato. Aquí la intuición falla feo: dos posiciones de 5% no son lo mismo si una " +
    "es un soberano y la otra una acción de semiconductores.", {
    x: 7.25, y: 3.05, w: 5.2, h: 1.5,
    fontFace: FUENTE, fontSize: 13.5, color: SUTIL, isTextBox: true, margin: 0,
  });

  destacado(s, 0.5, 5.15, 12.33, 0.85,
    "El screener contesta la primera. Black-Litterman contesta la segunda. " +
    "La frontera entre las dos es lo único que hay que memorizar.", "verde");
  s.addNotes("Si el comité se lleva una sola frase, que sea la de la banda verde.");
}

// ================================================================ 7 ETAPAS
{
  const s = laminaContenido("Qué construimos", "Un solo notebook. Se aprieta un botón y sale un Excel de nueve hojas.");
  const pasos = ["Universo", "Datos", "25 medidas", "Puntaje", "Frenos", "Views", "Cartera"];
  const detalle = ["447 candidatos", "2 años, Yahoo", "seis bloques", "relativo al grupo",
    "solo bajan", "Q y convicción", "con las bandas"];
  pasos.forEach((p, i) => {
    const x = 0.5 + i * 1.79;
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.28, y: 2.35, w: 1.0, h: 1.0,
      fill: { color: i < 5 ? VERDE : GRAFITO }, line: { color: i < 5 ? VERDE : GRAFITO },
    });
    s.addText(String(i + 1), {
      x: x + 0.28, y: 2.58, w: 1.0, h: 0.55,
      fontFace: FUENTE, fontSize: 22, bold: true, color: BLANCO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(p, {
      x: x - 0.05, y: 3.5, w: 1.65, h: 0.32,
      fontFace: FUENTE, fontSize: 12.5, bold: true, color: GRAFITO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(detalle[i], {
      x: x - 0.05, y: 3.8, w: 1.65, h: 0.32,
      fontFace: FUENTE, fontSize: 10.5, color: SUTIL,
      align: "center", isTextBox: true, margin: 0,
    });
    if (i < 6) {
      s.addShape(pres.ShapeType.rect, {
        x: x + 1.35, y: 2.83, w: 0.4, h: 0.035,
        fill: { color: GRIS }, line: { color: GRIS },
      });
    }
  });
  s.addText("Los pasos 1 a 5 son el screener: deciden sobre qué nombres hay una opinión y " +
    "qué tan fuerte. Los pasos 6 y 7 son Black-Litterman: deciden los pesos.", {
    x: 0.5, y: 4.55, w: 12.33, h: 0.6,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 5.35, 12.33, 0.75,
    "Nada se escribe entre las dos mitades. Las views son una variable en memoria que " +
    "la celda siguiente consume directo.", "oro");
}

// ================================================================ TRES REGLAS
{
  const s = laminaContenido("Tres cosas que conviene fijar de entrada", "");
  const reglas = [
    ["No lee ninguna cuenta", "No sabe qué tienes ni cuánto. No puede recomendarte algo por lo que ya está en el libro. Cada instrumento se evalúa por sus propios méritos."],
    ["No ejecuta nada", "Produce una propuesta. Cada decisión pasa por el gestor, y el archivo de views va a una carpeta de propuestas que el código no puede confundir con la de aprobadas."],
    ["El perfil no es una etiqueta", "Elegir Conservador o Agresivo reconfigura pesos, umbrales, frenos, tamaño de posición y liquidez mínima. El mismo nombre, el mismo día, puede salir OW en uno y UW en otro."],
  ];
  reglas.forEach((r, i) => {
    const x = 0.5 + i * 4.19;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.1, w: 3.95, h: 3.6, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.35, y: 2.45, w: 0.62, h: 0.62,
      fill: { color: [VERDE, ORO, GRAFITO][i] }, line: { color: [VERDE, ORO, GRAFITO][i] },
    });
    s.addText(r[0], {
      x: x + 0.35, y: 3.3, w: 3.25, h: 0.75,
      fontFace: FUENTE, fontSize: 17, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(r[1], {
      x: x + 0.35, y: 4.1, w: 3.25, h: 1.5,
      fontFace: FUENTE, fontSize: 12, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  s.addNotes("La tercera suele sorprender. Vale la pena detenerse: no es el mismo ranking " +
    "con otro corte, es otro modelo.");
}

// ================================================================ UNIVERSO 1
{
  const s = laminaContenido("Antes de puntuar: quién entra a la cancha",
    "La pregunta que nos hicieron y que no teníamos bien contestada.");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.05, w: 12.33, h: 0.95, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("«¿Cuál es el criterio para decidir qué acciones se evalúan y cuáles no, más allá " +
    "de que estén en el S&P, el Nasdaq o el Dow?»", {
    x: 0.85, y: 2.2, w: 11.6, h: 0.65,
    fontFace: FUENTE, fontSize: 15, italic: true, color: GRAFITO,
    valign: "middle", isTextBox: true, margin: 0,
  });
  s.addText("La respuesta honesta era que no había uno escrito: había una lista, y la lista " +
    "era el criterio. Ahora hay cuatro reglas que corren en orden, y el orden importa — cuando " +
    "un nombre falla, se reporta la primera que lo botó, para que el motivo sea accionable.", {
    x: 0.5, y: 3.15, w: 12.33, h: 0.7,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });
  const reglas = [
    ["Pertenencia", "¿Cotiza en EE.UU. y está en un índice del universo, o en la lista curada de ETFs?"],
    ["Producto", "Fuera apalancados, inversos, covered-call y ETNs: su retorno depende del camino, no del destino."],
    ["Historia", "¿Alcanza para calcular el modelo completo? Hoy son 53 semanas, y el piso lo fija el propio modelo."],
    ["Negociabilidad", "Precio, volumen diario y días para liquidar. El único de los cuatro que cambia con el perfil."],
  ];
  reglas.forEach((r, i) => {
    const x = 0.5 + i * 3.11;
    tarjetaNumerada(s, i + 1, x, 3.95, 2.91, 2.45, r[0], r[1]);
  });
  s.addNotes("El punto que hay que dejar claro aqui es que ninguna de las cuatro reglas " +
    "mira si el activo se ve bueno. Miran si se puede medir y si se puede operar. Esa " +
    "separacion es la que hace que el ranking despues signifique algo.");
}

// ================================================================ UNIVERSO 2
{
  const s = laminaContenido("Lo que la política prohíbe a propósito",
    "Esta es la mitad que normalmente no se escribe, y es la que más protege.");
  const filas = [
    [celda("Prohibido usar para filtrar", ENC), celda("Por qué", ENC)],
    [celda("Desempeño\nretorno, momentum, Sharpe, beta", { bold: true }),
     celda("Quitar a los perdedores antes de puntuar hace que los que quedan parezcan promedio. Es meter la respuesta dentro de la pregunta.")],
    [celda("Sector\nbalancear por industria", { bold: true }),
     celda("La nota es relativa al grupo: la composición define contra quién se mide cada nombre. Equilibrarla a mano decide el resultado por la puerta de atrás.")],
    [celda("Tamaño\ncapitalización de mercado", { bold: true }),
     celda("Más allá del piso de liquidez que ya impone la regla 4, filtrar por tamaño es una apuesta de factor disfrazada de higiene.")],
  ];
  s.addTable(filas, {
    x: 0.5, y: 2.1, w: 12.33, colW: [3.9, 8.43],
    border: { type: "solid", color: BLANCO, pt: 1 },
    fill: { color: FONDO }, rowH: 0.75, valign: "middle",
    margin: [6, 10, 6, 10],
  });
  destacado(s, 0.5, 5.15, 12.33, 0.9,
    "La regla detrás de las tres: el universo se define por lo que se puede medir y operar, " +
    "nunca por lo que se espera que rinda. Qué tan bueno se ve un activo es trabajo del " +
    "ranking, y el ranking viene después.", "verde");
  s.addNotes("Si alguien pregunta por que no filtramos por sector para tener un universo " +
    "balanceado: porque el puntaje es transversal. Si dejo diez tecnologicas y dos " +
    "electricas, cada tecnologica compite contra nueve pares y cada electrica contra una. " +
    "Cambiar esa mezcla cambia el ranking sin que ningun precio se haya movido.");
}

// ================================================================ UNIVERSO 3
{
  const s = laminaContenido("Cuántos activos son, en realidad",
    "Veníamos diciendo «unos 600» en todos los documentos. Nadie los había contado.");
  cifra(s, 0.5, 2.05, 2.6, "317", "acciones únicas\nS&P + Nasdaq-100 + Dow");
  cifra(s, 3.3, 2.05, 2.6, "130", "ETFs curados\namplios, sector, estilo, RF, intl, commodities");
  cifra(s, 6.1, 2.05, 2.6, "447", "candidatos", GRAFITO);
  s.addShape(pres.ShapeType.roundRect, {
    x: 9.1, y: 2.05, w: 3.73, h: 1.5, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("De las cuatro reglas, tres casi no botan a nadie — y no es un defecto: el " +
    "problema ya viene resuelto aguas arriba. La regla 1 no puede botar a nadie porque el " +
    "punto de partida es la lista de índices.", {
    x: 9.4, y: 2.2, w: 3.15, h: 1.25,
    fontFace: FUENTE, fontSize: 11, color: SUTIL, isTextBox: true, margin: 0,
  });
  s.addText("La que manda es la cuarta: negociabilidad. Y muerde distinto según el perfil.", {
    x: 0.5, y: 3.75, w: 12.33, h: 0.35,
    fontFace: FUENTE, fontSize: 14, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  const filas = [
    [celda("Perfil", ENC), celda("Volumen diario mín.", ENC), celda("Universo evaluado", ENC), celda("Qué se cae", ENC)],
    [celda("Agresivo"), celda("US$10 MM"), celda("~425–435", { bold: true, color: VERDE }), celda("casi nada")],
    [celda("Moderado"), celda("US$20 MM"), celda("~410–425", { bold: true, color: VERDE }), celda("ETFs de nicho")],
    [celda("Conservador"), celda("US$50 MM"), celda("~380–400", { bold: true, color: VERDE }), celda("+ commodities chicos, países individuales")],
    [celda("Cons. Defensivo"), celda("US$100 MM"), celda("~310–350", { bold: true, color: VERDE }), celda("+ la cola pequeña del S&P")],
  ];
  s.addTable(filas, {
    x: 0.5, y: 4.15, w: 12.33, colW: [2.4, 2.4, 2.4, 5.13],
    border: { type: "solid", color: BLANCO, pt: 1 },
    fill: { color: FONDO }, rowH: 0.38, valign: "middle",
    margin: [4, 10, 4, 10],
  });
  destacado(s, 0.5, 6.15, 12.33, 0.75,
    "Son rangos razonados, no medidos: el volumen real solo se sabe con datos en vivo. " +
    "El modelo publica el número exacto en cada corrida.", "oro");
  s.addNotes("Si preguntan de donde salen los rangos: en acciones la mediana del S&P negocia " +
    "entre 150 y 300 millones diarios, asi que veinte millones no le hace cosquillas a nadie. " +
    "En ETFs es al reves: de los cuatro mil listados en Estados Unidos, apenas doscientos " +
    "pasan de cien millones diarios. Un caso concreto que ya medimos: DBA negocia 30.6 " +
    "millones, entra en Moderado y Agresivo y se cae en los dos conservadores.");
}

// ================================================================ UNIVERSO 4
{
  const s = laminaContenido("Del universo a la cartera",
    "Conviene tener clara la diferencia entre lo que se evalúa y lo que se compra.");
  const etapas = [
    ["447", "Candidatos", "Las listas curadas", GRIS],
    ["~400", "Universo evaluado", "La política de selección,\nsobre todo la liquidez", GRIS],
    ["25", "Cesta del optimizador", "Top del ranking, con mínimo\n3 por clase de activo", VERDE],
    ["8–15", "Cartera ejecutada", "Lo que resuelve el optimizador,\ndespués del piso de 1%", VERDE],
  ];
  etapas.forEach((e, i) => {
    const x = 0.5 + i * 3.18;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.3, w: 2.85, h: 2.5, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addText(e[0], {
      x, y: 2.5, w: 2.85, h: 0.8,
      fontFace: FUENTE, fontSize: 40, bold: true, color: e[3] === VERDE ? VERDE : GRAFITO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(e[1], {
      x, y: 3.35, w: 2.85, h: 0.35,
      fontFace: FUENTE, fontSize: 14, bold: true, color: GRAFITO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(e[2], {
      x: x + 0.2, y: 3.75, w: 2.45, h: 0.9,
      fontFace: FUENTE, fontSize: 11, color: SUTIL,
      align: "center", isTextBox: true, margin: 0,
    });
    if (i < 3) {
      s.addText("▸", {
        x: x + 2.87, y: 3.25, w: 0.3, h: 0.4,
        fontFace: FUENTE, fontSize: 20, color: GRIS,
        align: "center", isTextBox: true, margin: 0,
      });
    }
  });
  destacado(s, 0.5, 5.15, 12.33, 0.9,
    "El embudo se estrecha por razones distintas en cada paso: liquidez, después puntaje, " +
    "después las bandas del mandato y lo que es ejecutable. Ninguno de los cuatro números " +
    "es una meta que alguien puso.", "verde");
  s.addNotes("Este es el cuadro que contesta la pregunta que mas se repite: si el universo " +
    "son cuatrocientos nombres, por que la cartera tiene doce. Son cuatro filtros distintos " +
    "y ninguno es una cuota.");
}

// ================================================================ BLOQUES
{
  const s = laminaContenido("Las 25 medidas viven en seis bloques",
    "Los pesos que se ven abajo son los del perfil Moderado. Cambian con el perfil.");
  const filas = [
    [celda("Bloque", ENC), celda("Peso", ENC), celda("Qué contesta", ENC)],
    [celda("Momentum y tendencia"), celda("25%", { bold: true, color: VERDE }), celda("¿Viene subiendo, y la tendencia se sostiene?")],
    [celda("Retorno ajustado por riesgo"), celda("21%", { bold: true, color: VERDE }), celda("¿El retorno compensó lo que se sufrió?")],
    [celda("Volatilidad y caídas"), celda("17%", { bold: true, color: VERDE }), celda("¿Cuánto duele tenerlo? Es el freno del momentum")],
    [celda("Sensibilidad al mercado"), celda("14%", { bold: true, color: VERDE }), celda("¿Lo que rinde es propio o es solo mercado?")],
    [celda("Valuación y carry"), celda("12%", { bold: true, color: VERDE }), celda("Proxies de mercado. Es el bloque más débil")],
    [celda("Liquidez y capacidad"), celda("11%", { bold: true, color: VERDE }), celda("¿Se puede entrar y salir del tamaño que quiero?")],
  ];
  s.addTable(filas, {
    x: 0.5, y: 2.1, w: 12.33, colW: [3.9, 1.2, 7.23],
    border: { type: "solid", color: BLANCO, pt: 1 },
    fill: { color: FONDO }, rowH: 0.42, valign: "middle",
    margin: [4, 10, 4, 10],
  });
  destacado(s, 0.5, 5.3, 12.33, 0.85,
    "Si falta un dato, el peso se reparte entre las medidas que sí están. Nunca se rellena " +
    "con cero: un cero afirma «este nombre es promedio», que es un dato que nadie midió.", "oro");
}

// ================================================================ Z-SCORE
{
  const s = laminaContenido("La idea central: nada se juzga solo",
    "Si mañana explican una sola cosa técnica, que sea esta.");
  s.addText("Un Sharpe de 1.2 no es bueno ni malo en abstracto. Depende de contra qué se " +
    "compare. Así que el modelo convierte cada medida cruda en una nota relativa al resto " +
    "del universo ese día.", {
    x: 0.5, y: 2.05, w: 7.6, h: 1.1,
    fontFace: FUENTE, fontSize: 15, color: GRAFITO, isTextBox: true, margin: 0,
  });
  const esc = [["−2", "muy por debajo", SUTIL], ["0", "el promedio", GRAFITO], ["+2", "muy por encima", VERDE]];
  esc.forEach((e, i) => {
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.7 + i * 2.5, y: 3.3, w: 1.15, h: 1.15,
      fill: { color: i === 2 ? VERDE : (i === 1 ? GRIS : FONDO) },
      line: { color: i === 2 ? VERDE : GRIS },
    });
    s.addText(e[0], {
      x: 0.7 + i * 2.5, y: 3.6, w: 1.15, h: 0.6,
      fontFace: FUENTE, fontSize: 24, bold: true,
      color: i === 2 ? BLANCO : GRAFITO, align: "center", isTextBox: true, margin: 0,
    });
    s.addText(e[1], {
      x: 0.35 + i * 2.5, y: 4.55, w: 1.85, h: 0.35,
      fontFace: FUENTE, fontSize: 11.5, color: SUTIL, align: "center", isTextBox: true, margin: 0,
    });
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.4, y: 2.05, w: 4.43, h: 3.9, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("Antes de calcular la nota", {
    x: 8.75, y: 2.35, w: 3.75, h: 0.4,
    fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText([
    { text: "Se recortan los extremos: el 5% de arriba y el 5% de abajo.", options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Un solo nombre con +300% comprimiría a todos los demás en un pañuelo y las diferencias reales entre ellos desaparecerían.", options: { bullet: true, breakLine: true, paraSpaceAfter: 8 } },
    { text: "Volatilidad, drawdown y beta se puntúan por separado entre ETFs y acciones: un ETF es estructuralmente menos volátil, y compararlos juntos le regalaría la nota a todos los ETFs.", options: { bullet: true } },
  ], {
    x: 8.75, y: 2.85, w: 3.75, h: 2.9,
    fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
  });
}

// ================================================================ CORRECCIÓN
{
  const s = laminaContenido("Y ahí estaba escondido un problema",
    "Lo encontramos auditando el modelo contra su propia documentación.");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.05, w: 12.33, h: 1.15, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("La documentación decía: «si el universo entero está mediocre no habrá muchos " +
    "Sobreponderar, porque los umbrales van sobre el z y no sobre el percentil».", {
    x: 0.9, y: 2.25, w: 11.5, h: 0.8,
    fontFace: FUENTE, fontSize: 14.5, italic: true, color: SUTIL, isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 3.4, 12.33, 0.75, "Eso es falso. Y falso en la dirección que favorece al modelo.", "verde");
  s.addText("Calcular la nota relativa es restar el promedio del grupo y dividir entre su " +
    "dispersión. Después de esa cuenta la distribución es siempre la misma: un umbral de " +
    "+0.50 nombra el tercio superior del grupo tanto si todos subieron 40% como si todos " +
    "cayeron 40%. Usar percentiles habría dado exactamente lo mismo.", {
    x: 0.5, y: 4.4, w: 12.33, h: 1.1,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("El sistema siempre dice cuáles son los mejores del grupo. Si los mejores del " +
    "grupo valen la pena es otra pregunta, y hasta ahora nadie la estaba haciendo.", {
    x: 0.5, y: 5.5, w: 12.33, h: 0.6,
    fontFace: FUENTE, fontSize: 14, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addNotes("Este es el punto donde conviene invitar preguntas. Es un error conceptual, " +
    "no un bug, y estaba escrito tanto en el código como en el documento.");
}

// ================================================================ PISO CALIDAD
{
  const s = laminaContenido("La solución: un piso absoluto para comprar",
    "Momentum de 12 meses positivo y Sharpe sobre cero. O sea, que le haya ganado al efectivo.");
  s.addChart(pres.ChartType.bar, [
    { name: "Sin el piso", labels: ["Alcista +20%", "Plano 0%", "Bajista −10%"], values: [6, 5, 3] },
    { name: "Con el piso", labels: ["Alcista +20%", "Plano 0%", "Bajista −10%"], values: [6, 0, 0] },
  ], {
    x: 0.5, y: 2.05, w: 7.4, h: 3.6,
    barDir: "col", chartColors: [GRIS, VERDE],
    showTitle: true, title: "Nombres en Sobreponderar, mismo universo",
    titleFontFace: FUENTE, titleFontSize: 13, titleColor: GRAFITO,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: FUENTE,
    dataLabelFontSize: 12, dataLabelColor: GRAFITO,
    catAxisLabelColor: GRAFITO, catAxisLabelFontFace: FUENTE, catAxisLabelFontSize: 11,
    valAxisLabelColor: SUTIL, valAxisLabelFontFace: FUENTE, valAxisLabelFontSize: 10,
    valGridLine: { color: GRIS, size: 1 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontFace: FUENTE, legendFontSize: 11,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.2, y: 2.05, w: 4.63, h: 3.6, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("El caso plano era el hueco", {
    x: 8.55, y: 2.35, w: 3.95, h: 0.4,
    fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("El freno de «cuchillo cayendo» exige media móvil rota Y momentum negativo, " +
    "así que no hace nada cuando el mercado va de lado.\n\nPero con la tasa libre de riesgo " +
    "en 4.25%, un año plano es Sharpe negativo. Sobreponderar algo que perdió contra letras " +
    "es exactamente lo que este piso evita.\n\nLa columna alcista es el control: si el freno " +
    "también mordiera cuando el universo sí es bueno, sería un error y no una protección.", {
    x: 8.55, y: 2.85, w: 3.95, h: 2.65,
    fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
  });
}

// ================================================================ FRENOS
{
  const s = laminaContenido("Los frenos de riesgo", "Se aplican después de puntuar y solo pueden bajar una recomendación, nunca subirla.");
  const frenos = [
    ["Calidad absoluta", "Sin momentum positivo y Sharpe sobre cero, no hay Sobreponderar."],
    ["Cuchillo cayendo", "Bajo la media de largo plazo y con momentum negativo, por barato que se vea."],
    ["Amplificador", "Caída histórica profunda combinada con beta alta."],
    ["Capacidad", "Si no se puede salir en el plazo definido, no es una idea accionable."],
    ["Techo de volatilidad", "Arriba de cierto nivel las estadísticas de 52 semanas dejan de ser confiables."],
    ["Redundancia", "IWM y VTWO son el mismo índice. Solo el mejor puntuado conserva el OW."],
  ];
  frenos.forEach((f, i) => {
    const x = 0.5 + (i % 3) * 4.19;
    const y = 2.1 + Math.floor(i / 3) * 1.75;
    s.addShape(pres.ShapeType.ellipse, {
      x, y: y + 0.05, w: 0.4, h: 0.4, fill: { color: ORO }, line: { color: ORO },
    });
    s.addText(f[0], {
      x: x + 0.55, y: y + 0.05, w: 3.4, h: 0.4,
      fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(f[1], {
      x: x + 0.55, y: y + 0.5, w: 3.35, h: 1.0,
      fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 5.65, 12.33, 0.9,
    "Un nombre puede activar varios a la vez. El castigo a la convicción se aplica una vez " +
    "por nombre, no una por freno: nada se cobra doble.", "verde");
}

// ================================================ SECCIÓN: BLACK-LITTERMAN
{
  const s = laminaSeccion("3", "Black-Litterman",
    "Qué problema resuelve, cómo funciona por dentro, y cómo le conectamos el screener.");
  s.addNotes("Diez láminas. Es la parte que más cuesta explicar y la que más se usa mal " +
    "en la industria, así que vale la pena la pausa.");
}

// ================================================================ MARKOWITZ
{
  const s = laminaContenido("Antes de BL: por qué Markowitz solo no alcanza",
    "La teoría es de 1952 y es correcta. El problema es lo que pasa cuando la usas con datos reales.");
  s.addText("Markowitz te pide retornos esperados y te devuelve pesos. Suena razonable hasta " +
    "que lo corres:", {
    x: 0.5, y: 2.05, w: 12.33, h: 0.45,
    fontFace: FUENTE, fontSize: 15, color: GRAFITO, isTextBox: true, margin: 0,
  });
  const probs = [
    ["Nadie sabe los retornos esperados", "Es el dato más difícil de estimar en finanzas, y el modelo lo pide como si fuera conocido."],
    ["Es hipersensible", "Mueves un pronóstico medio punto y la cartera cambia por completo. No es estabilidad, es azar."],
    ["Concentra", "Sin restricciones te pone 40% en un nombre porque su retorno estimado salió un pelo más alto."],
  ];
  probs.forEach((p, i) => {
    const x = 0.5 + i * 4.19;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.65, w: 3.95, h: 2.0, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addText(p[0], {
      x: x + 0.3, y: 2.95, w: 3.35, h: 0.7,
      fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(p[1], {
      x: x + 0.3, y: 3.65, w: 3.35, h: 0.85,
      fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 5.0, 12.33, 1.1,
    "El resultado práctico: carteras que ningún comité aprobaría, y que cambian de forma " +
    "cada vez que actualizas los datos. Por eso en la industria se optimiza «con restricciones» " +
    "— que es una manera elegante de decir que el optimizador se desactiva a mano.", "oro");
}

// ================================================================ LA IDEA BL
{
  const s = laminaContenido("La idea de Black y Litterman",
    "Goldman Sachs, 1990. Dan vuelta la pregunta.");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.1, w: 5.98, h: 1.5, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("La pregunta de siempre", {
    x: 0.85, y: 2.3, w: 5.3, h: 0.35,
    fontFace: FUENTE, fontSize: 12, color: SUTIL, isTextBox: true, margin: 0,
  });
  s.addText("«¿Qué va a rendir cada activo?»", {
    x: 0.85, y: 2.7, w: 5.3, h: 0.65,
    fontFace: FUENTE, fontSize: 19, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.85, y: 2.1, w: 5.98, h: 1.5, rectRadius: 0.06,
    fill: { color: VERDE }, line: { color: VERDE },
  });
  s.addText("La pregunta de Black-Litterman", {
    x: 7.2, y: 2.3, w: 5.3, h: 0.35,
    fontFace: FUENTE, fontSize: 12, color: BLANCO, isTextBox: true, margin: 0,
  });
  s.addText("«¿Qué tendría que rendir cada activo\npara que la cartera neutral sea la óptima?»", {
    x: 7.2, y: 2.66, w: 5.3, h: 0.85,
    fontFace: FUENTE, fontSize: 15, bold: true, color: BLANCO, isTextBox: true, margin: 0,
  });

  s.addText("Eso se llama optimización inversa, y cambia todo. Ya no tienes que pronosticar " +
    "veinte retornos. Partes de una cartera neutral que por definición es razonable, y solo " +
    "declaras aquello en lo que de verdad tienes una opinión. El modelo mezcla las dos cosas " +
    "en proporción a qué tan seguro estás.", {
    x: 0.5, y: 3.9, w: 12.33, h: 1.1,
    fontFace: FUENTE, fontSize: 15, color: GRAFITO, isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 5.1, 12.33, 1.0,
    "El cambio de fondo: pasas de «adivinar veinte números» a «declarar tres o cuatro " +
    "opiniones y decir qué tan convencido estás de cada una». Eso sí es algo que un comité " +
    "puede discutir y aprobar.", "verde");
  s.addNotes("Fischer Black y Robert Litterman, Goldman Sachs. El paper es de 1990-1992. " +
    "Nació justamente porque la mesa de Goldman no podía usar Markowitz en producción.");
}

// ================================================================ ANALOGÍA
{
  const s = laminaContenido("La analogía que mejor funciona", "Black-Litterman es un GPS al que le puedes hablar.");
  const pasos = [
    ["El GPS traza una ruta", "Con todo lo que sabe del tráfico. No es una ruta tonta: es la mejor con la información pública. Ese es el equilibrio de mercado."],
    ["Tú sabes algo más", "«Esa calle está cerrada». No redibujas el mapa entero: aportas un dato puntual. Esa es una view."],
    ["Y qué tan seguro estás", "No es lo mismo «acabo de pasar por ahí» que «me dijeron». El GPS debería tratarlos distinto. Esa es la convicción."],
    ["El GPS recalcula", "No ignora lo que sabía ni tampoco te obedece ciegamente. Mezcla. Eso es la parte bayesiana."],
  ];
  pasos.forEach((p, i) => {
    const y = 2.05 + i * 1.02;
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.5, y: y + 0.08, w: 0.62, h: 0.62,
      fill: { color: i === 3 ? GRAFITO : VERDE }, line: { color: i === 3 ? GRAFITO : VERDE },
    });
    s.addText(String(i + 1), {
      x: 0.5, y: y + 0.22, w: 0.62, h: 0.4,
      fontFace: FUENTE, fontSize: 17, bold: true, color: BLANCO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(p[0], {
      x: 1.3, y: y + 0.05, w: 3.3, h: 0.4,
      fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(p[1], {
      x: 4.7, y: y + 0.05, w: 8.1, h: 0.75,
      fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 6.15, 12.33, 0.65,
    "Si no dices nada, el GPS te lleva por su ruta. Si no hay views, la cartera es la neutral del mandato.", "oro");
}

// ================================================================ PASO 1
{
  const s = laminaContenido("Paso 1 — El punto de partida",
    "De dónde salen los retornos «que el mercado ya está asumiendo».");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.05, w: 12.33, h: 1.0, rectRadius: 0.06,
    fill: { color: GRAFITO }, line: { color: GRAFITO },
  });
  s.addText("π  =  λ  ×  Σ  ×  w", {
    x: 0.5, y: 2.2, w: 12.33, h: 0.7,
    fontFace: FUENTE, fontSize: 30, bold: true, color: VERDE,
    align: "center", isTextBox: true, margin: 0,
  });
  const term = [
    ["π", "Retorno implícito", "Lo que sale: qué tendría que rendir cada activo."],
    ["λ", "Aversión al riesgo", "Cuánto castiga el riesgo. En el modelo, 2.5."],
    ["Σ", "Matriz de riesgo", "Volatilidades y correlaciones entre los activos."],
    ["w", "Cartera de partida", "La cartera neutral. Aquí está toda la decisión."],
  ];
  term.forEach((t, i) => {
    const x = 0.5 + i * 3.13;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 3.35, w: 2.93, h: 1.75, rectRadius: 0.06,
      fill: { color: i === 3 ? ORO : FONDO }, line: { color: i === 3 ? ORO : FONDO },
    });
    s.addText(t[0], {
      x: x + 0.25, y: 3.5, w: 2.4, h: 0.5,
      fontFace: FUENTE, fontSize: 24, bold: true, color: i === 3 ? GRAFITO : VERDE,
      isTextBox: true, margin: 0,
    });
    s.addText(t[1], {
      x: x + 0.25, y: 4.02, w: 2.45, h: 0.35,
      fontFace: FUENTE, fontSize: 13, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(t[2], {
      x: x + 0.25, y: 4.38, w: 2.45, h: 0.65,
      fontFace: FUENTE, fontSize: 11, color: i === 3 ? GRAFITO : SUTIL, isTextBox: true, margin: 0,
    });
  });
  s.addText("Es una multiplicación, no una estimación. Lo que le pases como cartera de partida " +
    "ES la cartera neutral del modelo. Volveremos sobre esto: resultó ser la decisión más " +
    "grande de toda la asignación, y estaba mal puesta.", {
    x: 0.5, y: 5.3, w: 12.33, h: 0.9,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });
}

// ================================================================ PASO 2
{
  const s = laminaContenido("Paso 2 — Las views", "Una view es una opinión con dos números, y son independientes entre sí.");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.05, w: 5.98, h: 1.6, rectRadius: 0.06,
    fill: { color: VERDE }, line: { color: VERDE },
  });
  s.addText("Q", {
    x: 0.85, y: 2.2, w: 1, h: 0.55,
    fontFace: FUENTE, fontSize: 30, bold: true, color: BLANCO, isTextBox: true, margin: 0,
  });
  s.addText("Cuánto crees que va a rendir de más o de menos que lo que el mercado asume.", {
    x: 1.75, y: 2.3, w: 4.4, h: 1.1,
    fontFace: FUENTE, fontSize: 13.5, color: BLANCO, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.85, y: 2.05, w: 5.98, h: 1.6, rectRadius: 0.06,
    fill: { color: GRAFITO }, line: { color: GRAFITO },
  });
  s.addText("Ω", {
    x: 7.2, y: 2.2, w: 1, h: 0.55,
    fontFace: FUENTE, fontSize: 30, bold: true, color: ORO, isTextBox: true, margin: 0,
  });
  s.addText("Qué tanto le crees a esa opinión. Alimenta la matriz de incertidumbre de la view.", {
    x: 8.1, y: 2.3, w: 4.4, h: 1.1,
    fontFace: FUENTE, fontSize: 13.5, color: BLANCO, isTextBox: true, margin: 0,
  });

  s.addText("Puedes tener una opinión fuerte con poca confianza, o una modesta con mucha. " +
    "El modelo las trata como cosas distintas porque lo son.", {
    x: 0.5, y: 3.85, w: 12.33, h: 0.5,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 4.5, w: 5.98, h: 1.65, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("View absoluta", {
    x: 0.85, y: 4.7, w: 5.3, h: 0.35,
    fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("«SPY va a rendir −3.03%». Sobre un instrumento solo.", {
    x: 0.85, y: 5.08, w: 5.3, h: 0.9,
    fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.85, y: 4.5, w: 5.98, h: 1.65, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("View relativa", {
    x: 7.2, y: 4.7, w: 5.3, h: 0.35,
    fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("«NVDA va a rendir 1.83% más que QQQ». El Q se calcula con la volatilidad del " +
    "spread, no la de una pata sola.", {
    x: 7.2, y: 5.08, w: 5.3, h: 0.9,
    fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
  });
}

// ================================================================ PASO 3
{
  const s = laminaContenido("Paso 3 — La mezcla",
    "Aquí está la parte bayesiana, y es más simple de lo que suena.");
  s.addText("Imagina dos personas dándote un número. Una está muy segura, la otra dice « no sé, " +
    "creo que…». Tú no promedias los dos por igual: le das más peso al que está más seguro. " +
    "Eso es literalmente lo que hace la fórmula.", {
    x: 0.5, y: 2.05, w: 12.33, h: 0.85,
    fontFace: FUENTE, fontSize: 15, color: GRAFITO, isTextBox: true, margin: 0,
  });
  const lados = [
    ["El equilibrio", "Lo que el mercado asume", VERDE,
      "Su peso depende de τ, que fija cuánta incertidumbre le damos al punto de partida. En el modelo, 0.025 — o sea, le creemos bastante."],
    ["Tus views", "Donde no estás de acuerdo", GRAFITO,
      "Su peso depende de Ω, que sale de la convicción. Convicción alta = varianza baja = mueve más el resultado."],
  ];
  lados.forEach((l, i) => {
    const x = 0.5 + i * 6.35;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 3.05, w: 5.98, h: 1.9, rectRadius: 0.06,
      fill: { color: l[2] }, line: { color: l[2] },
    });
    s.addText(l[0], {
      x: x + 0.35, y: 3.25, w: 5.3, h: 0.4,
      fontFace: FUENTE, fontSize: 18, bold: true, color: BLANCO, isTextBox: true, margin: 0,
    });
    s.addText(l[1], {
      x: x + 0.35, y: 3.65, w: 5.3, h: 0.3,
      fontFace: FUENTE, fontSize: 12, color: i === 0 ? BLANCO : GRIS, isTextBox: true, margin: 0,
    });
    s.addText(l[3], {
      x: x + 0.35, y: 4.02, w: 5.3, h: 0.85,
      fontFace: FUENTE, fontSize: 11.5, color: i === 0 ? BLANCO : GRIS, isTextBox: true, margin: 0,
    });
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 6.28, y: 3.72, w: 0.78, h: 0.78, fill: { color: ORO }, line: { color: ORO },
  });
  s.addText("+", {
    x: 6.28, y: 3.83, w: 0.78, h: 0.55,
    fontFace: FUENTE, fontSize: 26, bold: true, color: GRAFITO,
    align: "center", isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 5.15, 12.33, 1.0,
    "El resultado no es ni el equilibrio ni tus views: es el punto intermedio que corresponde " +
    "a cuánta confianza declaraste. Si no dices nada, queda el equilibrio intacto. Si dices " +
    "algo con convicción máxima, se mueve mucho — pero nunca ignora del todo el punto de partida.", "verde");
}

// ================================================================ Q
{
  const s = laminaContenido("Cómo conectamos el screener con las views",
    "El puntaje es un ranking, no un pronóstico. Hay que traducir, y la traducción es explícita.");
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 2.05, w: 12.33, h: 0.95, rectRadius: 0.06,
    fill: { color: GRAFITO }, line: { color: GRAFITO },
  });
  s.addText("Q  =  0.08  ×  (z del nombre)  ×  (su volatilidad)", {
    x: 0.5, y: 2.2, w: 12.33, h: 0.65,
    fontFace: FUENTE, fontSize: 25, bold: true, color: VERDE,
    align: "center", isTextBox: true, margin: 0,
  });
  const props = [
    ["Está escalada por riesgo", "A igual posición en el ranking, un nombre más volátil recibe una expectativa mayor. Suena raro, pero es lo que necesita el optimizador: como divide retorno entre riesgo, una fórmula plana lo llevaría a cargar siempre en lo de baja volatilidad."],
    ["Está centrada", "Un nombre en la mitad del ranking recibe Q = 0 exacto. Ni positivo ni negativo. Esto es justo lo que estaba roto en el sistema anterior, donde un activo neutral recibía −1.40% por un problema de escalas entre señales."],
  ];
  props.forEach((p, i) => {
    const x = 0.5 + i * 6.35;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 3.2, w: 5.98, h: 1.95, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addText(p[0], {
      x: x + 0.35, y: 3.4, w: 5.3, h: 0.4,
      fontFace: FUENTE, fontSize: 16, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(p[1], {
      x: x + 0.35, y: 3.82, w: 5.3, h: 1.25,
      fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 5.35, 12.33, 1.0,
    "El 0.08 es un supuesto nuestro, no una medición. Es cuánta correlación asumimos entre " +
    "el ranking y los retornos que efectivamente ocurren. Lo pusimos bajo a propósito, y está " +
    "declarado como supuesto en el archivo, en la justificación y en el Excel.", "oro");
  s.addNotes("Si preguntan cómo se calibra de verdad: hay que guardar las recomendaciones " +
    "varios meses y compararlas contra lo que pasó. No lo tenemos todavía y no lo escondemos.");
}

// ================================================================ CONVICCIÓN
{
  const s = laminaContenido("La convicción no mide la fuerza de la señal",
    "Eso ya está en Q. La convicción mide cuánta confianza tenemos en la estimación.");
  const fact = [
    ["Fuerza", "Qué tan lejos del promedio está. Satura: un z de 3 no vale el triple que uno de 1, porque ambos ya dicen «claramente en un extremo»."],
    ["Acuerdo", "Cuántos de los seis bloques apuntan en la misma dirección. Un +1.5 con los seis de acuerdo no es lo mismo que un +1.5 que carga un bloque y cinco contradicen."],
    ["Cobertura", "Qué proporción de las medidas estaba realmente disponible para ese nombre."],
  ];
  fact.forEach((f, i) => {
    const x = 0.5 + i * 4.19;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.1, w: 3.95, h: 2.35, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.35, y: 2.4, w: 0.5, h: 0.5, fill: { color: VERDE }, line: { color: VERDE },
    });
    s.addText(f[0], {
      x: x + 1.0, y: 2.45, w: 2.7, h: 0.4,
      fontFace: FUENTE, fontSize: 16, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(f[1], {
      x: x + 0.35, y: 3.05, w: 3.3, h: 1.25,
      fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 4.65, w: 12.33, h: 1.5, rectRadius: 0.06,
    fill: { color: GRAFITO }, line: { color: GRAFITO },
  });
  s.addText("Y un castigo", {
    x: 0.9, y: 4.85, w: 4, h: 0.4,
    fontFace: FUENTE, fontSize: 17, bold: true, color: ORO, isTextBox: true, margin: 0,
  });
  s.addText("Si se activó cualquier freno de riesgo, la convicción se corta a la mitad. Ese castigo " +
    "siempre muerde, incluso en los nombres más fuertes — que son precisamente aquellos para " +
    "los que existen los frenos. La versión anterior recortaba al final, y los nombres muy " +
    "fuertes saturaban el techo antes de que el castigo llegara a aplicarse.", {
    x: 0.9, y: 5.28, w: 11.5, h: 0.8,
    fontFace: FUENTE, fontSize: 12.5, color: BLANCO, isTextBox: true, margin: 0,
  });
}

// ================================================================ EJEMPLO JPM
{
  const s = laminaContenido("Un ejemplo completo, con números reales", "Perfil Moderado. El mejor puntuado de la corrida fue JPM.");
  cifra(s, 0.5, 2.05, 2.6, "100", "puntaje de 0 a 100", VERDE);
  cifra(s, 3.2, 2.05, 2.6, "+1.53", "z compuesto", VERDE);
  cifra(s, 5.9, 2.05, 2.6, "21.5%", "volatilidad anual", GRAFITO);
  cifra(s, 8.6, 2.05, 2.6, "+2.63%", "Q resultante", GRAFITO);

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 3.75, w: 5.98, h: 1.05, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("Q = 0.08 × 1.53 × 0.215 = +2.63%", {
    x: 0.85, y: 4.0, w: 5.3, h: 0.5,
    fontFace: FUENTE, fontSize: 17, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 6.85, y: 3.75, w: 5.98, h: 1.05, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("Convicción = 0.85 × 0.77 × 0.92 × 0.95 = 0.57", {
    x: 7.2, y: 4.0, w: 5.4, h: 0.5,
    fontFace: FUENTE, fontSize: 16, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });

  s.addText("Por qué la convicción es 0.57 y no más alta: cinco bloques a favor y uno claramente " +
    "en contra. Valuación y carry salió en −1.12 mientras retorno ajustado por riesgo salió " +
    "en +1.59. La justificación que acompaña cada view escribe esto en texto, para que el " +
    "gestor vea de inmediato qué la sostiene y qué la debilita antes de aprobarla.", {
    x: 0.5, y: 5.0, w: 12.33, h: 1.15,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });
}

// ================================================================ PASO 4
{
  const s = laminaContenido("Paso 4 — La cartera, con el Procedimiento encima",
    "Maximiza retorno esperado menos una penalización por riesgo, sujeto a todo lo que manda el mandato.");
  const lim = [
    "Bandas mínimas y máximas por clase de activo",
    "Máximo de renta variable total",
    "Tope por emisor individual",
    "Exclusiones duras del Art. 170 RIV",
    "Prohibición de posiciones cortas",
    "Presupuesto de apalancamiento con el buffer de 95%",
  ];
  lim.forEach((l, i) => {
    const x = 0.5 + (i % 2) * 6.35;
    const y = 2.1 + Math.floor(i / 2) * 0.62;
    s.addShape(pres.ShapeType.ellipse, {
      x, y: y + 0.08, w: 0.28, h: 0.28, fill: { color: VERDE }, line: { color: VERDE },
    });
    s.addText(l, {
      x: x + 0.45, y, w: 5.6, h: 0.45,
      fontFace: FUENTE, fontSize: 13, color: GRAFITO, isTextBox: true, margin: 0,
    });
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 4.15, w: 12.33, h: 1.05, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("La cesta no puede ser el top-25 por puntaje. El ranking premia momentum y " +
    "riesgo-retorno, donde la renta variable domina, y con una cesta 100% equity el techo del " +
    "mandato queda por debajo del libro invertido: no hay solución y la hoja sale vacía. " +
    "Ahora la cesta garantiza representantes de cada clase disponible.", {
    x: 0.85, y: 4.32, w: 11.6, h: 0.8,
    fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 5.4, 12.33, 0.95,
    "Después de optimizar, el sistema compara la cartera contra cada límite y reporta lo que " +
    "se rompa. El código anterior escribía «Auditoría OK» sin comparar nada — una auditoría " +
    "que no puede fallar es peor que no tener ninguna.", "verde");
}

// ================================================ SECCIÓN: LA AUDITORÍA
{
  const s = laminaSeccion("4", "Qué encontramos al auditarlo",
    "Una revisión externa, más lo que encontró el propio sistema midiéndose a sí mismo.");
}

// ================================================================ ANCLA
{
  const s = laminaContenido("El hallazgo más grande: el punto de partida",
    "Con ocho views sobre veintitantos activos, cerca de tres cuartas partes de los pesos salen del ancla.");
  s.addChart(pres.ChartType.bar, [
    { name: "Techo del mandato", labels: ["Conservador", "Moderado", "Agresivo"], values: [50, 60, 80] },
    { name: "Con ancla de capitalización", labels: ["Conservador", "Moderado", "Agresivo"], values: [50.0, 59.4, 62.8] },
    { name: "Con ancla de política", labels: ["Conservador", "Moderado", "Agresivo"], values: [18.4, 27.4, 33.6] },
  ], {
    x: 0.5, y: 2.05, w: 7.4, h: 3.5,
    barDir: "col", chartColors: [GRIS, GRAFITO, VERDE],
    showTitle: true, title: "Renta variable resuelta, en % del libro",
    titleFontFace: FUENTE, titleFontSize: 13, titleColor: GRAFITO,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontFace: FUENTE,
    dataLabelFontSize: 11, dataLabelColor: GRAFITO,
    catAxisLabelColor: GRAFITO, catAxisLabelFontFace: FUENTE, catAxisLabelFontSize: 11,
    valAxisLabelColor: SUTIL, valAxisLabelFontFace: FUENTE, valAxisLabelFontSize: 10,
    valGridLine: { color: GRIS, size: 1 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontFace: FUENTE, legendFontSize: 11,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 8.2, y: 2.05, w: 4.63, h: 3.5, rectRadius: 0.06,
    fill: { color: FONDO }, line: { color: FONDO },
  });
  s.addText("Pegado al techo, exactamente", {
    x: 8.55, y: 2.3, w: 3.95, h: 0.4,
    fontFace: FUENTE, fontSize: 15, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addText("El sistema anterior normalizaba capitalización de acciones contra patrimonio de " +
    "ETFs. No son la misma unidad, y esa cuenta ancla cerca de 95% en renta variable — cuando " +
    "ningún mandato de la casa pasa de 80%.\n\nEl resultado: la cartera quedaba clavada en el " +
    "límite, o a un pelo de él. La banda regulatoria estaba haciendo la asignación de activos, " +
    "y el modelo solo repartía por dentro.", {
    x: 8.55, y: 2.78, w: 3.95, h: 2.55,
    fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 5.7, 12.33, 0.85,
    "Con el ancla nueva, Conservador resuelve en 18.4% contra un techo de 50%, y Agresivo en 33.6% contra 80%. El límite dejó de decidir.", "verde");
  s.addNotes("Estos numeros salen de una cesta que abarca las siete clases de activo, que es " +
    "el caso realista. Con una cesta cargada a renta variable el efecto es aun mas fuerte: ahi " +
    "el ancla de capitalizacion clava la solucion en el techo exacto de las tres estrategias. " +
    "Si preguntan por que Agresivo no llega al techo ni siquiera con el ancla vieja: porque a " +
    "80% de renta variable la penalizacion por riesgo ya pesa mas que el retorno esperado.");
}

// ================================================================ ANCLA 2
{
  const s = laminaContenido("Ahora el punto de partida es el propio mandato",
    "Cada clase en el punto medio de su banda; dentro de cada clase, por capitalización.");
  const pasos = [
    ["Entre clases: política", "El punto medio de cada banda, repartido sobre las clases que realmente están en la cesta, con el techo de renta variable aplicado al ancla misma."],
    ["Dentro de la clase: mercado", "Ahí sí comparar valores de mercado tiene sentido, porque estás comparando acciones contra acciones."],
    ["La propiedad que lo valida", "Sin views, el optimizador devuelve exactamente esa cartera. Verificado en las cuatro estrategias. Cuando el modelo no tiene nada que decir, el resultado es la asignación del mandato."],
  ];
  pasos.forEach((p, i) => {
    const y = 2.1 + i * 1.25;
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.5, y: y + 0.1, w: 0.55, h: 0.55,
      fill: { color: i === 2 ? ORO : VERDE }, line: { color: i === 2 ? ORO : VERDE },
    });
    s.addText(String(i + 1), {
      x: 0.5, y: y + 0.22, w: 0.55, h: 0.35,
      fontFace: FUENTE, fontSize: 15, bold: true, color: i === 2 ? GRAFITO : BLANCO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(p[0], {
      x: 1.25, y: y + 0.05, w: 3.6, h: 0.4,
      fontFace: FUENTE, fontSize: 15.5, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(p[1], {
      x: 4.95, y: y + 0.05, w: 7.85, h: 1.0,
      fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 5.95, 12.33, 0.9,
    "Aviso importante: el punto medio de una banda NO es una asignación estratégica. Eso lo " +
    "decide el Comité, y nuestros documentos dan bandas, no objetivos. El código acepta los " +
    "números reales el día que existan.", "oro");
  s.addNotes("Este es uno de los tres puntos donde hace falta una decisión del comité. " +
    "No es un pendiente técnico: es una decisión de política que nosotros no podemos tomar.");
}

// ================================================================ ALFA
{
  const s = laminaContenido("El alfa que en realidad era retorno",
    "El bloque de sensibilidad al mercado estaba pagando dos veces por el mismo rendimiento.");
  s.addText("Cuando la beta contra el índice es prácticamente cero, el alfa de Jensen deja de " +
    "ser «retorno que el mercado no explica» y pasa a ser simplemente el retorno del activo — " +
    "que el bloque de momentum ya está puntuando.", {
    x: 0.5, y: 2.05, w: 12.33, h: 0.8,
    fontFace: FUENTE, fontSize: 15, color: GRAFITO, isTextBox: true, margin: 0,
  });
  const filas = [
    [celda("Nombre", ENC), celda("R² contra el índice", ENC), celda("«Alfa» acreditado", ENC), celda("Retorno real del año", ENC)],
    [celda("LLY", { bold: true }), celda("0.005"), celda("+70%", { color: VERDE, bold: true }), celda("+73%", { bold: true })],
    [celda("XLV", { bold: true }), celda("0.021"), celda("+16.5%", { color: VERDE, bold: true }), celda("+23.3%", { bold: true })],
    [celda("TLT", { bold: true }), celda("0.033"), celda("−10.2%"), celda("−4.9%", { bold: true })],
  ];
  s.addTable(filas, {
    x: 0.5, y: 3.05, w: 12.33, colW: [2.5, 3.2, 3.3, 3.33],
    border: { type: "solid", color: BLANCO, pt: 1 },
    fill: { color: FONDO }, rowH: 0.44, valign: "middle", margin: [4, 10, 4, 10],
  });
  s.addText("La revisión externa proponía excluir renta fija y materias primas. Medido, la clase " +
    "de activo resultó ser el criterio equivocado: LLY y XLV son acciones, y el oro se explica " +
    "mejor por el índice (0.187) que Apple (0.178). Una lista por clase habría dejado adentro " +
    "el peor caso y botado uno sano.", {
    x: 0.5, y: 5.0, w: 12.33, h: 0.95,
    fontFace: FUENTE, fontSize: 13.5, color: SUTIL, isTextBox: true, margin: 0,
  });
  destacado(s, 0.5, 6.05, 12.33, 0.75,
    "Ahora el criterio es estadístico: si la beta no es significativa, se omite el bloque entero " +
    "y el puntaje se reparte entre los demás.", "verde");
}

// ================================================================ POSICIÓN MÍNIMA
{
  const s = laminaContenido("Posiciones demasiado chicas para operar",
    "Un optimizador no tiene ninguna noción de qué vale la pena ejecutar.");
  const filas = [
    [celda("", ENC), celda("Posiciones", ENC), celda("Exposición bruta", ENC), celda("Posición menor", ENC), celda("Incumplimientos", ENC)],
    [celda("Sin piso", { bold: true }), celda("15"), celda("111.91%"), celda("0.000%", { color: "B03A2E", bold: true }), celda("0")],
    [celda("Con piso de 1%", { bold: true }), celda("13"), celda("111.75%"), celda("1.99%", { color: VERDE, bold: true }), celda("0")],
  ];
  s.addTable(filas, {
    x: 0.5, y: 2.1, w: 12.33, colW: [3.0, 2.2, 2.6, 2.5, 2.03],
    border: { type: "solid", color: BLANCO, pt: 1 },
    fill: { color: FONDO }, rowH: 0.46, valign: "middle", margin: [4, 10, 4, 10],
  });
  s.addText("En la corrida de referencia devolvió XLF en 0.159% y otro nombre en prácticamente " +
    "cero. Nadie abre esa posición: cuesta una boleta, una línea en cada reporte y una " +
    "conciliación para siempre, a cambio de una contribución al riesgo que redondea a nada. " +
    "Sobre un libro de US$5MM, ese 0.159% son US$8,000.", {
    x: 0.5, y: 3.75, w: 12.33, h: 0.95,
    fontFace: FUENTE, fontSize: 14, color: GRAFITO, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.5, y: 4.8, w: 12.33, h: 1.45, rectRadius: 0.06,
    fill: { color: GRAFITO }, line: { color: GRAFITO },
  });
  s.addText("Lo importante no es el 1%, es cómo se aplica", {
    x: 0.9, y: 4.98, w: 11.5, h: 0.35,
    fontFace: FUENTE, fontSize: 15, bold: true, color: ORO, isTextBox: true, margin: 0,
  });
  s.addText("Lo obvio sería borrar esos pesos del resultado, y sería un error: el libro quedaría " +
    "corto de presupuesto y podría empujar a otro nombre por encima de su tope o a una clase " +
    "fuera de su banda. Lo que hace el sistema es forzar esos nombres a cero y volver a " +
    "optimizar. Cada pasada es una optimización restringida de verdad, así que todos los " +
    "límites siguen cumpliéndose exactos.", {
    x: 0.9, y: 5.38, w: 11.5, h: 0.8,
    fontFace: FUENTE, fontSize: 12.5, color: BLANCO, isTextBox: true, margin: 0,
  });
}

// ================================================================ DIAGNÓSTICOS
{
  const s = laminaContenido("El sistema ahora se mide a sí mismo",
    "Dos diagnósticos en cada corrida. Ninguno cambia una recomendación ni un peso.");
  const diag = [
    ["Correlación entre bloques", "El modelo declara seis bloques y le pone un peso a cada uno. Eso afirma que cada uno aporta algo que los otros no. Si dos van juntos al 0.90, sus pesos son una sola apuesta hecha dos veces y la cartera está menos diversificada de lo que promete la tabla."],
    ["Saturación de views", "La Q se recorta en ±5%. Si casi todas las views terminan pegadas al tope, el recorte pasó a ser la señal: nombres que el screener rankeó muy distinto llegan al optimizador con el mismo retorno esperado."],
  ];
  diag.forEach((d, i) => {
    const x = 0.5 + i * 6.35;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.1, w: 5.98, h: 2.75, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.35, y: 2.4, w: 0.5, h: 0.5,
      fill: { color: i === 0 ? VERDE : ORO }, line: { color: i === 0 ? VERDE : ORO },
    });
    s.addText(d[0], {
      x: x + 1.0, y: 2.45, w: 4.7, h: 0.4,
      fontFace: FUENTE, fontSize: 16, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(d[1], {
      x: x + 0.35, y: 3.05, w: 5.3, h: 1.7,
      fontFace: FUENTE, fontSize: 12, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 5.05, 12.33, 1.15,
    "El primero ya encontró algo que la revisión externa no vio: la posición en el rango anual " +
    "(bloque de valuación, puntúa al revés) es casi el espejo de la distancia al máximo de 52 " +
    "semanas (bloque de momentum, puntúa a favor). Son el mismo dato con el signo cambiado, en " +
    "dos bloques distintos. No lo tocamos: cambiarlo altera lo que el modelo pretende medir, y " +
    "esa es una decisión de la casa.", "verde");
}

// ================================================================ PENDIENTES
{
  const s = laminaContenido("Lo que necesita una decisión, no más código",
    "Tres números que pusimos nosotros y que alguien tiene que validar antes de operar con esto.");
  const pend = [
    ["El coeficiente de información", "0.08", "Cuánto le creemos al ranking como predictor. Calibrarlo de verdad requiere guardar las recomendaciones varios meses y compararlas contra lo que pasó."],
    ["El techo de materias primas", "5–25%", "El Procedimiento no tiene banda para oro ni commodities, y el optimizador solo restringe lo que aparece listado. Le pusimos un techo por perfil, pero el número lo inventamos nosotros."],
    ["La asignación estratégica", "punto medio", "Usamos el punto medio de cada banda como posición neutral. Es una lectura razonable de un límite, pero una asignación estratégica la decide el Comité."],
  ];
  pend.forEach((p, i) => {
    const y = 2.1 + i * 1.42;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.5, y, w: 12.33, h: 1.25, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addText(p[0], {
      x: 0.85, y: y + 0.18, w: 3.6, h: 0.4,
      fontFace: FUENTE, fontSize: 15.5, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(p[1], {
      x: 0.85, y: y + 0.62, w: 3.6, h: 0.45,
      fontFace: FUENTE, fontSize: 19, bold: true, color: VERDE, isTextBox: true, margin: 0,
    });
    s.addText(p[2], {
      x: 4.65, y: y + 0.2, w: 8.0, h: 0.9,
      fontFace: FUENTE, fontSize: 12.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  s.addText("Ninguno de los tres es un pendiente técnico. Son decisiones de política que el " +
    "modelo no puede tomar por su cuenta, y que están declaradas como supuestos en cada " +
    "archivo que genera.", {
    x: 0.5, y: 6.45, w: 12.33, h: 0.5,
    fontFace: FUENTE, fontSize: 13, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
  });
}

// ================================================================ LIMITACIONES
{
  const s = laminaContenido("Limitaciones, dichas antes de que las pregunten", "");
  const lim = [
    ["No hay fundamentales", "Ni P/E, ni márgenes, ni crecimiento. El bloque de valuación usa proxies de mercado, y con Yahoo pierde dos de sus cuatro medidas. Un nombre puede salir primero aquí y estar caro por cualquier métrica fundamental."],
    ["Los pesos no salen de un backtest", "Están escogidos para que cada perfil sea internamente coherente y para que la dirección de cada diferencia sea defendible. La dirección la sostenemos; los números exactos son discutibles."],
    ["52 semanas es una muestra corta", "Beta, alfa y las razones de captura tienen intervalos anchos. No hay que leer el tercer decimal."],
    ["Solo mira precios y volumen", "No sabe de resultados trimestrales, cambios de gerencia, litigios ni fusiones. Es apoyo al criterio, no un reemplazo."],
  ];
  lim.forEach((l, i) => {
    const x = 0.5 + (i % 2) * 6.35;
    const y = 2.0 + Math.floor(i / 2) * 2.15;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: 5.98, h: 1.9, rectRadius: 0.06,
      fill: { color: FONDO }, line: { color: FONDO },
    });
    s.addText(l[0], {
      x: x + 0.35, y: y + 0.22, w: 5.3, h: 0.4,
      fontFace: FUENTE, fontSize: 15.5, bold: true, color: GRAFITO, isTextBox: true, margin: 0,
    });
    s.addText(l[1], {
      x: x + 0.35, y: y + 0.68, w: 5.3, h: 1.05,
      fontFace: FUENTE, fontSize: 11.5, color: SUTIL, isTextBox: true, margin: 0,
    });
  });
  destacado(s, 0.5, 6.3, 12.33, 0.55,
    "Y un screening de hace dos semanas describe un mercado que ya no existe. Hay que volver a correrlo.", "oro");
}

// ================================================================ CIERRE
{
  const s = pres.addSlide();
  s.background = { color: GRAFITO };
  marca(s, true);
  s.addText("Dónde estamos", {
    x: 0.9, y: 1.5, w: 11.5, h: 0.7,
    fontFace: FUENTE, fontSize: 34, bold: true, color: VERDE, isTextBox: true, margin: 0,
  });
  const est = [
    ["624", "verificaciones automáticas\nque corren en cada cambio"],
    ["7", "hallazgos de la auditoría\nimplementados y medidos"],
    ["3", "supuestos que necesitan\ndecisión del Comité"],
  ];
  est.forEach((e, i) => {
    const x = 0.9 + i * 4.0;
    s.addShape(pres.ShapeType.ellipse, {
      x: x + 0.55, y: 2.5, w: 1.5, h: 1.5,
      fill: { color: i === 2 ? ORO : VERDE }, line: { color: i === 2 ? ORO : VERDE },
    });
    s.addText(e[0], {
      x: x + 0.55, y: 2.9, w: 1.5, h: 0.75,
      fontFace: FUENTE, fontSize: 34, bold: true, color: i === 2 ? GRAFITO : BLANCO,
      align: "center", isTextBox: true, margin: 0,
    });
    s.addText(e[1], {
      x: x, y: 4.2, w: 2.6, h: 0.85,
      fontFace: FUENTE, fontSize: 12, color: GRIS, align: "center", isTextBox: true, margin: 0,
    });
  });
  s.addText("El modelo no ejecuta, no lee cuentas y no puede aprobar nada por su cuenta. " +
    "Produce una propuesta con su justificación escrita al lado, y cada view pasa por un gestor.", {
    x: 0.9, y: 5.35, w: 11.5, h: 0.75,
    fontFace: FUENTE, fontSize: 14, color: BLANCO, isTextBox: true, margin: 0,
  });
  s.addShape(pres.ShapeType.rect, {
    x: 0.9, y: 6.3, w: 1.1, h: 0.045, fill: { color: ORO }, line: { color: ORO },
  });
  pie(s);
  s.addNotes("Cerrar invitando a que cuestionen los tres supuestos. Es lo que más valor " +
    "agrega de esta sesión.");
}

pres.writeFile({ fileName: "CCI_modelo.pptx" })
  .then(f => console.log("escrito:", f));
