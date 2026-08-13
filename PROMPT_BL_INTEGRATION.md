# Prompt: análisis del sistema Black-Litterman de CCI e integración con el screener

Prompt reutilizable. Entrada: el notebook `DEF_Black_Litterman_CCI_Colab_Notebook_2_0.ipynb`,
el `Documento Técnico: Sistema de Asignación Táctica Black-Litterman`, y este repositorio.
Salida: un informe de hallazgos y un puente de código verificado entre ambos sistemas.

---

## Rol

Actúas como analista cuantitativo senior revisando dos sistemas que van a
convivir en producción en una casa de bolsa regulada. El estándar es el de una
revisión de riesgo model: no basta con que el código corra, tiene que producir
números defendibles ante un comité.

---

## Fase 1 — Delimitar qué hace cada sistema

Antes de proponer integración alguna, establece qué problema resuelve cada uno.
No asumas que compiten.

1. Para el sistema Black-Litterman, identifica: qué toma como entrada, de dónde
   sale el universo de activos, qué produce, y en qué punto exacto interviene
   un humano.
2. Para el screener de este repositorio, lo mismo.
3. Declara explícitamente si son **sustitutos** (hacen lo mismo, hay que elegir
   uno) o **complementarios** (resuelven etapas distintas del proceso). Si son
   complementarios, nombra la frontera: dónde termina uno y empieza el otro.

Una integración que no pueda articular esa frontera en una frase va a duplicar
lógica y producir dos respuestas distintas a la misma pregunta.

---

## Fase 2 — Auditar el sistema Black-Litterman

Compara **tres** fuentes que pueden discrepar entre sí: el documento técnico
(lo que dice que hace), el código del notebook (lo que hace), y la salida
ejecutada que quedó guardada en el notebook (lo que hizo).

Reporta cada discrepancia con:

- **Qué**: la afirmación del documento y la línea de código que la contradice.
- **Evidencia**: preferiblemente la salida real del notebook. Si un bug es
  aritmético, **reprodúcelo numéricamente** y muestra que el número calculado
  coincide con el impreso. Una discrepancia sin evidencia numérica es una
  sospecha, no un hallazgo.
- **Consecuencia**: qué decisión de cartera cambia por causa de esto. Un
  hallazgo que no cambia ninguna asignación es cosmético; márcalo como tal.
- **Severidad**: bloqueante / corrige-antes-de-producción / cosmético.

Presta atención específica a:

- **Coherencia de escalas** en cualquier combinación ponderada de señales. Si
  se suman con pesos que suman 1.0 tres señales que viven en rangos distintos
  (un nivel en [0,1], un score con signo en [−1,1], una probabilidad), el
  resultado tiene un sesgo sistemático aunque cada componente sea correcto.
  Verifica qué produce el combinador ante una entrada **neutral en todas las
  dimensiones**: debe dar exactamente cero.
- **Señales constantes entre activos.** Una dimensión que vale lo mismo para
  todos los activos de la cesta no genera views diferenciadas; en
  Black-Litterman un desplazamiento uniforme del vector Q se absorbe en buena
  medida contra el equilibrio y aporta rotación sin información. Verifica cuánto
  del score combinado es constante entre activos.
- **Restricciones declaradas pero no implementadas** (apalancamiento, derivados,
  buffers).
- **Funciones de auditoría que no auditan.**
- **Valores fallback silenciosos** que entran en cálculos sensibles.
- **Parámetros calculados y nunca usados.**

---

## Fase 3 — Diseñar el puente

Diseña la integración respetando la frontera de la Fase 1. Reglas:

1. **No dupliques el motor matemático.** Si Black-Litterman ya hace la
   combinación bayesiana y la optimización restringida, el screener no debe
   proponer pesos de cartera; debe proponer *insumos*.
2. **El puente exporta al esquema que el sistema receptor ya consume.** No
   inventes un formato nuevo ni exijas cambios en el sistema receptor para
   aceptarlo.
3. **La traducción de señal a retorno esperado (Q) debe ser explícita y
   escalada por riesgo.** Un z-score transversal es un ranking, no un pronóstico
   de retorno. Convertirlo requiere un supuesto declarado (coeficiente de
   información) y debe escalar con la volatilidad del activo: a igual ranking, un
   activo más volátil justifica mayor retorno esperado, que es precisamente lo
   que un optimizador media-varianza necesita para dimensionar bien.
4. **Convicción y señal son cosas distintas.** La magnitud de la señal va en Q.
   La convicción va en Ω y debe reflejar *confianza en la estimación*: cuántos
   bloques del modelo coinciden en signo, cuánta cobertura de datos hubo, si se
   activó algún gate de riesgo. Una view sostenida por un solo bloque no merece
   la misma Ω que una donde los seis coinciden.
5. **Mapea los perfiles de riesgo a las estrategias del sistema receptor.** Si
   el receptor tiene cuatro estrategias y el screener tres perfiles, crea el que
   falta en vez de forzar un mapeo aproximado.

---

## Fase 4 — Implementar y verificar

Implementa el puente como módulo del repositorio, no como código suelto en un
notebook, y verifícalo con la misma disciplina que el resto del repo:

- El esquema exportado debe validarse **contra el consumidor real**: escribe una
  prueba que ejecute la función del sistema receptor (o una réplica fiel de su
  contrato) sobre lo que produce el puente.
- Prueba las propiedades donde un error silencioso produciría números
  plausibles pero equivocados: signo de Q respecto a la recomendación,
  respeto del tope de Q, que el filtro de convicción mínima descarte, que una
  view relativa se construya con la referencia correcta, que un nombre con gate
  activo no salga con convicción alta.
- Verifica que la matemática de Black-Litterman **acepta** las views generadas:
  Ω debe ser invertible y el posterior debe ser finito.

---

## Fase 5 — Reportar

Entrega:

1. La frontera entre ambos sistemas, en una frase.
2. La tabla de hallazgos de la Fase 2, ordenada por severidad, con el parche
   concreto para los bloqueantes.
3. El puente implementado, con el mapeo perfil ↔ estrategia.
4. Lo que **no** verificaste y por qué.

Sé explícito sobre las calibraciones que son juicio y no resultado empírico. Un
coeficiente de información elegido a mano es un supuesto, no un hallazgo; dilo
en esos términos.
