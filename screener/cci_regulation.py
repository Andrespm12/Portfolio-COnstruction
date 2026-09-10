"""
CCI's Investment Procedure as data: asset classes, bands and hard exclusions.

Kept separate from :mod:`screener.optimizer` on purpose. CCI's technical
document names this the guiding principle of the system -- the Investment
Procedure delimits risk boundaries and is never the starting point for weights,
while the market equilibrium is computed independently. Two files keep that
separation visible: policy changes here, mathematics changes there.

Asset classification without the Google Sheet
---------------------------------------------
CCI's system reads ``clase_activo`` from a manually maintained sheet. Running
the screener and the optimizer in one notebook removes the sheet, so the class
has to come from somewhere else: :data:`ASSET_CLASS` maps the curated ETF
universe to CCI's own labels, and anything not an ETF is ``Equity``.

Two consequences are worth stating rather than burying:

* **Commodity and precious-metal ETFs have no band in CCI's REGULACIONES.**
  Their optimizer only constrains classes that appear in ``bandas``, so gold or
  broad commodities would run unconstrained up to 100%. :data:`COMMODITY_BANDS`
  adds a ceiling per strategy. It is **not** from CCI's Investment Procedure --
  it is a placeholder chosen so the hole is closed rather than silent, and it
  needs a real number from Compliance before this is used to trade.
* **EM sovereign debt (EMB) is classified NoIG.** It is nominally sovereign,
  but its credit profile and drawdown behaviour sit with high yield, and the
  band that matters for risk is the NoIG one.
"""

from __future__ import annotations

from typing import Mapping

# --------------------------------------------------------------------------
# Bands, verbatim from CCI's REGULACIONES
# --------------------------------------------------------------------------

REGULACIONES: dict[str, dict] = {
    "Conservador_Defensivo": {
        "max_equity_total": 0.30, "max_equity_individual": 0.05,
        "leverage_max": 1.0, "derivados_max": 0.0,
        "bandas": {
            "RentaFija_Soberana_IG": (0.0, 1.0),
            "RentaFija_Corporativa_IG": (0.0, 0.50),
            "ETF_RentaFija": (0.0, 1.0),
            "ETF_RentaVariable": (0.0, 0.25),
            "Equity": (0.0, 0.25),
            "Efectivo_MM": (0.0, 1.0),
            "RentaFija_NoIG": (0.0, 0.05),
        },
    },
    "Conservador": {
        "max_equity_total": 0.50, "max_equity_individual": 0.10,
        "leverage_max": 1.0, "derivados_max": 0.0,
        "bandas": {
            "RentaFija_Soberana_IG": (0.0, 1.0),
            "RentaFija_Corporativa_IG": (0.0, 0.60),
            "ETF_RentaFija": (0.0, 1.0),
            "ETF_RentaVariable": (0.0, 0.35),
            "Equity": (0.0, 0.35),
            "Efectivo_MM": (0.0, 1.0),
            "RentaFija_NoIG": (0.0, 0.10),
        },
    },
    "Moderado": {
        "max_equity_total": 0.60, "max_equity_individual": 0.15,
        "leverage_max": 1.25, "derivados_max": 0.20,
        "bandas": {
            "RentaFija_Soberana_IG": (0.0, 1.0),
            "RentaFija_Corporativa_IG": (0.0, 0.70),
            "ETF_RentaFija": (0.0, 1.0),
            "ETF_RentaVariable": (0.0, 0.50),
            "Equity": (0.0, 0.50),
            "Efectivo_MM": (0.0, 1.0),
            "RentaFija_NoIG": (0.0, 0.15),
        },
    },
    "Agresivo": {
        "max_equity_total": 0.80, "max_equity_individual": 0.20,
        "leverage_max": 1.50, "derivados_max": 0.30,
        "bandas": {
            "RentaFija_Soberana_IG": (0.0, 1.0),
            "RentaFija_Corporativa_IG": (0.0, 0.80),
            "ETF_RentaFija": (0.0, 1.0),
            "ETF_RentaVariable": (0.0, 0.70),
            "Equity": (0.0, 0.70),
            "Efectivo_MM": (0.0, 1.0),
            "RentaFija_NoIG": (0.0, 0.25),
        },
    },
}

#: Tickers forced to zero weight. Art. 170 RIV -- related issuers.
EXCLUSIONES_DURAS: tuple[str, ...] = ("CCI", "EMISOR_VINCULADO_1")


# --------------------------------------------------------------------------
# Modelo de Asignación de Mercado Internacional
# --------------------------------------------------------------------------
#
# The desired allocation from the Investment Procedure, verbatim. This is the
# strategic asset allocation the band midpoints were standing in for: bands are
# ceilings, these are targets, and only the second is an allocation.
#
# The Procedimiento states four lines; the engine carries eight classes, so
# each line is a *group* of classes and the budget is split inside it by market
# value, exactly as weights are split inside a single class. Two mapping
# decisions are recorded here rather than buried:
#
# * "Renta Fija Corporativa" covers investment grade **and** high yield /
#   emerging debt, per the desk. RentaFija_NoIG keeps its own regulatory
#   ceiling inside the group (5 / 10 / 15 / 25%), so the line cannot become all
#   high yield.
# * Broad aggregate bond funds (AGG, BND) sit with the government line. They are
#   investment grade and majority treasury, and neither line names them. This
#   one is a judgement call, not the Procedimiento's -- move ETF_RentaFija to
#   the corporate group if the committee reads it the other way.
#
# Commodities have no line in the Procedimiento, so they carry **no target**:
# the neutral portfolio holds none, and gold enters only when a view pushes it
# there. That is the honest reading of a document that does not allocate to
# them, and it is separate from COMMODITY_BANDS, which is a ceiling we invented
# to stop an unconstrained position.

GRUPOS_ASIGNACION: dict[str, tuple[str, ...]] = {
    "Renta Fija Gubernamental IG": ("RentaFija_Soberana_IG", "ETF_RentaFija"),
    "Renta Fija Corporativa": ("RentaFija_Corporativa_IG", "RentaFija_NoIG"),
    "Acciones y ETFs indexados": ("Equity", "ETF_RentaVariable"),
    "Efectivo / Money Market": ("Efectivo_MM",),
}

MODELO_ASIGNACION: dict[str, dict[str, float]] = {
    "Conservador_Defensivo": {
        "Renta Fija Gubernamental IG": 0.45,
        "Renta Fija Corporativa": 0.25,
        "Acciones y ETFs indexados": 0.20,
        "Efectivo / Money Market": 0.10,
    },
    "Conservador": {
        "Renta Fija Gubernamental IG": 0.40,
        "Renta Fija Corporativa": 0.20,
        "Acciones y ETFs indexados": 0.30,
        "Efectivo / Money Market": 0.10,
    },
    "Moderado": {
        "Renta Fija Gubernamental IG": 0.30,
        "Renta Fija Corporativa": 0.15,
        "Acciones y ETFs indexados": 0.50,
        "Efectivo / Money Market": 0.05,
    },
    "Agresivo": {
        "Renta Fija Gubernamental IG": 0.20,
        "Renta Fija Corporativa": 0.10,
        "Acciones y ETFs indexados": 0.65,
        "Efectivo / Money Market": 0.05,
    },
}


def clase_a_grupo(clase: str) -> str | None:
    """Which line of the Modelo de Asignación an engine class belongs to."""
    for grupo, clases in GRUPOS_ASIGNACION.items():
        if clase in clases:
            return grupo
    return None


# --------------------------------------------------------------------------
# Sector concentration
# --------------------------------------------------------------------------
#
# NOT from CCI's Investment Procedure. Its bands are by **asset class**, and
# nothing in it constrains industry. A live Agresivo run came out with roughly
# 35% of the book in one semiconductor chain -- eleven single names plus EWY,
# which is largely Samsung and SK Hynix -- and passed the band audit clean,
# because every one of those positions is "Equity" and Equity was inside its
# ceiling. The audit was correct and the portfolio was still a sector fund.
#
# The mechanism is structural, not bad luck: the factor model weights momentum
# at 25-36% depending on profile, momentum is serially correlated *within* an
# industry, so whatever sector ran hardest supplies most of the top of the
# ranking, and mean-variance then buys the whole cluster because their pairwise
# correlations still look moderate over a two-year window.
#
# These ceilings are the desk's, proposed by this engine and not by any
# document. They must be confirmed by the Investment Committee before they
# govern a trade. Set a value to None to disable the constraint for a strategy
# and the run will say the sleeve is unconstrained rather than pretend.
#
# EL TOPE ES FRACCIÓN DEL SLEEVE DE RENTA VARIABLE, NO DEL LIBRO.
# ---------------------------------------------------------------
# Medido sobre el libro completo, el tope quedaba más flojo cuanto más
# conservador era el mandato, que es exactamente al revés de lo que se quiere.
# Los valores anteriores, traducidos a fracción del sleeve de acciones:
#
#     Conservador Defensivo   15% del libro / 20% en acciones  =  75% del sleeve
#     Conservador             18% / 30%                        =  60%
#     Moderado                22% / 50%                        =  44%
#     Agresivo                25% / 65%                        =  38%
#
# En Defensivo permitía tres cuartas partes del sleeve en un solo sector: no
# restringía nada. En Agresivo, 38% apenas superaba el peso que tecnología ya
# tiene en el S&P 500, así que era el único de los cuatro que mordía. El tope
# hacía su trabajo solo en el mandato que menos lo necesitaba.
#
# Ahora se mide contra la exposición a renta variable del propio libro
# (``exposicion_sector <= tope * exposicion_rv``), que sigue siendo lineal en
# los pesos y por lo tanto convexa: el solver la aplica igual. Los valores
# parten del peso del mayor sector del índice — tecnología ronda el 33% del
# S&P 500 — y se aprietan con el conservadurismo del mandato.
SECTOR_CAPS: dict[str, float | None] = {
    "Conservador_Defensivo": 0.30,
    "Conservador": 0.33,
    "Moderado": 0.36,
    # 38% y no 40%: es exactamente lo que el tope viejo del 25% del libro
    # permitía sobre un sleeve del 65%. Agresivo era el único mandato donde el
    # tope de verdad ataba, así que el cambio lo deja como estaba y aprieta los
    # otros tres. Una regla de concentración nueva que afloja alguna cartera no
    # es una regla de concentración, es una excusa.
    "Agresivo": 0.38,
}


# --------------------------------------------------------------------------
# Risk appetite
# --------------------------------------------------------------------------
#
# Aversión al riesgo por mandato, el coeficiente ``lambda`` de
# ``max w'mu - (lambda/2) w'Sigma w``.
#
# Hasta aquí las cuatro estrategias resolvían con **el mismo 2.5**. La única
# diferencia entre una cartera Agresiva y una Conservadora era el ancho de sus
# bandas, y una banda es un techo: nada obligaba a la Agresiva a *usarlo*. Dos
# mandatos con distinto apetito de riesgo que optimizan la misma función no son
# dos mandatos, son el mismo con distinto papel.
#
# Un lambda alto compra tranquilidad y un lambda bajo compra retorno esperado,
# que es exactamente la diferencia que el cliente firmó.
#
# Consecuencia que conviene saber: lambda también escala el equilibrio
# (``pi = lambda * Sigma * w``). Con lambda bajo los retornos de equilibrio son
# más chicos, así que **las views pesan relativamente más en la Agresiva**. Es
# coherente con el mandato — quien pide retorno paga por convicción — pero es un
# efecto, no una casualidad.
RISK_AVERSION_BY_STRATEGY: dict[str, float] = {
    "Conservador_Defensivo": 8.0,
    "Conservador": 5.0,
    "Moderado": 2.5,          # el valor que traía el documento técnico de CCI
    "Agresivo": 1.5,
}

# --------------------------------------------------------------------------
# Supuestos de mercado — de dónde salen las bandas de riesgo y retorno
# --------------------------------------------------------------------------
#
# Las bandas de más abajo NO son números escogidos: son el resultado de aplicar
# estos supuestos al Modelo de Asignación de cada mandato. Están acá arriba y
# no enterrados en un comentario porque es lo que el Comité tiene que discutir.
# Cambiar un supuesto y volver a derivar es honesto; cambiar una banda sin
# tocar el supuesto que la produjo deja el modelo diciendo dos cosas distintas.
#
# Son de largo plazo, en dólares y anualizados. La sensibilidad más grande de
# todo el bloque es el retorno de la renta variable: bajarlo de 8,0% a 7,0%
# comprime la escalera de retorno esperado en unos 45 puntos básicos entre el
# mandato Defensivo y el Agresivo.

#: ``clase -> (volatilidad anual, retorno esperado anual)``.
SUPUESTOS_CLASE: dict[str, tuple[float, float]] = {
    "Efectivo / Money Market": (0.004, 0.040),
    "Renta Fija Gubernamental IG": (0.055, 0.047),
    "Renta Fija Corporativa": (0.080, 0.056),
    # 17% y no el 16% del índice: el sleeve son 20-25 nombres más ETFs, no el
    # S&P 500 entero, y esa concentración se paga en volatilidad.
    "Acciones y ETFs indexados": (0.170, 0.080),
}

#: Correlaciones entre clases. Efectivo va con 0,00 contra todo.
SUPUESTOS_CORRELACION: dict[frozenset[str], float] = {
    frozenset({"Renta Fija Gubernamental IG", "Renta Fija Corporativa"}): 0.80,
    frozenset({"Renta Fija Gubernamental IG", "Acciones y ETFs indexados"}): 0.10,
    frozenset({"Renta Fija Corporativa", "Acciones y ETFs indexados"}): 0.45,
}


#: Volatilidad anual esperada de cada mandato: ``(mínima, máxima)``.
#:
#: NO salen del Procedimiento de Inversión, que no habla de volatilidad. Se
#: derivan de :data:`SUPUESTOS_CLASE` aplicados al :data:`MODELO_ASIGNACION`:
#:
#:   * el **piso** es el 80% de la volatilidad de la asignación neutral. Un
#:     libro por debajo de eso dejó de ser el mandato que el cliente firmó.
#:   * el **techo** es la volatilidad de esa misma asignación con la renta
#:     variable en su tope regulatorio, redondeada hacia arriba. Es el máximo
#:     riesgo que el mandato puede cargar sin salirse del Procedimiento.
#:
#: Los valores anteriores (1,5-7,0 / 3,5-10,5 / 6,5-15,0 / 10,0-24,0) tenían dos
#: defectos que este cálculo destapó:
#:
#:   1. El techo Defensivo de 7,0% era **inferior** a la volatilidad que su
#:      propia política produce en el tope de renta variable (7,1%). Un libro
#:      Defensivo totalmente invertido y perfectamente conforme incumplía su
#:      propio techo: la restricción contradecía al mandato.
#:   2. El techo Agresivo de 24% era inalcanzable — el mandato no pasa de 13,9%
#:      ni forzándolo. Como el techo es lo que el solver impone, la restricción
#:      de volatilidad de la Agresiva no hacía absolutamente nada.
#:
#: Los dos extremos siguen sin tratarse igual, y la razón es matemática, no de
#: criterio. ``w'Sigma w <= max**2`` es una restricción convexa y el solver la
#: aplica. ``w'Sigma w >= min**2`` es convexa al revés: no se puede pedir. Así
#: que el techo se **impone** y el piso se **audita** — si una cartera Agresiva
#: sale por debajo de su piso, la corrida lo reporta como incumplimiento, que es
#: lo que es: un cliente que firmó Agresivo no contrató una cartera Moderada.
#:
#: Las bandas contiguas se solapan a propósito: los mandatos se solapan en el
#: riesgo que pueden producir, y fingir lo contrario obligaría a inventar cortes
#: que ninguna política respalda.
RISK_TARGETS: dict[str, tuple[float, float]] = {
    "Conservador_Defensivo": (0.050, 0.075),
    "Conservador": (0.055, 0.100),
    "Moderado": (0.075, 0.115),
    "Agresivo": (0.090, 0.145),
}

#: Retorno anual esperado de cada mandato, **estratégico y de largo plazo**.
#:
#: Es el retorno de la asignación neutral bajo :data:`SUPUESTOS_CLASE`. No es un
#: pronóstico ni una promesa al cliente: es lo que la política de cada mandato
#: debería rendir en promedio si los supuestos se cumplen.
#:
#: **No confundir con el retorno esperado que reporta la corrida.** Ese sale del
#: posterior de Black-Litterman y está condicionado a las views y al lambda de
#: la corrida. Son dos números distintos y viven en campos distintos a
#: propósito: mezclarlos es exactamente la confusión de lambda que costó una
#: corrida entera, cuando la cartera Agresiva reportaba menos retorno esperado
#: que la Moderada por un artefacto de escala.
#:
#: La escalera está comprimida — 138 puntos básicos entre el mandato más
#: defensivo y el más agresivo — y eso es real, no un defecto del cálculo: con
#: el efectivo cerca del 4%, la prima por asumir 45 puntos más de renta variable
#: es genuinamente pequeña en el entorno de tasas actual.
RETORNO_ESPERADO: dict[str, float] = {
    "Conservador_Defensivo": 0.055,
    "Conservador": 0.058,
    "Moderado": 0.065,
    "Agresivo": 0.069,
}

#: Caída máxima tolerada por mandato, como número negativo.
#:
#: Un múltiplo de la volatilidad del techo: **2,5 sigma** para los dos mandatos
#: conservadores y **3,0 sigma** para los dos agresivos. La diferencia no es
#: arbitraria. Un 60/40 cayó cerca de 31% en 2008 y 21% en 2022, y 2,5 sigma
#: sobre su techo da 27%: encaja. Un 80/20 cayó cerca de 42% en 2008, que son
#: 3 sigma y no 2,5. Las colas de la renta variable son más gordas que la
#: normal, y usar el mismo múltiplo en los cuatro subestima justo donde el
#: cliente más lo va a sentir.
#:
#: Es una **tolerancia**, no una restricción: el drawdown de una cartera no es
#: una función convexa de sus pesos y no se le puede pedir al solver. Se audita
#: contra la caída en muestra que reporta ``drawdown_metrics``, con la
#: advertencia que esa función ya lleva: esos pesos no existían entonces.
DRAWDOWN_TOLERADO: dict[str, float] = {
    "Conservador_Defensivo": -0.16,
    "Conservador": -0.22,
    "Moderado": -0.28,
    "Agresivo": -0.40,
}

#: Tracking error objetivo contra el ancla, por mandato.
#:
#: Existe porque ``RISK_AVERSION_BY_STRATEGY`` **no es aversión al riesgo en el
#: sentido clásico**. El objetivo de este optimizador penaliza el riesgo
#: *activo contra el ancla*, no el riesgo total, así que lambda es en la
#: práctica un presupuesto de tracking error: cuánto se le permite a las views
#: alejarse de la asignación del Comité.
#:
#: Eso cambia cómo se calibra. El lambda implícito en las propias asignaciones
#: sería 4,1 / 3,7 / 2,7 / 2,1, y los valores en uso son 8,0 / 5,0 / 2,5 / 1,5 —
#: una escalera bastante más empinada. Está bien que lo sea, precisamente porque
#: no mide lo mismo: es defendible que un mandato defensivo se desvíe menos de
#: su política. Pero entonces lambda no se calibra contra volatilidad total, se
#: calibra contra esto.
#:
#: El tracking error resultante depende de la cesta y de las views de cada
#: corrida, así que no se puede fijar por decreto: la corrida lo **mide** y lo
#: reporta contra este objetivo. Si queda sistemáticamente fuera, se ajusta
#: lambda y se vuelve a correr. Dos iteraciones bastan.
TRACKING_ERROR_OBJETIVO: dict[str, tuple[float, float]] = {
    "Conservador_Defensivo": (0.010, 0.015),
    "Conservador": (0.015, 0.020),
    "Moderado": (0.025, 0.035),
    "Agresivo": (0.040, 0.060),
}


def risk_aversion_for(strategy: str, default: float = 2.5) -> float:
    """El ``lambda`` del mandato. Se usa en el equilibrio y en el objetivo."""
    return float(RISK_AVERSION_BY_STRATEGY.get(strategy, default))


def retorno_esperado_for(strategy: str) -> float | None:
    """Retorno estratégico de largo plazo del mandato, o ``None``."""
    valor = RETORNO_ESPERADO.get(strategy)
    return None if valor is None else float(valor)


def drawdown_tolerado_for(strategy: str) -> float | None:
    """Caída máxima tolerada del mandato (negativa), o ``None``."""
    valor = DRAWDOWN_TOLERADO.get(strategy)
    return None if valor is None else float(valor)


def tracking_error_objetivo_for(strategy: str) -> tuple[float, float] | None:
    """Banda de tracking error contra el ancla, o ``None``."""
    valor = TRACKING_ERROR_OBJETIVO.get(strategy)
    return None if valor is None else (float(valor[0]), float(valor[1]))


def riesgo_de_asignacion(pesos_por_clase: Mapping[str, float]
                         ) -> tuple[float, float]:
    """
    Volatilidad y retorno anual de una asignación por clase de activo.

    Es la función que produjo :data:`RISK_TARGETS`, :data:`RETORNO_ESPERADO` y
    :data:`DRAWDOWN_TOLERADO`. Vive en el código y no en una hoja de cálculo
    perdida para que el Comité pueda cambiar un supuesto y volver a derivar,
    en vez de discutir un número cuyo origen nadie recuerda.
    """
    import math

    clases = [c for c in pesos_por_clase if c in SUPUESTOS_CLASE]
    if not clases:
        return (0.0, 0.0)

    var = 0.0
    ret = 0.0
    for i, ci in enumerate(clases):
        wi = float(pesos_por_clase[ci])
        si, mi = SUPUESTOS_CLASE[ci]
        ret += wi * mi
        for j, cj in enumerate(clases):
            wj = float(pesos_por_clase[cj])
            sj, _ = SUPUESTOS_CLASE[cj]
            rho = 1.0 if i == j else SUPUESTOS_CORRELACION.get(
                frozenset({ci, cj}), 0.0)
            var += wi * wj * si * sj * rho
    return (math.sqrt(max(var, 0.0)), ret)


# --------------------------------------------------------------------------
# Core index exposures
# --------------------------------------------------------------------------
#
# Which market exposures the core should be *able* to hold, and the vehicles
# that deliver each one.
#
# The problem this fixes: ``select_basket`` used to hand the optimizer the top
# names by score plus a floor of three per asset class. In a live run that
# produced exactly three equity ETFs -- XBI (biotech), EWT (Taiwan) and EWY
# (Korea) -- because those were the highest-scoring ETFs in a momentum-weighted
# ranking. SPY ranked #149, IWM #115, EEM #119. None of them was ever offered
# to the optimizer, so "the model did not choose SPY" was never true: the model
# never saw it.
#
# Broad market exposure is an allocation decision, so policy decides which
# exposures are available. Which *vehicle* delivers an exposure is a comparison
# of like with like, so the screener's own score decides that -- the best
# eligible candidate in each line wins its slot. That is the split the desk
# asked for: rank the exposure first, then compare the wrappers.
#
# What is deliberately absent: factor and style funds (SCHD, VTV, VLUE, IWD).
# A value or dividend tilt is a position, not core beta, and it should have to
# earn its way in through the ranking like any other bet. Sector funds are
# absent for the same reason.
#
# The list is policy, and it is the kind of hand-maintained list that
# ``screener.seleccion`` warns about. The difference is that this one is
# declared, versioned and printed in the run, so it can be argued with.
EXPOSICIONES_NUCLEO: dict[str, tuple[str, ...]] = {
    "EEUU amplio": ("SPY", "IVV", "VOO", "VTI", "SPLG", "SCHX"),
    "EEUU pequeña capitalización": ("IWM", "IJR"),
    "Desarrollados ex-EEUU": ("EFA", "IEFA", "VEA"),
    "Emergentes": ("IEMG", "EEM", "VWO"),
    "Nasdaq / crecimiento": ("QQQ", "QQQM", "VUG", "IWF"),
}

#: Ceiling for commodity and precious-metal ETFs, per strategy.
#:
#: NOT from CCI's Investment Procedure. Their REGULACIONES has no commodity
#: class, and their optimizer only constrains classes present in ``bandas``, so
#: without this an unconstrained gold position could take the whole book. These
#: are placeholders that close the hole; Compliance has to supply the real
#: numbers before this drives a trade.
COMMODITY_BANDS: dict[str, tuple[float, float]] = {
    "Conservador_Defensivo": (0.0, 0.05),
    "Conservador": (0.0, 0.10),
    "Moderado": (0.0, 0.15),
    "Agresivo": (0.0, 0.25),
}

CLASE_COMMODITIES = "ETF_Commodities"

#: Classes whose bands the screener's own labels can populate.
CLASE_EQUITY = "Equity"
CLASE_ETF_RV = "ETF_RentaVariable"


# --------------------------------------------------------------------------
# Asset classification
# --------------------------------------------------------------------------

_SOBERANA_IG = {"TLT", "IEF", "SHY", "GOVT", "TIP", "MBB"}
_CORPORATIVA_IG = {"LQD", "VCIT", "VCSH", "FLOT", "MUB"}
_AGREGADO = {"BND", "AGG"}
#: EMB is nominally sovereign; its credit and drawdown profile sit with high
#: yield, and the NoIG band is the one that governs that risk.
_NO_IG = {"HYG", "JNK", "BKLN", "SRLN", "EMB"}
_EFECTIVO = {"BIL", "SGOV"}
_COMMODITIES = {
    "GLD", "IAU", "SLV", "GDX", "GDXJ", "DBA", "DBC", "USO", "UNG", "PDBC",
    "PPLT", "COPX", "URA", "WOOD", "CORN", "WEAT", "LIT", "REMX", "OIH", "XOP",
}

ASSET_CLASS: dict[str, str] = {
    **{t: "RentaFija_Soberana_IG" for t in _SOBERANA_IG},
    **{t: "RentaFija_Corporativa_IG" for t in _CORPORATIVA_IG},
    **{t: "ETF_RentaFija" for t in _AGREGADO},
    **{t: "RentaFija_NoIG" for t in _NO_IG},
    **{t: "Efectivo_MM" for t in _EFECTIVO},
    **{t: CLASE_COMMODITIES for t in _COMMODITIES},
}


def classify_for_bands(ticker: str, asset_type: str) -> str:
    """
    CCI's ``clase_activo`` for one instrument.

    ``asset_type`` is the screener's ETF/STOCK label, which comes from exact
    membership in the curated universe rather than a description regex.
    """
    tkr = ticker.upper()
    if tkr in ASSET_CLASS:
        return ASSET_CLASS[tkr]
    return CLASE_ETF_RV if asset_type == "ETF" else CLASE_EQUITY


def bands_for(strategy: str) -> dict[str, tuple[float, float]]:
    """Every band that applies to a strategy, including the commodity addition."""
    if strategy not in REGULACIONES:
        raise KeyError(
            f"Estrategia desconocida: {strategy!r}. "
            f"Opciones: {sorted(REGULACIONES)}"
        )
    bands = dict(REGULACIONES[strategy]["bandas"])
    bands[CLASE_COMMODITIES] = COMMODITY_BANDS[strategy]
    return bands


def unbanded_classes(classes: Mapping[str, str], strategy: str) -> set[str]:
    """
    Classes present in the basket that no band constrains.

    An unbanded class is unconstrained up to 100% of the book. Callers surface
    this rather than letting it pass quietly -- it is the failure mode that
    prompted :data:`COMMODITY_BANDS`.
    """
    return {c for c in set(classes.values()) if c not in bands_for(strategy)}
