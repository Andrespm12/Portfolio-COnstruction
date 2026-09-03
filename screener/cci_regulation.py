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
SECTOR_CAPS: dict[str, float | None] = {
    "Conservador_Defensivo": 0.15,
    "Conservador": 0.18,
    "Moderado": 0.22,
    "Agresivo": 0.25,
}


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
