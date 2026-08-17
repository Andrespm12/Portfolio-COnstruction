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
