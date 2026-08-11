"""
Universe definition and eligibility filtering.

Scope, per the screening mandate
--------------------------------
US-listed securities that are a member of at least one of:

  * S&P 500 / S&P Composite 1500  (``SP500``)
  * Nasdaq-100 or Nasdaq Composite (``NDX`` / ``NASDAQ``)
  * Dow Jones Industrial Average   (``DJIA``)

plus **ETFs, which are screened and ranked as individual securities** rather
than being treated as a separate asset class. An ETF competes for capital
against a single stock on identical factor definitions.

Membership data
---------------
This environment has no outbound network access to index-constituent sources,
so membership lists below are a **static snapshot** and carry a
``SNAPSHOT_DATE``. They are the one part of this package that goes stale on
its own. :func:`load_membership_override` reads a CSV to replace them without
touching code -- wire that to a real constituent feed in production.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .config import ELIGIBILITY, EXCLUDED_PRODUCT_PATTERNS

#: Date the hardcoded membership lists were captured. Refresh quarterly --
#: index reconstitution happens on a known calendar.
SNAPSHOT_DATE = "2026-08-11"


# --------------------------------------------------------------------------
# Index membership (static snapshot)
# --------------------------------------------------------------------------

DJIA: frozenset[str] = frozenset({
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
})

NDX: frozenset[str] = frozenset({
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN", "BIIB", "BKNG", "BKR",
    "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD", "CSCO",
    "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG",
    "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN",
    "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR",
    "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MU", "NFLX",
    "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP",
    "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM", "TMUS",
    "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
})

#: Large-cap S&P 500 members. Deliberately a curated liquid subset rather than
#: a fabricated "complete" 500 -- an inaccurate constituent list is worse than
#: an explicitly partial one. Override via CSV for the full index.
SP500: frozenset[str] = frozenset({
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "ACN", "CSCO",
    "TXN", "QCOM", "INTU", "AMAT", "NOW", "IBM", "MU", "LRCX", "ADI", "KLAC",
    "PANW", "SNPS", "CDNS", "MSI", "ANET", "APH", "NXPI", "MCHP", "FTNT", "IT",
    "GLW", "HPQ", "DELL", "WDC", "STX", "ON", "TER", "SWKS", "MPWR", "ZBRA",
    # Communication services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR",
    "EA", "TTWO", "OMC", "IPG", "WBD", "LYV", "NWSA", "FOXA", "MTCH", "PARA",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "LOW", "BKNG", "TJX", "SBUX", "NKE", "ORLY",
    "AZO", "CMG", "MAR", "HLT", "GM", "F", "ROST", "YUM", "DHI", "LEN",
    "EBAY", "APTV", "LVS", "WYNN", "MGM", "RCL", "CCL", "NCLH", "DRI", "ULTA",
    # Consumer staples
    "WMT", "PG", "COST", "KO", "PEP", "PM", "MO", "MDLZ", "CL", "KMB",
    "GIS", "SYY", "KR", "STZ", "HSY", "KHC", "CHD", "CLX", "TSN", "MKC",
    # Health care
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "AMGN",
    "BMY", "GILD", "ISRG", "VRTX", "MDT", "SYK", "CI", "ELV", "REGN", "BSX",
    "ZTS", "HCA", "MCK", "CVS", "BDX", "EW", "IQV", "A", "IDXX", "BIIB",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI",
    "BLK", "SCHW", "C", "CB", "MMC", "PGR", "AON", "ICE", "CME", "PNC",
    "USB", "TFC", "COF", "BK", "AIG", "MET", "PRU", "TRV", "ALL", "AFL",
    # Industrials
    "CAT", "GE", "RTX", "HON", "UNP", "BA", "LMT", "DE", "UPS", "ADP",
    "ETN", "ITW", "NOC", "GD", "CSX", "NSC", "EMR", "PH", "CMI", "FDX",
    "WM", "PCAR", "ROK", "CTAS", "PAYX", "FAST", "ODFL", "URI", "IR", "AME",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HES", "DVN", "FANG", "HAL", "BKR", "TRGP", "MRO", "APA",
    # Materials
    "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "DD", "NUE", "PPG",
    "VMC", "MLM", "IFF", "ALB", "CE", "STLD", "PKG", "IP", "CF", "MOS",
    # Utilities
    "NEE", "SO", "DUK", "SRE", "AEP", "D", "EXC", "XEL", "ED", "PEG",
    "WEC", "ES", "AWK", "DTE", "PPL", "AEE", "CMS", "CNP", "ATO", "NRG",
    # Real estate
    "PLD", "AMT", "EQIX", "CCI", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VTR", "INVH", "ARE", "MAA", "ESS", "EXR", "IRM", "HST", "KIM",
})

#: ETFs eligible for the screen. Broad-market, sector, style, factor and
#: single-asset-class funds -- all screened head-to-head with single stocks.
#: Leveraged, inverse, single-stock and option-income products are excluded by
#: :func:`is_excluded_product`, not by omission from this list.
ETF_UNIVERSE: frozenset[str] = frozenset({
    # Broad US market
    "SPY", "IVV", "VOO", "VTI", "QQQ", "QQQM", "DIA", "IWM", "IWB", "RSP",
    "VTWO", "IJH", "IJR", "MDY", "SCHX", "ITOT", "SPLG", "VV", "VXF", "SPTM",
    # Sector (SPDR Select + thematic)
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB", "XLRE",
    "XLC", "SMH", "SOXX", "IBB", "XBI", "KRE", "KBE", "XOP", "OIH", "ITB",
    "IYT", "IGV", "HACK", "SKYY", "ARKK", "ROBO", "TAN", "ICLN", "LIT", "REMX",
    # Style / factor
    "MTUM", "QUAL", "USMV", "VLUE", "SIZE", "SPHQ", "SPLV", "MOAT", "COWZ", "CALF",
    "VUG", "VTV", "IWF", "IWD", "SCHD", "VIG", "DGRO", "NOBL", "VYM", "HDV",
    # Fixed income
    "TLT", "IEF", "SHY", "GOVT", "BND", "AGG", "LQD", "HYG", "JNK", "TIP",
    "BIL", "SGOV", "MUB", "EMB", "VCIT", "VCSH", "MBB", "FLOT", "BKLN", "SRLN",
    # International
    "EFA", "IEFA", "VEA", "EEM", "IEMG", "VWO", "VXUS", "EWJ", "EWG", "EWU",
    "FXI", "MCHI", "EWZ", "EWY", "EWT", "INDA", "EWC", "EWA", "EPP", "ILF",
    # Commodities / real assets
    "GLD", "IAU", "SLV", "GDX", "GDXJ", "DBA", "DBC", "USO", "UNG", "PDBC",
    "VNQ", "SCHH", "IYR", "RWO", "PPLT", "COPX", "URA", "WOOD", "CORN", "WEAT",
})


def all_index_members() -> dict[str, frozenset[str]]:
    return {"SP500": SP500, "NDX": NDX, "DJIA": DJIA, "ETF": ETF_UNIVERSE}


def index_tags(ticker: str) -> list[str]:
    """Every index/list the ticker belongs to."""
    t = ticker.upper().strip()
    return [name for name, members in all_index_members().items() if t in members]


def in_universe(ticker: str) -> bool:
    """True when the ticker is in at least one mandated index or the ETF list."""
    return bool(index_tags(ticker))


def load_membership_override(csv_path: str | Path) -> dict[str, frozenset[str]]:
    """
    Replace the static snapshot from a CSV with columns ``ticker,index``.

    Production wiring point: point this at a maintained constituent feed and
    the staleness problem disappears.
    """
    buckets: dict[str, set[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            idx = (row.get("index") or "").strip().upper()
            tkr = (row.get("ticker") or "").strip().upper()
            if idx and tkr:
                buckets.setdefault(idx, set()).add(tkr)
    return {k: frozenset(v) for k, v in buckets.items()}


# --------------------------------------------------------------------------
# Product exclusions
# --------------------------------------------------------------------------

_EXCLUDE_RE = [re.compile(p, re.IGNORECASE) for p in EXCLUDED_PRODUCT_PATTERNS]


def is_excluded_product(description: str) -> tuple[bool, str | None]:
    """
    Identify leveraged, inverse, single-stock and option-income wrappers.

    A search for any liquid ticker on IBKR returns a long tail of these
    (``GRANITESH 2X LNG NVDA ETF``, ``YIELDMAX NVDA OPTION INC ETF``,
    ``DIREXION DAILY NVDA BEAR 1X``). Their returns are path-dependent or
    option-truncated, so factor metrics computed on them are not comparable to
    those of ordinary equities. Excluding them is a correctness requirement,
    not a stylistic preference.
    """
    desc = (description or "").upper()
    for rx in _EXCLUDE_RE:
        if rx.search(desc):
            return True, rx.pattern
    return False, None


def classify_asset_type(description: str, sections: list[str] | None = None) -> str:
    """Best-effort STOCK vs ETF classification from the IBKR description."""
    desc = (description or "").upper()
    if re.search(r"\b(ETF|FUND|TRUST|SHARES|INDEX|PORTFOLIO|SPDR|ISHARES|VANGUARD)\b", desc):
        return "ETF"
    return "STOCK"


# --------------------------------------------------------------------------
# Eligibility screen
# --------------------------------------------------------------------------

def screen_eligibility(instrument: dict, metrics: dict) -> tuple[bool, list[str]]:
    """
    Apply hard filters. Returns ``(is_eligible, [reasons_for_failure])``.

    Runs before scoring: an ineligible name is removed from the cross-section
    entirely so it cannot distort the z-score distribution of its peers.
    """
    reasons: list[str] = []
    rules = ELIGIBILITY

    excluded, pattern = is_excluded_product(instrument.get("name", ""))
    if excluded:
        reasons.append(f"excluded product type (matched /{pattern}/)")

    country = (instrument.get("country_code") or rules.required_country).upper()
    if country != rules.required_country:
        reasons.append(f"not US-listed (country={country})")

    exch = (instrument.get("exchange") or "").upper()
    if exch and exch not in rules.allowed_exchanges:
        reasons.append(f"exchange {exch} not in allowed US venues")

    if not instrument.get("indices"):
        reasons.append("not a member of S&P / Nasdaq / Dow, and not an eligible ETF")

    price = metrics.get("_last_price")
    if price is not None and price < rules.min_price:
        reasons.append(f"price ${price:.2f} below ${rules.min_price:.2f} minimum")

    adv = metrics.get("_adv_usd")
    if adv is not None and adv < rules.min_adv_usd:
        reasons.append(f"ADV ${adv/1e6:.1f}MM below ${rules.min_adv_usd/1e6:.0f}MM minimum")

    bars = metrics.get("_bars")
    if bars is not None and bars < rules.min_history_bars:
        reasons.append(f"only {bars} price observations (need {rules.min_history_bars})")

    return (not reasons), reasons
