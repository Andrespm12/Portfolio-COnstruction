"""
Configuration for the quantitative stock/ETF screening engine.

Everything that is a *judgement call* lives here: factor definitions, block
weights, winsorization limits, recommendation bands and hard risk gates.
The rest of the package is mechanical.

Design principles
-----------------
1. Every metric declares a DIRECTION (+1 = higher is better, -1 = lower is
   better) so the scoring engine never has to special-case signs.
2. Every metric declares whether it is scored PEER-RELATIVE (z-scored inside
   its asset-type cohort: ETF vs single stock). This matters because ETFs are
   structurally lower-volatility and lower-beta than single names; scoring
   them in one undifferentiated pool would mechanically hand every ETF a top
   risk score and every single stock a bottom one.
3. Block weights sum to 1.0. Metric weights sum to 1.0 *within* each block.
   Missing metrics cause weights to re-normalize rather than silently score 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# --------------------------------------------------------------------------
# Global run parameters
# --------------------------------------------------------------------------

BENCHMARK_TICKER = "SPY"

#: Annualized risk-free rate used for Sharpe / Sortino / excess return.
#: Default anchored to front-end US T-Bill yields. Override at runtime.
RISK_FREE_RATE = 0.0425

#: Bars per year for the history frequency in use (weekly bars).
PERIODS_PER_YEAR = 52

#: Winsorization percentiles applied to every raw metric before z-scoring.
#: Prevents a single blow-up name (e.g. a 300% momentum outlier) from
#: compressing the entire cross-section into a narrow band.
WINSOR_LOWER_PCT = 5.0
WINSOR_UPPER_PCT = 95.0

#: Z-scores are clipped to this absolute value after standardization.
Z_CLIP = 3.0


# --------------------------------------------------------------------------
# Eligibility filters (hard screens applied BEFORE scoring)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EligibilityRules:
    """Hard filters. A name failing any of these never reaches the scorer."""

    #: Minimum share price. Removes sub-$5 names where bid/ask spread and
    #: tick-size effects dominate any factor signal.
    min_price: float = 5.0

    #: Minimum 90-day average daily traded value, USD. This is the single most
    #: important institutional filter: it defines whether a position can
    #: actually be built and exited without paying up in market impact.
    min_adv_usd: float = 20_000_000.0

    #: Maximum acceptable share of ADV that a target position may represent.
    #: Used to compute `days_to_liquidate`; names that would take longer than
    #: `max_days_to_liquidate` to unwind at this participation rate are flagged.
    max_participation_rate: float = 0.20
    max_days_to_liquidate: float = 3.0

    #: Minimum number of usable price observations required to compute risk
    #: statistics. Below this, risk metrics are not statistically meaningful.
    min_history_bars: int = 30

    #: Listing venues considered "US-listed" for universe purposes.
    allowed_exchanges: tuple[str, ...] = (
        "NASDAQ", "NYSE", "ARCA", "AMEX", "BATS", "IEX", "NYSENAT", "PSE",
    )

    #: Country code required on the contract record. Filters out the foreign
    #: cross-listings (MEXI/EBS/TSE lines) that IBKR returns for most tickers.
    required_country: str = "US"


ELIGIBILITY = EligibilityRules()


#: Description patterns that identify products we deliberately exclude from a
#: *stock-picking* universe. These are not investments in a business or a
#: diversified basket -- they are structured trading vehicles whose return is
#: driven by path-dependence (daily rebalancing) or by a sold-option overlay.
#: Scoring them alongside ordinary equities produces garbage rankings.
EXCLUDED_PRODUCT_PATTERNS: tuple[str, ...] = (
    # Leveraged / inverse -- daily-reset compounding, not buy-and-hold assets
    r"\b\d+(\.\d+)?\s*X\b", r"\bULTRA(PRO|SHORT)?\b", r"\bLEVERAGE",
    r"\bBULL\b", r"\bBEAR\b", r"\bINVERSE\b", r"\bDAILY\b.*\b(BULL|BEAR)\b",
    r"\bSHT\b", r"\b2XSHT\b", r"\bLNG\b",
    # Option-income / covered-call overlays -- return profile is truncated
    r"YIELDMAX", r"YLDMX", r"YIELDBOOST", r"OPTION\s+INC", r"WEEKLYPAY",
    r"COVERED\s+CALL", r"GROWTH\s*&\s*INCOME", r"PREM\s+INC", r"OIS\s+ETF",
    # Notes and non-1940-Act wrappers -- carry issuer credit risk
    r"\bETN\b", r"\bETRACS\b",
)


# --------------------------------------------------------------------------
# Factor model
# --------------------------------------------------------------------------

Direction = Literal[1, -1]


@dataclass(frozen=True)
class Metric:
    """A single scored input."""

    key: str
    label: str
    direction: Direction
    weight: float
    peer_relative: bool = False
    description: str = ""


@dataclass(frozen=True)
class Block:
    """A weighted group of metrics representing one investment dimension."""

    key: str
    label: str
    weight: float
    metrics: tuple[Metric, ...]
    rationale: str = ""

    def normalized_metric_weights(self) -> dict[str, float]:
        total = sum(m.weight for m in self.metrics)
        return {m.key: m.weight / total for m in self.metrics}


FACTOR_MODEL: tuple[Block, ...] = (
    Block(
        key="momentum",
        label="Momentum & Trend",
        weight=0.22,
        rationale=(
            "Cross-sectional momentum is the most robust price-based anomaly in "
            "US equities. The 12-month-less-1-month construction is deliberate: "
            "the most recent month exhibits short-term reversal, so including it "
            "dilutes the signal."
        ),
        metrics=(
            Metric("mom_12_1", "12M-1M total return", 1, 0.35,
                   description="Classic Jegadeesh-Titman momentum, skipping the reversal month."),
            Metric("mom_6m", "6M total return", 1, 0.20,
                   description="Intermediate-horizon trend confirmation."),
            Metric("mom_3m", "3M total return", 1, 0.15,
                   description="Near-term trend; catches regime inflections earlier."),
            Metric("pct_from_52w_high", "Proximity to 52w high", 1, 0.15,
                   description="Distance below the 52-week high. Names near highs keep making highs."),
            Metric("above_40w_ma", "Price vs 40W moving average", 1, 0.10,
                   description="Percentage above/below the 40-week (200-day) moving average."),
            Metric("ma_slope_13w", "40W MA slope (13w)", 1, 0.05,
                   description="Is the long-term trend itself rising or rolling over."),
        ),
    ),
    Block(
        key="risk_adjusted",
        label="Risk-Adjusted Return",
        weight=0.18,
        rationale=(
            "Raw return is not the objective -- return per unit of risk taken is. "
            "Sortino and Calmar are weighted alongside Sharpe because investors "
            "do not experience upside volatility as risk, and because the path of "
            "the drawdown drives real-world capital retention."
        ),
        metrics=(
            Metric("sharpe_1y", "Sharpe ratio (1Y)", 1, 0.30,
                   description="Excess return over risk-free, per unit of total volatility."),
            Metric("sortino_1y", "Sortino ratio (1Y)", 1, 0.30,
                   description="Excess return per unit of *downside* deviation only."),
            Metric("calmar_1y", "Calmar ratio (1Y)", 1, 0.25,
                   description="Annual return divided by max drawdown. Rewards smooth compounding."),
            Metric("pct_positive_periods", "Hit rate (% positive weeks)", 1, 0.15,
                   description="Consistency of returns. High hit-rate names are easier to hold."),
        ),
    ),
    Block(
        key="risk",
        label="Volatility & Drawdown",
        weight=0.15,
        rationale=(
            "Scored peer-relative (ETF vs single stock) because the two cohorts "
            "have structurally different volatility distributions. Lower is better "
            "throughout: this block is the brake on the momentum block's tendency "
            "to load into high-beta melt-ups."
        ),
        metrics=(
            Metric("volatility_1y", "Realized volatility (1Y ann.)", -1, 0.30, peer_relative=True,
                   description="Annualized standard deviation of weekly returns."),
            Metric("max_drawdown", "Maximum drawdown (1Y)", -1, 0.30, peer_relative=True,
                   description="Worst peak-to-trough decline. Magnitude, so lower is better."),
            Metric("downside_deviation", "Downside deviation", -1, 0.20, peer_relative=True,
                   description="Volatility of negative returns only."),
            Metric("ulcer_index", "Ulcer index", -1, 0.20, peer_relative=True,
                   description="Depth *and duration* of drawdowns. Penalizes long underwater periods."),
        ),
    ),
    Block(
        key="market_sensitivity",
        label="Market Sensitivity & Alpha Quality",
        weight=0.12,
        rationale=(
            "Beta is cheap -- an investor can buy it in SPY for 3bp. What justifies "
            "single-name risk is idiosyncratic return and asymmetric capture. This "
            "block rewards names that outperform on up-moves more than they "
            "underperform on down-moves."
        ),
        metrics=(
            Metric("capture_spread", "Up/down capture spread", 1, 0.40,
                   description="Upside capture minus downside capture vs benchmark. Positive = convex."),
            Metric("alpha_annual", "Jensen's alpha (ann.)", 1, 0.35,
                   description="Return not explained by market beta."),
            Metric("idio_vol_share", "Idiosyncratic vol share", 1, 0.15,
                   description="Share of variance NOT explained by the market. Diversifying return source."),
            Metric("beta_1y", "Beta vs benchmark", -1, 0.10, peer_relative=True,
                   description="Lower beta preferred at equal return -- less market risk per dollar."),
        ),
    ),
    Block(
        key="liquidity",
        label="Liquidity & Tradability",
        weight=0.10,
        rationale=(
            "A screen that ignores capacity produces ideas the portfolio cannot "
            "actually implement. Scaled to the account's own net liquidation "
            "value, so 'liquid enough' is defined relative to real position sizes."
        ),
        metrics=(
            Metric("adv_usd_log", "90D avg daily $ volume (log)", 1, 0.50,
                   description="Log of average daily traded value. Primary capacity measure."),
            Metric("days_to_liquidate", "Days to liquidate target position", -1, 0.35,
                   description="At max participation rate, days needed to exit a full position."),
            Metric("turnover_stability", "Volume stability", 1, 0.15,
                   description="Inverse coefficient of variation of weekly volume. Penalizes episodic liquidity."),
        ),
    ),
    Block(
        key="valuation_carry",
        label="Valuation Proxy & Carry",
        weight=0.10,
        rationale=(
            "IBKR's market-data surface exposes no fundamental valuation ratios, so "
            "this block uses market-implied proxies rather than pretending to have "
            "P/E or EV/EBITDA. IV-vs-HV spread is a genuine richness signal: options "
            "priced far above realized vol signal crowded positioning and elevated "
            "expected turbulence. See docs for wiring a fundamentals feed."
        ),
        metrics=(
            Metric("dividend_yield", "Dividend yield", 1, 0.30,
                   description="Cash carry. Also a mild quality/maturity proxy."),
            Metric("iv_hv_spread", "IV minus realized vol", -1, 0.30,
                   description="Options rich vs realized. High spread = expensive hedges, crowded name."),
            Metric("iv_percentile", "IV percentile (52w)", -1, 0.20,
                   description="Where implied vol sits in its own 1-year range."),
            Metric("range_position", "52-week range position", -1, 0.20,
                   description="Position within the 52w high-low band. Mean-reversion counterweight to momentum."),
        ),
    ),
    Block(
        key="portfolio_fit",
        label="Portfolio Fit",
        weight=0.13,
        rationale=(
            "This is what separates a screen from a portfolio decision. A name that "
            "scores well standalone but correlates 0.95 with an existing overweight "
            "adds no diversification and concentrates the book. Computed against the "
            "live IBKR account, not a generic benchmark."
        ),
        metrics=(
            Metric("corr_to_portfolio", "Correlation to current book", -1, 0.40,
                   description="Weekly return correlation to the existing portfolio's return stream."),
            Metric("diversification_benefit", "Marginal diversification benefit", 1, 0.35,
                   description="Reduction in portfolio volatility from a marginal allocation."),
            Metric("existing_overlap", "Existing exposure overlap", -1, 0.25,
                   description="Current weight in the book. Penalizes doubling down on held risk."),
        ),
    ),
)


def block_weight_total() -> float:
    return sum(b.weight for b in FACTOR_MODEL)


def all_metrics() -> dict[str, Metric]:
    return {m.key: m for b in FACTOR_MODEL for m in b.metrics}


# --------------------------------------------------------------------------
# Recommendation bands
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RecommendationBands:
    """
    Composite-z thresholds mapping to portfolio stance.

    Bands are set on the *z-score of the composite*, not on raw percentile, so
    that a uniformly mediocre universe does not mechanically produce a top
    decile of Overweights. If nothing is good, nothing gets overweighted.
    """

    overweight_z: float = 0.50
    underweight_z: float = -0.50

    #: A name must clear this many populated blocks to receive a non-neutral
    #: call. Thin data => Market Weight by default, never a conviction call.
    min_populated_blocks: int = 5


BANDS = RecommendationBands()

OVERWEIGHT = "OVERWEIGHT"
MARKET_WEIGHT = "MARKET WEIGHT"
UNDERWEIGHT = "UNDERWEIGHT"


# --------------------------------------------------------------------------
# Hard risk gates -- applied AFTER scoring, can only downgrade, never upgrade
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskGates:
    """
    Overrides that cap the recommendation regardless of composite score.

    Rationale: a purely additive factor score can be dragged into Overweight by
    one extreme block (typically momentum). These gates encode the constraints
    a risk manager would impose on top of the model.
    """

    #: Falling knife: below long-term trend AND negative long-horizon momentum.
    #: Cannot be Overweight no matter how cheap or high-carry it looks.
    trend_gate_enabled: bool = True

    #: High drawdown + high beta => caps at Market Weight. The combination
    #: means the name will amplify exactly the risk the book already carries.
    max_drawdown_limit: float = -0.45
    beta_limit: float = 1.30

    #: Redundancy gate: highly correlated to the existing book AND already a
    #: meaningful position => caps at Market Weight to prevent concentration.
    corr_limit: float = 0.85
    existing_weight_limit: float = 0.05

    #: Liquidity failure forces Underweight -- if it cannot be traded at size,
    #: it is not an actionable overweight regardless of its factor profile.
    liquidity_gate_enabled: bool = True

    #: Volatility ceiling for an Overweight. Two independent reasons:
    #: (1) Estimation error. Every ratio in this model (Sharpe, Sortino, beta,
    #:     alpha) is estimated from ~52 weekly observations. The standard error
    #:     of a Sharpe estimate scales with volatility; above ~60% annualized
    #:     the confidence interval is wide enough that the point estimate is
    #:     not a reliable ranking input.
    #: (2) Sizing degeneracy. Inverse-vol sizing shrinks a 76%-vol name to a
    #:     ~1.4% weight. A "high conviction" call that can only be expressed in
    #:     1.4% of the book is not meaningfully an overweight.
    #: Set to None to disable.
    max_volatility_for_overweight: float | None = 0.60

    #: Redundancy ceiling. Two instruments correlating above this threshold are
    #: the same exposure in different wrappers (e.g. IWM vs VTWO, both Russell
    #: 2000). Overweighting both double-counts a single bet, so only the
    #: higher-scoring one may hold an Overweight.
    duplicate_corr_limit: float = 0.95


GATES = RiskGates()


# --------------------------------------------------------------------------
# Position sizing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SizingParams:
    """
    Translates a recommendation into an indicative portfolio weight using
    inverse-volatility (risk-parity style) scaling tilted by conviction.
    """

    #: Annualized volatility each position is scaled toward.
    target_position_vol: float = 0.20

    #: Base weight for a Market Weight name before vol scaling.
    base_weight: float = 0.030

    #: Conviction multipliers applied to the base weight.
    overweight_multiplier: float = 1.75
    market_weight_multiplier: float = 1.00
    underweight_multiplier: float = 0.35

    #: Hard caps on any single indicative weight.
    min_weight: float = 0.005
    max_weight: float = 0.080


SIZING = SizingParams()
