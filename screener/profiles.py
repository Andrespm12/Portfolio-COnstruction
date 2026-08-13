"""
Risk profiles: Conservative / Moderate / Aggressive.

What a profile changes
----------------------
A profile is not a cosmetic label on the same ranking. It rewires four
independent parts of the model, and each one changes the output on its own:

1. **Block weights** -- what the composite score rewards.
2. **Recommendation bands** -- how much score is required to earn an
   Overweight, and how little to earn an Underweight. These are deliberately
   asymmetric per profile.
3. **Risk gates** -- the hard ceilings applied after scoring, which can only
   downgrade a call.
4. **Sizing and eligibility** -- target position volatility, weight caps, and
   the minimum liquidity a name must clear to be scored at all.

Standalone by construction
--------------------------
These profiles score a security **on its own merits**. The Portfolio Fit block
is removed from the model entirely rather than zero-weighted, and no
account data is read.

Removing it is not the same as passing an empty book: with no positions,
:func:`screener.portfolio.compute_portfolio_fit` still returns
``existing_overlap = 0.0`` -- a real number, identical for every name. That
constant would be z-scored across the cross-section and the block would count
as *populated* in the band logic, so an empty account would quietly influence
the composite. Dropping the block is the only way to get a genuinely
independent screen.

Calibration
-----------
The numbers below are judgement calls, chosen so that each profile is
internally coherent rather than tuned to any backtest. The direction of every
difference is defensible from the block rationales in :mod:`screener.config`:
momentum is the block that loads into high-beta melt-ups, so it falls as risk
tolerance falls; the volatility/drawdown block is the brake, so it rises.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .config import (
    FACTOR_MODEL, Block, EligibilityRules, RecommendationBands, RiskGates,
    SizingParams,
)

#: The block dropped in every profile. See the module docstring.
EXCLUDED_BLOCK = "portfolio_fit"


@dataclass(frozen=True)
class RiskProfile:
    """A complete, self-consistent parameterization of the model."""

    key: str
    label: str
    summary: str
    #: Weights over the six standalone blocks. Must sum to 1.0.
    block_weights: Mapping[str, float]
    bands: RecommendationBands
    gates: RiskGates
    sizing: SizingParams
    eligibility: EligibilityRules
    #: Dollar size of the position the liquidity block assumes. Drives
    #: ``days_to_liquidate``; it is a sizing assumption, not account data.
    default_position_usd: float

    def model(self) -> tuple[Block, ...]:
        """The factor model for this profile, with Portfolio Fit removed."""
        blocks = tuple(b for b in FACTOR_MODEL if b.key != EXCLUDED_BLOCK)
        missing = {b.key for b in blocks} - set(self.block_weights)
        if missing:
            raise ValueError(f"{self.key}: no weight declared for {sorted(missing)}")

        total = sum(self.block_weights[b.key] for b in blocks)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{self.key}: block weights sum to {total}, not 1.0")

        return tuple(replace(b, weight=self.block_weights[b.key]) for b in blocks)

    def describe(self) -> str:
        lines = [f"{self.label} — {self.summary}", ""]
        lines.append("Pesos por bloque")
        for block in self.model():
            lines.append(f"  {block.weight:6.0%}  {block.label}")
        lines += [
            "",
            "Umbrales de recomendación",
            f"  Overweight  z >= {self.bands.overweight_z:+.2f}",
            f"  Underweight z <= {self.bands.underweight_z:+.2f}",
            "",
            "Techos de riesgo",
            f"  Volatilidad máxima para un Overweight   "
            f"{_pct(self.gates.max_volatility_for_overweight)}",
            f"  Beta máxima antes de degradar            "
            f"{self.gates.beta_limit:.2f}",
            f"  Drawdown que activa el gate              "
            f"{self.gates.max_drawdown_limit:.0%}",
            "",
            "Dimensionamiento",
            f"  Volatilidad objetivo por posición        "
            f"{self.sizing.target_position_vol:.0%}",
            f"  Peso máximo por nombre                   "
            f"{self.sizing.max_weight:.1%}",
            "",
            "Elegibilidad",
            f"  Precio mínimo                            "
            f"${self.eligibility.min_price:,.0f}",
            f"  Volumen diario mínimo                    "
            f"${self.eligibility.min_adv_usd / 1e6:,.0f}MM",
        ]
        return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "sin techo" if value is None else f"{value:.0%}"


# --------------------------------------------------------------------------
# The three profiles
# --------------------------------------------------------------------------

CONSERVADOR_DEFENSIVO = RiskProfile(
    key="conservador_defensivo",
    label="Conservador Defensivo",
    summary="mandato de mínima volatilidad; la tendencia casi no puntúa",
    # The most defensive mandate in CCI's Investment Procedure. Momentum falls
    # to a third of the Conservative weight and the volatility/drawdown block
    # becomes a third of the entire model: this book is judged on the path, not
    # the endpoint. Liquidity is the second-largest block because an inability
    # to exit is itself a drawdown for a mandate defined by drawdown.
    block_weights={
        "momentum": 0.08,
        "risk_adjusted": 0.22,
        "risk": 0.34,
        "market_sensitivity": 0.08,
        "liquidity": 0.16,
        "valuation_carry": 0.12,
    },
    # The most asymmetric bands in the set: an Overweight needs a full standard
    # deviation of edge, while a fifth of one is enough to step aside.
    bands=RecommendationBands(
        overweight_z=1.00, underweight_z=-0.20, min_populated_blocks=5,
    ),
    gates=RiskGates(
        trend_gate_enabled=True,
        max_drawdown_limit=-0.20,
        beta_limit=0.85,
        corr_limit=1.01,             # inert: no book is read
        existing_weight_limit=1.01,  # inert: no book is read
        liquidity_gate_enabled=True,
        max_volatility_for_overweight=0.22,
        duplicate_corr_limit=0.88,
    ),
    sizing=SizingParams(
        target_position_vol=0.08, base_weight=0.020,
        overweight_multiplier=1.35, market_weight_multiplier=1.00,
        underweight_multiplier=0.25,
        min_weight=0.005, max_weight=0.035,
    ),
    eligibility=EligibilityRules(
        min_price=10.0, min_adv_usd=100_000_000.0,
        max_participation_rate=0.10, max_days_to_liquidate=1.5,
    ),
    default_position_usd=250_000.0,
)

CONSERVADOR = RiskProfile(
    key="conservador",
    label="Conservador",
    summary="preservación de capital; penaliza volatilidad y drawdown por encima de todo",
    # Volatility & drawdown becomes the single largest block, and momentum is
    # roughly halved: momentum is precisely the factor that loads into
    # high-beta melt-ups, which is the exposure this mandate cannot carry.
    # Liquidity rises because an investor who cannot tolerate drawdown needs
    # to be able to exit before one develops.
    block_weights={
        "momentum": 0.12,
        "risk_adjusted": 0.24,
        "risk": 0.28,
        "market_sensitivity": 0.10,
        "liquidity": 0.14,
        "valuation_carry": 0.12,
    },
    # Asymmetric on purpose: an Overweight must be clearly earned (+0.80),
    # while mild underperformance is enough to step aside (-0.30).
    bands=RecommendationBands(
        overweight_z=0.80, underweight_z=-0.30, min_populated_blocks=5,
    ),
    gates=RiskGates(
        trend_gate_enabled=True,
        max_drawdown_limit=-0.30,
        beta_limit=1.00,
        corr_limit=1.01,            # inert: no book is read
        existing_weight_limit=1.01,  # inert: no book is read
        liquidity_gate_enabled=True,
        max_volatility_for_overweight=0.30,
        duplicate_corr_limit=0.90,
    ),
    sizing=SizingParams(
        target_position_vol=0.12, base_weight=0.025,
        overweight_multiplier=1.50, market_weight_multiplier=1.00,
        underweight_multiplier=0.30,
        min_weight=0.005, max_weight=0.050,
    ),
    eligibility=EligibilityRules(
        min_price=10.0, min_adv_usd=50_000_000.0,
        max_participation_rate=0.15, max_days_to_liquidate=2.0,
    ),
    default_position_usd=250_000.0,
)

MODERADO = RiskProfile(
    key="moderado",
    label="Moderado",
    summary="equilibrio entre tendencia y riesgo; el modelo base del repo",
    # The weights declared in config.py, renormalized over the six standalone
    # blocks after Portfolio Fit is removed.
    block_weights={
        "momentum": 0.25,
        "risk_adjusted": 0.21,
        "risk": 0.17,
        "market_sensitivity": 0.14,
        "liquidity": 0.11,
        "valuation_carry": 0.12,
    },
    bands=RecommendationBands(
        overweight_z=0.50, underweight_z=-0.50, min_populated_blocks=5,
    ),
    gates=RiskGates(
        trend_gate_enabled=True,
        max_drawdown_limit=-0.45,
        beta_limit=1.30,
        corr_limit=1.01,
        existing_weight_limit=1.01,
        liquidity_gate_enabled=True,
        max_volatility_for_overweight=0.60,
        duplicate_corr_limit=0.95,
    ),
    sizing=SizingParams(
        target_position_vol=0.20, base_weight=0.030,
        overweight_multiplier=1.75, market_weight_multiplier=1.00,
        underweight_multiplier=0.35,
        min_weight=0.005, max_weight=0.080,
    ),
    eligibility=EligibilityRules(
        min_price=5.0, min_adv_usd=20_000_000.0,
        max_participation_rate=0.20, max_days_to_liquidate=3.0,
    ),
    default_position_usd=500_000.0,
)

AGRESIVO = RiskProfile(
    key="agresivo",
    label="Agresivo",
    summary="busca retorno; premia tendencia y alfa idiosincrático, tolera volatilidad",
    # Momentum roughly triples relative to Conservative and alpha quality
    # doubles: if the mandate is to take single-name risk, that risk has to be
    # paid for in idiosyncratic return, not in beta. The risk block drops but
    # is never zeroed -- it remains the only brake in the model.
    block_weights={
        "momentum": 0.36,
        "risk_adjusted": 0.18,
        "risk": 0.08,
        "market_sensitivity": 0.20,
        "liquidity": 0.09,
        "valuation_carry": 0.09,
    },
    # Mirror image of Conservative: easier to earn an Overweight (+0.30),
    # harder to be pushed to Underweight (-0.70).
    bands=RecommendationBands(
        overweight_z=0.30, underweight_z=-0.70, min_populated_blocks=5,
    ),
    gates=RiskGates(
        trend_gate_enabled=True,
        max_drawdown_limit=-0.60,
        beta_limit=1.80,
        corr_limit=1.01,
        existing_weight_limit=1.01,
        liquidity_gate_enabled=True,
        # Still capped. Above ~90% annualized the Sharpe/beta/alpha estimates
        # from ~52 weekly bars are too noisy to rank on, regardless of mandate.
        max_volatility_for_overweight=0.90,
        duplicate_corr_limit=0.97,
    ),
    sizing=SizingParams(
        target_position_vol=0.30, base_weight=0.040,
        overweight_multiplier=2.00, market_weight_multiplier=1.00,
        underweight_multiplier=0.40,
        min_weight=0.005, max_weight=0.120,
    ),
    eligibility=EligibilityRules(
        min_price=5.0, min_adv_usd=10_000_000.0,
        max_participation_rate=0.25, max_days_to_liquidate=5.0,
    ),
    default_position_usd=500_000.0,
)

PROFILES: dict[str, RiskProfile] = {
    p.key: p for p in (CONSERVADOR_DEFENSIVO, CONSERVADOR, MODERADO, AGRESIVO)
}

#: CCI's four Mercado Internacional strategies, in the exact spelling their
#: Google Sheet tabs and REGULACIONES dictionary use, mapped to the profile
#: that expresses the same risk appetite.
CCI_STRATEGIES: dict[str, str] = {
    "Conservador_Defensivo": "conservador_defensivo",
    "Conservador": "conservador",
    "Moderado": "moderado",
    "Agresivo": "agresivo",
}


def profile_for_strategy(strategy: str) -> RiskProfile:
    """Resolve a CCI strategy name to its screening profile."""
    key = CCI_STRATEGIES.get(strategy.strip())
    if key is None:
        raise KeyError(
            f"Estrategia CCI desconocida: {strategy!r}. "
            f"Opciones: {sorted(CCI_STRATEGIES)}"
        )
    return PROFILES[key]


def get_profile(name: str) -> RiskProfile:
    """Look up a profile by key or label, accent- and case-insensitively."""
    needle = (name or "").strip().lower()
    needle = (needle.replace("á", "a").replace("é", "e").replace("í", "i")
                    .replace("ó", "o").replace("ú", "u"))
    if needle in PROFILES:
        return PROFILES[needle]
    for profile in PROFILES.values():
        if needle == profile.label.lower():
            return profile
    raise KeyError(f"Perfil desconocido: {name!r}. Opciones: {sorted(PROFILES)}")


def apply_profile(profile: RiskProfile) -> RiskProfile:
    """
    Install a profile process-wide.

    Rebinds through :mod:`screener.tuning` rather than assigning to
    :mod:`screener.config`, because ``scoring``, ``report`` and ``universe``
    bind these names at import time -- assigning to config alone would leave
    the scorer running the previous profile while the run still produced
    numbers.
    """
    from .tuning import rebind

    for attribute, value in (
        ("FACTOR_MODEL", profile.model()),
        ("BANDS", profile.bands),
        ("GATES", profile.gates),
        ("SIZING", profile.sizing),
        ("ELIGIBILITY", profile.eligibility),
    ):
        if not rebind(attribute, value):
            raise RuntimeError(
                f"{attribute} is not bound in any imported screener module; "
                "the profile would be silently ignored."
            )
    return profile
