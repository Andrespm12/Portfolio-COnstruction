"""
Cross-sectional scoring: raw metrics -> z-scores -> composite -> recommendation.

Method
------
1. **Winsorize** each metric at the 5th/95th percentile of the cross-section.
2. **Standardize** to z-scores, either against the whole universe or against
   the asset-type peer group (ETF vs single stock) for metrics flagged
   ``peer_relative``.
3. **Orient** by multiplying by the metric's direction so that a positive z
   always means "better".
4. **Aggregate** within block (metric weights), then across blocks (block
   weights), re-normalizing whenever inputs are missing rather than imputing
   zeros -- imputing a zero silently asserts "average", which is a real
   claim about a name we have no data for.
5. **Standardize the composite** and map to Overweight / Market Weight /
   Underweight bands, then apply hard risk gates that can only downgrade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import (
    BANDS, FACTOR_MODEL, GATES, MARKET_WEIGHT, OVERWEIGHT, SIZING, UNDERWEIGHT,
    WINSOR_LOWER_PCT, WINSOR_UPPER_PCT, Z_CLIP, all_metrics,
)


# --------------------------------------------------------------------------
# Statistical primitives
# --------------------------------------------------------------------------

def winsorize(values: np.ndarray, lower_pct: float = WINSOR_LOWER_PCT,
              upper_pct: float = WINSOR_UPPER_PCT) -> np.ndarray:
    """Clip to percentile bounds, ignoring NaNs when computing them."""
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return values
    lo = np.percentile(finite, lower_pct)
    hi = np.percentile(finite, upper_pct)
    return np.clip(values, lo, hi)


def zscore(values: np.ndarray) -> np.ndarray:
    """
    Standardize, preserving NaN. Falls back to a rank-based transform when the
    distribution is degenerate (zero variance).
    """
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < 3:
        return out
    sample = values[mask]
    sd = np.std(sample, ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return out
    out[mask] = (sample - np.mean(sample)) / sd
    return np.clip(out, -Z_CLIP, Z_CLIP)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Percentile rank 0-100, NaN-preserving."""
    out = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    n = int(mask.sum())
    if n == 0:
        return out
    sample = values[mask]
    order = sample.argsort().argsort().astype(float)
    out[mask] = (order / max(n - 1, 1)) * 100.0
    return out


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

@dataclass
class ScoredInstrument:
    ticker: str
    name: str
    asset_type: str
    indices: list[str]
    sector: str | None
    raw_metrics: dict[str, float | None]
    metric_z: dict[str, float] = field(default_factory=dict)
    block_scores: dict[str, float] = field(default_factory=dict)
    block_coverage: dict[str, float] = field(default_factory=dict)
    composite_raw: float = float("nan")
    composite_z: float = float("nan")
    score_0_100: float = float("nan")
    recommendation: str = MARKET_WEIGHT
    pre_gate_recommendation: str = MARKET_WEIGHT
    gates_triggered: list[str] = field(default_factory=list)
    #: Tickers in the universe whose return series is effectively identical to
    #: this one (correlation above the duplicate threshold).
    duplicates: list[str] = field(default_factory=list)
    indicative_weight: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    eligible: bool = True
    ineligibility_reasons: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Scoring engine
# --------------------------------------------------------------------------

def _metric_matrix(rows: list[ScoredInstrument], key: str) -> np.ndarray:
    return np.asarray(
        [r.raw_metrics.get(key) if r.raw_metrics.get(key) is not None else np.nan for r in rows],
        dtype=float,
    )


def score_universe(rows: list[ScoredInstrument]) -> list[ScoredInstrument]:
    """
    Score every instrument in the cross-section. Mutates and returns `rows`.

    Requires at least 3 names; z-scores are meaningless on a smaller sample.
    """
    if len(rows) < 3:
        raise ValueError(
            f"Cross-sectional scoring needs at least 3 instruments, got {len(rows)}."
        )

    metric_defs = all_metrics()
    asset_types = np.asarray([r.asset_type for r in rows])

    # ---- Step 1-3: winsorize, standardize, orient -------------------------
    for key, meta in metric_defs.items():
        raw = _metric_matrix(rows, key)
        z = np.full(raw.shape, np.nan, dtype=float)

        if meta.peer_relative:
            # Standardize inside each asset-type cohort. ETFs and single stocks
            # have structurally different volatility and beta distributions;
            # pooling them would make every ETF look like a risk-management win
            # purely by construction.
            for cohort in np.unique(asset_types):
                idx = np.where(asset_types == cohort)[0]
                if idx.size >= 3:
                    z[idx] = zscore(winsorize(raw[idx]))
                else:
                    # Cohort too small to standardize within: fall back to the
                    # full cross-section rather than dropping the metric.
                    pass
            unresolved = np.isnan(z) & np.isfinite(raw)
            if unresolved.any():
                fallback = zscore(winsorize(raw))
                z[unresolved] = fallback[unresolved]
        else:
            z = zscore(winsorize(raw))

        z = z * meta.direction
        for i, row in enumerate(rows):
            if np.isfinite(z[i]):
                row.metric_z[key] = float(z[i])

    # ---- Step 4: aggregate to blocks, then composite ----------------------
    for row in rows:
        composite_num, composite_den = 0.0, 0.0

        for block in FACTOR_MODEL:
            weights = block.normalized_metric_weights()
            num, den = 0.0, 0.0
            for metric in block.metrics:
                z = row.metric_z.get(metric.key)
                if z is None:
                    continue
                num += weights[metric.key] * z
                den += weights[metric.key]

            coverage = den
            row.block_coverage[block.key] = coverage
            if den > 0:
                block_score = num / den  # re-normalize over available metrics
                row.block_scores[block.key] = block_score
                composite_num += block.weight * block_score
                composite_den += block.weight

        row.composite_raw = composite_num / composite_den if composite_den > 0 else float("nan")

    # ---- Step 5: standardize composite, rank, recommend ------------------
    composites = np.asarray([r.composite_raw for r in rows], dtype=float)
    comp_z = zscore(composites)
    comp_pct = percentile_rank(composites)

    for i, row in enumerate(rows):
        row.composite_z = float(comp_z[i]) if np.isfinite(comp_z[i]) else float("nan")
        row.score_0_100 = float(comp_pct[i]) if np.isfinite(comp_pct[i]) else float("nan")
        row.pre_gate_recommendation = _band(row)
        row.recommendation = row.pre_gate_recommendation

    return rows


def _band(row: ScoredInstrument) -> str:
    """Map composite z to a stance, defaulting to neutral on thin coverage."""
    populated = sum(1 for v in row.block_coverage.values() if v > 0)
    if populated < BANDS.min_populated_blocks or not np.isfinite(row.composite_z):
        return MARKET_WEIGHT
    if row.composite_z >= BANDS.overweight_z:
        return OVERWEIGHT
    if row.composite_z <= BANDS.underweight_z:
        return UNDERWEIGHT
    return MARKET_WEIGHT


# --------------------------------------------------------------------------
# Risk gates -- downgrade only
# --------------------------------------------------------------------------

_RANK = {UNDERWEIGHT: 0, MARKET_WEIGHT: 1, OVERWEIGHT: 2}
_UNRANK = {v: k for k, v in _RANK.items()}


def _cap(current: str, ceiling: str) -> str:
    return _UNRANK[min(_RANK[current], _RANK[ceiling])]


def apply_risk_gates(rows: list[ScoredInstrument]) -> list[ScoredInstrument]:
    """
    Apply hard overrides after scoring.

    Gates can only lower a recommendation. A high composite score is a
    necessary but not sufficient condition for an Overweight -- these encode
    the constraints a risk manager imposes on top of the factor model.
    """
    for row in rows:
        m = row.raw_metrics
        gates: list[str] = []

        if not row.eligible:
            row.recommendation = UNDERWEIGHT
            row.gates_triggered = ["INELIGIBLE: " + "; ".join(row.ineligibility_reasons)]
            continue

        # Absolute quality floor. Everything upstream of this point is
        # cross-sectional: the composite is standardized, so its top third is
        # the top third of whatever was screened, in a bull market or a bear
        # one. This is the only test in the model that asks whether a name is
        # any good on its own terms rather than relative to its peers, and it
        # is the reason a falling universe no longer yields a slate of
        # Overweights.
        floors: list[str] = []
        mom_floor = GATES.min_momentum_for_overweight
        mom_12_1 = m.get("mom_12_1")
        if mom_floor is not None and mom_12_1 is not None and mom_12_1 < mom_floor:
            floors.append(f"momentum 12M-1M {mom_12_1:+.1%} bajo el piso de {mom_floor:+.1%}")

        sharpe_floor = GATES.min_sharpe_for_overweight
        sharpe = m.get("sharpe_1y")
        if sharpe_floor is not None and sharpe is not None and sharpe < sharpe_floor:
            floors.append(f"Sharpe 1A {sharpe:.2f} bajo el piso de {sharpe_floor:.2f}")

        if floors:
            row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
            gates.append(
                "CALIDAD ABSOLUTA: " + "; ".join(floors)
                + " -- puede seguir siendo lo mejor disponible, pero no es una "
                "instrucción de tener más que el benchmark"
            )

        # Falling knife: below the 40-week trend AND negative 12M-1M momentum.
        if GATES.trend_gate_enabled:
            above_ma, mom = m.get("above_40w_ma"), m.get("mom_12_1")
            if above_ma is not None and mom is not None and above_ma < 0 and mom < 0:
                row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
                gates.append("TREND: below 40W MA with negative 12M-1M momentum")

        # Amplifier: deep drawdown history combined with high market beta.
        mdd, beta = m.get("max_drawdown"), m.get("beta_1y")
        if mdd is not None and beta is not None:
            if -mdd <= GATES.max_drawdown_limit and beta > GATES.beta_limit:
                row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
                gates.append(
                    f"RISK: {mdd:.0%} max drawdown with {beta:.2f} beta -- "
                    "drawdown depth and market sensitivity compound"
                )

        # Redundancy: already held AND highly correlated to the current book.
        corr, existing = m.get("corr_to_portfolio"), m.get("existing_overlap")
        if corr is not None and existing is not None:
            if corr > GATES.corr_limit and existing > GATES.existing_weight_limit:
                row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
                gates.append(
                    f"CONCENTRATION: {corr:.2f} correlation to book at {existing:.1%} existing weight"
                )

        # Capacity: cannot be exited inside the liquidation-horizon budget.
        if GATES.liquidity_gate_enabled:
            dtl = m.get("days_to_liquidate")
            from .config import ELIGIBILITY
            if dtl is not None and dtl > ELIGIBILITY.max_days_to_liquidate:
                row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
                gates.append(f"CAPACITY: {dtl:.1f} days to liquidate a target position")

        # Estimation reliability + sizing degeneracy at extreme volatility.
        ceiling = GATES.max_volatility_for_overweight
        vol = m.get("volatility_1y")
        if ceiling is not None and vol is not None and vol > ceiling:
            row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
            gates.append(
                f"VOLATILITY: {vol:.0%} annualized exceeds the {ceiling:.0%} overweight "
                "ceiling -- factor estimates unreliable and inverse-vol sizing "
                "collapses the position to an immaterial weight"
            )

        row.gates_triggered = gates

    _apply_duplicate_gate(rows)
    return rows


def _apply_duplicate_gate(rows: list[ScoredInstrument]) -> None:
    """
    Prevent overweighting the same exposure twice.

    Within each set of mutually near-identical instruments only the
    highest-scoring member may keep an Overweight; the rest are capped at
    Market Weight. Without this, a screen that contains both IWM and VTWO
    (correlation ~0.997, same underlying index) would happily recommend
    doubling a single small-cap bet.
    """
    by_ticker = {r.ticker: r for r in rows}
    for row in rows:
        if not row.duplicates or row.recommendation != OVERWEIGHT:
            continue
        peers = [by_ticker[t] for t in row.duplicates if t in by_ticker]
        better = [p for p in peers
                  if (p.composite_z if np.isfinite(p.composite_z) else -99)
                  > (row.composite_z if np.isfinite(row.composite_z) else -99)]
        if better:
            best = max(better, key=lambda p: p.composite_z)
            row.recommendation = _cap(row.recommendation, MARKET_WEIGHT)
            row.gates_triggered.append(
                f"REDUNDANT: duplicate exposure to {best.ticker} "
                f"(correlation > {GATES.duplicate_corr_limit:.2f}); "
                f"{best.ticker} scores higher and carries the allocation"
            )


# --------------------------------------------------------------------------
# Indicative sizing
# --------------------------------------------------------------------------

def assign_indicative_weights(rows: list[ScoredInstrument]) -> list[ScoredInstrument]:
    """
    Inverse-volatility sizing tilted by conviction.

    weight = base * conviction_multiplier * (target_vol / realized_vol)

    Equal *risk* contribution rather than equal dollars: a 60%-vol
    semiconductor name and a 12%-vol utility ETF at the same dollar weight are
    not the same position. Capped, then reported as indicative only -- final
    sizing is a portfolio-construction decision, not a screening output.
    """
    for row in rows:
        vol = row.raw_metrics.get("volatility_1y")
        multiplier = {
            OVERWEIGHT: SIZING.overweight_multiplier,
            MARKET_WEIGHT: SIZING.market_weight_multiplier,
            UNDERWEIGHT: SIZING.underweight_multiplier,
        }[row.recommendation]

        weight = SIZING.base_weight * multiplier
        if vol and vol > 0:
            weight *= SIZING.target_position_vol / vol

        row.indicative_weight = float(
            min(SIZING.max_weight, max(SIZING.min_weight, weight))
        )
    return rows


def detect_duplicates(rows: list[ScoredInstrument],
                      returns_by_ticker: dict[str, np.ndarray]) -> None:
    """
    Flag pairs of instruments whose return series are effectively identical.

    Populates ``ScoredInstrument.duplicates`` in place. Runs before the gates
    so :func:`_apply_duplicate_gate` can act on it.
    """
    from .metrics import align_returns

    tickers = [r.ticker for r in rows if returns_by_ticker.get(r.ticker) is not None
               and returns_by_ticker[r.ticker].size >= 12]
    by_ticker = {r.ticker: r for r in rows}

    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            ra, rb = align_returns(returns_by_ticker[a], returns_by_ticker[b])
            if ra.size < 12 or np.std(ra) == 0 or np.std(rb) == 0:
                continue
            c = np.corrcoef(ra, rb)[0, 1]
            if np.isfinite(c) and c > GATES.duplicate_corr_limit:
                by_ticker[a].duplicates.append(b)
                by_ticker[b].duplicates.append(a)


def run_scoring_pipeline(rows: list[ScoredInstrument]) -> list[ScoredInstrument]:
    """Score, gate and size in the required order."""
    score_universe(rows)
    apply_risk_gates(rows)
    assign_indicative_weights(rows)
    return sorted(
        rows,
        key=lambda r: (r.composite_z if np.isfinite(r.composite_z) else -99),
        reverse=True,
    )
