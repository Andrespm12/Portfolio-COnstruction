"""
Measurements on the model's own output, for model-risk review.

Nothing here changes a recommendation, a view or a weight. These are the two
numbers an external reviewer asked for that the model was not reporting, and
both are cheap enough to run on every screen:

**Block correlation.** The factor model declares six blocks and spends a block
weight on each, which asserts that each one carries information the others do
not. If two blocks correlate at 0.90 across the cross-section, their weights add
up to a single bet made twice, and the composite is less diversified than its
weight table claims. This is a *measurement of the claim*, not a fix -- the fix
(orthogonalizing the blocks, or folding two of them together) is a real change
to the factor model and should only be made once the measurement says it is
needed, on a universe that resembles the one actually being screened.

**View saturation.** ``Q`` is clipped at +/-5% to respect the calibration in
CCI's technical document. The clip is a safety rail, but if most views land on
it, the rail has become the signal: several names that the screener ranked very
differently arrive at the optimizer with an identical expected return, and the
ranking that justified the whole exercise is discarded at the last step. Two
views at the cap is a rail doing its job; six is a calibration problem, and the
answer is to lower the information coefficient, not to raise the cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .config import FACTOR_MODEL
from .scoring import ScoredInstrument

#: |correlation| at or above which two blocks are reported as overlapping.
#: Not a hard threshold in any statistical sense -- it is the level at which a
#: reviewer should be told, because two blocks this correlated are close to
#: being one input entered twice.
REDUNDANT_CORR = 0.70

#: Minimum names required before a block correlation is worth reporting. Below
#: this the estimate is noise, and reporting it invites acting on noise.
MIN_OBSERVATIONS = 12

#: Share of the cross-section a block must be scored on to enter the matrix.
#: A block populated on a handful of names would, under listwise deletion, drag
#: the sample for *every other* pair down to that handful -- one thin block
#: would silence the whole diagnostic.
MIN_BLOCK_COVERAGE = 0.50

#: Standard deviations at or below this are treated as zero variance. Not a
#: comparison against 0.0: ``np.std`` of fifty identical values returns ~1.7e-16,
#: not zero, so an exact test would classify a constant block as live and then
#: report its floating-point noise as a correlation.
FLAT_SD = 1e-10


# --------------------------------------------------------------------------
# 1. Do the six blocks actually measure six things?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockCorrelation:
    """Cross-sectional correlation between block scores."""

    #: Blocks actually measured, in factor-model order. Not necessarily every
    #: declared block -- see :attr:`excluded_blocks`.
    blocks: tuple[str, ...]
    weights: tuple[float, ...]
    matrix: np.ndarray
    n_observations: int
    #: Participation ratio of the correlation matrix eigenvalues. Equals the
    #: measured block count when the blocks are mutually independent and falls
    #: toward 1 as they collapse onto a single common factor.
    effective_factors: float
    #: ``(block_a, block_b, correlation)`` for every pair at or above
    #: :data:`REDUNDANT_CORR`, strongest first.
    redundant_pairs: tuple[tuple[str, str, float], ...]
    #: Combined block weight of every block appearing in a redundant pair --
    #: the share of the composite that is not as diversified as it looks.
    overlapping_weight: float
    #: ``(block, reason)`` for each declared block left out of the matrix.
    excluded_blocks: tuple[tuple[str, str], ...] = ()
    #: How many blocks the factor model declares, measured or not.
    n_declared: int = 0

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def reliable(self) -> bool:
        return self.n_observations >= MIN_OBSERVATIONS and self.n_blocks >= 2


def block_correlation(scored: Sequence[ScoredInstrument], *,
                      threshold: float = REDUNDANT_CORR) -> BlockCorrelation:
    """
    Correlate every pair of block scores across the scored cross-section.

    Two exclusions happen before the matrix is built, and both are reported
    rather than applied silently:

    *Thinly populated blocks.* Rows are dropped listwise -- only names scored on
    every measured block enter the sample -- because pairwise deletion computes
    each cell on a different subsample and can return a matrix that is not even
    positive semi-definite, which would make the eigenvalue summary below
    meaningless rather than merely imprecise. The cost of listwise deletion is
    that one sparsely populated block takes the whole sample down with it: in
    standalone mode ``portfolio_fit`` is never scored, and requiring it would
    leave zero usable names. So a block scored on less than
    :data:`MIN_BLOCK_COVERAGE` of the cross-section is dropped first.

    *Flat blocks.* A block with no cross-sectional variance has no ranking to
    correlate. It is excluded outright instead of being carried as a row of
    NaN, so that the returned matrix is either fully populated or explicitly
    unreliable -- never a mix a caller has to remember to check.
    """
    declared = tuple(b.key for b in FACTOR_MODEL)
    weights = {b.key: b.weight for b in FACTOR_MODEL}
    excluded: list[tuple[str, str]] = []

    n_scored = len(scored)
    populated = {
        k: sum(1 for r in scored if np.isfinite(r.block_scores.get(k, np.nan)))
        for k in declared
    }
    keys = []
    for k in declared:
        if n_scored and populated[k] >= MIN_BLOCK_COVERAGE * n_scored:
            keys.append(k)
        else:
            excluded.append((k, f"puntuado en {populated[k]} de {n_scored} nombres"))

    rows = [
        [row.block_scores[k] for k in keys]
        for row in scored
        if all(np.isfinite(row.block_scores.get(k, np.nan)) for k in keys)
    ]
    observed = (np.asarray(rows, dtype=float) if rows
                else np.empty((0, len(keys))))

    def _empty(reason_keys: list[str]) -> BlockCorrelation:
        return BlockCorrelation(
            blocks=tuple(reason_keys),
            weights=tuple(weights[k] for k in reason_keys),
            matrix=np.full((len(reason_keys), len(reason_keys)), np.nan),
            n_observations=int(observed.shape[0]),
            effective_factors=float("nan"),
            redundant_pairs=(),
            overlapping_weight=0.0,
            excluded_blocks=tuple(excluded),
            n_declared=len(declared),
        )

    if observed.shape[0] < 3 or len(keys) < 2:
        return _empty(keys)

    sd = observed.std(axis=0, ddof=1)
    live = sd > FLAT_SD
    for k, alive in zip(keys, live):
        if not alive:
            excluded.append((k, "sin variación transversal"))
    keys = [k for k, alive in zip(keys, live) if alive]
    observed = observed[:, live]

    if len(keys) < 2:
        return _empty(keys)

    matrix = np.corrcoef(observed, rowvar=False)

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c = matrix[i, j]
            if np.isfinite(c) and abs(c) >= threshold:
                pairs.append((keys[i], keys[j], float(c)))
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)

    involved = {k for a, b, _ in pairs for k in (a, b)}

    return BlockCorrelation(
        blocks=tuple(keys),
        weights=tuple(weights[k] for k in keys),
        matrix=matrix,
        n_observations=int(observed.shape[0]),
        effective_factors=_participation_ratio(matrix),
        redundant_pairs=tuple(pairs),
        overlapping_weight=float(sum(weights[k] for k in involved)),
        excluded_blocks=tuple(excluded),
        n_declared=len(declared),
    )


def _participation_ratio(corr: np.ndarray) -> float:
    """
    Effective number of independent dimensions in a correlation matrix.

    ``(sum lambda)^2 / sum lambda^2``. The eigenvalues of a k x k correlation
    matrix sum to k, so this is ``k^2 / sum lambda^2``: it returns k when every
    eigenvalue is 1 (mutually uncorrelated blocks) and 1 when a single
    eigenvalue carries everything (all blocks are the same signal).
    """
    if corr.size == 0 or corr.shape[0] < 2 or not np.isfinite(corr).all():
        return float("nan")
    eig = np.linalg.eigvalsh(corr)
    eig = np.clip(eig, 0.0, None)  # tiny negatives are floating-point noise
    denom = float(np.sum(eig ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(eig) ** 2 / denom)


def format_block_correlation(diag: BlockCorrelation) -> str:
    """Spanish console summary, written to be pasted into a review note."""
    dropped = "".join(f"\n  {k}: excluido — {why}" for k, why in diag.excluded_blocks)

    if not diag.reliable:
        return (
            f"Correlación entre bloques: {diag.n_observations} nombres con los "
            f"{diag.n_blocks} bloques medibles completos; se necesitan al menos "
            f"{MIN_OBSERVATIONS} y dos bloques para que la medición signifique "
            f"algo. No se reporta.{dropped}"
        )

    width = max(len(k) for k in diag.blocks) + 2
    header = " " * width + "".join(f"{k[:7]:>9}" for k in diag.blocks)
    lines = [
        f"Correlación entre bloques ({diag.n_observations} nombres)",
        header,
    ]
    for i, key in enumerate(diag.blocks):
        cells = "".join(
            "      n/d" if not np.isfinite(diag.matrix[i, j])
            else f"{diag.matrix[i, j]:>9.2f}"
            for j in range(diag.n_blocks)
        )
        lines.append(f"{key:<{width}}{cells}")

    lines.append("")
    lines.append(
        f"Factores efectivos: {diag.effective_factors:.2f} de {diag.n_blocks} "
        "bloques medidos."
    )
    if diag.excluded_blocks:
        lines.append(
            f"({diag.n_declared - diag.n_blocks} de {diag.n_declared} bloques "
            "declarados quedaron fuera de la medición:" + dropped + ")"
        )
    if diag.redundant_pairs:
        lines.append(
            f"Pares solapados (|r| >= {REDUNDANT_CORR:.2f}) — el peso de estos "
            f"bloques suma {diag.overlapping_weight:.0%} del compuesto:"
        )
        for a, b, c in diag.redundant_pairs:
            lines.append(f"  {a} ~ {b}: {c:+.2f}")
        lines.append(
            "Interpretación: ese porcentaje del compuesto está apostando dos "
            "veces a lo mismo. No es un error del cálculo; es que la tabla de "
            "pesos promete más diversificación de la que hay."
        )
    else:
        lines.append(
            f"Ningún par de bloques supera |r| = {REDUNDANT_CORR:.2f}. Los pesos "
            "por bloque representan apuestas razonablemente distintas."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 2. How much of the ranking survives the +/-5% clip?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ViewSaturation:
    """How binding the ``max_abs_q`` cap was on this run's views."""

    n_views: int
    n_at_cap: int
    cap: float
    #: ``(etiqueta, Q, Q_sin_recorte)`` for each view sitting on the cap.
    clipped: tuple[tuple[str, float, float], ...]
    #: Largest ``|Q_bruto| / cap`` observed. 1.0 means nothing was clipped.
    max_overshoot: float

    @property
    def share_at_cap(self) -> float:
        return self.n_at_cap / self.n_views if self.n_views else 0.0

    @property
    def rank_collapse(self) -> bool:
        """
        True when the cap has erased a ranking the screener actually made.

        Two views clipped to the same number are only a problem if the model
        had distinguished them before the clip. If both would have been 5.1%,
        nothing was lost.
        """
        if self.n_at_cap < 2:
            return False
        raw = [abs(r) for _, _, r in self.clipped if np.isfinite(r)]
        if len(raw) < 2:
            return False
        # A tenth of the cap apart is the point at which the pre-clip spread is
        # wide enough that flattening it discards a distinction the model made.
        return (max(raw) - min(raw)) > 0.10 * self.cap


def view_label(view: Mapping[str, Any]) -> str:
    """``TICKER`` for an absolute view, ``LONG/SHORT`` for a relative one."""
    if view.get("tipo") == "absoluto":
        return str(view.get("activo", "?"))
    return f"{view.get('activo_long', '?')}/{view.get('activo_short', '?')}"


def view_saturation(views: Sequence[Mapping[str, Any]],
                    params: Any) -> ViewSaturation:
    """
    Count how many views landed on the ``+/-max_abs_q`` cap.

    Reads the pre-clip value from the private ``_q_bruto`` key that
    :func:`screener.black_litterman.build_views` attaches; views arriving from
    elsewhere (a hand-edited file, CCI's own generator) simply report the
    clipped value as their own raw value, which understates the overshoot
    rather than inventing one.
    """
    cap = float(getattr(params, "max_abs_q", 0.05))
    tol = cap * 1e-9

    clipped: list[tuple[str, float, float]] = []
    overshoot = 1.0
    for view in views:
        q = float(view.get("Q", 0.0))
        raw = float(view.get("_q_bruto", q))
        if np.isfinite(raw) and cap > 0:
            overshoot = max(overshoot, abs(raw) / cap)
        if abs(q) >= cap - tol:
            clipped.append((view_label(view), q, raw))

    return ViewSaturation(
        n_views=len(views),
        n_at_cap=len(clipped),
        cap=cap,
        clipped=tuple(clipped),
        max_overshoot=float(overshoot),
    )


def format_view_saturation(diag: ViewSaturation) -> str:
    """Spanish console summary."""
    if diag.n_views == 0:
        return "Saturación de views: no se generó ninguna view."

    lines = [
        f"Saturación de views: {diag.n_at_cap} de {diag.n_views} "
        f"({diag.share_at_cap:.0%}) tocan el tope de {diag.cap:.1%}."
    ]
    if not diag.clipped:
        lines.append(
            "El tope no está limitando nada: las views salen del modelo dentro "
            "de la banda por sí solas."
        )
        return "\n".join(lines)

    for label, q, raw in diag.clipped:
        if abs(raw) > diag.cap:
            lines.append(f"  {label}: {q:+.2%} (sin recorte habría sido {raw:+.2%})")
        else:
            lines.append(f"  {label}: {q:+.2%}")

    if diag.max_overshoot > 1.0:
        lines.append(
            f"El Q más extremo antes del recorte era {diag.max_overshoot:.1f}x el tope."
        )
    if diag.rank_collapse:
        lines.append(
            "ATENCIÓN: hay views recortadas que el modelo sí distinguía entre sí. "
            "Llegan al optimizador con el mismo retorno esperado, así que esa "
            "parte del ranking se pierde. La corrección es bajar el information "
            "coefficient hasta que el tope deje de morder, no subir el tope: el "
            "tope viene de la calibración del documento técnico de CCI."
        )
    elif diag.n_at_cap > diag.n_views / 2:
        lines.append(
            "Más de la mitad de las views están en el tope. Revisar el "
            "information coefficient antes de la próxima corrida."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------

def run_diagnostics(scored: Sequence[ScoredInstrument],
                    views: Sequence[Mapping[str, Any]],
                    params: Any) -> str:
    """Both measurements, formatted as one block for the notebook."""
    return "\n\n".join((
        format_block_correlation(block_correlation(scored)),
        format_view_saturation(view_saturation(views, params)),
    ))
