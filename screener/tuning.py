"""
Runtime overrides for the factor model.

Why this is not just ``config.FACTOR_MODEL = ...``
--------------------------------------------------
:mod:`screener.scoring` and :mod:`screener.report` do
``from .config import FACTOR_MODEL``, which binds the tuple into their own
namespaces at import time. Reassigning ``screener.config.FACTOR_MODEL`` after
that leaves both modules still pointing at the original object, so the
weights would appear to change while the scorer kept using the old ones --
a silent no-op that is worse than an error, because the run still produces
numbers. :func:`set_block_weights` rebinds every module that holds a
reference.

Intended for interactive use (a Colab notebook, a sensitivity sweep). For a
permanent change, edit :mod:`screener.config` instead.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import fields, replace
from typing import Iterator, Mapping

from . import config as _config
from .config import Block, FACTOR_MODEL

#: Pristine values captured at import, before any override can run. These are
#: what :func:`reset_block_weights` restores to.
#:
#: This module must never be rebound by :func:`rebind`: it imports
#: ``FACTOR_MODEL`` itself, so a rebind pass that included it would overwrite
#: the only surviving reference to the original model and make reset a no-op
#: that silently leaves the overridden weights in place.
_PRISTINE: dict[str, object] = {
    "FACTOR_MODEL": _config.FACTOR_MODEL,
    "BANDS": _config.BANDS,
    "GATES": _config.GATES,
    "SIZING": _config.SIZING,
    "ELIGIBILITY": _config.ELIGIBILITY,
}

_SELF = __name__


def _screener_modules() -> Iterator[object]:
    """
    Every imported module of this package except this one.

    See :data:`_PRISTINE` for why the exclusion is load-bearing rather than
    tidiness.
    """
    for name, module in list(sys.modules.items()):
        if module is None or name == _SELF:
            continue
        if name == "screener" or name.startswith("screener."):
            yield module


def rebind(attribute: str, value: object) -> list[str]:
    """
    Set ``attribute`` on every ``screener`` module that already holds it.

    Returns the module names touched. An empty list means the name was never
    imported anywhere -- treated as an error by the callers below, because a
    silently ineffective override is the whole hazard this module exists for.
    """
    touched: list[str] = []
    for module in _screener_modules():
        if hasattr(module, attribute):
            setattr(module, attribute, value)
            touched.append(module.__name__)
    return touched


def pristine(attribute: str) -> object:
    """The value ``attribute`` held in :mod:`screener.config` at import time."""
    if attribute not in _PRISTINE:
        raise KeyError(f"No pristine copy of {attribute!r}. Known: {sorted(_PRISTINE)}")
    return _PRISTINE[attribute]


def reset(attribute: str) -> object:
    """Restore one config singleton to its pristine value everywhere."""
    value = pristine(attribute)
    rebind(attribute, value)
    return value


def reset_all() -> None:
    """Restore every overridable config singleton. Safe to call repeatedly."""
    for attribute in _PRISTINE:
        rebind(attribute, _PRISTINE[attribute])


def _rebind(model: tuple[Block, ...]) -> None:
    rebind("FACTOR_MODEL", model)


def override(attribute: str, **changes: object) -> object:
    """
    Replace fields on one of config's frozen dataclass singletons
    (``GATES``, ``BANDS``, ``SIZING``, ``ELIGIBILITY``) and rebind it everywhere.

    ``override("GATES", max_volatility_for_overweight=None)`` disables the
    volatility ceiling for the rest of the session.
    """
    current = getattr(_config, attribute, None)
    if current is None:
        raise AttributeError(f"screener.config has no attribute {attribute!r}")

    unknown = set(changes) - {f.name for f in fields(current)}
    if unknown:
        valid = sorted(f.name for f in fields(current))
        raise TypeError(f"{attribute} has no field(s) {sorted(unknown)}. Valid: {valid}")

    updated = replace(current, **changes)
    touched = rebind(attribute, updated)
    if not touched:
        raise RuntimeError(f"{attribute} is not bound in any imported screener module.")
    return updated


def _pristine_model() -> tuple[Block, ...]:
    """The model as declared in config, never a previously overridden one."""
    return _PRISTINE["FACTOR_MODEL"]  # type: ignore[return-value]


def current_block_weights() -> dict[str, float]:
    return {b.key: b.weight for b in _config.FACTOR_MODEL}


def build_model(block_weights: Mapping[str, float],
                base: tuple[Block, ...] | None = None) -> tuple[Block, ...]:
    """
    Return a copy of the factor model with block weights replaced.

    Weights are renormalized to sum to 1.0, so passing relative sizes
    (``{"momentum": 2, "risk": 1, ...}``) works. Unlisted blocks keep their
    existing weight before renormalization.
    """
    base = base or _pristine_model()
    known = {b.key for b in base}
    unknown = set(block_weights) - known
    if unknown:
        raise KeyError(
            f"Unknown block(s): {sorted(unknown)}. Valid keys: {sorted(known)}"
        )

    merged = {b.key: float(block_weights.get(b.key, b.weight)) for b in base}
    if any(w < 0 for w in merged.values()):
        raise ValueError("Block weights must be non-negative.")

    total = sum(merged.values())
    if total <= 0:
        raise ValueError("Block weights sum to zero -- nothing would be scored.")

    return tuple(replace(b, weight=merged[b.key] / total) for b in base)


def set_block_weights(block_weights: Mapping[str, float]) -> dict[str, float]:
    """
    Override block weights process-wide. Returns the normalized weights applied.

    A block set to 0.0 is still computed and reported, but contributes nothing
    to the composite -- the clean way to ask "what does this model say without
    momentum?"
    """
    model = build_model(block_weights)
    _rebind(model)
    return {b.key: b.weight for b in model}


def reset_block_weights() -> dict[str, float]:
    """Restore the weights declared in :mod:`screener.config`."""
    model = _pristine_model()
    _rebind(model)
    return {b.key: b.weight for b in model}


@contextmanager
def block_weights(overrides: Mapping[str, float]) -> Iterator[dict[str, float]]:
    """Scoped override, restored on exit. Use for sensitivity sweeps."""
    previous = _config.FACTOR_MODEL
    try:
        yield set_block_weights(overrides)
    finally:
        _rebind(previous)
