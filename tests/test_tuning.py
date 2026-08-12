"""
Tests for runtime factor-model overrides.

The failure mode being guarded against is specific: rebinding only
``screener.config.FACTOR_MODEL`` leaves ``screener.scoring`` holding the
original tuple, so an override would appear to succeed while the scorer kept
using the old weights. That produces a plausible-looking run with the wrong
model, which is why these tests assert on *scoring output*, not just on the
value of the config attribute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import screener.config as config  # noqa: E402
import screener.report as report  # noqa: E402
import screener.scoring as scoring  # noqa: E402
from screener.run_screen import run  # noqa: E402
from screener.tuning import (  # noqa: E402
    block_weights, build_model, current_block_weights, reset_block_weights,
    set_block_weights,
)
from screener.yahoo_adapter import build_market_data  # noqa: E402
from test_yahoo_adapter import make_yf_frame  # noqa: E402

PASSED = 0
FAILED = 0

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "JPM", "LLY"]
PORTFOLIO = {"net_liquidation": 10_000_000.0, "positions": []}


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


def payload() -> dict:
    frame = make_yf_frame(TICKERS, dividends={"SPY": 6.0, "JPM": 4.0})
    return build_market_data(frame, TICKERS, benchmark="SPY")


def scores(data: dict) -> dict[str, float]:
    scored, _ = run(data, PORTFOLIO)
    return {r.ticker: r.composite_z for r in scored}


def test_normalization() -> None:
    model = build_model({"momentum": 2.0, "risk": 1.0})
    total = sum(b.weight for b in model)
    check("rebuilt model weights sum to 1.0", abs(total - 1.0) < 1e-12, f"got {total}")

    weights = {b.key: b.weight for b in model}
    check("relative sizes are honoured after renormalization",
          abs(weights["momentum"] / weights["risk"] - 2.0) < 1e-12)

    check("unlisted blocks keep their relative standing",
          weights["liquidity"] > 0)


def test_validation() -> None:
    try:
        build_model({"not_a_block": 1.0})
        check("unknown block key raises", False, "no exception")
    except KeyError as exc:
        check("unknown block key raises and lists valid keys", "momentum" in str(exc))

    try:
        build_model({"momentum": -1.0})
        check("negative weight raises", False, "no exception")
    except ValueError:
        check("negative weight raises", True)

    try:
        build_model({key: 0.0 for key in current_block_weights()})
        check("all-zero weights raise", False, "no exception")
    except ValueError:
        check("all-zero weights raise", True)


def test_rebinding_reaches_every_module() -> None:
    try:
        set_block_weights({"momentum": 0.5})
        check("config is rebound", config.FACTOR_MODEL is not None)
        check("scoring sees the SAME model object as config",
              scoring.FACTOR_MODEL is config.FACTOR_MODEL)
        check("report sees the SAME model object as config",
              report.FACTOR_MODEL is config.FACTOR_MODEL)
    finally:
        reset_block_weights()

    check("reset restores scoring's binding too",
          scoring.FACTOR_MODEL is config.FACTOR_MODEL)


def test_override_changes_actual_scores() -> None:
    data = payload()
    baseline = scores(data)

    try:
        # Momentum is 22% of the model; zeroing it must move the composite.
        set_block_weights({"momentum": 0.0})
        without_momentum = scores(data)
    finally:
        reset_block_weights()

    moved = [t for t in baseline if abs(baseline[t] - without_momentum[t]) > 1e-6]
    check("zeroing a block actually changes composite scores",
          len(moved) >= len(baseline) // 2,
          f"only {len(moved)}/{len(baseline)} names moved")

    restored = scores(data)
    check("scores return to baseline after reset",
          all(abs(restored[t] - baseline[t]) < 1e-9 for t in baseline))


def test_reset_survives_repeated_overrides() -> None:
    """
    Regression: the generic rebind pass originally included screener.tuning
    itself, which imports FACTOR_MODEL. The first override overwrote the only
    reference to the original model, so reset restored the *overridden* weights
    and every subsequent run silently used them.
    """
    data = payload()
    baseline = scores(data)
    original = current_block_weights()

    for override_weights in ({"momentum": 0.0}, {"risk": 5.0}, {"liquidity": 0.0}):
        set_block_weights(override_weights)
        reset_block_weights()
        check(f"reset after {list(override_weights)[0]} restores declared weights",
              current_block_weights() == original,
              f"got {current_block_weights()}")

    check("scores are identical to baseline after three override/reset cycles",
          all(abs(scores(data)[t] - baseline[t]) < 1e-9 for t in baseline))


def test_context_manager_restores() -> None:
    data = payload()
    baseline = scores(data)
    before = current_block_weights()

    with block_weights({"momentum": 0.0, "risk": 3.0}) as applied:
        check("context manager reports the normalized weights it applied",
              abs(sum(applied.values()) - 1.0) < 1e-12)
        check("weights differ inside the context",
              applied["momentum"] == 0.0)
        inside = scores(data)

    check("weights are restored on exit", current_block_weights() == before)
    check("scores are restored on exit",
          all(abs(scores(data)[t] - baseline[t]) < 1e-9 for t in baseline))
    check("the scoped run produced a genuinely different ranking",
          any(abs(inside[t] - baseline[t]) > 1e-6 for t in baseline))


def test_context_manager_restores_on_exception() -> None:
    before = current_block_weights()
    try:
        with block_weights({"momentum": 0.0}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("weights are restored even when the body raises",
          current_block_weights() == before)


def test_override_rebinds_dataclass_singletons() -> None:
    import screener.universe as universe
    from screener.tuning import override

    original_gates = config.GATES
    original_elig = config.ELIGIBILITY
    try:
        override("GATES", max_volatility_for_overweight=None)
        check("GATES override reaches config",
              config.GATES.max_volatility_for_overweight is None)
        check("GATES override reaches scoring (the module that enforces it)",
              scoring.GATES is config.GATES)
        check("unrelated GATES fields are preserved",
              config.GATES.beta_limit == original_gates.beta_limit)

        override("ELIGIBILITY", min_adv_usd=1.0)
        check("ELIGIBILITY override reaches universe (the module that screens)",
              universe.ELIGIBILITY is config.ELIGIBILITY
              and universe.ELIGIBILITY.min_adv_usd == 1.0)
    finally:
        rebind_all(original_gates, original_elig)

    check("gates restored", config.GATES.max_volatility_for_overweight
          == original_gates.max_volatility_for_overweight)


def rebind_all(gates, elig) -> None:
    from screener.tuning import rebind
    rebind("GATES", gates)
    rebind("ELIGIBILITY", elig)


def test_override_rejects_bad_fields() -> None:
    from screener.tuning import override

    try:
        override("GATES", not_a_field=1)
        check("unknown dataclass field raises", False, "no exception")
    except TypeError as exc:
        check("unknown dataclass field raises and lists valid ones",
              "beta_limit" in str(exc))

    try:
        override("NOT_A_THING", x=1)
        check("unknown config attribute raises", False, "no exception")
    except AttributeError:
        check("unknown config attribute raises", True)


def test_override_changes_gate_behaviour() -> None:
    """An override that does not change output would be a silent no-op."""
    from screener.tuning import override

    data = payload()
    original = config.GATES
    try:
        # Force the redundancy gate to bind on everything correlated at all.
        override("GATES", corr_limit=-1.0, existing_weight_limit=-1.0)
        scored, _ = run(data, {
            "net_liquidation": 10_000_000.0,
            "positions": [{"ticker": "SPY", "market_value": 5_000_000.0,
                           "quantity": 6500, "asset_class": "STK"}],
        })
        gated = [r for r in scored if r.gates_triggered]
        check("a loosened gate threshold actually fires more gates",
              len(gated) > 0, "no gates triggered")
    finally:
        from screener.tuning import rebind
        rebind("GATES", original)


def test_zeroed_block_is_still_reported() -> None:
    data = payload()
    try:
        set_block_weights({"momentum": 0.0})
        scored, _ = run(data, PORTFOLIO)
        spy = next(r for r in scored if r.ticker == "SPY")
        check("a zero-weight block is still computed and shown",
              np.isfinite(spy.block_scores.get("momentum", float("nan"))))
    finally:
        reset_block_weights()


def main() -> int:
    for fn in [
        test_normalization,
        test_validation,
        test_rebinding_reaches_every_module,
        test_override_changes_actual_scores,
        test_reset_survives_repeated_overrides,
        test_context_manager_restores,
        test_context_manager_restores_on_exception,
        test_override_rebinds_dataclass_singletons,
        test_override_rejects_bad_fields,
        test_override_changes_gate_behaviour,
        test_zeroed_block_is_still_reported,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
