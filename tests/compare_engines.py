"""
Diffs the JavaScript engine's output against the Python engine's output.

Run `node tests/verify_js_engine.js` first to produce tests/_js_output.json.
Any mismatch beyond floating-point noise is a port bug and fails the check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from screener.run_screen import load_json, run

ROOT = Path(__file__).resolve().parents[1]
TOL = 1e-6

BLOCK_KEYS = ["momentum", "risk_adjusted", "risk", "market_sensitivity",
              "liquidity", "valuation_carry", "portfolio_fit"]


def close(a, b, tol=TOL) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))


def main() -> int:
    js_path = ROOT / "tests" / "_js_output.json"
    if not js_path.exists():
        print("Run `node tests/verify_js_engine.js` first.")
        return 1

    js = json.loads(js_path.read_text(encoding="utf-8"))
    py_rows, py_meta = run(
        load_json(ROOT / "data" / "raw" / "market_data.json"),
        load_json(ROOT / "data" / "portfolio_ibkr.json"),
    )

    js_rows = js["rows"]
    problems: list[str] = []

    if len(js_rows) != len(py_rows):
        problems.append(f"row count: JS {len(js_rows)} vs Python {len(py_rows)}")

    for i, (j, p) in enumerate(zip(js_rows, py_rows), start=1):
        where = f"[{i}] {p.ticker}"
        if j["ticker"] != p.ticker:
            problems.append(f"{where}: order differs — JS has {j['ticker']}")
            continue
        if j["rec"] != p.recommendation:
            problems.append(f"{where}: recommendation JS {j['rec']} vs PY {p.recommendation}")
        if not close(j["cz"], p.composite_z):
            problems.append(f"{where}: composite z JS {j['cz']:.9f} vs PY {p.composite_z:.9f}")
        if not close(j["score"], p.score_0_100):
            problems.append(f"{where}: score JS {j['score']:.9f} vs PY {p.score_0_100:.9f}")
        if not close(j["weight"], p.indicative_weight):
            problems.append(f"{where}: weight JS {j['weight']:.9f} vs PY {p.indicative_weight:.9f}")
        for b in BLOCK_KEYS:
            if not close(j["blocks"].get(b), p.block_scores.get(b)):
                problems.append(
                    f"{where}: block {b} JS {j['blocks'].get(b)} vs PY {p.block_scores.get(b)}")
        for label, jv, pv in (
            ("1Y return", j["ret_1y"], p.diagnostics.get("return_1y")),
            ("volatility", j["vol"], p.diagnostics.get("volatility")),
            ("max drawdown", j["maxdd"], p.diagnostics.get("max_drawdown")),
            ("beta", j["beta"], p.diagnostics.get("beta")),
            ("sharpe", j["sharpe"], p.raw_metrics.get("sharpe_1y")),
            ("corr to book", j["corr"], p.raw_metrics.get("corr_to_portfolio")),
        ):
            if not close(jv, pv):
                problems.append(f"{where}: {label} JS {jv} vs PY {pv}")
        if len(j["gates"]) != len(p.gates_triggered):
            problems.append(
                f"{where}: gate count JS {len(j['gates'])} vs PY {len(p.gates_triggered)}")

    print(f"Compared {len(js_rows)} rows across "
          f"{len(BLOCK_KEYS)} factor blocks and 6 diagnostics each.")
    if problems:
        print(f"\nFAIL — {len(problems)} mismatch(es):")
        for msg in problems[:40]:
            print(f"  {msg}")
        return 1

    print("PASS — JavaScript engine reproduces the Python engine exactly "
          f"(tolerance {TOL:g}).")
    print(f"       recommendations, composite z, 0-100 score, indicative weight,")
    print(f"       all 7 block scores and gate counts match on every name.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
