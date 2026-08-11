"""
Entry point: load market data, compute metrics, score, gate, size, report.

Usage
-----
    python -m screener.run_screen \
        --market-data data/raw/market_data.json \
        --portfolio data/portfolio_ibkr.json \
        --out-csv output/screen_results.csv \
        --out-md output/screen_report.md

Data acquisition is deliberately decoupled from scoring. ``market_data.json``
is produced by whatever adapter has IBKR access -- an agent calling the IBKR
MCP tools, or :mod:`screener.ibkr_adapter` hitting the Client Portal Web API
directly when run on a machine with a gateway. The scoring engine never talks
to a broker, which keeps it deterministic and testable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import BENCHMARK_TICKER, ELIGIBILITY, RISK_FREE_RATE, SIZING
from .metrics import compute_metrics, diagnostics, simple_returns
from .portfolio import (
    build_portfolio_return_series, compute_portfolio_fit, concentration_stats,
    existing_weights, portfolio_volatility,
)
from .report import console_summary, write_csv, write_markdown
from .scoring import ScoredInstrument, detect_duplicates, run_scoring_pipeline
from .universe import index_tags, is_excluded_product, screen_eligibility


def _closes(instrument: dict) -> np.ndarray:
    hist = instrument.get("history") or {}
    closes = [c for c in (hist.get("close") or []) if isinstance(c, (int, float))]
    return np.asarray(closes, dtype=float)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(market_data: dict, portfolio: dict,
        rf: float = RISK_FREE_RATE) -> tuple[list[ScoredInstrument], dict]:
    instruments: list[dict] = market_data.get("instruments", [])
    benchmark_ticker = market_data.get("benchmark", BENCHMARK_TICKER)
    rf = float(market_data.get("risk_free_rate", rf))

    if not instruments:
        raise ValueError("market_data.json contains no instruments.")

    # ---- Benchmark return series ----------------------------------------
    bench = next((i for i in instruments if i.get("ticker", "").upper() == benchmark_ticker), None)
    if bench is None:
        raise ValueError(f"Benchmark {benchmark_ticker} missing from market data.")
    bench_returns = simple_returns(_closes(bench))
    if bench_returns.size < 12:
        raise ValueError(f"Benchmark {benchmark_ticker} has insufficient price history.")

    # ---- Existing book ---------------------------------------------------
    net_liq = float(portfolio.get("net_liquidation") or 0.0)
    positions = portfolio.get("positions") or []
    history_by_ticker = {
        i["ticker"].upper(): _closes(i) for i in instruments if i.get("ticker")
    }
    port_returns, _, _ = build_portfolio_return_series(positions, history_by_ticker)
    weights_by_nlv = existing_weights(positions, net_liq)

    # ---- Metrics ---------------------------------------------------------
    rows: list[ScoredInstrument] = []
    for inst in instruments:
        ticker = (inst.get("ticker") or "").upper()
        if not ticker:
            continue

        inst.setdefault("indices", index_tags(ticker))

        m = compute_metrics(
            inst, bench_returns,
            net_liq=net_liq,
            participation=ELIGIBILITY.max_participation_rate,
            base_position_weight=SIZING.base_weight,
            rf=rf,
        )
        m.update(compute_portfolio_fit(inst, port_returns, weights_by_nlv))

        diag = diagnostics(inst, bench_returns, rf=rf)

        gate_inputs = {
            "_last_price": diag.get("last_price"),
            "_adv_usd": diag.get("adv_usd"),
            "_bars": diag.get("bars"),
        }
        eligible, reasons = screen_eligibility(inst, gate_inputs)

        rows.append(ScoredInstrument(
            ticker=ticker,
            name=inst.get("name", ticker),
            asset_type=inst.get("asset_type", "STOCK"),
            indices=inst.get("indices", []),
            sector=inst.get("sector"),
            raw_metrics=m,
            diagnostics=diag,
            eligible=eligible,
            ineligibility_reasons=reasons,
        ))

    eligible_rows = [r for r in rows if r.eligible]
    if len(eligible_rows) < 3:
        raise ValueError(
            f"Only {len(eligible_rows)} eligible securities; need at least 3 for "
            "cross-sectional scoring. Loosen filters or widen the universe."
        )

    # Identify duplicate exposures before gating so the redundancy gate can act.
    returns_by_ticker = {
        t: simple_returns(p) for t, p in history_by_ticker.items()
    }
    detect_duplicates(eligible_rows, returns_by_ticker)

    scored = run_scoring_pipeline(eligible_rows)

    # Ineligible names are reported but never scored -- they would distort the
    # cross-sectional distribution of the names that passed.
    ineligible = [r for r in rows if not r.eligible]
    for r in ineligible:
        r.recommendation = "EXCLUDED"

    meta = {
        "n_screened": len(rows),
        "n_eligible": len(eligible_rows),
        "n_excluded": len(ineligible),
        "benchmark": benchmark_ticker,
        "risk_free_rate": rf,
        "net_liquidation": net_liq,
        "portfolio_stats": concentration_stats(positions, net_liq),
        "portfolio_vol": portfolio_volatility(port_returns),
        "data_source": market_data.get("data_source", "Interactive Brokers"),
        "history_desc": market_data.get("history_desc", "weekly bars, 1 year"),
        "as_of": market_data.get("as_of"),
        "excluded": [(r.ticker, r.ineligibility_reasons) for r in ineligible],
    }
    return scored, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Quantitative stock/ETF screening engine")
    ap.add_argument("--market-data", default="data/raw/market_data.json")
    ap.add_argument("--portfolio", default="data/portfolio_ibkr.json")
    ap.add_argument("--out-csv", default="output/screen_results.csv")
    ap.add_argument("--out-md", default="output/screen_report.md")
    ap.add_argument("--risk-free-rate", type=float, default=RISK_FREE_RATE)
    args = ap.parse_args(argv)

    market_data = load_json(args.market_data)
    portfolio = load_json(args.portfolio) if Path(args.portfolio).exists() else {
        "net_liquidation": 0.0, "positions": []
    }

    scored, meta = run(market_data, portfolio, rf=args.risk_free_rate)

    write_csv(scored, args.out_csv)
    write_markdown(scored, meta, args.out_md)
    print(console_summary(scored, meta))

    if meta["excluded"]:
        print("\nEXCLUDED BY HARD FILTERS:")
        for ticker, reasons in meta["excluded"]:
            print(f"  {ticker}: {'; '.join(reasons)}")

    print(f"\nCSV  -> {args.out_csv}")
    print(f"REPORT -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
