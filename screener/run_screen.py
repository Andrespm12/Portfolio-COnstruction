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
from .seleccion import evaluar, resumen as resumen_seleccion
from .universe import index_tags, is_excluded_product, screen_eligibility


def _closes(instrument: dict) -> np.ndarray:
    hist = instrument.get("history") or {}
    closes = [c for c in (hist.get("close") or []) if isinstance(c, (int, float))]
    return np.asarray(closes, dtype=float)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(market_data: dict, portfolio: dict,
        rf: float = RISK_FREE_RATE, *,
        standalone: bool = False,
        target_position_usd: float | None = None,
        ) -> tuple[list[ScoredInstrument], dict]:
    """
    Score the cross-section.

    ``standalone=True`` scores every name on its own merits: no account data is
    read and no Portfolio Fit metric is computed. Use it with
    :func:`screener.profiles.apply_profile`, which also removes the Portfolio
    Fit block from the factor model -- passing an empty book here is not
    equivalent, because ``existing_overlap`` would still be emitted as ``0.0``
    for every name and the block would count as populated.

    ``target_position_usd`` sets the position size the liquidity block assumes
    when computing ``days_to_liquidate``. It is a sizing assumption, not
    account data, and is required in standalone mode -- without a position size
    there is nothing to divide by average daily volume.
    """
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
    history_by_ticker = {
        i["ticker"].upper(): _closes(i) for i in instruments if i.get("ticker")
    }

    if standalone:
        if not target_position_usd or target_position_usd <= 0:
            raise ValueError(
                "standalone=True requires a positive target_position_usd -- the "
                "liquidity block needs a position size to divide by ADV."
            )
        # net_liq * base_position_weight is how compute_metrics derives the
        # target position; setting the weight to 1.0 makes net_liq the position
        # size directly, with no notion of an account behind it.
        net_liq, base_position_weight = float(target_position_usd), 1.0
        positions: list[dict] = []
        port_returns = np.asarray([], dtype=float)
        weights_by_nlv: dict[str, float] = {}
    else:
        net_liq = float(portfolio.get("net_liquidation") or 0.0)
        base_position_weight = SIZING.base_weight
        positions = portfolio.get("positions") or []
        port_returns, _, _ = build_portfolio_return_series(positions, history_by_ticker)
        weights_by_nlv = existing_weights(positions, net_liq)

    # ---- Metrics ---------------------------------------------------------
    rows: list[ScoredInstrument] = []
    decisiones: list = []
    for inst in instruments:
        ticker = (inst.get("ticker") or "").upper()
        if not ticker:
            continue

        inst.setdefault("indices", index_tags(ticker))

        m = compute_metrics(
            inst, bench_returns,
            net_liq=net_liq,
            participation=ELIGIBILITY.max_participation_rate,
            base_position_weight=base_position_weight,
            rf=rf,
        )
        if not standalone:
            m.update(compute_portfolio_fit(inst, port_returns, weights_by_nlv))

        diag = diagnostics(inst, bench_returns, rf=rf)

        gate_inputs = {
            "_last_price": diag.get("last_price"),
            "_adv_usd": diag.get("adv_usd"),
            "_bars": diag.get("bars"),
        }
        decision = evaluar(inst, gate_inputs)
        eligible, reasons = decision.admitido, (
            [] if decision.admitido else [decision.detalle])
        decisiones.append(decision)

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
        "standalone": standalone,
        "net_liquidation": 0.0 if standalone else net_liq,
        "target_position_usd": target_position_usd if standalone else None,
        "portfolio_stats": {} if standalone else concentration_stats(positions, net_liq),
        "portfolio_vol": None if standalone else portfolio_volatility(port_returns),
        "data_source": market_data.get("data_source", "Interactive Brokers"),
        "history_desc": market_data.get("history_desc", "weekly bars, 1 year"),
        "as_of": market_data.get("as_of"),
        "excluded": [(r.ticker, r.ineligibility_reasons) for r in ineligible],
        # Rastro completo de la política de selección: una decisión por
        # candidato, admitido o no. Es lo que hace visible una exclusión.
        "seleccion": decisiones,
        "seleccion_resumen": resumen_seleccion(decisiones),
    }
    return scored, meta


def run_standalone(market_data: dict, profile: "RiskProfile | str" = "moderado",
                   position_usd: float | None = None,
                   rf: float = RISK_FREE_RATE,
                   ) -> tuple[list[ScoredInstrument], dict]:
    """
    Score a universe on its own merits under a named risk profile.

    No account data is read and the Portfolio Fit block is removed from the
    factor model, so nothing about an existing book can reach the ranking. The
    profile rewires block weights, recommendation bands, risk gates, sizing and
    eligibility together -- see :mod:`screener.profiles`.

    ``position_usd`` is the position size the liquidity block assumes when
    computing ``days_to_liquidate``; it defaults to the profile's own. It is a
    sizing assumption, not a statement about anyone's capital.
    """
    from .profiles import apply_profile, get_profile

    resolved = get_profile(profile) if isinstance(profile, str) else profile
    apply_profile(resolved)

    size = float(position_usd or resolved.default_position_usd)
    scored, meta = run(market_data, {}, rf=rf,
                       standalone=True, target_position_usd=size)
    meta["profile"] = resolved.key
    meta["profile_label"] = resolved.label
    meta["profile_summary"] = resolved.summary
    meta["block_weights"] = {b.key: b.weight for b in resolved.model()}
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
