"""Output formatting: CSV, markdown report and console summary."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from .config import FACTOR_MODEL, MARKET_WEIGHT, OVERWEIGHT, UNDERWEIGHT
from .scoring import ScoredInstrument

_ORDER = {OVERWEIGHT: 0, MARKET_WEIGHT: 1, UNDERWEIGHT: 2}


def _f(value, spec: str = ".2f", dash: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return dash
    return format(value, spec)


def write_csv(rows: list[ScoredInstrument], path: str | Path,
              standalone: bool = False) -> None:
    """
    Write the results table.

    ``standalone=True`` omits the two book-relative columns entirely rather
    than emitting them empty: an independent screen has no book to correlate
    against, and a blank column invites the reader to assume the number was
    merely unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    block_keys = [b.key for b in FACTOR_MODEL]
    book_columns = [] if standalone else ["corr_to_portfolio", "existing_overlap"]
    header = (
        ["rank", "ticker", "name", "asset_type", "indices", "recommendation",
         "score_0_100", "composite_z", "indicative_weight"]
        + [f"block_{k}" for k in block_keys]
        + ["last_price", "return_1y", "volatility", "max_drawdown", "beta",
           "alpha_annual", "sharpe", "sortino", "calmar", "adv_usd"]
        + book_columns + ["gates"]
    )

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(rows, start=1):
            d = r.diagnostics
            writer.writerow(
                [i, r.ticker, r.name, r.asset_type, "|".join(r.indices), r.recommendation,
                 _f(r.score_0_100, ".1f"), _f(r.composite_z, ".3f"), _f(r.indicative_weight, ".4f")]
                + [_f(r.block_scores.get(k), ".3f") for k in block_keys]
                + [_f(d.get("last_price"), ".2f"), _f(d.get("return_1y"), ".4f"),
                   _f(d.get("volatility"), ".4f"), _f(d.get("max_drawdown"), ".4f"),
                   _f(d.get("beta"), ".3f"), _f(d.get("alpha_annual"), ".4f"),
                   _f(r.raw_metrics.get("sharpe_1y"), ".3f"),
                   _f(r.raw_metrics.get("sortino_1y"), ".3f"),
                   _f(r.raw_metrics.get("calmar_1y"), ".3f"),
                   _f(d.get("adv_usd"), ".0f")]
                + ([] if standalone else
                   [_f(r.raw_metrics.get("corr_to_portfolio"), ".3f"),
                    _f(r.raw_metrics.get("existing_overlap"), ".4f")])
                + [" | ".join(r.gates_triggered)]
            )


def _bucket(rows: list[ScoredInstrument], rec: str) -> list[ScoredInstrument]:
    return [r for r in rows if r.recommendation == rec]


def build_markdown(rows: list[ScoredInstrument], meta: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    add = lines.append

    add("# Stock & ETF Screening Results")
    add("")
    add(f"**Generated:** {ts}  ")
    add(f"**Universe screened:** {meta.get('n_screened', 0)} securities  ")
    add(f"**Eligible after hard filters:** {meta.get('n_eligible', 0)}  ")
    add(f"**Benchmark:** {meta.get('benchmark', 'SPY')}  ")
    add(f"**Risk-free rate:** {meta.get('risk_free_rate', 0):.2%}  ")
    add(f"**Data source:** {meta.get('data_source', 'IBKR')}  ")
    add(f"**Price history:** {meta.get('history_desc', 'weekly bars, 1 year')}")
    if meta.get("standalone"):
        add(f"  \n**Risk profile:** {meta.get('profile_label', '—')}  ")
        add(f"**Assumed position size:** ${meta.get('target_position_usd') or 0:,.0f}  ")
        add("**Book:** none — every name is scored on its own merits, with no "
            "account data read and no Portfolio Fit block in the model.")
    add("")

    ow, mw, uw = _bucket(rows, OVERWEIGHT), _bucket(rows, MARKET_WEIGHT), _bucket(rows, UNDERWEIGHT)
    add(f"**Distribution:** {len(ow)} Overweight / {len(mw)} Market Weight / {len(uw)} Underweight")
    add("")

    pf = meta.get("portfolio_stats") or {}
    if pf:
        add("## Current book context")
        add("")
        add(f"- Net liquidation value: **${meta.get('net_liquidation', 0):,.0f}**")
        add(f"- Positions with price history: **{pf.get('positions', 0)}**")
        add(f"- Largest position: **{pf.get('largest_position', 0):.1%}** of NLV")
        add(f"- Top-5 concentration: **{pf.get('top5_concentration', 0):.1%}** of NLV")
        add(f"- Effective number of positions (1/HHI): **{pf.get('effective_positions', 0):.1f}**")
        if meta.get("portfolio_vol") is not None:
            add(f"- Realized volatility of the equity/ETF sleeve: **{meta['portfolio_vol']:.1%}** annualized")
        add("")

    add("## Ranking")
    add("")
    # In standalone mode there is no book, so correlation-to-book is not a
    # column that could ever hold a value. Report annualized alpha instead of
    # printing an empty one.
    standalone = bool(meta.get("standalone"))
    last_label = "Alpha" if standalone else "Corr"
    add(f"| # | Ticker | Type | Rec | Score | z | 1Y Ret | Vol | MaxDD | Beta | "
        f"Sharpe | {last_label} | Wgt |")
    add("|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, start=1):
        d = r.diagnostics
        last = (_f(d.get("alpha_annual"), "+.1%") if standalone
                else _f(r.raw_metrics.get("corr_to_portfolio"), ".2f"))
        add(
            f"| {i} | **{r.ticker}** | {r.asset_type} | {r.recommendation} | "
            f"{_f(r.score_0_100, '.0f')} | {_f(r.composite_z, '+.2f')} | "
            f"{_f(d.get('return_1y'), '+.1%')} | {_f(d.get('volatility'), '.1%')} | "
            f"{_f(d.get('max_drawdown'), '.1%')} | {_f(d.get('beta'), '.2f')} | "
            f"{_f(r.raw_metrics.get('sharpe_1y'), '.2f')} | {last} | "
            f"{_f(r.indicative_weight, '.1%')} |"
        )
    add("")

    add("## Factor block detail")
    add("")
    block_keys = [b.key for b in FACTOR_MODEL]
    add("| Ticker | " + " | ".join(b.label for b in FACTOR_MODEL) + " |")
    add("|:---|" + "---:|" * len(block_keys))
    for r in rows:
        add(f"| **{r.ticker}** | " +
            " | ".join(_f(r.block_scores.get(k), "+.2f") for k in block_keys) + " |")
    add("")

    for rec, bucket in ((OVERWEIGHT, ow), (UNDERWEIGHT, uw)):
        if not bucket:
            continue
        add(f"## {rec} rationale")
        add("")
        for r in bucket:
            drivers = sorted(r.block_scores.items(), key=lambda kv: kv[1], reverse=(rec == OVERWEIGHT))
            top = ", ".join(
                f"{next(b.label for b in FACTOR_MODEL if b.key == k)} ({v:+.2f})"
                for k, v in drivers[:3]
            )
            add(f"**{r.ticker}** — {r.name} · score {_f(r.score_0_100, '.0f')}/100 (z {_f(r.composite_z, '+.2f')})")
            add(f"- Primary drivers: {top}")
            d = r.diagnostics
            add(f"- 1Y return {_f(d.get('return_1y'), '+.1%')}, vol {_f(d.get('volatility'), '.1%')}, "
                f"max DD {_f(d.get('max_drawdown'), '.1%')}, beta {_f(d.get('beta'), '.2f')}, "
                f"alpha {_f(d.get('alpha_annual'), '+.1%')}")
            if r.gates_triggered:
                add(f"- Gates: {'; '.join(r.gates_triggered)}")
            add("")

    gated = [r for r in rows if r.gates_triggered and r.pre_gate_recommendation != r.recommendation]
    if gated:
        add("## Downgraded by risk gates")
        add("")
        add("Names whose factor score qualified them for a higher stance but which were capped:")
        add("")
        for r in gated:
            add(f"- **{r.ticker}**: {r.pre_gate_recommendation} → {r.recommendation}. "
                f"{'; '.join(r.gates_triggered)}")
        add("")

    add("## Methodology")
    add("")
    for b in FACTOR_MODEL:
        add(f"**{b.label}** ({b.weight:.0%}) — {b.rationale}")
        add("")

    return "\n".join(lines)


def write_markdown(rows: list[ScoredInstrument], meta: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(rows, meta), encoding="utf-8")


def console_summary(rows: list[ScoredInstrument], meta: dict) -> str:
    out: list[str] = []
    out.append("=" * 108)
    out.append(f"SCREENING RESULTS — {meta.get('n_eligible', 0)} eligible of {meta.get('n_screened', 0)} screened")
    out.append("=" * 108)
    standalone = bool(meta.get("standalone"))
    if standalone and meta.get("profile_label"):
        out.append(f"Perfil: {meta['profile_label']}   "
                   f"posición asumida ${meta.get('target_position_usd', 0):,.0f}")
        out.append("-" * 108)
    out.append(f"{'#':>3} {'TICKER':<8} {'TYPE':<6} {'RECOMMENDATION':<15} {'SCORE':>6} "
               f"{'Z':>7} {'1Y':>8} {'VOL':>7} {'MAXDD':>8} {'BETA':>6} {'SHRP':>6} "
               f"{'ALPHA' if standalone else 'CORR':>6} {'WGT':>6}")
    out.append("-" * 108)
    for i, r in enumerate(rows, start=1):
        d = r.diagnostics
        last = (_f(d.get("alpha_annual"), "+.1%") if standalone
                else _f(r.raw_metrics.get("corr_to_portfolio"), ".2f"))
        out.append(
            f"{i:>3} {r.ticker:<8} {r.asset_type:<6} {r.recommendation:<15} "
            f"{_f(r.score_0_100, '.0f'):>6} {_f(r.composite_z, '+.2f'):>7} "
            f"{_f(d.get('return_1y'), '+.1%'):>8} {_f(d.get('volatility'), '.1%'):>7} "
            f"{_f(d.get('max_drawdown'), '.1%'):>8} {_f(d.get('beta'), '.2f'):>6} "
            f"{_f(r.raw_metrics.get('sharpe_1y'), '.2f'):>6} {last:>6} "
            f"{_f(r.indicative_weight, '.1%'):>6}"
        )
    out.append("-" * 108)
    ow, mw, uw = _bucket(rows, OVERWEIGHT), _bucket(rows, MARKET_WEIGHT), _bucket(rows, UNDERWEIGHT)
    out.append(f"OVERWEIGHT ({len(ow)}): {', '.join(r.ticker for r in ow) or '—'}")
    out.append(f"MARKET WEIGHT ({len(mw)}): {', '.join(r.ticker for r in mw) or '—'}")
    out.append(f"UNDERWEIGHT ({len(uw)}): {', '.join(r.ticker for r in uw) or '—'}")
    out.append("=" * 108)
    return "\n".join(out)
