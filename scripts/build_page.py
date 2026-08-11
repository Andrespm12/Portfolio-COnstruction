"""
Injects the IBKR data into web/template.html and writes web/screener.html.

The published page is self-contained: the scoring engine is a JavaScript port of
`screener/`, and the market data is embedded so the page always renders even when
no IBKR connector is reachable. When one *is* reachable, the page re-pulls and
re-scores in place.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "/*__EMBEDDED_DATA__*/null"


def main() -> None:
    market = json.loads((ROOT / "data" / "raw" / "market_data.json").read_text(encoding="utf-8"))
    portfolio = json.loads((ROOT / "data" / "portfolio_ibkr.json").read_text(encoding="utf-8"))
    template = (ROOT / "web" / "template.html").read_text(encoding="utf-8")

    if PLACEHOLDER not in template:
        raise SystemExit(f"placeholder {PLACEHOLDER!r} not found in template.html")

    payload = json.dumps(
        {"market": market, "portfolio": portfolio},
        separators=(",", ":"),
        ensure_ascii=False,
    )
    # The payload is injected into a <script> body, so the one sequence that could
    # break out of it must be neutralized.
    payload = payload.replace("</", "<\\/")

    out = ROOT / "web" / "screener.html"
    out.write_text(template.replace(PLACEHOLDER, payload), encoding="utf-8")

    kb = out.stat().st_size / 1024
    print(f"web/screener.html written — {kb:.0f} KB, "
          f"{len(market['instruments'])} instruments, {len(portfolio['positions'])} positions")


if __name__ == "__main__":
    main()
