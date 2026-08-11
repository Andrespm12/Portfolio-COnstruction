# Portfolio Construction — Quantitative Stock & ETF Screener

A systematic screen over US-listed equities and ETFs that produces a composite
score and an **Overweight / Market Weight / Underweight** call for every name,
scored against a live Interactive Brokers account.

Two deliverables:

1. **[`PROMPT_SCREENING.md`](PROMPT_SCREENING.md)** — the reusable LLM prompt
   encoding the full methodology.
2. **`screener/`** — the Python implementation of that methodology, which
   computes the metrics, scores the cross-section, applies risk gates and sizes
   positions. Plus an executed run against real IBKR data in `output/`.

---

## Quick start

```bash
pip install pandas numpy
python3 scripts/build_market_data.py     # regenerate data/ from the IBKR pull
python3 -m screener.run_screen           # score, gate, size, report
python3 tests/test_scoring.py            # 12 correctness assertions
```

Outputs land in `output/screen_results.csv` and `output/screen_report.md`.

---

## Executed run — 2026-08-11

21 securities (11 stocks, 10 ETFs), weekly bars over 1 year, benchmarked to SPY,
scored against a $15.84MM IBKR account.

| # | Ticker | Type | Call | Score | 1Y | Vol | MaxDD | Beta | Corr to book |
|---:|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| 1 | LLY | STOCK | **Overweight** | 100 | +73.2% | 32.8% | −18.7% | 0.18 | 0.09 |
| 2 | MU | STOCK | Market Weight ⚠ | 95 | +618.6% | 76.2% | −27.4% | 2.93 | 0.61 |
| 3 | JPM | STOCK | **Overweight** | 90 | +24.6% | 20.1% | −14.1% | 0.55 | 0.09 |
| 4 | AAPL | STOCK | **Overweight** | 85 | +31.7% | 25.2% | −11.1% | 0.87 | 0.36 |
| 5 | SMH | ETF | **Overweight** | 80 | +93.8% | 32.7% | −18.1% | 1.96 | 0.69 |
| 6 | GOOGL | STOCK | **Overweight** | 75 | +68.6% | 35.7% | −20.2% | 1.72 | 0.45 |
| 7 | XLE | ETF | **Overweight** | 70 | +42.4% | 23.1% | −14.9% | **−0.76** | −0.48 |
| 8 | IWM | ETF | **Overweight** | 65 | +32.5% | 16.0% | −8.9% | 0.94 | 0.58 |
| 9 | XLV | ETF | **Overweight** | 60 | +23.3% | 15.9% | −10.6% | 0.19 | 0.11 |
| 10 | VTWO | ETF | Market Weight | 55 | +32.5% | 15.9% | −8.7% | 0.96 | 0.59 |
| 11 | SPY | ETF | Market Weight ⚠ | 50 | +19.7% | 12.2% | −8.6% | 1.00 | 0.86 |
| 12 | AMZN | STOCK | Market Weight | 45 | +17.9% | 36.6% | −19.6% | 1.76 | 0.45 |
| 13 | QQQ | ETF | Market Weight | 40 | +24.4% | 18.3% | −10.6% | 1.39 | 0.85 |
| 14 | NVDA | STOCK | Market Weight | 35 | +20.5% | 32.4% | −17.3% | 1.65 | 0.57 |
| 15 | DBA | ETF | Underweight | 30 | +2.0% | 9.7% | −8.0% | 0.06 | 0.07 |
| 16 | TLT | ETF | Underweight ⚠ | 25 | −4.9% | 8.7% | −10.1% | 0.13 | 0.32 |
| 17 | MSFT | STOCK | Underweight | 20 | −3.2% | 36.6% | −31.9% | 1.58 | 0.41 |
| 18 | AVGO | STOCK | Underweight | 15 | +35.8% | 47.0% | −25.4% | 2.73 | 0.66 |
| 19 | GLD | ETF | Underweight | 10 | +30.4% | 23.5% | −23.8% | 0.83 | 0.72 |
| 20 | TSLA | STOCK | Underweight | 5 | +0.7% | 41.8% | −35.3% | 2.01 | 0.56 |
| 21 | META | STOCK | Underweight ⚠ | 0 | −23.7% | 40.9% | −33.0% | 2.04 | 0.69 |

⚠ = a risk gate fired. Four did:

- **MU** — factor score ranked it #2, but 76% annualized volatility exceeds the
  60% Overweight ceiling. Capped to Market Weight.
- **SPY** — 0.86 correlation to the book at a 16.5% existing weight triggers the
  concentration gate. Already Market Weight, so the stance is unchanged, but the
  gate is on the record: adding SPY here buys risk you already own.
- **TLT, META** — below the 40-week moving average with negative 12M−1M
  momentum. Cannot be Overweight regardless of score.

### Three findings worth a second look

**XLE's beta is genuinely −0.76.** Verified independently: only 40% of weeks moved
in the same direction as SPY, and in SPY-down weeks XLE averaged **+1.9%**. Energy
was a real diversifier this window, which is what pushes it to Overweight despite
a middling absolute return. This is the single most regime-dependent call in the
table — it rests on a correlation that has historically been positive.

**IWM and VTWO correlate 0.9969** and returned +32.52% vs +32.47%. They are the
same Russell 2000 exposure in different wrappers, yet score 0.65z apart — driven
almost entirely by liquidity ($7.6bn vs ~$159MM ADV) and by the existing VTWO
short. The redundancy gate ensures only the higher-scoring one can carry an
Overweight. Do not treat the score gap as a return forecast.

**The momentum block is doing most of the work at the top.** LLY, SMH and GOOGL
rank high largely on trend. That is the model behaving as designed, and it is
also the exposure most likely to reverse. The volatility and drawdown blocks
(15%) plus the gates are the only brakes.

---

## Factor model

| Block | Weight | What it measures |
|:--|--:|:--|
| Momentum & Trend | 22% | 12M−1M, 6M, 3M returns; distance from 52w high; 40W MA position and slope |
| Risk-Adjusted Return | 18% | Sharpe, Sortino, Calmar, hit rate |
| Volatility & Drawdown | 15% | Realized vol, max drawdown, downside deviation, Ulcer index |
| Market Sensitivity | 12% | Up/down capture spread, Jensen's alpha, idiosyncratic vol share, beta |
| Liquidity & Tradability | 10% | ADV, days-to-liquidate at 20% participation, volume stability |
| Valuation Proxy & Carry | 10% | Dividend yield, IV−HV spread, IV percentile, 52w range position |
| Portfolio Fit | 13% | Correlation to your book, marginal diversification benefit, existing overlap |

Volatility, drawdown and beta are scored **peer-relative** (ETF cohort vs stock
cohort). Without that, every ETF would top the risk block by construction and the
screen would degenerate into "prefer diversified things."

Weights **re-normalize** over available data rather than imputing zeros. Imputing
a zero silently asserts "average," which is a factual claim about a security you
have no measurement for.

---

## What this does NOT measure — read before using

**There is no fundamental data in this run.** No P/E, EV/EBITDA, ROIC, margins,
revenue growth, FCF or leverage. The execution environment blocks all outbound
network access except PyPI — Yahoo Finance, Wikipedia and every other data source
returned HTTP 403 at the proxy — and IBKR's market-data endpoints expose price,
volatility, liquidity and dividend yield but no financial statements.

Concretely: **Value, Quality and Growth are unmeasured.** The "Valuation Proxy &
Carry" block (10%) uses dividend yield, implied-vs-realized vol and 52-week range
position. Those are real signals, but they are not a substitute for knowing what
a business earns. A name can rank #1 here and be expensive on every fundamental
metric — the model has no way to see it.

To close that gap, feed a fundamentals source into `screener/metrics.py` and add
Value/Quality/Growth blocks to `FACTOR_MODEL` in `config.py`. The re-normalizing
weight logic means they slot in without touching the scoring engine.

**Other limitations, stated plainly:**

- **Universe size.** 21 names, not the full S&P/Nasdaq/Dow. Each security costs
  ~3 IBKR MCP round-trips (resolve, snapshot, history), and the agent-in-the-loop
  transport caps a single session well short of 500 names. The engine itself is
  universe-agnostic; run `screener/` against a direct IBKR Client Portal
  connection to screen the full index without the agent in the path.
- **Index membership is a static snapshot** (`universe.SNAPSHOT_DATE`), hardcoded
  because constituent sources are unreachable from here. `load_membership_override()`
  reads a CSV to replace it — wire that to a maintained feed in production.
- **~52 weekly observations is a short sample.** Beta, alpha and capture ratios
  carry wide confidence intervals. Do not read the third decimal.
- **One-year lookback only**, and that year contained a semiconductor melt-up
  (MU +618%) and a mega-cap tech drawdown (META −24%, MSFT −3%). The
  cross-section reflects that regime, not a stable equilibrium.
- **Indicative weights are indicative.** Sizing is a portfolio-construction
  decision; this screen ranks, it does not allocate.

---

## Layout

```
PROMPT_SCREENING.md          The prompt (primary deliverable)
screener/
  config.py                  Factor weights, bands, risk gates, sizing — every judgement call
  universe.py                Index membership, product exclusions, hard eligibility filters
  metrics.py                 Metric computation from IBKR bars and snapshots
  scoring.py                 Winsorize -> z-score -> composite -> stance -> gates
  portfolio.py               Portfolio-fit metrics against the live IBKR book
  report.py                  CSV / markdown / console output
  run_screen.py              CLI entry point
scripts/build_market_data.py Captured IBKR pull -> data/*.json
tests/test_scoring.py        12 assertions, mostly on metric direction
data/                        market_data.json, portfolio_ibkr.json
output/                      screen_results.csv, screen_report.md
```

## Design decisions worth knowing

**Data acquisition is decoupled from scoring.** The engine never talks to a
broker; it consumes `market_data.json`. That keeps it deterministic and testable,
and lets the same code run behind an agent, a cron job or a live gateway.

**Gates can only downgrade.** An additive score can be dragged into Overweight by
one extreme block — usually momentum. Making the gates one-directional means they
can never justify a position the factor model does not already support.

**Structured products are excluded by pattern, not by omission.** A search for any
liquid ticker on IBKR returns a long tail of `GRANITESH 2X LNG NVDA ETF`,
`YIELDMAX NVDA OPTION INC ETF`, `MCRSCTRS 3X LG SEMI CON ETN`. Their returns are
path-dependent or option-truncated, so factor metrics computed on them are not
comparable to ordinary equities. `is_excluded_product()` catches them and the
test suite asserts it against 9 real IBKR descriptions.

**The tests target sign errors.** The highest-consequence bug in a screener is an
inverted direction on a risk metric: the output looks entirely plausible while
recommending exactly the wrong names. `test_strong_name_outranks_weak_name` and
`test_risk_block_prefers_lower_volatility` exist specifically to catch that.
