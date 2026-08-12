# Portfolio Construction — Quantitative Stock & ETF Screener

A systematic screen over US-listed equities and ETFs that produces a composite
score and an **Overweight / Market Weight / Underweight** call for every name,
scored against a live Interactive Brokers account.

Four deliverables:

1. **[`PROMPT_SCREENING.md`](PROMPT_SCREENING.md)** — the reusable LLM prompt
   encoding the full methodology.
2. **`screener/`** — the Python implementation of that methodology, which
   computes the metrics, scores the cross-section, applies risk gates and sizes
   positions. Plus an executed run against real IBKR data in `output/`.
3. **`web/`** — a standalone page that runs the whole model in the browser and
   re-pulls from IBKR on demand.
4. **`notebooks/`** — a Colab notebook that runs the same engine over a
   ~600-name universe on live Yahoo Finance data, no broker session required.

---

## Quick start

**In the browser, nothing to install:**
[open the notebook in Colab](https://colab.research.google.com/github/Andrespm12/Portfolio-COnstruction/blob/claude/stock-picking-screening-metrics-b0b2dh/notebooks/screener_colab.ipynb)
→ *Runtime → Run all*.

**Locally:**

```bash
pip install pandas numpy
python3 scripts/build_market_data.py     # regenerate data/ from the IBKR pull
python3 -m screener.run_screen           # score, gate, size, report
python3 scripts/build_page.py            # bundle the page -> web/screener.html
python3 scripts/build_notebook.py        # bundle the notebook -> notebooks/

python3 tests/test_scoring.py            # engine correctness
python3 tests/test_yahoo_adapter.py      # Yahoo -> payload conversion
python3 tests/test_tuning.py             # runtime config overrides
python3 tests/test_notebook.py           # notebook drift + full execution
node tests/verify_js_engine.js && python3 tests/compare_engines.py   # JS/Python parity
```

Outputs land in `output/screen_results.csv` and `output/screen_report.md`.

---

## The notebook

`notebooks/screener_colab.ipynb` runs the **same engine** as this repo — the
`screener` package is embedded as a gzipped tarball and verified by SHA256 at
startup — but sources its data from Yahoo Finance instead of IBKR. That trade
buys two things the IBKR path cannot give: a universe of ~600 names instead of a
21-name captured snapshot, and prices that are live rather than frozen at
capture time.

What Yahoo cannot supply is stated rather than faked. `iv_percentile` needs a
*history* of implied volatility; Yahoo publishes today's option chain only, so
the metric is **omitted from the payload, not zero-filled** — the scorer
renormalizes the block over its surviving metrics. `iv_hv_spread` is available
behind an opt-in flag that costs two requests per ticker. A coverage cell prints
exactly which metrics went missing, and at what rate, before any ranking is
shown.

Two price series are carried deliberately: adjusted close feeds every
return-based metric (a dividend must not read as a price decline), while raw
close, high and low feed the snapshot, because the `min_price` filter and the
liquidity block are about the price a share actually trades at. Mixing them
would corrupt `pct_from_52w_high` and `range_position`, which compare a last
price against a 52-week band.

Block weights and risk gates are judgement calls, so the notebook can change
them at runtime via `screener.tuning` — `set_block_weights({'momentum': 0.0})`
answers "what does this model say without momentum?", and `block_weights(...)`
scopes a sweep. This is not a one-liner: `scoring` and `report` bind
`FACTOR_MODEL` at import time, so assigning to `config.FACTOR_MODEL` alone is a
silent no-op that leaves the scorer on the old weights while the run still
produces numbers. `tuning` rebinds every holder and `tests/test_tuning.py`
asserts on scoring *output*, not on the config attribute.

The notebook is a build artifact of `scripts/build_notebook.py`, never
hand-edited. `tests/test_notebook.py` rebuilds it from current source and fails
on any drift, checks each embedded module byte-for-byte against the repo, then
executes every code cell — engine unpack, parameters, coverage, scoring, both
styled tables, the detail view, export and tuning — with only the network call
stubbed.

---

## The page

`web/screener.html` is a single self-contained file that carries the **entire
scoring engine ported to JavaScript** plus an embedded snapshot of the IBKR data.
Open it and the model runs — ranking, factor heatmap, gates and indicative
sizing, all computed in the browser, nothing precomputed.

**Press "Correr modelo" and, when an IBKR connector is reachable, the page
re-pulls live snapshots, price history and account positions, then re-scores the
full cross-section in place.** Without a connector it runs on the embedded
snapshot and says so — the page never silently shows stale numbers as live ones.

The JS port is not trusted on faith. `tests/verify_js_engine.js` extracts the
engine straight out of the built HTML, runs it, and `tests/compare_engines.py`
diffs every recommendation, composite z, 0–100 score, indicative weight, all
seven block scores and gate counts against the Python engine at `1e-6`. A port
that silently diverges would be worse than no port at all: the page would show
confident numbers that disagree with this repo's own results.

Design notes: the ranking, heatmap and score bars use a blue↔orange diverging
pair (`#0369A1`/`#C2410C` light, `#2E9BD6`/`#D97706` dark) with slate as the
neutral midpoint — validated for colorblind separation rather than eyeballed
(worst adjacent pair ΔE 20.1 protan). Recommendations are never encoded by color
alone; every pill carries its label.

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
  yahoo_adapter.py           Yahoo Finance -> the same payload, for broker-free runs
  tuning.py                  Runtime weight/gate overrides that actually reach the scorer
web/
  template.html              The page: JS port of the engine, styling, IBKR live-refresh
  screener.html              Built artifact (template + embedded data) — open this
notebooks/
  screener_colab.ipynb       Built artifact (engine + Yahoo pull) — open this in Colab
scripts/
  build_market_data.py       Captured IBKR pull -> data/*.json
  build_page.py              template.html + data -> web/screener.html
  build_notebook.py          screener/ -> notebooks/screener_colab.ipynb
tests/
  test_scoring.py            12 assertions, mostly on metric direction
  test_yahoo_adapter.py      50 assertions on the Yahoo -> payload conversion
  test_tuning.py             31 assertions that overrides reach the scorer
  test_notebook.py           Rebuilds, diffs and executes every notebook cell
  verify_js_engine.js        Extracts and runs the engine out of the built page
  compare_engines.py         Diffs JS output against Python at 1e-6
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
