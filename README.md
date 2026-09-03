# Portfolio Construction — Quantitative Stock & ETF Screener

A systematic screen over US-listed equities and ETFs that produces a composite
score and an **Overweight / Market Weight / Underweight** call for every name,
scored against a live Interactive Brokers account.

Five deliverables:

1. **[`PROMPT_SCREENING.md`](PROMPT_SCREENING.md)** — the reusable LLM prompt
   encoding the full methodology.
2. **`screener/`** — the Python implementation of that methodology, which
   computes the metrics, scores the cross-section, applies risk gates and sizes
   positions. Plus an executed run against real IBKR data in `output/`.
3. **`web/`** — a standalone page that runs the whole model in the browser and
   re-pulls from IBKR on demand.
4. **`notebooks/`** — a Colab notebook that runs the same engine over a
   447-name candidate universe on live Yahoo Finance data, with **no account data and no
   broker session**, under one of four risk profiles.
5. **[`PROMPT_BL_INTEGRATION.md`](PROMPT_BL_INTEGRATION.md)**, `screener/black_litterman.py`
   and `screener/optimizer.py` — the analysis prompt, the bridge that turns
   rankings into Black-Litterman views, and the constrained optimizer that turns
   those into a CCI-compliant portfolio.

---

## Quick start

**In the browser, nothing to install:**
[open the notebook in Colab](https://colab.research.google.com/github/Andrespm12/Portfolio-COnstruction/blob/claude/stock-picking-screening-metrics-b0b2dh/notebooks/screener_colab.ipynb)
→ *Runtime → Run all*.

Upload **only the `.ipynb`** — the engine travels inside it. There is nothing
else to put in the Colab filesystem, and the transparency section downloads the
ETF composition it needs on its own.

**The whole model, no notebook:**

```bash
pip install pandas numpy yfinance openpyxl cvxpy scikit-learn
python3 scripts/bajar_tenencias.py                # ETF composition, once
python3 scripts/correr_modelo.py                  # screen, views, diagnostics, portfolio
python3 scripts/correr_modelo.py --perfil Agresivo --ancla mercado
python3 scripts/correr_modelo.py --tickers SPY,QQQ,TLT,GLD,AAPL --con-nombres
```

Same package, same order, same output as the notebook — `screening.xlsx` plus
the proposals JSON under `propuestas/`. `--help` lists every flag.

`bajar_tenencias.py` is optional: without it, section 8b reports 0% look-through
coverage instead of estimating what it cannot see, and everything else runs
unchanged.

**A self-contained zip, for a machine without the repo:**

```bash
python3 scripts/build_bundle.py           # -> output/modelo_cci.zip
```

The package, the two runnable programs and their requirements — no tests, no
notebook, no web. `tests/test_bundle.py` extracts it and runs both programs from
the extracted copy, so a missing module fails here rather than on the recipient's
machine.

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
python3 tests/test_profiles.py           # risk profiles + account-independence
python3 tests/test_black_litterman.py    # BL bridge, against CCI's own solver
python3 tests/test_optimizer.py          # posterior, regulatory bands, audit
python3 tests/test_notebook.py           # notebook drift + full execution
python3 tests/test_diagnostics.py        # block overlap + view saturation
python3 tests/test_correr_modelo.py      # the runner script, end to end
python3 -m pytest tests/test_bundle.py tests/test_tenencias_yahoo.py
node tests/verify_js_engine.js && python3 tests/compare_engines.py   # JS/Python parity
```

Outputs land in `output/screen_results.csv` and `output/screen_report.md`.

---

## What constrains the portfolio

CCI's Investment Procedure bands are by **asset class**, and two things they do
not constrain turned out to decide most of a live Agresivo book:

**Industry concentration.** That run came out ~35% in one semiconductor chain —
eleven single names plus EWY, which is largely Samsung and SK Hynix — and passed
its band audit clean, because every one of those positions is "Equity" and
Equity was inside its ceiling. The audit was right and the portfolio was still a
sector fund. `SECTOR_CAPS` adds a per-sector ceiling (15/18/22/25% by strategy)
enforced *through the funds*, so a sector ETF and a single name in the same
industry compete for one limit. Those numbers are the desk's, not the
Procedimiento's, and every run says so.

Two details that make the difference between a real limit and a fiction:

- Sector labels are normalized. Yahoo spells a fund's sector `technology` and a
  stock's `Technology`; left alone they are two buckets, each gets the full
  ceiling, and a 25% cap permits 50% in one industry.
- Stock sectors are fetched for the basket even when `--con-nombres` is off,
  which is the default. Without that the ceiling would see only the ETFs and
  wave through the single-name concentration it exists to stop.

Anything with no sector data stays **outside** the ceiling and is named in the
run. An unconstrained sleeve reported as such beats a compliance number computed
over half the book.

**What the optimizer is even offered.** `select_basket` used to hand over the
top names by score plus a floor per asset class. Momentum is 25-36% of the
factor model and clusters by industry, so the top of the ranking is whatever ran
hardest — in that run the only equity ETFs in the basket were XBI, EWT and EWY,
while SPY sat at #149, IWM at #115 and EEM at #119. None of them was ever
offered, so "the model rejected broad index exposure" was never true.

`EXPOSICIONES_NUCLEO` now puts one vehicle per core exposure in the basket by
construction. Which exposures exist is policy; which fund delivers one is the
screener's score picking among near-identical products. A ranking decides what
runs well; it must not also decide what is available.

**Risk that matches the mandate.** All four strategies used to maximize the
same utility with `lambda = 2.5`, so the only difference between an Aggressive
book and a Conservative one was the width of its bands -- and a band is a
ceiling, so nothing made the Aggressive one use it. Each mandate now carries its
own risk aversion (8.0 / 5.0 / 2.5 / 1.5), and the penalty applies to **active
risk against the Modelo de Asignación**, not to total risk.

That distinction is the whole design. Penalizing total risk with a per-mandate
lambda breaks the property the anchor exists for: with no views a defensive book
came out 54 points away from the allocation the Committee approved, because the
optimizer mixed the anchor with the minimum-variance portfolio. Penalizing the
deviation gives both -- with no views the book *is* the Modelo for every
mandate, whatever the lambda, and with views the lambda decides how far that
mandate is willing to move away from it. A conservative mandate deviating less
for the same view is what being conservative means.

`RISK_TARGETS` states the volatility range each mandate is expected to land in,
and `risk_profile_table` solves all four with the same basket and the same views
so the comparison isolates the mandate. It reports expected return, volatility,
in-sample max drawdown, worst rolling 12 months and a parametric 95% one-year
loss, and flags any inversion -- an aggressive book taking less risk than a
moderate one contradicts what the client signed.

The floor and the ceiling are not treated alike, for a mathematical reason
rather than a stylistic one: `w'Sigma w <= max**2` is convex and `w'Sigma w >=
min**2` is not. So a book that solves below its floor is re-solved maximizing
return against the ceiling -- the risk budget becomes something the portfolio is
built to use -- and both ends are audited. Those findings stay **out** of
`breaches`, which means the Investment Procedure was violated; a volatility
range the Committee has not approved must not fire a compliance signal.

**Views the book cannot contradict.** A relative view long A / short B now
constrains `w_A >= w_B`. A live run held MU at 5.84% against LRCX at 5.85% on a
view saying MU wins -- the portfolio was marginally short its own call. It is a
floor and not a margin: `0 >= 0` satisfies it, so it forbids the contradiction
without forcing a position into the book. Long-only cannot express the short
leg, and sizing to the spread anyway would put on a bet nobody approved.

---

## The notebook

`notebooks/screener_colab.ipynb` runs the **same engine** as this repo — the
`screener` package is embedded as a gzipped tarball and verified by SHA256 at
startup — but sources its data from Yahoo Finance instead of IBKR. That trade
buys two things the IBKR path cannot give: a candidate universe of 447 names
(317 stocks + 130 ETFs, before the selection policy filters it) instead of a
21-name captured snapshot, and prices that are live rather than frozen at
capture time.

### Independent of any account

The notebook reads no account data. Every name is scored on its own merits, and
the Portfolio Fit block is **removed from the factor model rather than
zero-weighted**.

That distinction is load-bearing, not pedantic. Passing an empty book would not
have been equivalent: with no positions, `compute_portfolio_fit` still returns
`existing_overlap = 0.0` for every name — a real number, identical across the
cross-section — which the scorer would z-score and count as a populated block in
the recommendation-band logic. An empty account would still have shaped the
composite. Dropping the block is the only way to get a genuinely standalone
screen, and `tests/test_profiles.py` asserts the metrics are *absent*, not zero.

The account-aware path is untouched: `run()` still takes a book, still computes
Portfolio Fit, and still matches the JavaScript engine at `1e-6`.

### Risk profiles

`PERFIL` selects **Conservador**, **Moderado** or **Agresivo**, which rewires
four things together — a profile is not a label on the same ranking:

| | Conservador | Moderado | Agresivo |
|---|---|---|---|
| Momentum weight | 12% | 25% | 36% |
| Volatility & drawdown weight | 28% | 17% | 8% |
| Overweight threshold | z ≥ +0.80 | z ≥ +0.50 | z ≥ +0.30 |
| Underweight threshold | z ≤ −0.30 | z ≤ −0.50 | z ≤ −0.70 |
| Vol ceiling for an Overweight | 30% | 60% | 90% |
| Beta limit | 1.00 | 1.30 | 1.80 |
| Max position weight | 5.0% | 8.0% | 12.0% |
| Min average daily volume | $50MM | $20MM | $10MM |

The bands are asymmetric on purpose: the conservative profile makes an
Overweight hard to earn and an Underweight easy to trigger, and the aggressive
one inverts that. The volatility ceiling is loosened but never removed — above
roughly 90% annualized, Sharpe, beta and alpha estimated from ~52 weekly bars
are too noisy to rank on regardless of mandate.

A comparison cell runs all three over the same data. A name that holds an
Overweight in all three is a robust call; one that only survives in Aggressive
is telling you its score depends on forgiving its volatility.

Because each profile applies its own liquidity floor, the three do not score the
same set of names — a $30MM-ADV ETF is eligible for Aggressive and Moderate but
not Conservative. The comparison marks those cells `n/e` and prints the reason
(`ADV $39.3MM below $50MM minimum`) rather than failing. An earlier version
indexed one profile's results by another's tickers and raised `KeyError` on the
first such name; `tests/test_notebook.py` now runs the notebook against a
fixture built specifically to straddle two liquidity floors.

Output is a single `screening.xlsx` with seven sheets — ranking, block scores,
profile comparison, the generated views, the basket, metric coverage, and the
parameters the run used, so the file still explains itself six months later.

A second file, `{Estrategia}_screener_propuestas_{fecha}.json`, is written only
when the BL export is ticked. It is machine input, not a report:
`black_litterman_core` does `json.load()` and expects heterogeneous dicts — an
absolute view carries `activo`, a relative one carries
`activo_long`/`activo_short` — which a flat sheet would force into empty cells,
and Excel silently coerces types on a number that feeds the view-variance matrix
directly. The same views are in the workbook for reading; the JSON exists for the
engine.

Optionally it is written straight into `CCI_BlackLitterman/propuestas/` on
Drive, so the handoff needs no download-then-upload.

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

## Black-Litterman bridge

CCI's tactical allocation system and this screener are complementary, and the
boundary is sharp: **the screener decides which names carry a view and how
strong it is; Black-Litterman decides the weights.**

`screener/black_litterman.py` exports only tactical inputs -- the `Q` vector and
the conviction feeding the view-variance matrix -- in the exact schema
`flujo_aprobacion` and `black_litterman_core` already consume, so the receiving
notebook needs no change. It deliberately does **not** export portfolio weights:
under Black-Litterman those are an output of the constrained optimizer, and
shipping a second unconstrained set alongside them invites exactly the confusion
a model-risk review exists to prevent.

A cross-sectional z-score is a ranking, not a forecast, so the translation is
explicit and risk-scaled:

```
Q_i = IC * z_i * sigma_i          clipped to +/-5%
```

At equal ranking a more volatile name earns more expected return -- which is what
a mean-variance optimizer needs to size correctly -- and a name at the middle of
the cross-section gets exactly zero. **`IC` is a declared assumption, not an
estimate**: the 0.08 default is not calibrated against any backtest, and the
exported JSON says so.

Conviction is kept separate from signal magnitude. Magnitude belongs in `Q`;
conviction measures confidence in the estimate -- how many of the six blocks
agree in sign, how much data backed them, whether a risk gate fired. It is built
as a ceiling scaled by factors each bounded by 1.0 rather than a product clipped
at the end, so a gate always reduces conviction, including on the strongest
names, which are the ones gates exist to restrain.

The four CCI strategies map to four profiles:

| Estrategia | Momentum | Vol/DD | OW `z >=` | Vol ceiling | Max weight | Min ADV |
|---|---:|---:|---:|---:|---:|---:|
| Conservador_Defensivo | 8% | 34% | +1.00 | 22% | 3.5% | $100MM |
| Conservador | 12% | 28% | +0.80 | 30% | 5.0% | $50MM |
| Moderado | 25% | 17% | +0.50 | 60% | 8.0% | $20MM |
| Agresivo | 36% | 8% | +0.30 | 90% | 12.0% | $10MM |

### Proposals are not approvals

Screener output goes to `propuestas/`, never `aprobadas/`, and `write_views`
raises if asked to write anywhere under the latter. This is a guard, not a
convention: CCI's `flujo_aprobacion` writes `{estrategia}_views_{fecha}.json`
into `aprobadas/` *after* a manager has reviewed and justified each view. The
bridge originally used that identical filename, so dropping its output in that
folder would have silently replaced signed-off decisions with unreviewed machine
output — the exact trail the system's Supervised Execution principle exists to
preserve. The proposal filename is now distinct on sight.

`snippets/cci_bl_cargar_propuestas.py` is a paste-in cell for the CCI notebook
that loads the latest proposals, warns when they are stale, and merges them with
the engine's own — deduplicating by conviction, since the same asset proposed by
both sources would otherwise become two near-identical rows of `P`, narrowing
`Ω` artificially and giving that bet more weight than either source justifies
alone. The manager still approves, edits or rejects every view.

`tests/test_black_litterman.py` carries a **verbatim copy** of CCI's
`black_litterman_core` and runs it on the exported views: a test written against
a paraphrase of the consumer proves nothing about the consumer. It also executes
the paste-in snippet against real bridge output rather than shipping it unrun.

---

## The optimizer

The notebook runs the whole chain in one process: screen, views, allocation.
Nothing is written between the halves — `views` is a Python list that the
optimizer cell consumes directly, which is why the file question stopped
mattering.

`screener/optimizer.py` follows CCI's technical document: inverse optimization
for the equilibrium, Ledoit-Wolf shrinkage on daily returns, the Bayesian
posterior, and mean-variance maximization under `screener/cci_regulation.py`.
The two files are split because CCI's own architecture insists on it — the
Investment Procedure delimits risk boundaries and is never the starting point
for weights.

Four defects in the original are not carried over, each a behaviour change:

| | CCI's implementation | Here |
|---|---|---|
| Solver | `ECOS`, absent from a stock Colab — the saved run died there with no allocation | `CLARABEL`, bundled with CVXPY, with fallbacks |
| Leverage | `leverage_max` 1.25/1.50 declared; optimizer hard-coded `sum(w) == 1` | a real gross-exposure budget with the documented 95% buffer |
| Band audit | wrote `"Auditoría OK"` unconditionally | compares every limit and returns the breaches |
| Equilibrium anchor | market cap normalized against ETF AUM — anchors near 95% equity | `policy_weights`: the mandate's own neutral portfolio |

An audit that cannot fail is worse than none: it leaves a document asserting
compliance that nothing verified.

### The anchor is most of the answer

`π = λ · Σ · w` passes `w` straight through. With eight views over twenty-odd
assets, roughly three quarters of the final weights come from the anchor — it is
the largest single decision in the allocation, not a technicality.

Normalizing single-stock market capitalization against ETF assets under
management gets it wrong twice. The units do not match: a company's market cap is
what the company is worth, a fund's AUM is how much money sits in that wrapper,
and for an equity ETF it double-counts shares already priced elsewhere in the
basket. And it ignores the mandate — that calculation anchors near 95% equity,
which no strategy here permits.

The second problem is the visible one. Under the market anchor the solved
portfolio runs into the equity ceiling; under the policy anchor it lands well
inside. Total equity weight on the seven-class test basket (17 names, 8 views):

| Strategy | Ceiling | Market anchor | Policy anchor |
|:--|--:|--:|--:|
| Conservador | 50% | 50.0% | 18.4% |
| Moderado | 60% | 59.4% | 27.4% |
| Agresivo | 80% | 62.8% | 33.6% |

Pinned to the limit — as Conservador is, exactly — means the band was making the
asset-allocation decision and the model was only choosing what was left over
inside it. (These are fixture numbers, not a forecast: they move with the basket.
Reproduce them by re-solving `world()` from `tests/test_optimizer.py` under both
anchors.)

`policy_weights` anchors on the mandate instead: each class at its band midpoint,
renormalized over the classes actually in the basket, with the equity ceiling
applied to the anchor itself. Within a class the split is by market cap, which is
where comparing market values is meaningful. The property that matters: **with no
views, the optimizer returns the anchor** — verified to within 7e-5 across all
four strategies — so the model's output with nothing to say is the mandate's own
allocation, and views tilt away from it.

**Band midpoints are not a strategic asset allocation**, and the code says so in
its notes. A real SAA is an Investment Committee decision and CCI's documents
supply bands, not targets. Pass `policy_weights(..., targets={...})` when those
numbers exist. Until then, note that renormalizing over the classes present means
the anchor moves with basket composition in a way a genuine SAA would not.

**One gap needs Compliance, not code.** CCI's `REGULACIONES` has no asset class
for commodities, and their optimizer constrains only classes present in
`bandas` — an unconstrained gold sleeve could take the entire book.
`COMMODITY_BANDS` closes it with a ceiling per strategy, but those numbers are
placeholders chosen so the hole is loud rather than silent. They are not from
the Investment Procedure.

Market caps are never defaulted. CCI's code substitutes `1e9` on any lookup
failure, straight into the equilibrium; `market_weights` returns the missing
names so the run can report them, and `policy_weights` degrades a single class
to an equal split rather than letting one gap move the whole portfolio.

### Nobody chooses the number of positions

Worth stating because it is a natural thing to assume and it is not true. There
is no diversification target and no maximum-holdings constraint anywhere in the
model. The count falls out of two steps:

1. `select_basket` picks the *candidate* universe — `TOP_N_CARTERA` names by
   score (25 by default) plus at least three per asset class so the bands are
   reachable. That is a ceiling on what may be considered, not a target for what
   is held.
2. The optimizer puts weight wherever it improves `w'μ − (λ/2)w'Σw` subject to
   the constraints. How many names end up non-zero is an outcome of that.

So the answer to "why fifteen positions?" is "because that is what the
mathematics produced," which is fine as far as it goes — and is exactly why the
floor below is needed.

### Positions too small to trade

An optimizer has no notion of what is worth executing. On the reference run it
returned XLF at **0.159%** and a second name at effectively zero: positions that
cost a ticket, a line on every report and a reconciliation forever, in exchange
for a risk contribution that rounds to nothing. On a US$5MM book, 0.159% is
US$8,000.

`min_position` (default 1%, `MIN_POSITION`) removes them. It is enforced by
**re-solving with those names forced to zero**, not by deleting weights from the
answer — deleting would leave the book short of its budget and could push a
survivor past its individual cap or a class past its band, so the weights would
no longer solve any stated problem while still being presented as the solution
to one. Each pass is a genuine constrained optimization, and the audit stays
clean:

| | Positions | Gross | Smallest | Breaches |
|:--|--:|--:|--:|--:|
| No floor | 15 | 111.91% | 0.000% | 0 |
| 1% floor | 13 | 111.75% | 1.99% | 0 |

The loop only ever removes names, so it terminates on its own. If dropping them
would make the problem infeasible, the last feasible portfolio is kept and the
reason is written into the notes rather than an empty book being returned.

This is a desk convention, not a regulatory limit — CCI's Investment Procedure
sets ceilings, never floors — which is why it is a parameter rather than a
constant.

### The basket must span the bands

`select_basket` picks the optimizer's universe class-aware rather than as the
top N by score, and that is a correctness requirement, not a refinement. The
screen ranks on momentum and risk-adjusted return, which equities dominate, so
the top of the list is routinely all equity — and every mandate caps total
equity below the amount the book must invest. The solver returns infeasible and
the portfolio comes out empty.

CCI's system never hits this because its basket comes from a hand-maintained
sheet that deliberately spans bonds, credit, cash and equity. Replacing the
sheet means reproducing that property.

`feasibility_report` states the structural reason before the solver runs, so an
empty result carries its cause — "la cesta es solo renta variable, y Moderado la
limita a 60%" — instead of a bare `infeasible`.

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
| Market Sensitivity | 12% | Up/down capture spread, Jensen's alpha, idiosyncratic vol share, beta — withheld entirely when the benchmark does not explain the name (see below) |
| Liquidity & Tradability | 10% | ADV, days-to-liquidate at 20% participation, volume stability |
| Valuation Proxy & Carry | 10% | Dividend yield, IV−HV spread, IV percentile, 52w range position |
| Portfolio Fit | 13% | Correlation to your book, marginal diversification benefit, existing overlap |

Volatility, drawdown and beta are scored **peer-relative** (ETF cohort vs stock
cohort). Without that, every ETF would top the risk block by construction and the
screen would degenerate into "prefer diversified things."

Weights **re-normalize** over available data rather than imputing zeros. Imputing
a zero silently asserts "average," which is a factual claim about a security you
have no measurement for.

### The bands are relative; the floor is not

Standardizing the composite carries no information about absolute quality. A
threshold of z ≥ 0.50 names roughly the top third of *this* universe whether
every name doubled or every name halved — z and percentile are monotone
transforms of each other, and neither knows whether the universe is any good.
(An earlier version of `config.py` claimed the opposite. It was wrong, and wrong
in the direction that flatters the model.)

The absolute bar lives in the gates, which run after scoring and can only
downgrade: an Overweight requires positive 12M−1M momentum and a Sharpe above
zero — it beat cash. Measured across regimes on the same universe:

| Regime | Overweights without the floor | With it |
|:--|--:|--:|
| Bull, +20%/yr | 6 | 6 |
| Flat, 0%/yr | 5 | 0 |
| Bear, −10%/yr | 3 | 0 |

The flat case is the one that mattered — the trend gate needs a broken moving
average *and* negative momentum, so it does nothing when prices go sideways, yet
at a 4.25% risk-free rate a flat year is a negative Sharpe. The bull column is
the control: a rail that fires when the universe is genuinely good would be a
bug.

### Market sensitivity is withheld, not faked

All four metrics in that block are defined against a relationship to SPY. When
beta does not clear a t-statistic of 2 — `sqrt((n−2)·R²/(1−R²))`, evaluated at
the sample size actually available — there is no relationship to interpret, and
the block is dropped so the composite renormalizes over the rest.

The trigger is statistical, not categorical, because asset class turned out to be
the wrong predictor: LLY regresses on SPY at R² 0.005 and XLV at 0.021, while GLD
explains better (0.187) than AAPL (0.178). An "exclude fixed income" rule would
have kept the worst case and thrown out a sound one.

What goes wrong if the block stays: at beta ≈ 0, Jensen's alpha collapses into
the asset's own excess return, which the momentum block already scores. LLY was
credited a "+70% alpha" against a +73% total return — the same performance paid
for twice, under a label that reads as skill. Keeping only beta and idiosyncratic
share is worse than it sounds, since both are functions of R² and block scores
renormalize: uncorrelated names then sweep both survivors with no return term
left anywhere, which ranked TLT fifth on "alpha quality" in a year it lost
4.9%, and DBA second on a +2.0% return.

### One year means one year

`mom_12_1` is a 48-week return skipping the most recent 4, so it needs 53
observations, and the download used to retain exactly 53. One missing Friday and
the model's highest-weighted metric returned None, silently. The window is now 60
bars.

That creates the opposite hazard, so `RISK_WINDOW_BARS` pins every "1Y" statistic
to 52 return observations regardless of how much history is retained — otherwise
the extra bars would quietly stretch volatility, Sharpe and beta into 14-month
figures while every label still said one year. The guarantee is tested in its
strong form: 60 bars and the same series cut to 53 produce identical values for
all 25 metrics.

---

## Diagnostics on the model itself

`screener/diagnostics.py` runs on every screen (notebook section 10b) and
changes nothing — it exists to tell you how far to trust the output above it.

**Block correlation.** The model declares six blocks and spends a weight on
each, which asserts each carries information the others do not. If two correlate
at 0.90, their weights are one bet placed twice and the composite is less
diversified than the weight table implies. The report gives the full matrix, the
*effective factor count* (participation ratio of the eigenvalues: equal to the
block count when the blocks are independent, falling toward 1 as they collapse
onto one), and the combined weight of every overlapping pair.

Blocks are excluded rather than reported as noise when they cannot be measured:
below half the cross-section scored (otherwise Portfolio Fit, never populated in
standalone mode, would empty the sample under listwise deletion), or with no
cross-sectional variance — tested on a tolerance, since `np.std` of fifty
identical floats returns ~1.7e-16 rather than zero and an exact comparison would
publish that as a correlation.

**View saturation.** `Q` is clipped at ±5% per CCI's technical document. That
clip is a safety rail, but if most views land on it the rail has become the
signal: names the screener ranked very differently arrive at the optimizer with
identical expected returns, and that part of the ranking is discarded at the last
step. The report counts views at the cap, shows what they would have been
without it, and flags the case where the clip erased a distinction the model
actually made. The correction is to lower the information coefficient, not to
raise the cap — the cap is the calibration.

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
- **52 weekly observations is a short sample.** Beta, alpha and capture ratios
  carry wide confidence intervals. Do not read the third decimal. Where the
  interval is wide enough to swallow the estimate entirely, the block is
  withheld rather than reported — but that is a floor, not a guarantee of
  precision above it.
- **The six blocks do not measure six independent things.** `diagnostics.py`
  reports the effective factor count on every run, and it is well below the
  block count. Part of that is structural: `range_position` (scored −1, inside
  Valuation) is a monotone inverse of `pct_from_52w_high` (scored +1, inside
  Momentum), so those two blocks partly cancel by construction. Read the block
  weights as a stated intent, not as a measured decomposition of risk.
- **Yahoo supplies no implied volatility**, so two of the four Valuation & Carry
  metrics (IV−HV spread, IV percentile — half the block's declared weight) are
  permanently absent on that path. The block renormalizes onto dividend yield and
  range position, which is less than the name promises.
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
  profiles.py                Four CCI mandates; drops the Portfolio Fit block
  black_litterman.py         Ranking -> BL views (Q, conviction)
  cci_regulation.py          CCI's Investment Procedure as data: classes, bands, exclusions
  optimizer.py               Equilibrium anchor, Bayesian posterior, constrained allocation
  diagnostics.py             Measurements on the model itself: block overlap, view saturation
  tuning.py                  Runtime weight/gate overrides that actually reach the scorer
web/
  template.html              The page: JS port of the engine, styling, IBKR live-refresh
  screener.html              Built artifact (template + embedded data) — open this
notebooks/
  screener_colab.ipynb       Built artifact (engine + Yahoo pull) — open this in Colab
scripts/
  correr_modelo.py           The whole chain in one command, no notebook
  build_market_data.py       Captured IBKR pull -> data/*.json
  build_page.py              template.html + data -> web/screener.html
  build_notebook.py          screener/ -> notebooks/screener_colab.ipynb
tests/
  test_scoring.py            12 assertions, mostly on metric direction
  test_yahoo_adapter.py      50 assertions on the Yahoo -> payload conversion
  test_tuning.py             31 assertions that overrides reach the scorer
  test_profiles.py           67 assertions on profiles and account-independence
  test_black_litterman.py    74 assertions, incl. CCI's own solver on the export
  test_optimizer.py          86 assertions on posterior, bands and the audit
snippets/
  cci_bl_cargar_propuestas.py  Paste-in cell for CCI's notebook
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
can never justify a position the factor model does not already support. It is
also why the absolute quality floor is a gate rather than a band: bands are
cross-sectional by construction, and a one-directional gate is the only place in
the pipeline where a name can be measured against something other than its peers.

**A metric that cannot be measured is withheld, never imputed.** This is the same
rule in four places — a missing block, an unpopulated metric, an alpha with no
market model behind it, a correlation on a flat series. Substituting zero asserts
"average," which is a factual claim about something never observed, and it is
indistinguishable from a real measurement downstream.

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
