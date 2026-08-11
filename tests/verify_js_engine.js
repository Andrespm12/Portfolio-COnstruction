/**
 * Parity check: the JavaScript engine embedded in the published page must
 * reproduce the Python engine's output exactly.
 *
 * A port that silently diverges is worse than no port at all -- the page would
 * show confident numbers that disagree with the repo's own results. This script
 * extracts the engine out of web/screener.html, runs it, and emits JSON for
 * tests/compare_engines.py to diff against the Python CSV.
 */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(ROOT, "web", "screener.html"), "utf8");

const START = '"use strict";';
const END = "/* ================================================================\n   Render";

const s = html.indexOf(START);
const e = html.indexOf(END);
if (s < 0 || e < 0) {
  console.error("Could not locate the engine block in screener.html");
  process.exit(1);
}

const engine = html.slice(s, e);
const runner = engine + "\n;module.exports={runModel,DATA};";

const mod = { exports: {} };
new Function("module", "exports", runner)(mod, mod.exports);

const { runModel, DATA } = mod.exports;
const res = runModel(DATA.market, DATA.portfolio);

const out = res.rows.map((r, i) => ({
  rank: i + 1,
  ticker: r.ticker,
  rec: { OW: "OVERWEIGHT", MW: "MARKET WEIGHT", UW: "UNDERWEIGHT" }[r.rec],
  score: r.score,
  cz: r.cz,
  weight: r.weight,
  blocks: r.blocks,
  ret_1y: r.dg.return_1y,
  vol: r.dg.volatility,
  maxdd: r.dg.max_drawdown,
  beta: r.dg.beta,
  sharpe: r.raw.sharpe_1y,
  corr: r.raw.corr_to_portfolio,
  gates: r.gates.map((g) => (typeof g === "string" ? g : g.t)),
}));

fs.writeFileSync(
  path.join(ROOT, "tests", "_js_output.json"),
  JSON.stringify({ rows: out, meta: res.meta }, null, 1),
);
console.log(`JS engine ran: ${out.length} rows -> tests/_js_output.json`);
for (const r of out.slice(0, 5)) {
  console.log(`  ${r.rank} ${r.ticker.padEnd(6)} ${r.rec.padEnd(14)} score ${r.score.toFixed(1)} z ${r.cz.toFixed(4)}`);
}
