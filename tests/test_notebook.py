"""
Executes the Colab notebook and checks it against the repo.

Two distinct things are verified here.

**Drift.** The notebook embeds a copy of the ``screener`` package. A copy that
silently falls behind the repo is the whole hazard of shipping one, so this
rebuilds the notebook from current source and fails if the checked-in file
differs. Fix by re-running ``scripts/build_notebook.py``.

**Execution.** Every code cell is run in order in a temporary directory, with
one substitution: the Yahoo download is replaced by a synthetic frame, because
CI has no outbound network. Everything downstream of the download -- unpacking
the engine, the parameter logic, coverage, scoring, both styled tables, the
per-name detail view, export and the tuning helpers -- runs for real against
the same code Colab will run.

What this does NOT prove: that ``yfinance.download`` returns what the adapter
expects. That contract is asserted separately against yfinance's documented
output shape in ``test_yahoo_adapter.py``, but the live call is unexercised
until the notebook is run with internet access.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "screener_colab.ipynb"

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Drift
# --------------------------------------------------------------------------

def test_notebook_matches_source() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_notebook as builder

    fresh = json.dumps(builder.build_notebook(), indent=1, ensure_ascii=False) + "\n"
    on_disk = NOTEBOOK.read_text(encoding="utf-8")
    check("checked-in notebook matches a fresh build from screener/",
          fresh == on_disk,
          "run: python3 scripts/build_notebook.py")

    check("the build is deterministic (rebuild is byte-identical)",
          json.dumps(builder.build_notebook(), indent=1, ensure_ascii=False) + "\n" == fresh)


def test_embedded_engine_is_current() -> None:
    """The embedded tarball must contain byte-identical module sources."""
    import base64
    import gzip
    import io
    import re
    import tarfile

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    setup = next(c for c in nb["cells"]
                 if c["cell_type"] == "code" and "ENGINE_B64" in "".join(c["source"]))
    source = "".join(setup["source"])
    blob = "".join(re.findall(r'^\s*"([A-Za-z0-9+/=]+)"\s*$', source, re.M))

    raw = gzip.decompress(base64.b64decode(blob))
    mismatches: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members = tar.getnames()
        for name in members:
            embedded = tar.extractfile(name).read()
            actual = (ROOT / name).read_bytes()
            if embedded != actual:
                mismatches.append(name)

    check("every embedded module matches the repo byte for byte",
          not mismatches, f"stale: {mismatches}")
    check("all engine modules are embedded",
          len(members) == len(__import__("build_notebook").MODULES),
          f"got {len(members)}")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

OFFLINE_FETCH = """
from test_yahoo_adapter import make_yf_frame
from screener.yahoo_adapter import build_market_data

_frame = make_yf_frame(TICKERS, dividends={'SPY': 6.0, 'JPM': 4.0})
market_data = build_market_data(
    _frame, TICKERS, benchmark=BENCHMARK,
    risk_free_rate=TASA_LIBRE_RIESGO,
)
print(f'{len(market_data["instruments"])} instrumentos (fixture offline)')
"""


def executable_cells(nb: dict) -> list[str]:
    """Code cells, with magics stripped and the network call substituted."""
    out: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])

        if "fetch_market_data(" in source:
            out.append(OFFLINE_FETCH)
            continue
        # %pip / %load_ext are IPython-only; nothing downstream depends on them
        # here because the test environment already has yfinance installed.
        source = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("%")
        )
        out.append(source)
    return out


def test_cells_execute() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = executable_cells(nb)

    # A small custom universe keeps the fixture fast and exercises the
    # custom-list branch of the parameter cell.
    tickers = ["SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "NVDA", "JPM", "LLY"]
    patched = 0
    for i, source in enumerate(cells):
        if "UNIVERSO =" in source:
            cells[i] = source.replace(
                'UNIVERSO = "Completo (S&P + Nasdaq + Dow + ETFs)"',
                'UNIVERSO = "Lista personalizada"',
            ).replace(
                'TICKERS_PERSONALIZADOS = ""',
                f'TICKERS_PERSONALIZADOS = "{",".join(tickers)}"',
            )
            patched += 1
    check("parameter cell exposes the knobs the test needs to patch", patched == 1)

    namespace: dict = {"__name__": "__main__"}
    cwd = os.getcwd()
    workdir = tempfile.mkdtemp(prefix="nb-exec-")
    failures: list[tuple[int, str]] = []

    try:
        os.chdir(workdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, source in enumerate(cells):
                try:
                    exec(compile(source, f"<cell {i}>", "exec"), namespace)
                except Exception as exc:  # noqa: BLE001 - reporting, not handling
                    failures.append((i, f"{type(exc).__name__}: {exc}"))
    finally:
        os.chdir(cwd)

    check("every code cell executes without raising",
          not failures,
          "; ".join(f"cell {i}: {msg}" for i, msg in failures))

    # ---- the namespace the notebook leaves behind -------------------------
    if failures:
        return

    check("engine unpacked and checksum verified",
          "ENGINE_SHA256" in namespace and "FACTOR_MODEL" in namespace)
    check("scored results exist",
          len(namespace.get("scored", [])) >= 8,
          f"got {len(namespace.get('scored', []))}")
    check("ranking table was built with one row per scored name",
          len(namespace["tabla"]) == len(namespace["scored"]))
    check("factor heatmap has a column per block",
          len(namespace["mapa"].columns) == len(namespace["FACTOR_MODEL"]))
    check("coverage report was computed",
          not namespace["_cov"].empty)

    exported = Path(workdir) / "screen_results.csv"
    report = Path(workdir) / "screen_report.md"
    check("CSV export is written", exported.exists() and exported.stat().st_size > 0)
    check("Markdown report is written", report.exists() and report.stat().st_size > 0)

    header = exported.read_text(encoding="utf-8").splitlines()[0]
    check("exported CSV carries the full schema",
          "block_momentum" in header and "indicative_weight" in header)


def test_tuning_cell_leaves_config_untouched() -> None:
    """
    The tuning cell ships its examples commented out. If one were live, every
    run of the notebook would silently score on a different model than
    config.py declares.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"]
                if c["cell_type"] == "code"
                and "set_block_weights" in "".join(c["source"]))
    body = "".join(cell["source"])

    live = [line for line in body.splitlines()
            if line.strip() and not line.strip().startswith("#")]
    mutating = [line for line in live
                if any(call in line for call in
                       ("set_block_weights(", "override(", "block_weights({"))
                and "import" not in line]
    check("no mutating tuning call is live in the shipped notebook",
          not mutating, f"live: {mutating}")


def test_no_outputs_committed() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    with_outputs = [i for i, c in enumerate(nb["cells"]) if c.get("outputs")]
    check("notebook ships with no stale cell outputs", not with_outputs,
          f"cells {with_outputs}")

    counts = [c.get("execution_count") for c in nb["cells"]
              if c["cell_type"] == "code"]
    check("no execution counts committed", all(c is None for c in counts))


def main() -> int:
    for fn in [
        test_notebook_matches_source,
        test_embedded_engine_is_current,
        test_no_outputs_committed,
        test_tuning_cell_leaves_config_untouched,
        test_cells_execute,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
