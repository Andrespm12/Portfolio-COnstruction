"""
Tests for the standalone runner script.

The script exists so the model can be run without a notebook, and the whole
point of it is that it does *the same thing* the notebook does. So the load
bearing tests here are not "it produces a file" -- they are:

* every stage actually ran (a script that silently skips the optimizer and
  still writes a workbook is worse than one that crashes),
* the workbook it writes has the same eleven sheets as the notebook's,
* the proposals JSON lands under ``propuestas/`` and carries no internal keys,
* and the command-line flags reach the model rather than being parsed and
  ignored -- which is the failure mode that leaves someone running Moderado
  for a week while believing they set Agresivo.

The Yahoo download is the one thing that cannot run here, so it is stubbed
exactly as ``test_notebook.py`` stubs it, against the same fixture.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from datetime import date
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import correr_modelo  # noqa: E402
import screener.yahoo_adapter as _ya  # noqa: E402
from screener.black_litterman import DRIVE_PROPOSALS_DIR  # noqa: E402
from screener.tuning import reset_all  # noqa: E402
from screener.yahoo_adapter import build_market_data  # noqa: E402
from test_yahoo_adapter import make_yf_frame  # noqa: E402

PASSED = 0
FAILED = 0

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "DBC", "TLT", "LQD", "BIL", "HYG",
           "XLE", "XLV", "XLF", "AAPL", "MSFT", "NVDA", "JPM", "LLY"]


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS {label}")
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Offline harness
# --------------------------------------------------------------------------

def run_script(*argv: str) -> tuple[str, Path]:
    """
    Run the script's main() with the network calls stubbed.

    Returns its stdout and the output directory. The stubs replace only the two
    functions that reach Yahoo; everything downstream is the real code path.
    """
    tmp = Path(tempfile.mkdtemp(prefix="correr-"))

    real_fetch = _ya.fetch_market_data
    real_caps = _ya.fetch_market_caps
    real_universe = _ya.default_universe

    def fake_fetch(tickers, *, benchmark="SPY", risk_free_rate=0.0425, **kw):
        frame = make_yf_frame(list(tickers), dividends={"SPY": 6.0, "JPM": 4.0})
        # Thin volume on one name so the per-profile liquidity floors actually
        # differ, which is what makes the comparison table exercise `n/e`.
        if ("Volume", "DBC") in frame.columns:
            frame[("Volume", "DBC")] = 250_000.0
        data = build_market_data(frame, list(tickers), benchmark=benchmark,
                                 risk_free_rate=risk_free_rate)
        return (data, frame) if kw.get("with_frame") else data

    def fake_caps(tickers, **kw):
        return {t: 1e10 + 1e9 * i for i, t in enumerate(tickers)}

    _ya.fetch_market_data = fake_fetch
    _ya.fetch_market_caps = fake_caps
    _ya.default_universe = lambda groups, benchmark="SPY": list(TICKERS)
    # The script imports these names into its own frame at call time, so the
    # module-level patch above is what it will actually see.
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            code = correr_modelo.main([*argv, "--salida", str(tmp)])
    finally:
        _ya.fetch_market_data = real_fetch
        _ya.fetch_market_caps = real_caps
        _ya.default_universe = real_universe
        reset_all()

    assert code == 0, f"script returned {code}"
    return buffer.getvalue(), tmp


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_every_stage_runs() -> None:
    """
    A run that quietly skips a stage and still writes a workbook is the
    failure this guards against.
    """
    out, tmp = run_script()

    for stage in ("1 · UNIVERSO", "2 · DESCARGA", "3 · COBERTURA",
                  "4 · SCREENING", "5 · COMPARACIÓN", "6 · VIEWS",
                  "7 · DIAGNÓSTICOS", "8 · CARTERA", "9 · ARCHIVOS"):
        check(f"stage ran: {stage}", stage in out,
              "missing from stdout")

    check("the optimizer reported a status",
          "estado: optimal" in out, out[-600:])
    check("the band audit ran and is reported",
          "Auditoría de bandas" in out or "AUDITORÍA" in out)
    check("both diagnostics printed",
          "Correlación entre bloques" in out and "Saturación de views" in out)
    check("the run reports its equilibrium anchor by asset class",
          "Ancla (politica) por clase de activo" in out)


def test_workbook_matches_the_notebook_shape() -> None:
    """Same eleven sheets the notebook writes, in the same order."""
    from openpyxl import load_workbook

    _, tmp = run_script()
    book = tmp / "screening.xlsx"
    check("the workbook was written", book.exists())

    wb = load_workbook(book)
    expected = ["Ranking", "Bloques", "Perfiles", "Views BL",
                "Cartera", "Sectores", "Riesgo", "Cesta", "Universo",
                "Cobertura", "Parametros"]
    check("eleven sheets, matching the notebook", wb.sheetnames == expected,
          str(wb.sheetnames))
    check("the Cartera sheet is never blank -- positions or a stated reason",
          wb["Cartera"].max_row > 1)
    check("the Ranking sheet has one row per scored name",
          wb["Ranking"].max_row > 5, f"{wb['Ranking'].max_row} rows")

    params = {row[0].value: row[1].value
              for row in wb["Parametros"].iter_rows(min_row=2)}
    check("Parametros records that no account was read",
          params.get("Portafolio", "").startswith("ninguno"))
    check("Parametros records the anchor actually used",
          params.get("Ancla del equilibrio") == "politica",
          str(params.get("Ancla del equilibrio")))
    check("Parametros records the absolute quality floor",
          "Sharpe" in str(params.get("Piso absoluto para Overweight", "")),
          str(params.get("Piso absoluto para Overweight")))
    check("Parametros still flags the IC as an assumption",
          "no calibrado" in str(params.get("Nota sobre el IC", "")))
    check("Parametros names the Modelo de Asignación as the anchor's source",
          "Modelo de Asignación" in str(params.get("Nota sobre el ancla", "")),
          str(params.get("Nota sobre el ancla"))[:160])
    check("Parametros no longer claims the anchor is band midpoints",
          "puntos medios" not in str(params.get("Nota sobre el ancla", "")),
          str(params.get("Nota sobre el ancla"))[:160])


def _companyfacts(anio: int, base: float) -> dict:
    """Un companyfacts con la forma real, para el EDGAR sustituido."""
    fin, ini = f"{anio}-12-31", f"{anio}-01-01"
    filed = f"{anio + 1}-02-15"

    def flujo(v):
        return {"units": {"USD": [{"val": v, "start": ini, "end": fin,
                                   "filed": filed, "form": "10-K",
                                   "accn": "a", "fy": anio, "fp": "FY"}]}}

    def saldo(v):
        return {"units": {"USD": [{"val": v, "end": fin, "filed": filed,
                                   "form": "10-K", "accn": "a", "fy": anio,
                                   "fp": "FY"}]}}

    return {"cik": 1, "entityName": "PRUEBA INC", "facts": {"us-gaap": {
        "Revenues": flujo(1000.0 * base),
        "NetIncomeLoss": flujo(100.0 * base),
        "OperatingIncomeLoss": flujo(150.0 * base),
        "DepreciationDepletionAndAmortization": flujo(50.0 * base),
        "NetCashProvidedByUsedInOperatingActivities": flujo(180.0 * base),
        "PaymentsToAcquirePropertyPlantAndEquipment": flujo(40.0 * base),
        "EarningsPerShareDiluted": {"units": {"USD/shares": [
            {"val": base, "start": ini, "end": fin, "filed": filed,
             "form": "10-K", "accn": "a", "fy": anio, "fp": "FY"}]}},
        "Assets": saldo(2000.0 * base),
        "StockholdersEquity": saldo(800.0 * base),
        "CashAndCashEquivalentsAtCarryingValue": saldo(100.0 * base),
        "LongTermDebtCurrent": saldo(50.0 * base),
        "LongTermDebtNoncurrent": saldo(350.0 * base),
        "LiabilitiesCurrent": saldo(300.0 * base),
        "CommonStockSharesOutstanding": {"units": {"shares": [
            {"val": 100.0, "end": fin, "filed": filed, "form": "10-K",
             "accn": "a", "fy": anio, "fp": "FY"}]}},
    }}}


def _almacen_falso(tickers) -> Path:
    """Un almacén de EDGAR con un ejercicio anual completo por nombre."""
    from screener.edgar import escribir_hechos
    from test_fundamentales import emisor  # noqa: E402

    from screener.edgar import Hecho

    # El ejercicio tiene que ser reciente de verdad: el módulo descarta un
    # cierre de más de 550 días porque un emisor que dejó de reportar no
    # describe a la empresa de hoy. Con un año fijo la prueba caducaría sola.
    destino = Path(tempfile.mkdtemp(prefix="fund-"))
    for n, ticker in enumerate(tickers):
        filas = emisor(ticker, anio=date.today().year - 1,
                       eps=float(n + 1), utilidad=100.0 * (n + 1))
        escribir_hechos(destino, ticker,
                        [Hecho(**{k: (v or None) if k in ("inicio", "fp")
                                  else v for k, v in f.items()})
                         for f in filas])
    return destino


def test_the_fundamental_block_runs_on_real_edgar_facts() -> None:
    """
    Phase 3 end to end: the store on disk becomes point-in-time ratios, and
    those ratios land in the workbook with the period they came from.

    A P/E with no date cannot be audited against the 10-K that produced it,
    which is why the sheet carries `periodo` and `filed` next to every ratio.
    """
    from openpyxl import load_workbook

    almacen = _almacen_falso([t for t in TICKERS if t not in
                              ("SPY", "QQQ", "IWM", "GLD", "TLT", "XLE",
                               "XLV", "DBC")])
    out, tmp = run_script("--fundamentales", str(almacen))

    check("the run announces the fundamental stage",
          "2b · FUNDAMENTALES" in out, out[:400])
    check("it says how many names got ratios",
          "con ratios" in out, out[:400])
    check("it reports the cohort per ratio, not just a total",
          "earnings_yield" in out and "puntuable" in out)
    check("quality ratios are computed but declared unscored",
          "roe" in out and "ponerla a puntuar exige decidir su peso" in out)

    wb = load_workbook(tmp / "screening.xlsx")
    check("the workbook gained the Fundamentales sheet",
          "Fundamentales" in wb.sheetnames, str(wb.sheetnames))
    encabezados = [c.value for c in wb["Fundamentales"][1]]
    check("every ratio carries the fiscal year it came from",
          {"periodo", "filed"} <= set(encabezados), str(encabezados))
    check("the scored yields are there", "earnings_yield" in encabezados,
          str(encabezados))
    check("the unscored quality ratios are NOT in the scored columns",
          "roe" not in encabezados, str(encabezados))
    check("one row per name with fundamentals",
          wb["Fundamentales"].max_row > 3,
          f"{wb['Fundamentales'].max_row} rows")


def test_the_run_fills_an_empty_store_by_itself() -> None:
    """
    The point of wiring the download into the run: an empty folder and a
    contact address are enough. Nobody has to remember a separate step.
    """
    import screener.descarga as descarga
    import screener.edgar as edgar
    from openpyxl import load_workbook
    from test_bajar_fundamentales import CUERPOS, TICKERS_EXCHANGE

    from screener.yahoo_adapter import classify

    # Un ETF no tiene estados financieros, así que la corrida ni se los pide.
    acciones = [t for t in TICKERS if classify(t) != "ETF"]
    anio = date.today().year - 1
    mapa = {str(i): {"cik_str": 900000 + i, "ticker": t}
            for i, t in enumerate(acciones)}
    mapa |= {str(i): {"cik_str": 500000 + i, "ticker": f"T{i:05d}"}
             for i in range(len(acciones), edgar.MIN_EMISORES + 1)}
    cuerpos = {f"{900000 + i:010d}": _companyfacts(anio, float(i + 1))
               for i in range(len(acciones))}

    vacio = Path(tempfile.mkdtemp(prefix="fund-vacio-"))
    real_json, real_text = edgar.fetch_json, edgar.fetch_text
    edgar.fetch_json = lambda url, **kw: (
        mapa if url == edgar.SEC_TICKERS
        else TICKERS_EXCHANGE if url == edgar.SEC_TICKERS_EXCHANGE
        else cuerpos[url.rsplit("CIK", 1)[-1].removesuffix(".json")])
    edgar.fetch_text = lambda url, **kw: ""
    real_rps = edgar.SEC_MAX_RPS
    edgar.SEC_MAX_RPS = 0
    try:
        out, tmp = run_script("--fundamentales", str(vacio),
                              "--contacto-sec", "Pruebas x@y.com")
    finally:
        edgar.fetch_json, edgar.fetch_text = real_json, real_text
        edgar.SEC_MAX_RPS = real_rps

    check("the run says it is filling the store",
          "Bajando de EDGAR" in out, out[:800])
    check("it downloaded one file per name",
          all((vacio / f"{t}.csv").exists() for t in acciones),
          str(sorted(p.name for p in vacio.glob("*.csv"))))
    check("and then scored on what it just downloaded",
          "Fundamentales" in load_workbook(tmp / "screening.xlsx").sheetnames)

    # Segunda corrida sobre el mismo almacén: nada que bajar.
    edgar.fetch_json = lambda url, **kw: (_ for _ in ()).throw(
        AssertionError("no debió salir a la red"))
    edgar.fetch_text = lambda url, **kw: ""
    try:
        out2, _ = run_script("--fundamentales", str(vacio),
                             "--contacto-sec", "Pruebas x@y.com")
    finally:
        edgar.fetch_json, edgar.fetch_text = real_json, real_text
    check("the second run downloads nothing",
          "Bajando de EDGAR" not in out2, out2[:800])
    check("and says everything was already there",
          "ya estaban" in out2, out2[:600])


def test_without_a_contact_the_run_explains_instead_of_failing() -> None:
    """
    The SEC blocks by IP whoever does not identify. An empty block that
    explains itself is fine; one that looks like a bug is not.
    """
    vacio = Path(tempfile.mkdtemp(prefix="fund-sin-contacto-"))
    out, _ = run_script("--fundamentales", str(vacio))
    check("it names the reason and the remedy",
          "Sin --contacto-sec" in out and "bloquea por IP" in out, out[:600])
    check("and the model still runs", "8 · CARTERA" in out)


def test_without_a_store_the_model_runs_exactly_as_before() -> None:
    """The store is optional. What is not optional is saying which one ran."""
    from openpyxl import load_workbook

    out, tmp = run_script("--fundamentales", "")
    check("no fundamental stage without a store", "2b · FUNDAMENTALES" not in out)
    wb = load_workbook(tmp / "screening.xlsx")
    check("and no Fundamentales sheet either",
          "Fundamentales" not in wb.sheetnames, str(wb.sheetnames))


def test_proposals_land_in_the_right_folder() -> None:
    """
    The governance boundary. Proposals go to propuestas/, never aprobadas/,
    and the internal diagnostic key must not reach the file a manager signs.
    """
    _, tmp = run_script()
    proposals = list((tmp / DRIVE_PROPOSALS_DIR).glob("*.json"))
    check("the proposals JSON was written under propuestas/",
          len(proposals) == 1, str(list(tmp.rglob('*.json'))))

    payload = json.loads(proposals[0].read_text(encoding="utf-8"))
    check("the file names the strategy it targets",
          payload.get("estrategia") == "Moderado")
    check("the file states the screen read no account",
          "sin datos de cuenta" in payload.get("origen", ""))
    check("no internal diagnostic keys reached the file",
          all(not k.startswith("_") for v in payload["views"] for k in v))
    check("the calibration block travels with the views",
          "calibracion" in payload and
          "nota" in payload["calibracion"])

    _, tmp2 = run_script("--sin-json")
    check("--sin-json suppresses the proposals file",
          not list(tmp2.rglob("*.json")),
          str(list(tmp2.rglob("*.json"))))


def test_flags_actually_reach_the_model() -> None:
    """
    A flag that parses but never reaches the model is the worst kind of bug
    here: the run looks right and is scored under a different mandate.
    """
    from openpyxl import load_workbook

    _, tmp = run_script("--perfil", "Agresivo", "--estrategia", "Agresivo",
                        "--ic", "0.03", "--top-n", "12", "--max-views", "3")
    params = {row[0].value: row[1].value
              for row in load_workbook(tmp / "screening.xlsx")["Parametros"]
              .iter_rows(min_row=2)}

    check("--perfil reached the screen", params.get("Perfil") == "Agresivo",
          str(params.get("Perfil")))
    check("--estrategia reached the export",
          params.get("Estrategia CCI destino") == "Agresivo")
    check("--ic reached the view translation", params.get("IC supuesto (views)") == 0.03,
          str(params.get("IC supuesto (views)")))
    check("--perfil rewired the gates, not just the label",
          params.get("Beta máxima") == "1.80", str(params.get("Beta máxima")))
    check("--perfil rewired the block weights",
          params.get("Peso — Momentum & Trend") == "36%",
          str(params.get("Peso — Momentum & Trend")))

    proposals = json.loads(
        next((tmp / DRIVE_PROPOSALS_DIR).glob("*.json")).read_text(encoding="utf-8"))
    check("--max-views capped the export", len(proposals["views"]) <= 3,
          f"{len(proposals['views'])} views")


def test_market_anchor_flag_changes_the_anchor() -> None:
    """Both anchors must run, so the two can actually be compared."""
    out, tmp = run_script("--ancla", "mercado")
    check("--ancla mercado runs end to end", "estado: optimal" in out)
    check("the market anchor is reported as such",
          "Ancla (mercado) por clase de activo" in out)

    from openpyxl import load_workbook
    params = {row[0].value: row[1].value
              for row in load_workbook(tmp / "screening.xlsx")["Parametros"]
              .iter_rows(min_row=2)}
    check("the workbook records which anchor was used",
          params.get("Ancla del equilibrio") == "mercado")


def test_custom_ticker_list() -> None:
    """--tickers implies the custom universe and warns about the product filter."""
    args = correr_modelo.parse_args(["--tickers", "SPY,QQQ,TLT"])
    check("--tickers switches the universe to the custom list",
          args.universo == "lista", args.universo)

    out, _ = run_script("--tickers", "SPY,QQQ,IWM,GLD,TLT,LQD,BIL,AAPL,MSFT,JPM")
    check("a custom list runs end to end", "9 · ARCHIVOS" in out)
    check("it warns that the leveraged-product filter is blind without names",
          "AVISO" in out and "apalancado" in out)


def test_profile_and_strategy_mismatch_is_flagged() -> None:
    """
    Screening under one mandate and exporting for another is legal but almost
    always a mistake, so it has to be said out loud.
    """
    out, _ = run_script("--perfil", "Conservador", "--estrategia", "Agresivo")
    check("a profile/strategy mismatch is called out",
          "AVISO" in out and "otro mandato" in out,
          out[:400])


def main() -> int:
    for fn in [
        test_every_stage_runs,
        test_workbook_matches_the_notebook_shape,
        test_proposals_land_in_the_right_folder,
        test_flags_actually_reach_the_model,
        test_market_anchor_flag_changes_the_anchor,
        test_custom_ticker_list,
        test_profile_and_strategy_mismatch_is_flagged,
    ]:
        fn()

    print("-" * 70)
    print(f"{PASSED}/{PASSED + FAILED} passed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
