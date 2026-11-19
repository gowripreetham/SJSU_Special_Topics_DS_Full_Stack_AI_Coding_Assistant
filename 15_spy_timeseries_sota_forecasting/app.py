"""Project 15 — SPY probabilistic forecasting: forecast studio, hold-out leaderboard, cost-aware backtests,
executable leakage audit and a stress simulator."""
from __future__ import annotations
import inspect
import sys
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
_s: dict = {}


def M() -> dict:
    if "m" not in _s:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _s["m"]


def model() -> dict:
    if "model" not in _s:
        _s["model"] = load_or_train(ROOT / "artifacts" / "model.joblib", pipeline.main)
        df = pipeline.load()
        _s["df"], _s["X"] = df, pipeline.features(df).dropna()
    return _s["model"]


class Day(BaseModel):
    date: str


class Stress(BaseModel):
    shock: float = Field(-3.0, ge=-10, le=10, description="today's return, %")
    vol_mult: float = Field(2.0, ge=0.5, le=5)


def pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def p_studio() -> list[dict]:
    m = M()
    dates = m["bands"]["1"]["dates"]
    return [H("Forecast studio", "Pick any session of the 2019 hold-out. The model sees only data up to that close and forecasts the next 1 and 5 sessions as a P10–P50–P90 price band. Then the future is revealed."),
            FORM("api/forecast", [F("date", "Forecast made at the close of", "select", dates[150], options=dates[::-1])], submit="Forecast", auto=True, run_on_load=True)]


def lb_blocks(h: str) -> list[dict]:
    r = M()["results"][h]
    sel, ho = {x["model"]: x for x in r["select"]}, sorted(r["holdout"], key=lambda x: x["pinball"])
    return [GRID(2, CHART("hbar", "2019 hold-out pinball loss (bp, lower is better)", [S("pinball", [x["pinball"] for x in ho], colors=["--axis" if x["model"] in ("Unconditional quantiles", "EWMA-vol Gaussian") else "1" for x in ho])], [x["model"] for x in ho], height=250),
                 CHART("bar", "P10–P90 band coverage (target 80%)", [S("selection years", [sel[x["model"]]["coverage80"] for x in ho]), S("2019 hold-out", [x["coverage80"] for x in ho], color="2")], [x["model"].split(" ")[0] for x in ho], ymin=0.6, ymax=1, height=250)),
            TABLE(f"h = {h} session(s) — ensemble members chosen on 2014–2018: {', '.join(r['members'])}", ["model", "pinball 2014-18", "pinball 2019", "coverage 2019", "band width % 2019", "sign hit rate 2019", "corr(P50, outcome)"],
                  [[x["model"], sel[x["model"]]["pinball"], x["pinball"], x["coverage80"], x["width"], x["hit_rate"], x["p50_corr"]] for x in ho], digits=3)]


def p_board() -> list[dict]:
    m = M()
    return [H("Leaderboard", "Six forecasters, two horizons. Selection years (2014–2018) drive every choice; 2019 is reported once."), TABS(("Next day (h = 1)", lb_blocks("1")), ("Next week (h = 5)", lb_blocks("5"))),
            CALLOUT("info", "Reading the table", f"Conditional models beat the unconditional band by a few percent of pinball loss — that gain comes from **forecasting volatility** (narrow bands in calm markets, wide in stress). The sign hit rate ({pct(m['hit']['rate'])}) is *identical* to the share of up days in 2019 ({pct(m['hit']['up_day_rate'])}): the median forecast is simply 'slightly up', every day. There is no directional skill here, and the dashboard does not pretend otherwise."),
            MD("## AutoResearch on the LightGBM quantile model\nObjective: walk-forward pinball loss on 2014–2018 (h = 1).", card=False), *ar_blocks(m["autoresearch"], "pinball (bp)")]


def p_backtest() -> list[dict]:
    m = M()
    bt, bs = m["backtest"], m["backtest_select"]
    d = m["bands"]["1"]["dates"]
    row = lambda k, v: [k, v["sharpe"], v["sortino"], pct(v["total_return"]), pct(v["max_drawdown"]), pct(v["vol"]), round(v["exposure"], 2), round(v["turnover"], 1)]  # noqa: E731
    return [H("Backtests with costs", f"Positions are set at the close of day t from the forecast and earn the return of day t+1; every change of position pays {pipeline.COST_BPS:.0f} bp."),
            CHART("line", "Equity curves on the 2019 hold-out (start = 1.0)", [S(k, v["equity"]) for k, v in bt.items()], d, height=320),
            GRID(2, TABLE("2019 hold-out", ["strategy", "Sharpe", "Sortino", "return", "max drawdown", "ann. vol", "avg exposure", "turnover"], [row(k, v) for k, v in bt.items()], digits=2),
                 TABLE("2014–2018 selection years (same rules)", ["strategy", "Sharpe", "Sortino", "return", "max drawdown", "ann. vol", "avg exposure", "turnover"], [row(k, v) for k, v in bs.items()], digits=2)),
            CALLOUT("warn", "An honest reading", f"A Sharpe near 2 in 2019 looks spectacular — and it is exactly what **buy & hold** earned, because 2019 was a +{100 * bt['Buy & hold']['total_return']:.0f}% year. The directional strategies coincide with buy & hold since the median forecast never turned negative. "
                    "Volatility targeting is the only strategy that uses the model, and what it changes is *risk* (volatility and drawdown), not return. Anyone reporting a Sharpe above 2 from daily index forecasts should be asked which year they tested on and what buy & hold did.")]


def p_audit() -> list[dict]:
    m = M()
    return [H("Leakage audit", "Six executable tests run inside the pipeline on every training run. Each one can fail."),
            KPIS(K("Tests passed", f"{sum(a['passed'] for a in m['audit'])} / {len(m['audit'])}"), K("Purge + embargo", f"h + {pipeline.EMBARGO} sessions"), K("Hold-out year", pipeline.HOLDOUT_YEAR, hint="scored once")),
            TABLE("Audit results", ["test", "result", "evidence"], [[a["test"], PILL("pass", "good") if a["passed"] else PILL("FAIL", "crit"), a["detail"]] for a in m["audit"]]),
            TABLE("Walk-forward folds (h = 5)", ["test year", "train ends", "test starts", "sessions purged between", "train rows", "test rows"], [[f["test_year"], f["train_end"], f["test_start"], f["gap_sessions"], f["n_train"], f["n_test"]] for f in m["folds"]]),
            TABS(("features()", [CODE("Backward-looking features — pipeline.py", inspect.getsource(pipeline.features))]), ("folds()", [CODE("Purged, embargoed walk-forward", inspect.getsource(pipeline.folds))]),
                 ("leakage_tests()", [CODE("The audit itself", inspect.getsource(pipeline.leakage_tests))]), ("backtest()", [CODE("Cost-aware backtest", inspect.getsource(pipeline.backtest))]))]


def p_stress() -> list[dict]:
    return [H("Stress simulator", "Take the most recent session, impose a shock, and see how the forecast distribution responds. (No macro series such as VIX are in this dataset, so shocks are expressed through SPY's own return and volatility.)"),
            FORM("api/stress", [F("shock", "Today's return (%)", "range", -3, min=-8, max=8, step=0.5), F("vol_mult", "Volatility multiplier", "range", 2, min=0.5, max=4, step=0.25)], submit="Apply shock", auto=True, run_on_load=True)]


def p_eda() -> list[dict]:
    m = M()
    e = m["eda"]
    xs = [f"{-5 + i / 6 + 1 / 12:.1f}" for i in range(60)]
    return [H("Data understanding", f"SPY daily bars {m['range'][0]} → {m['range'][1]} ({m['n_sessions']:,} sessions)."),
            KPIS(K("Annualised return", pct(e["ann_ret"])), K("Annualised volatility", pct(e["ann_vol"])), K("Excess kurtosis", e["kurtosis"], digits=2, hint="0 for a normal distribution"), K("Skewness", e["skew"], digits=2)),
            GRID(2, CHART("line", "Close price", [S("SPY", e["close"])], e["dates"], height=240), CHART("line", "21-day realised volatility (annualised)", [S("volatility", e["vol_series"], color="2")], e["dates"], height=240, sub="Volatility clusters — calm and turbulent regimes persist.")),
            GRID(2, CHART("bar", "Daily returns vs a normal distribution", [S("observed", e["ret_hist"]), S("normal with same mean/σ", e["normal_hist"], color="2", extra={"type": "line", "pointRadius": 0, "borderWidth": 2})], xs, x="daily return %", y="days", height=250, sub="Fat tails and a sharper peak: Gaussian bands are too wide on quiet days and far too narrow in crashes."),
                 CHART("bar", "Autocorrelation: returns vs |returns|", [S("returns", e["acf_ret"]), S("absolute returns", e["acf_abs"], color="2")], list(range(1, 21)), x="lag (days)", height=250, sub="Returns are almost uncorrelated (direction is unpredictable); their magnitude is strongly autocorrelated (volatility is predictable). The whole project follows from this chart.")),
            CHART("bar", "Calendar-year return", [S("return", list(e["yearly"].values()), colors=["3" if v > 0 else "8" for v in e["yearly"].values()])], list(e["yearly"]), height=200)]


CRISPDM = [
    ("Business understanding", "A risk desk wants tomorrow's and next week's *range* for SPY — for position sizing and limits — not a price target. Success = better-calibrated, sharper bands than naive ones, verified on a year the research never touched, and an honest statement about directional skill."),
    ("Data understanding", "10 years of daily OHLCV. Returns are fat-tailed and serially uncorrelated; absolute returns are strongly autocorrelated. Volatility is forecastable, direction is not."),
    ("Data preparation", "23 backward-looking features (multi-horizon returns, realised/EWMA volatility, range, gap, RSI, MACD, Bollinger z, distance to the 200-day average, drawdown, volume z-score, calendar). Targets are forward log returns. Scalers are fit per training window."),
    ("Modeling", "Unconditional and EWMA baselines, GARCH(1,1)-t, linear quantile regression, LightGBM quantile regression tuned by AutoResearch, and an ensemble of the two best models on 2014–2018."),
    ("Evaluation", "Purged + embargoed walk-forward; pinball loss and band coverage; sign hit rate compared with the up-day base rate; cost-aware backtests against buy & hold; six executable leakage tests."),
    ("Deployment", "`POST api/forecast` (band + TreeSHAP for a date), `POST api/stress` (shock scenarios). Models are frozen before the hold-out year."),
]


def p_spec() -> list[dict]:
    return [H("Design spec & UML", "The full spec with Mermaid package, class, sequence, activity and state diagrams lives in `15_spy_timeseries_sota_forecasting/docs/design_spec.md` — GitHub renders the diagrams inline."),
            MD((ROOT / "docs" / "design_spec.md").read_text())]


app = make_app(__file__, "15", "SPY Probabilistic Forecasting", "Quantile forecasts · purged walk-forward · executable leakage audit · cost-aware backtests", [
    ("studio", "Forecast studio", "App", p_studio), ("stress", "Stress simulator", "App", p_stress),
    ("leaderboard", "Leaderboard", "Data science", p_board), ("backtest", "Backtests", "Data science", p_backtest), ("audit", "Leakage audit", "Data science", p_audit), ("eda", "Data understanding", "Data science", p_eda),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]), ("spec", "Design spec & UML", "Docs", p_spec),
])


@app.post("/api/forecast")
def forecast(d: Day) -> dict:
    m, mm = M(), model()
    b1, b5 = m["bands"]["1"], m["bands"]["5"]
    if d.date not in b1["dates"]:
        raise HTTPException(422, "date must be a 2019 trading session")
    i = b1["dates"].index(d.date)
    df, X = _s["df"], _s["X"]
    pos = df.index.get_loc(d.date)
    hist = df.close.iloc[pos - 59:pos + 1]
    fut = df.close.iloc[pos + 1:pos + 6]
    labels = [x.strftime("%m-%d") for x in hist.index] + [f"t+{k}" for k in range(1, 6)]
    c0 = float(hist.iloc[-1])
    pad = [None] * 59
    j = b5["dates"].index(d.date) if d.date in b5["dates"] else None
    fan = lambda lo, hi, k: [c0 + (v - c0) * s / 5 for s in range(1, 6)] if k == 5 else None  # noqa: E731
    series = [S("close (known at forecast time)", hist.round(2).tolist() + [None] * 5, color="--ink"), S("what happened next", pad + [c0] + fut.round(2).tolist(), color="2", dash=True, points=True)]
    if j is not None:
        series += [S("_lo5", pad + [c0] + [c0 + (b5["p10"][j] - c0) * np.sqrt(s / 5) for s in range(1, 6)], width=0, color="1"), S("P10–P90 cone to t+5", pad + [c0] + [c0 + (b5["p90"][j] - c0) * np.sqrt(s / 5) for s in range(1, 6)], width=0, color="1", fill="-1")]
    row = X.loc[[d.date]][mm["columns"]]
    contrib = mm["models"][1][1].predict(row, pred_contrib=True)[0]
    top = sorted(zip(mm["columns"], (contrib[:-1] * 1e4).tolist()), key=lambda kv: -abs(kv[1]))[:8]
    y1, q1 = b1["y"][i], b1["q"][i]
    inside = q1[0] <= y1 <= q1[2]
    kp = [K("Close on " + d.date, f"${c0:.2f}"), K("Next-day band (P10 – P90)", f"${b1['p10'][i]:.2f} – ${b1['p90'][i]:.2f}", hint=f"{q1[0]:+.2f}% … {q1[2]:+.2f}%, median {q1[1]:+.3f}%"),
          K("Actual next close", f"${b1['actual'][i]:.2f}", hint=f"{y1:+.2f}% — {'inside' if inside else 'OUTSIDE'} the band")]
    if j is not None:
        kp += [K("5-day band", f"${b5['p10'][j]:.2f} – ${b5['p90'][j]:.2f}"), K("Actual close t+5", f"${b5['actual'][j]:.2f}", hint="inside" if b5["q"][j][0] <= b5["y"][j] <= b5["q"][j][2] else "OUTSIDE the band")]
    return {"date": d.date, "band_1d": [b1["p10"][i], b1["p50"][i], b1["p90"][i]], "blocks": [
        KPIS(*kp), CHART("line", "Last 60 sessions, the forecast cone, and the realised path", series, labels, height=320, note="The cone is the ensemble's 5-day P10–P90 band, interpolated with √time."),
        CHART("hbar", "TreeSHAP for the next-day median (LightGBM P50), in basis points", [S("contribution (bp)", [v for _, v in top], colors=["3" if v > 0 else "8" for _, v in top])], [k for k, _ in top], height=250,
              note="Contributions are a few basis points — the median barely moves. The features matter far more for the *width* of the band than for its centre.")]}


@app.post("/api/stress")
def stress(s: Stress) -> dict:
    mm = model()
    X = _s["X"]
    base = X.iloc[[-1]][mm["columns"]].copy()
    sh = base.copy()
    r = s.shock / 100
    sh["ret_1"] = r
    for k in ["ret_2", "ret_5", "ret_10", "ret_21", "ret_63"]:
        sh[k] = base[k] + r - base["ret_1"].iloc[0]
    for k in ["vol_5", "vol_21", "ewma_vol", "range", "range_5"]:
        sh[k] = base[k] * s.vol_mult
    sh["vol_ratio"] = sh["vol_5"] / base["vol_63"]
    sh["drawdown"] = np.minimum(0, base["drawdown"] + min(0, r))
    sh["boll_z"] = base["boll_z"] + r / max(1e-4, float(base["vol_21"].iloc[0]) / np.sqrt(252))
    out, rows = {}, []
    for h in pipeline.HORIZONS:
        qb = np.sort([float(m_.predict(base)[0]) for m_ in mm["models"][h]])
        qs = np.sort([float(m_.predict(sh)[0]) for m_ in mm["models"][h]])
        out[h] = (qb, qs)
        rows.append([f"{h} session(s)", f"{100 * qb[0]:+.2f}% … {100 * qb[2]:+.2f}%", f"{100 * qs[0]:+.2f}% … {100 * qs[2]:+.2f}%", f"{(qs[2] - qs[0]) / (qb[2] - qb[0]):.2f}×", f"{1e4 * (qs[1] - qb[1]):+.1f} bp"])
    lab = ["P10", "P50", "P90"]
    return {"blocks": [KPIS(K("Reference session", str(X.index[-1].date())), K("Scenario", f"{s.shock:+.1f}% day, volatility ×{s.vol_mult:g}"), K("Next-day band widens", f"{(out[1][1][2] - out[1][1][0]) / (out[1][0][2] - out[1][0][0]):.2f}×")),
                       GRID(2, *[CHART("bar", f"{h}-session forward return quantiles (%)", [S("baseline", (100 * out[h][0]).round(3).tolist()), S("stressed", (100 * out[h][1]).round(3).tolist(), color="8")], lab, height=240) for h in pipeline.HORIZONS]),
                       TABLE("Scenario summary", ["horizon", "baseline band", "stressed band", "width ratio", "median shift"], rows),
                       CALLOUT("info", "Model limits", "Tree models cannot extrapolate beyond volatility levels seen in 2010–2018 training data, so very large multipliers saturate. That is a property of the model class worth knowing before relying on it in a crisis.")]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8015)
