"""Project 12 — TimePulse: forecast studio with prediction fans, rolling-origin backtests and series diagnostics."""
from __future__ import annotations
import sys
import time
from pathlib import Path

import pandas as pd
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


class Req(BaseModel):
    series: str = "airline"
    model: str = "Holt-Winters (ETS)"
    horizon: int = Field(24, ge=1, le=60)
    level: float = Field(0.8, ge=0.5, le=0.99)


class Ser(BaseModel):
    series: str = "airline"


SERIES_OPT = [[k, v[0]] for k, v in pipeline.SERIES.items()]


def p_studio() -> list[dict]:
    return [H("Forecast studio", "Choose a series, a model and a horizon. The model is re-fit live on the full history and returns a forecast fan."),
            FORM("api/forecast", [F("series", "Series", "select", "airline", options=SERIES_OPT), F("model", "Model", "select", "Holt-Winters (ETS)", options=pipeline.MODELS),
                                  F("horizon", "Horizon (months)", "range", 24, min=6, max=60, step=6), F("level", "Interval level", "select", "0.8", options=["0.5", "0.8", "0.95"])],
                 submit="Forecast", auto=True, run_on_load=True)]


def backtest_blocks(key: str) -> list[dict]:
    s = M()["series"][key]
    lb = s["leaderboard"]
    hist = 36
    return [KPIS(K("Best model", lb[0]["model"]), K("Its MASE", lb[0]["mase"], digits=3, hint="< 1 beats in-sample seasonal naive"), K("Observations", s["n"], hint=f"{s['start'][:7]} → {s['end'][:7]}"),
                 K("Backtest folds", M()["origins"], hint=f"each forecasts {M()['horizon']} months ahead")),
            GRID(2, CHART("hbar", "Mean MASE across all folds and horizons", [S("MASE", [r["mase"] for r in lb])], [r["model"] for r in lb], height=260),
                 CHART("line", "Error grows with horizon", [S(r["model"], r["by_horizon"]) for r in lb[:5]], list(range(1, 13)), x="months ahead", y="MASE", height=260)),
            CHART("line", "Final fold: every model's forecast of the last 12 months vs what actually happened",
                  [S("actual", s["values"][-hist:], width=3, color="--ink")] + [S(r["model"], [None] * (hist - 12) + s["last_fold"][r["model"]], dash=True) for r in lb[:4]], s["dates"][-hist:], height=300),
            TABLE("Leaderboard", ["model", "MASE (all)", "MASE h=1", "MASE h=12", "80% interval coverage", "per-fold MASE"],
                  [[r["model"], r["mase"], r["mase_h1"], r["mase_h12"], f"{100 * r['coverage80']:.0f}%" if r["coverage80"] is not None else "no interval", " · ".join(f"{v:.2f}" for v in r["folds"])] for r in lb], digits=3)]


def p_backtest() -> list[dict]:
    return [H("Rolling-origin backtest", "Train on everything up to an origin, forecast 12 months, slide the origin 6 months forward, repeat 8 times. Errors are scaled by the in-sample seasonal-naive error (MASE)."),
            TABS(*[(v[0], backtest_blocks(k)) for k, v in pipeline.SERIES.items()]),
            CALLOUT("info", "Reading interval coverage", "An 80% interval should contain ~80% of future points. Theta's are too wide (100%), CO₂'s SARIMA/ETS bands are too narrow (67–72%) — a calibration gap you would fix with conformal adjustment before trusting them operationally.")]


def diag_blocks(key: str) -> list[dict]:
    s = M()["series"][key]
    ci = s["ci"]
    lags = list(range(1, 41))
    bar = lambda title, v, col: CHART("bar", title, [S("correlation", v[1:], color=col), S("_+ci", [ci] * 40, extra={"type": "line", "borderDash": [4, 4], "borderWidth": 1, "pointRadius": 0}, color="--muted"),  # noqa: E731
                                                     S("_-ci", [-ci] * 40, extra={"type": "line", "borderDash": [4, 4], "borderWidth": 1, "pointRadius": 0}, color="--muted")], lags, x="lag (months)", height=220)
    st = s["stl"]
    unit = "log scale" if s["log"] else "level"
    return [CHART("line", s["title"], [S("observed", s["values"])], s["dates"], height=230),
            GRID(3, CHART("line", f"STL trend ({unit})", [S("trend", st["trend"], color="2")], s["dates"], height=200), CHART("line", "STL seasonal", [S("seasonal", st["seasonal"], color="3")], s["dates"], height=200),
                 CHART("line", "STL remainder", [S("remainder", st["resid"], color="7")], s["dates"], height=200)),
            KPIS(K("Trend strength", st["strength_trend"], digits=3, hint="1 − Var(R)/Var(T+R)"), K("Seasonal strength", st["strength_seasonal"], digits=3, hint="1 − Var(R)/Var(S+R)"), K("SARIMA order (AIC)", f"{tuple(s['sarima_order'][0])}×{tuple(s['sarima_order'][1])}")),
            GRID(2, bar("ACF of the differenced series (40 lags)", s["acf"], "1"), bar("PACF of the differenced series (40 lags)", s["pacf"], "2")),
            GRID(2, TABLE("Stationarity tests", ["series", "ADF p (H₀: unit root)", "KPSS p (H₀: stationary)", "verdict"], [[t["series"], t["adf_p"], t["kpss_p"], PILL(t["verdict"], "good" if t["verdict"] == "stationary" else "warn")] for t in s["tests"]], digits=3,
                          sub="ADF and KPSS have opposite null hypotheses; agreeing verdicts are convincing."),
                 CHART("bar", "Seasonal profile (month mean ÷ overall mean)", [S("index", s["seasonal_profile"], color="3")], ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], ymin=min(s["seasonal_profile"]) - 0.02, height=230))]


def p_diag() -> list[dict]:
    return [H("Series diagnostics", "Decomposition, autocorrelation and stationarity — the evidence behind every modelling choice."), TABS(*[(v[0], diag_blocks(k)) for k, v in pipeline.SERIES.items()])]


CRISPDM = [
    ("Business understanding", "Capacity planning needs monthly forecasts 1–12 months out **with uncertainty**. Success = beat seasonal-naive (MASE < 1) at every horizon and report calibrated intervals."),
    ("Data understanding", "Airline passengers: strong trend, multiplicative seasonality (swings grow with the level). Mauna Loa CO₂: smooth trend, additive seasonality. STL strengths and ACF spikes at lag 12 confirm both."),
    ("Data preparation", "Monthly index, log transform where seasonality is multiplicative, differencing guided by ADF + KPSS. Lag features for the tree model use only past values."),
    ("Modeling", "Seasonal naive and drift baselines, Holt-Winters, SARIMA (order chosen by AIC on the first training window only), Theta, and gradient boosting on lagged differences with recursive multi-step forecasting."),
    ("Evaluation", "Rolling-origin backtest: 8 origins × 12 horizons per series, MASE by horizon, and empirical coverage of the 80% intervals. No random splits, no look-ahead."),
    ("Deployment", "`POST api/forecast` re-fits the chosen model on demand (≈1 s) and returns mean + interval; the studio page plots the fan."),
]

app = make_app(__file__, "12", "TimePulse Forecasting", "Six forecasters · rolling-origin backtest · forecast fans · ACF/PACF · STL", [
    ("studio", "Forecast studio", "App", p_studio),
    ("backtest", "Backtest leaderboard", "Data science", p_backtest),
    ("diagnostics", "Series diagnostics", "Data science", p_diag),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/forecast")
def forecast(r: Req) -> dict:
    m = M()
    if r.series not in pipeline.SERIES or r.model not in pipeline.MODELS:
        from fastapi import HTTPException
        raise HTTPException(422, "unknown series or model")
    y, s = pipeline.load(r.series), m["series"][r.series]
    order = (tuple(s["sarima_order"][0]), tuple(s["sarima_order"][1]))
    t0 = time.time()
    f = pipeline.forecast(r.model, y, r.horizon, s["log"], order, level=r.level)
    ms = (time.time() - t0) * 1000
    hist = min(len(y), 72)
    fut = [d.strftime("%Y-%m") for d in pd.date_range(y.index[-1], periods=r.horizon + 1, freq="MS")[1:]]
    pad = [None] * (hist - 1)
    last = float(y.iloc[-1])
    series = [S("history", y.iloc[-hist:].round(2).tolist() + [None] * r.horizon, color="--ink"), S("forecast", pad + [last] + f["mean"].round(2).tolist(), color="1", width=2.5)]
    if f["lo"] is not None:
        series += [S("_lo", pad + [last] + f["lo"].round(2).tolist(), color="1", width=0), S(f"{int(r.level * 100)}% interval", pad + [last] + f["hi"].round(2).tolist(), color="1", width=0, fill="-1")]
    bt = next(x for x in s["leaderboard"] if x["model"] == r.model)
    return {"mean": f["mean"].tolist(), "blocks": [
        CHART("line", f"{s['title']} — {r.model}, next {r.horizon} months", series, s["dates"][-hist:] + fut, height=340,
              note=None if f["lo"] is not None else "This model has no native prediction interval — its uncertainty is reported through the backtest instead."),
        KPIS(K("Fit + forecast time", f"{ms:.0f} ms"), K("Forecast, month +1", float(f["mean"][0]), digits=1), K(f"Forecast, month +{r.horizon}", float(f["mean"][-1]), digits=1),
             K("Backtest MASE", bt["mase"], digits=3, hint="rank " + str(1 + [x["model"] for x in s["leaderboard"]].index(r.model)) + f" of {len(pipeline.MODELS)}"),
             K("Interval coverage in backtest", f"{100 * bt['coverage80']:.0f}%" if bt["coverage80"] is not None else "n/a", hint="target 80%"))]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8012)
