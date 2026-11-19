"""Project 13 — NYC TLC Mobility Platform: fare estimator plus a fully transparent audit console
(data quality, EDA, spatial analytics, AutoResearch, explainability, MLOps, static code audit, CRISP-DM paper)."""
from __future__ import annotations
import inspect
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403
from common.audit import audit_project  # noqa: E402

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
_s: dict = {}
LOG: deque = deque(maxlen=500)  # in-memory request log for the monitoring page
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
AUDIT_CFG = dict(files=["pipeline.py"], temporal=True, forbidden=("tip", "tolls", "total", "dropoff", "payment", "fare"))


def M() -> dict:
    if "m" not in _s:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _s["m"]


def model() -> dict:
    if "model" not in _s:
        _s["model"] = load_or_train(ROOT / "artifacts" / "model.joblib", pipeline.main)
        _s["zones"] = pipeline.zones()
    return _s["model"]


class Trip(BaseModel):
    pickup_zone: str
    dropoff_zone: str
    hour: int = Field(18, ge=0, le=23)
    dow: int = Field(4, ge=0, le=6)
    color: str = "yellow"
    passengers: int = Field(1, ge=1, le=6)
    distance: float | None = Field(None, ge=0, le=60)


class Load(BaseModel):
    n: int = Field(300, ge=10, le=3000)


def predict_core(t: Trip) -> dict:
    mm = model()
    Z = _s["zones"]
    if t.pickup_zone not in Z or t.dropoff_zone not in Z:
        raise HTTPException(422, "unknown taxi zone")
    a, b = Z[t.pickup_zone], Z[t.dropoff_zone]
    crow = float(pipeline.haversine(a["c"][0], a["c"][1], b["c"][0], b["c"][1])) / 1.609
    dist = t.distance if t.distance else max(0.4, 1.3 * crow)  # road distance ≈ 1.3 × straight line when the caller has no route
    df = pd.DataFrame([{"pickup": pd.Timestamp(2019, 3, 4 + t.dow, t.hour), "pickup_zone": t.pickup_zone, "dropoff_zone": t.dropoff_zone, "pickup_borough": a["borough"], "dropoff_borough": b["borough"],
                        "distance": dist, "passengers": t.passengers, "color": t.color}])
    X = pipeline.features(df, Z)
    import xgboost
    c = mm["model"].get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)[0]
    fare = float(c.sum())
    return {"fare": fare, "lo": max(2.5, fare - mm["q80"]), "hi": fare + mm["q80"], "distance": dist, "base": float(c[-1]), "contrib": sorted(zip(mm["features"], c[:-1].tolist()), key=lambda kv: -abs(kv[1])),
            "boroughs": (a["borough"], b["borough"]), "version": mm["version"]}


# ----------------------------------------------------------------------------- pages
def p_estimator() -> list[dict]:
    t = M()["test"]
    return [H("Fare estimator", f"Predicts the metered fare before the ride. On the held-out final week it is within $1 for {100 * t['within_1usd']:.0f}% of trips and within $2 for {100 * t['within_2usd']:.0f}%."), CUSTOM("fare")]


def p_quality() -> list[dict]:
    q = M()["quality"]
    return [H("Data quality scorecard", "Six dimensions, each defined by an explicit rule that anyone can re-compute."),
            KPIS(K("Overall quality", f"{100 * q['overall']:.1f}%"), K("Raw rows", q["rows"]), K("Rows after cleaning", M()["clean_rows"]), K("Removed", q["rows"] - M()["clean_rows"])),
            TABLE("Scorecard", ["dimension", "score", "rule"], [[d["name"], {"bar": d["score"], "text": f"{100 * d['score']:.2f}%"}, d["rule"]] for d in q["dims"]]),
            TABLE("Missing values by column", ["column", "missing rows"], [[k, v] for k, v in q["missing"].items()]),
            CODE("The cleaning rule, verbatim — pipeline.py", inspect.getsource(pipeline.clean))]


def p_eda() -> list[dict]:
    e = M()["eda"]
    dist_tab = []
    for r in e["iqr"]:
        ed = r["edges"]
        dist_tab.append(CHART("bar", f"{r['col']} — mean {r['mean']:.2f}, median {r['median']:.2f}", [S("trips", r["hist"])], [f"{(a + b) / 2:.1f}" for a, b in zip(ed, ed[1:])], height=210,
                              note=f"Tukey fences [{r['lo']:.2f}, {r['hi']:.2f}] flag **{r['outliers']}** outliers (IQR = {r['q3'] - r['q1']:.2f})."))
    o = e["ols"]
    return [H("EDA dashboard", "Six views of March 2019."),
            TABS(("Distributions & outliers", [GRID(2, *dist_tab)]),
                 ("Bivariate", [CHART("scatter", f"Fare vs distance — OLS: fare ≈ {o['intercept']:.2f} + {o['slope']:.2f} × miles (r = {o['r']:.3f})", [S("trip", e["scatter"], r=2.5), S("_ols", [{"x": 0, "y": o["intercept"]}, {"x": 25, "y": o["intercept"] + 25 * o["slope"]}], line=True, color="2")], x="miles", y="fare ($)", height=340,
                                      sub="The fitted intercept and slope recover the real TLC tariff: $2.50 flag-drop + $2.50 per mile. Horizontal bands at $52 are the JFK flat fare.")]),
                 ("Temporal", [HEAT("Trips by weekday × hour", list(range(24)), DOW, e["heat_trips"], digits=0, labw=50), HEAT("Median speed (mph) by weekday × hour", list(range(24)), DOW, e["heat_speed"], digits=0, labw=50),
                               CHART("bar", "Trips per day", [S("trips", [v for _, v in e["daily"]])], [k[5:] for k, _ in e["daily"]], height=200)]),
                 ("Boroughs", [TABLE("Pickup borough analytics", ["borough", "trips", "share", "mean fare", "median miles", "mean tip %"], [[b, n, f"{100 * s:.1f}%", round(f, 2), round(d, 2), round(t, 1)] for b, n, s, f, d, t in e["borough"]])]),
                 ("Correlations", [GRID(2, HEAT("Pearson", e["corr_cols"], e["corr_cols"], e["pearson"], diverging=True, min=-1, max=1, labw=80), HEAT("Spearman (rank)", e["corr_cols"], e["corr_cols"], e["spearman"], diverging=True, min=-1, max=1, labw=80))]),
                 ("Categoricals", [GRID(3, CHART("doughnut", "Payment type", [S("trips", list(e["payment"].values()))], list(e["payment"]), height=220), CHART("doughnut", "Cab colour", [S("trips", list(e["color"].values()))], list(e["color"]), height=220),
                                        CHART("bar", "Mean recorded tip % by payment", [S("tip %", list(e["tip_by_payment"].values()), color="3")], list(e["tip_by_payment"]), height=220, note="Cash tips are not recorded by the meter — a measurement artefact, and one reason `tip` is unusable as a feature."))]))]


def p_spatial() -> list[dict]:
    s = M()["spatial"]
    return [H("Spatial analytics", "Trips aggregated to the 263 official TLC taxi zones, drawn offline from the zone polygons."),
            GRID(2, CUSTOM("choropleth", {"values": s["trips"], "clusters": s["centers"], "note": "Darker = more pickups. Coloured dots = K-Means (k = 6) demand centres."}, title="Pickup demand & demand clusters"),
                 CUSTOM("choropleth", {"values": s["fare"], "note": "Mean fare by pickup zone (zones with ≥ 5 trips). Airports and outer boroughs start the expensive rides."}, title="Mean fare by pickup zone")),
            GRID(2, TABLE("Demand clusters", ["cluster", "trips", "mean fare", "mean miles", "tip %", "anchor zone"], [[c, n, round(f, 2), round(d, 2), round(t, 1), z] for c, n, f, d, t, z in s["clusters"]]),
                 TABLE("Busiest pickup zones", ["zone", "trips", "mean fare", "tip %"], [[z, n, round(f, 2), round(t, 1)] for z, n, f, t in s["top_zones"]]))]


def p_research() -> list[dict]:
    m = M()
    lb, ab = m["leaderboard"], m["ablation"]
    return [H("AutoResearch: tournament, HPO and ablation", "Every number on this page is a TimeSeriesSplit cross-validation MAE computed inside the training window. The test week is not involved."),
            GRID(2, CHART("hbar", "Model tournament — CV MAE ($), lower is better", [S("MAE", [r["cv_mae"] for r in lb], colors=["--axis" if "baseline" in r["model"] else "1" for r in lb])], [r["model"] for r in lb], height=300),
                 CHART("bar", "Ablation — CV MAE after removing one feature group", [S("MAE", [r["cv_mae"] for r in ab], colors=["3"] + ["2"] * (len(ab) - 1))], [r["removed"] for r in ab], height=300,
                       note="Removing **distance** multiplies the error; the other groups add smaller, honest gains.")),
            CALLOUT("info", "Why XGBoost is the deployed model", "LightGBM and HistGradientBoosting edge it by about one cent of CV MAE — inside fold-to-fold noise. XGBoost is shipped because its native TreeSHAP (`pred_contribs`) powers the per-trip waterfall on the estimator page without an extra dependency."),
            TABLE("Ablation detail", ["feature group removed", "features left", "CV MAE", "Δ vs full"], [[r["removed"], r["n_features"], r["cv_mae"], r["delta"]] for r in ab], digits=3),
            MD("## Hyper-parameter hill-climb\nOne change per experiment, kept only when CV MAE drops.", card=False), *ar_blocks(m["autoresearch"], "CV MAE")]


def p_xai() -> list[dict]:
    m = M()
    sg = m["shap_global"][:14]
    sl = m["slices"]
    return [H("Explainable AI", "Global TreeSHAP attributions and error slices. Per-trip waterfalls are on the estimator page."),
            GRID(2, CHART("hbar", "Global importance — mean |SHAP| in dollars (test week)", [S("mean |SHAP|", [v for _, v in sg])], [k for k, _ in sg], height=380, sub=f"Additive: prediction = ${m['shap_base']:.2f} base value + Σ contributions."),
                 CHART("scatter", "Predicted vs actual fare (test week)", [S("trip", m["pred_vs_actual"], r=2.5), S("_ideal", [{"x": 0, "y": 0}, {"x": 80, "y": 80}], line=True, color="--axis")], x="actual $", y="predicted $", height=380)),
            GRID(3, TABLE("Error by pickup borough", ["borough", "trips", "MAE $", "mean fare"], [[b, n, round(e, 2), round(f, 2)] for b, n, e, f in sl["by_borough"]]),
                 TABLE("Error by trip length", ["distance", "trips", "MAE $", "MAPE"], [[a, n, round(e, 2), f"{100 * p:.1f}%"] for a, n, e, p in sl["by_distance"]]),
                 CHART("bar", "MAE by pickup hour", [S("MAE $", sl["by_hour"], color="2")], list(range(24)), height=230))]


def p_mlops() -> list[dict]:
    m = M()
    r, t = m["registry"], m["test"]
    lat = [x["ms"] for x in LOG]
    return [H("MLOps console", "Model registry, drift monitor, live load test and request log."),
            KPIS(K("Model version", r["version"]), K("Test MAE", f"${t['mae']:.2f}", hint=f"meter-formula baseline ${t['baseline_meter_mae']:.2f}"), K("Test R²", t["r2"], digits=3), K("80% interval coverage", f"{100 * t['coverage80']:.0f}%"),
                 K("Requests this session", len(LOG)), K("p50 latency", f"{statistics.median(lat):.1f} ms" if lat else "—")),
            GRID(2, TABLE("Model registry entry", ["field", "value"], [["version (sha1 of data + params + features)", {"code": r["version"]}], ["data sha256", {"code": r["data_sha256"]}], ["algorithm", r["algorithm"]], ["features", r["n_features"]], ["trained rows", r["trained_rows"]],
                                                                         ["train window", " → ".join(r["train_window"])], ["test window", " → ".join(r["test_window"])], ["hyper-parameters", ", ".join(f"{k}={v}" for k, v in r["params"].items())], ["libraries", ", ".join(f"{k} {v}" for k, v in r["libs"].items())]]),
                 TABLE("Drift monitor — PSI of the test week against the training window", ["feature", "PSI", "status"], [[d["feature"], d["psi"], PILL("stable", "good") if d["psi"] < 0.1 else PILL("watch", "warn") if d["psi"] < 0.25 else PILL("drift", "crit")] for d in m["drift"]], digits=4, maxh=330,
                       sub="PSI < 0.1 stable · 0.1–0.25 watch · > 0.25 retrain trigger.")),
            FORM("api/loadtest", [F("n", "Sequential requests", "range", 300, min=50, max=2000, step=50)], title="Load test the inference path (in-process, single worker)", submit="Run load test"),
            TABLE("Latest requests", ["time", "route", "fare $", "latency ms", "model"], [[x["at"], x["route"], round(x["fare"], 2), round(x["ms"], 2), {"code": x["version"]}] for x in list(LOG)[-12:][::-1]] or [["—", "no requests yet — use the estimator", None, None, ""]]),
            MD("```\ncurl -X POST http://127.0.0.1:8013/api/predict -H 'Content-Type: application/json' \\\n  -d '{\"pickup_zone\":\"Midtown Center\",\"dropoff_zone\":\"JFK Airport\",\"hour\":18,\"dow\":4}'\n```\nOpenAPI docs: `/docs`. Invalid zones or out-of-range fields return HTTP 422.", title="Inference REST API")]


def p_auditor() -> list[dict]:
    a = audit_project(ROOT, **AUDIT_CFG)
    f = a["files"][0]
    return [H("Code auditor workbench", "A static AST audit of this project's own training pipeline, re-run every time this page loads."),
            KPIS(K("Grade", a["grade"]), K("Score", a["overall"], digits=1), K("Checks passed", a["checks_passed"]), K("Fail / warn / info", f"{a['counts']['fail']} / {a['counts']['warn']} / {a['counts']['info']}"), K("Lines audited", a["loc"])),
            GRID(2, CHART("hbar", "Score by dimension", [S("score", list(a["dimensions"].values()))], list(a["dimensions"]), xmin=0, xmax=100, height=240),
                 TABLE("Checks passed", ["rule", "dimension", "evidence"], [[{"code": p["rule"]}, p["dimension"], p["message"]] for p in f["passes"]], maxh=260)),
            TABLE("Findings", ["severity", "rule", "line", "message", "code / waiver"], [[PILL(x["severity"], {"fail": "crit", "warn": "warn", "info": "info", "waived": ""}[x["severity"]]), {"code": x["rule"]}, x["line"], x["message"], x["waiver"] or x["code"]] for x in f["findings"]] or [[PILL("clean", "good"), "", None, "no findings", ""]]),
            TABS(("features()", [CODE("Booking-time features only", inspect.getsource(pipeline.features))]), ("Chronological CV", [CODE("pipeline.py — main() excerpt", "\n".join(inspect.getsource(pipeline.main).splitlines()[5:22]))]),
                 ("PSI", [CODE("Population Stability Index", inspect.getsource(pipeline.psi))]))]


def p_paper() -> list[dict]:
    m = M()
    t, q, r, e = m["test"], m["quality"], m["registry"], m["eda"]
    lb = m["leaderboard"]
    ab = {a["removed"]: a for a in m["ablation"]}
    return [H("CRISP-DM research report", "Generated from the pipeline's artifacts — every figure below is read from `artifacts/metrics.json`, none is typed by hand."), MD(f"""
## Abstract
We build and audit a fare-prediction service for New York City taxis from {q['rows']:,} trips recorded in March 2019. A gradient-boosted model using only booking-time information attains a mean absolute error of **${t['mae']:.2f}** on a chronologically held-out final week (R² = {t['r2']:.3f}), against ${t['baseline_meter_mae']:.2f} for a distance-only tariff formula and ${t['baseline_median_mae']:.2f} for the median fare. The pipeline is leak-free by construction and by static audit.

## 1 · Business understanding
**Decision supported**: quote a fare before the ride. **Users**: riders (budgeting), dispatch (pricing sanity checks). **Success criteria**: beat the published tariff formula on unseen future trips; state uncertainty honestly; remain auditable. **Non-goals**: predicting tips (unrecorded for cash) or surge pricing (not present in medallion-cab data).

## 2 · Data understanding
The dataset holds pickup/drop-off timestamps and TLC zones, distance, fare, tip, tolls, total, payment type and cab colour. Fare is dominated by distance: an OLS fit gives fare ≈ {e['ols']['intercept']:.2f} + {e['ols']['slope']:.2f}·miles (r = {e['ols']['r']:.3f}), reproducing the regulated tariff ($2.50 flag-drop, $2.50/mile). Demand is concentrated in Manhattan with a weekday evening peak. Data quality averages {100 * q['overall']:.1f}% across six rule-based dimensions.

## 3 · Data preparation
Rows failing validity rules are removed ({q['rows'] - m['clean_rows']} of {q['rows']:,}). {r['n_features']} features are derived in four groups — distance, temporal, spatial (zone centroids, borough indicators, airport and cross-borough flags) and trip attributes. **Excluded on principle**: tip, tolls, total, payment type and drop-off time, all of which are only known after the ride.

## 4 · Modeling
Trips are ordered by pickup time. The training window is {r['train_window'][0]} → {r['train_window'][1]}; the test window is {r['test_window'][0]} → {r['test_window'][1]}. Inside the training window, 4-fold expanding-window cross-validation ranks seven candidates; the best CV MAE is ${lb[0]['cv_mae']:.3f} ({lb[0]['model']}). A hill-climb of {m['autoresearch']['n_trials']} experiments tunes XGBoost. Ablation shows distance features are indispensable (CV MAE rises to ${ab['distance']['cv_mae']:.2f} without them) while temporal and spatial groups contribute Δ {ab['temporal']['delta']:+.3f} and Δ {ab['spatial']['delta']:+.3f}.

## 5 · Evaluation
| Metric (test week, n = {t['n_test']:,}) | Value |
|---|---|
| MAE | ${t['mae']:.3f} |
| RMSE | ${t['rmse']:.3f} |
| R² | {t['r2']:.4f} |
| MAPE | {100 * t['mape']:.1f}% |
| Within $1 / $2 | {100 * t['within_1usd']:.1f}% / {100 * t['within_2usd']:.1f}% |
| Baseline: tariff formula MAE | ${t['baseline_meter_mae']:.3f} |
| Baseline: median fare MAE | ${t['baseline_median_mae']:.3f} |
| 80% interval half-width / empirical coverage | ±${t['q80']:.2f} / {100 * t['coverage80']:.1f}% |

RMSE far above MAE signals a heavy tail: a small number of flat-fare, negotiated or mis-metered trips that distance cannot explain. Errors grow with trip length in dollars but shrink in percentage terms.

## 6 · Deployment
The model is served by FastAPI with Pydantic validation, versioned by a hash of data, parameters and feature list (`{r['version']}`), monitored with PSI drift statistics, and load-tested in-process. A static AST auditor re-checks the training code on every page load.

## Limitations
One month, {m['clean_rows']:,} trips, medallion cabs only. Route distance is an input (it must come from a routing service at quote time). The interval is a constant-width conformal band, so it is conservative for short rides and optimistic for long ones.
""")]


SKILLS = [("CRISP-DM phase gating", "process", "six phases, each with an exit artefact on this site"), ("Rule-based data-quality scoring", "data", "pipeline.quality"), ("Tukey IQR outlier fences", "data", "EDA › distributions"),
          ("OLS trend + Pearson/Spearman", "data", "EDA › bivariate, correlations"), ("Zone-level spatial aggregation", "data", "Spatial analytics"), ("K-Means demand clustering", "modeling", "Spatial analytics"),
          ("Leak-free feature engineering", "modeling", "pipeline.features"), ("Chronological split + TimeSeriesSplit", "evaluation", "pipeline.main"), ("Model tournament with baselines", "modeling", "AutoResearch"),
          ("Hill-climbing HPO", "modeling", "common/autoresearch.py"), ("Feature-group ablation", "evaluation", "AutoResearch"), ("TreeSHAP attribution", "explainability", "Estimator, Explainable AI"),
          ("Error slicing", "evaluation", "Explainable AI"), ("Split-conformal intervals", "evaluation", "Estimator"), ("PSI drift monitoring", "mlops", "MLOps console"), ("Content-hashed model registry", "mlops", "MLOps console"),
          ("Load testing & latency percentiles", "mlops", "MLOps console"), ("Typed REST API (FastAPI + Pydantic)", "engineering", "app.py"), ("Static AST code audit", "governance", "common/audit.py"),
          ("Declarative UI from JSON blocks", "engineering", "common/kit/kit.js"), ("Offline vector map rendering", "engineering", "common/kit/nycmap.js"), ("Browser end-to-end tests", "engineering", "scripts/shots.py")]


def p_arch() -> list[dict]:
    return [H("Architecture & skills matrix", "How this application is built, and which data-science skills each part exercises."),
            MD("""
```
 data/taxis.csv ──► pipeline.py ──► artifacts/metrics.json + model.joblib ──► app.py (FastAPI) ──► JSON blocks ──► kit.js (browser)
                     │  quality → clean → features → tournament → HPO → ablation → test → SHAP → drift → registry
                     └─ audited statically by common/audit.py  (shown on the Code auditor page)
```
- **Training and serving share one `features()` function** — there is no train/serve skew by construction.
- **The dashboard never recomputes metrics**: pages render what the pipeline wrote, so the site cannot disagree with the experiment.
- **No build step**: the front end is one vanilla-JS runtime that renders declarative blocks.""", title="System diagram"),
            TABLE("Skills matrix", ["skill", "category", "where to see it"], [[a, PILL(b, "info"), c] for a, b, c in SKILLS])]


app = make_app(__file__, "13", "NYC TLC Mobility Platform", "Auditable CRISP-DM: quality · EDA · spatial · AutoResearch · XAI · MLOps · code audit", [
    ("estimator", "Fare estimator", "App", p_estimator),
    ("quality", "Data quality", "Audit console", p_quality), ("eda", "EDA dashboard", "Audit console", p_eda), ("spatial", "Spatial analytics", "Audit console", p_spatial),
    ("research", "AutoResearch", "Audit console", p_research), ("xai", "Explainable AI", "Audit console", p_xai), ("mlops", "MLOps console", "Audit console", p_mlops), ("auditor", "Code auditor", "Audit console", p_auditor),
    ("paper", "CRISP-DM report", "Docs", p_paper), ("architecture", "Architecture & skills", "Docs", p_arch),
])


@app.post("/api/predict")
def predict(t: Trip) -> dict:
    t0 = time.perf_counter()
    r = predict_core(t)
    ms = (time.perf_counter() - t0) * 1000
    LOG.append({"at": time.strftime("%H:%M:%S"), "route": f"{t.pickup_zone} → {t.dropoff_zone}", "fare": r["fare"], "ms": ms, "version": r["version"]})
    top = r["contrib"][:9]
    rest = sum(v for _, v in r["contrib"][9:])
    labels = ["base value"] + [k for k, _ in top] + ["other features"]
    vals = [r["base"]] + [v for _, v in top] + [rest]
    return {"fare": r["fare"], "interval": [r["lo"], r["hi"]], "distance": r["distance"], "model_version": r["version"], "latency_ms": ms,
            "blocks": [KPIS(K("Predicted fare", f"${r['fare']:.2f}", hint=f"80% interval ${r['lo']:.2f} – ${r['hi']:.2f}"), K("Route distance", f"{r['distance']:.1f} mi"), K("Boroughs", " → ".join(r["boroughs"])), K("Latency", f"{ms:.1f} ms", hint=f"model {r['version']}"))],
            "detail": [CHART("bar", "TreeSHAP waterfall — how the model builds this fare (dollars)", [S("contribution", vals, colors=["--axis"] + ["2" if v > 0 else "1" for v in vals[1:]])], labels, height=260,
                             note=f"Base value ${r['base']:.2f} + contributions = **${r['fare']:.2f}**. Orange raises the fare, blue lowers it.")]}


@app.post("/api/loadtest")
def loadtest(l: Load) -> dict:
    model()
    zs = [z for z, v in M()["spatial"]["trips"].items() if v >= 20]
    rs = np.random.RandomState(0)
    lat = []
    t0 = time.perf_counter()
    for _ in range(l.n):
        a, b = rs.choice(zs, 2)
        t1 = time.perf_counter()
        predict_core(Trip(pickup_zone=a, dropoff_zone=b, hour=int(rs.randint(24)), dow=int(rs.randint(7))))
        lat.append((time.perf_counter() - t1) * 1000)
    wall = time.perf_counter() - t0
    lat = np.array(lat)
    return {"rps": l.n / wall, "blocks": [KPIS(K("Requests", l.n), K("Throughput", f"{l.n / wall:.0f} req/s"), K("p50", f"{np.percentile(lat, 50):.2f} ms"), K("p95", f"{np.percentile(lat, 95):.2f} ms"), K("p99", f"{np.percentile(lat, 99):.2f} ms"), K("Errors", 0)),
                                           hist_chart(lat, "Latency distribution (ms)", bins=30, x="ms"),
                                           CALLOUT("info", "What this measures", "Feature construction + XGBoost inference + TreeSHAP for each request, executed sequentially inside one Python worker — a floor for a single uvicorn process, without network overhead.")]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8013)
