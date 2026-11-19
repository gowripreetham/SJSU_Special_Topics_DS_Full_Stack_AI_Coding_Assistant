"""Project 01 — NYC Taxi Trip Duration Predictor: FastAPI serving + interactive map estimator + DS admin dashboard."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
_state: dict = {}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def M() -> dict:
    if "m" not in _state:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _state["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _state["m"]


def model() -> dict:
    if "model" not in _state:
        _state["model"] = load_or_train(ROOT / "artifacts" / "model.joblib", pipeline.main)
    return _state["model"]


class Trip(BaseModel):
    pickup_lon: float = Field(ge=-74.3, le=-73.6)
    pickup_lat: float = Field(ge=40.5, le=41.0)
    dropoff_lon: float = Field(ge=-74.3, le=-73.6)
    dropoff_lat: float = Field(ge=40.5, le=41.0)
    hour: float = Field(17, ge=0, lt=24)
    dow: int = Field(2, ge=0, le=6)
    passengers: int = Field(1, ge=1, le=6)
    month: int = Field(4, ge=1, le=12)


def meter_fare(minutes: float, road_km: float, hour: float, dow: int) -> float:
    """2016 TLC standard metered rate (rule-based, not learned): $2.50 flag drop + $0.50 per 1/5 mile above 12 mph,
    or per minute below it, + $0.80 surcharges, + $0.50 overnight / $1 weekday rush."""
    miles = road_km / 1.609
    slow_min = max(0.0, minutes - miles / 12 * 60)
    fare = 2.50 + 2.50 * miles + 0.50 * slow_min + 0.80
    if hour >= 20 or hour < 6:
        fare += 0.50
    if dow < 5 and 16 <= hour < 20:
        fare += 1.00
    return round(fare, 2)


def predict_core(t: Trip) -> dict:
    mm = model()
    df = pd.DataFrame([{"pickup_datetime": pd.Timestamp(2016, t.month, 4 + t.dow, int(t.hour), int(t.hour % 1 * 60)),  # 2016-04-04 was a Monday
                        "pickup_longitude": t.pickup_lon, "pickup_latitude": t.pickup_lat, "dropoff_longitude": t.dropoff_lon,
                        "dropoff_latitude": t.dropoff_lat, "passenger_count": t.passengers, "vendor_id": 2}])
    if t.month != 4:
        df["pickup_datetime"] = pd.Timestamp(2016, t.month, 1, int(t.hour)) + pd.offsets.Week(weekday=t.dow)
    X = pipeline.features(df, mm["kmeans"])[mm["columns"]]
    import xgboost
    contrib = mm["model"].get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)[0]
    logp = float(contrib.sum())
    secs, lo, hi = np.expm1(logp), np.expm1(logp - mm["q80"]), np.expm1(logp + mm["q80"])
    road = float(X.manhattan_km.iloc[0])
    return {"minutes": secs / 60, "lo": lo / 60, "hi": hi / 60, "dist_km": float(X.dist_km.iloc[0]), "road_km": road,
            "speed_kmh": road / (secs / 3600), "fare": meter_fare(secs / 60, road, t.hour, t.dow),
            "contrib": sorted(zip(mm["columns"], contrib[:-1].tolist()), key=lambda kv: -abs(kv[1]))[:8], "base": float(contrib[-1]), "log_pred": logp}


def by_hour(t: Trip) -> list[float]:
    return [predict_core(t.model_copy(update={"hour": h}))["minutes"] for h in range(24)]


# ----------------------------------------------------------------------------- pages
def p_estimator() -> list[dict]:
    m = M()
    return [H("Trip estimator", "Gradient-boosted model trained on real 2016 NYC yellow-cab trips. Every estimate ships with an honest 80% conformal interval."),
            CUSTOM("trip", {"points": m["eda"]["pickups"]}, card=True)]


def p_eda() -> list[dict]:
    e, m = M()["eda"], M()
    ed = e["dur_edges"]
    return [H("Exploratory data analysis", f"{m['quality']['clean_rows']:,} cleaned trips · {m['date_range'][0][:10]} → {m['date_range'][1][:10]}"),
            GRID(2, CHART("bar", "Trip duration (minutes)", [S("trips", e["dur_hist"])], [f"{(ed[i] + ed[i + 1]) / 2:.0f}" for i in range(30)], x="minutes (clipped at 60)", y="trips", sub="Right-skewed — which is why we model log1p(duration)."),
                 CHART("bar", "log1p(duration seconds) — the modelling target", [S("trips", e["log_hist"], color="3")], [f"{(a + b) / 2:.1f}" for a, b in zip(e["log_edges"], e["log_edges"][1:])], x="log seconds", y="trips", sub="Near-Gaussian after the log transform.")),
            HEAT("Median straight-line speed (km/h) by day × hour", list(range(24)), DOW, e["speed_heat"], digits=0, sub="Weekday daytime congestion is the dominant temporal signal; nights and Sunday mornings are fastest.", labw=50),
            GRID(2, CHART("scatter", "Distance vs duration", [S("trip", e["scatter"], r=2.5)], x="haversine km", y="minutes", sub="Roughly linear with fan-shaped noise from traffic."),
                 CHART("line", "Demand and duration by hour", [S("median minutes", e["dur_by_hour"], color="2")], list(range(24)), x="pickup hour", y="median minutes")),
            GRID(2, CHART("bar", "Trips by pickup hour", [S("trips", e["trips_by_hour"])], list(range(24)), x="hour", y="trips"),
                 HEAT("Pearson correlation", e["corr_cols"], e["corr_cols"], e["corr"], diverging=True, min=-1, max=1, labw=80)),
            TABLE("Descriptive statistics", ["stat", "minutes", "distance km", "speed km/h", "passengers"], e["desc"])]


def p_models() -> list[dict]:
    m = M()
    lb = m["leaderboard"]
    return [H("Model tournament & AutoResearch", "All selection is done with cross-validation on the training split. The 20% test split is scored once, at the end."),
            GRID(2, CHART("hbar", "5-fold CV RMSLE (lower is better)", [S("RMSLE", [r["cv_rmsle"] for r in lb])], [r["model"] for r in lb], height=260),
                 TABLE("Leaderboard", ["model", "CV RMSLE", "CV MAE (min)", "fit secs"], [[r["model"], r["cv_rmsle"], r["cv_mae_min"], r["secs"]] for r in lb], digits=4)),
            MD("## AutoResearch hill-climb\nStarting from default XGBoost, the agent proposes **one hyper-parameter change per experiment** and keeps it only when 3-fold validation RMSLE improves — the same keep/discard loop as the `autoresearch` skill, fully logged below.", card=False),
            *ar_blocks(m["autoresearch"], "RMSLE")]


def p_admin() -> list[dict]:
    m = M()
    t, q, e = m["test"], m["quality"], m["eda"]
    imp = m["importance"][:12]
    return [H("Data-science admin dashboard", "Everything a reviewer needs: held-out metrics, data quality, explainability, residual diagnostics and calibration of the uncertainty band."),
            KPIS(K("Test RMSLE", t["rmsle"], digits=4, hint=f"median baseline {t['baseline_rmsle']:.3f}"), K("Test MAE", f"{t['mae_min']:.2f} min"),
                 K("Median abs % error", f"{100 * t['median_ape']:.1f}%"), K("R² (log space)", t["r2_log"], digits=3),
                 K("80% interval coverage", f"{100 * t['coverage80']:.1f}%", hint="target 80% — conformal band is calibrated")),
            GRID(2, CHART("hbar", "Permutation importance (test set)", [S("ΔRMSE when shuffled", [v for _, v in imp])], [k for k, _ in imp], height=330, sub="Model-agnostic: how much the error grows when one feature is scrambled."),
                 CHART("scatter", "Predicted vs actual (test)", [S("trip", e["pred_vs_actual"], r=2.5), S("_ideal", [{"x": 0, "y": 0}, {"x": 60, "y": 60}], line=True, color="--axis")], x="actual minutes", y="predicted minutes", height=330, xmax=60, ymax=60)),
            GRID(2, CHART("scatter", "Residuals vs prediction", [S("trip", e["resid"], r=2.5, color="2")], x="predicted minutes", y="actual − predicted (min)", sub="No strong funnel after the log transform; long trips are slightly under-predicted."),
                 CHART("bar", "Residual distribution (log space)", [S("trips", e["resid_hist"], color="7")], [f"{-1.5 + 0.1 * i + 0.05:.2f}" for i in range(30)], x="log residual", y="trips")),
            GRID(2, TABLE("Data quality gate", ["check", "rows"], [["raw rows", q["raw_rows"]], ["missing cells", q["missing_cells"]], ["duplicate ids", q["duplicate_ids"]], ["dropped: duration outside 1 min–3 h", q["dropped_duration"]], ["dropped: outside NYC bounding box", q["dropped_out_of_nyc"]], ["dropped: invalid passenger count", q["dropped_passengers"]], ["dropped: impossible speed", q["dropped_speed"]], ["clean rows", q["clean_rows"]]]),
                 MD(f"""- **Model**: XGBoost regressor on `log1p(duration)`, {len(m['features'])} engineered features
- **Train / test**: {t['n_train']:,} / {t['n_test']:,} trips (random 80/20, seed 42)
- **Leakage controls**: no drop-off timestamp or speed in features; spatial K-Means fit on train rows only; test scored once
- **Uncertainty**: split-conformal band from out-of-fold residuals (±{t['conformal_q80']:.3f} in log space)
- **Limits**: 6k-trip sample of the 1.46M-row competition file, 2016 traffic; no weather or events. Fare is a rule-based meter estimate.
- **Pipeline runtime**: {m['train_secs']} s on 2 CPU cores""", title="Model card"))]


CRISPDM = [
    ("Business understanding", "Riders and dispatchers want to know *how long will this trip take?* before it starts. Success = beat a naive median by a wide margin on RMSLE (the Kaggle metric) **and** report an honest uncertainty band."),
    ("Data understanding", "Kaggle *NYC Taxi Trip Duration* sample: pickup time, GPS pickup/drop-off, passenger count, vendor, duration. EDA shows a log-normal target, strong hour×weekday congestion, and a near-linear distance relationship."),
    ("Data preparation", "Quality gate (duration 1 min–3 h, NYC bounding box, plausible speed). Features: haversine & Manhattan distance, bearing, grid-rotated coordinates, cyclic hour, rush-hour flag, airport proximity, and K-Means pickup/drop-off clusters fit on training rows only."),
    ("Modeling", "Tournament of median baseline → Ridge → Random Forest → HistGradientBoosting → XGBoost under 5-fold CV, then an AutoResearch hill-climb over 8 XGBoost hyper-parameters with every experiment logged."),
    ("Evaluation", "Single final score on the 20% hold-out, permutation importance, residual diagnostics, and a coverage check proving the 80% conformal interval actually contains ~80% of test trips."),
    ("Deployment", "FastAPI `POST api/predict` (Pydantic-validated, ~10 ms), an interactive offline NYC zone map, per-request TreeSHAP attributions, and this admin dashboard for monitoring."),
]

API_DOC = """
### `POST api/predict`
```
curl -X POST http://127.0.0.1:8001/api/predict -H 'Content-Type: application/json' \\
  -d '{"pickup_lon":-73.9855,"pickup_lat":40.758,"dropoff_lon":-73.7781,"dropoff_lat":40.6413,"hour":17,"dow":2,"passengers":1}'
```
Returns the point estimate (minutes), the 80% conformal interval, distances, the metered-fare estimate and per-feature TreeSHAP contributions.
Inputs are validated: coordinates must fall inside the NYC bounding box, `hour ∈ [0,24)`, `dow ∈ 0..6`, `passengers ∈ 1..6` — bad requests get HTTP 422.

Interactive OpenAPI docs are generated automatically at **`/docs`**.
"""

app = make_app(__file__, "01", "NYC Taxi Trip Predictor", "CRISP-DM spatial regression · XGBoost · AutoResearch · conformal intervals", [
    ("estimator", "Trip estimator", "App", p_estimator),
    ("eda", "EDA", "Data science", p_eda),
    ("models", "Models & AutoResearch", "Data science", p_models),
    ("admin", "Admin dashboard", "Data science", p_admin),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report", "The six phases, as executed for this project."), CRISP(CRISPDM)]),
    ("api", "REST API", "Docs", lambda: [H("REST API"), MD(API_DOC)]),
])


@app.post("/api/predict")
def predict(t: Trip) -> dict:
    r = predict_core(t)
    hrs = by_hour(t)
    return {**{k: r[k] for k in ("minutes", "lo", "hi", "dist_km", "fare")},
            "blocks": [KPIS(K("Estimated duration", f"{r['minutes']:.1f} min", hint=f"80% interval {r['lo']:.0f}–{r['hi']:.0f} min"),
                            K("Metered fare (rule)", f"${r['fare']:.2f}", hint="2016 TLC rate card"),
                            K("Distance", f"{r['dist_km']:.1f} km", hint=f"~{r['road_km']:.1f} km on the grid"),
                            K("Avg speed", f"{r['speed_kmh']:.0f} km/h"))],
            "detail": [GRID(2, CHART("line", "Same trip at every hour of the day", [S("minutes", hrs)], list(range(24)), x="pickup hour", y="minutes", height=240, sub="Counterfactual sweep — the model is re-queried 24 times."),
                            CHART("hbar", "Why this estimate? (TreeSHAP, log-seconds)", [S("contribution", [v for _, v in r["contrib"]], colors=["2" if v > 0 else "1" for _, v in r["contrib"]])], [k for k, _ in r["contrib"]], height=240, sub="Orange pushes the duration up, blue pulls it down, relative to the average trip."))]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
