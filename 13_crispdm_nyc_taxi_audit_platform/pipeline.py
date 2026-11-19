"""Project 13 — NYC TLC mobility platform: an auditable CRISP-DM pipeline for taxi-fare prediction.

Data: 6,433 real NYC yellow/green taxi trips from March 2019 (the TLC sample distributed with seaborn), joined to TLC taxi-zone
geometry. Target: metered fare in dollars, predicted from what is known when the ride is booked.
Integrity rules enforced here and re-checked by the built-in code auditor:
  * chronological split — train on trips before 2019-03-24, test on the final 8 days; CV uses TimeSeriesSplit
  * no post-trip columns in features (tip, tolls, total, drop-off time, payment)
  * model selection on training folds only; the test window is scored once
Run:  python pipeline.py
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from lightgbm import LGBMRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hill_climb, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42
CUTOFF = pd.Timestamp("2019-03-24")
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx"]
GROUPS = {"distance": ["distance", "log_distance", "centroid_km"], "temporal": ["hour", "dow", "is_weekend", "rush", "night"],
          "spatial": ["p_lon", "p_lat", "d_lon", "d_lat", "same_zone", "airport", "cross_borough"] + [f"p_{b}" for b in BOROUGHS] + [f"d_{b}" for b in BOROUGHS],
          "trip": ["passengers", "is_yellow"]}
FEATURES = [f for g in GROUPS.values() for f in g]


def zones() -> dict[str, dict]:
    return {z["zone"]: z for z in json.loads((ROOT.parent / "common" / "kit" / "nyc_zones.json").read_text())}


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2))


def load_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "taxis.csv", parse_dates=["pickup", "dropoff"])


def quality(df: pd.DataFrame) -> dict:
    """Six-dimension data-quality scorecard, each a share of rows passing an explicit rule."""
    dur = (df.dropoff - df.pickup).dt.total_seconds() / 60
    surcharge = df.total - (df.fare + df.tip + df.tolls)
    dims = {"Completeness": (float(1 - df.isna().any(axis=1).mean()), "rows with no missing field"),
            "Uniqueness": (float(1 - df.duplicated().mean()), "rows that are not exact duplicates"),
            "Validity": (float(((df.distance > 0) & (df.fare >= 2.5) & df.passengers.between(1, 6)).mean()), "distance > 0, fare ≥ $2.50 flag-drop, 1–6 passengers"),
            "Consistency": (float(surcharge.between(0, 8).mean()), "total − (fare + tip + tolls) is a plausible surcharge of $0–8"),
            "Timeliness": (float(((df.pickup >= "2019-03-01") & (dur > 0) & (dur < 180)).mean()), "pickup inside March 2019 and 0 < duration < 3 h"),
            "Accuracy": (float((df.fare / df.distance.clip(lower=0.1)).between(1.5, 30).mean()), "fare per mile between $1.50 and $30")}
    return {"dims": [{"name": k, "score": v[0], "rule": v[1]} for k, v in dims.items()], "overall": float(np.mean([v[0] for v in dims.values()])),
            "missing": {c: int(n) for c, n in df.isna().sum().items() if n}, "rows": len(df)}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    ok = (df.distance > 0) & (df.fare >= 2.5) & df.passengers.between(1, 6) & df.pickup_zone.notna() & df.dropoff_zone.notna() & (df.fare / df.distance.clip(lower=0.1)).between(1.5, 30) & (df.pickup >= "2019-03-01")
    return df[ok].sort_values("pickup").reset_index(drop=True)


def features(df: pd.DataFrame, Z: dict[str, dict] | None = None) -> pd.DataFrame:
    """Booking-time information only. Never reads the fare, the gratuity or anything recorded after the ride ends."""
    Z = Z or zones()
    X = pd.DataFrame(index=df.index)
    pc = np.array([Z[z]["c"] for z in df.pickup_zone])
    dc = np.array([Z[z]["c"] for z in df.dropoff_zone])
    X["distance"] = df.distance.to_numpy()
    X["log_distance"] = np.log1p(X.distance)
    X["centroid_km"] = haversine(pc[:, 0], pc[:, 1], dc[:, 0], dc[:, 1])
    t = df.pickup.dt
    X["hour"], X["dow"] = t.hour + t.minute / 60, t.dayofweek
    X["is_weekend"] = (X.dow >= 5).astype(int)
    X["rush"] = ((X.dow < 5) & X.hour.between(16, 20)).astype(int)
    X["night"] = ((X.hour >= 20) | (X.hour < 6)).astype(int)
    X["p_lon"], X["p_lat"], X["d_lon"], X["d_lat"] = pc[:, 0], pc[:, 1], dc[:, 0], dc[:, 1]
    X["same_zone"] = (df.pickup_zone.to_numpy() == df.dropoff_zone.to_numpy()).astype(int)
    X["airport"] = (df.pickup_zone.str.contains("Airport") | df.dropoff_zone.str.contains("Airport")).astype(int).to_numpy()
    X["cross_borough"] = (df.pickup_borough.to_numpy() != df.dropoff_borough.to_numpy()).astype(int)
    for b in BOROUGHS:
        X[f"p_{b}"], X[f"d_{b}"] = (df.pickup_borough == b).astype(int).to_numpy(), (df.dropoff_borough == b).astype(int).to_numpy()
    X["passengers"] = df.passengers.to_numpy()
    X["is_yellow"] = (df.color == "yellow").astype(int).to_numpy()
    return X[FEATURES]


def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of b (live) against a (reference). < 0.1 stable · 0.1–0.25 watch · > 0.25 drift."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([a.min() - 1e-9, (a.min() + a.max()) / 2, a.max() + 1e-9])
    pa = np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a) + 1e-4
    pb = np.histogram(np.clip(b, edges[0], edges[-1]), edges)[0] / len(b) + 1e-4
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def xgb(**kw) -> XGBRegressor:
    base = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=SEED, n_jobs=2, tree_method="hist", objective="reg:absoluteerror")
    return XGBRegressor(**{**base, **kw})


def main() -> None:
    t0 = time.time()
    raw = load_raw()
    Z = zones()
    q = quality(raw)
    df = clean(raw)
    X, y = features(df, Z), df.fare.to_numpy()
    tr, te = np.where(df.pickup < CUTOFF)[0], np.where(df.pickup >= CUTOFF)[0]
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    tscv = TimeSeriesSplit(4)
    print(f"clean rows {len(df):,}; train {len(tr):,} (≤ {CUTOFF.date()}), test {len(te):,}")

    def cv_mae(model, cols=FEATURES) -> float:
        errs = []
        for a, b in tscv.split(Xtr):  # expanding window: always train on the past, validate on the future
            m = model.fit(Xtr.iloc[a][cols], ytr[a])
            errs.append(np.mean(np.abs(m.predict(Xtr.iloc[b][cols]) - ytr[b])))
        return float(np.mean(errs))

    class Median:
        def fit(self, X, y):
            self.m = float(np.median(y)); return self

        def predict(self, X):
            return np.full(len(X), self.m)

    print("== tournament (TimeSeriesSplit CV, MAE $) ==")
    cands = {"Median fare (baseline)": (Median(), FEATURES), "Meter formula: OLS on distance (baseline)": (LinearRegression(), ["distance"]),
             "Ridge (all features)": (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 10))), FEATURES),
             "Random forest": (RandomForestRegressor(300, min_samples_leaf=3, n_jobs=2, random_state=SEED), FEATURES),
             "HistGradientBoosting": (HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=SEED), FEATURES),
             "LightGBM": (LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, objective="mae", subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, n_jobs=2, verbose=-1), FEATURES),
             "XGBoost": (xgb(), FEATURES)}
    board = []
    for name, (m, cols) in cands.items():
        t1 = time.time()
        board.append({"model": name, "cv_mae": cv_mae(m, cols), "secs": round(time.time() - t1, 1)})
        print(f"  {name:<42} CV MAE ${board[-1]['cv_mae']:.3f}")

    print("== AutoResearch / HPO on XGBoost ==")
    ar = hill_climb(lambda c: cv_mae(xgb(**c)), {"max_depth": [3, 4, 5, 6, 8], "learning_rate": [0.02, 0.03, 0.05, 0.08], "n_estimators": [200, 400, 600, 900], "subsample": [0.6, 0.8, 1.0],
                                                 "colsample_bytree": [0.5, 0.7, 0.8, 1.0], "min_child_weight": [1, 3, 5, 10], "reg_lambda": [0.5, 1, 3, 10]},
                    dict(max_depth=5, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1), n_iter=20, seed=SEED, maximize=False)
    board.append({"model": "XGBoost (AutoResearch-tuned)", "cv_mae": ar["best_score"], "secs": None})

    print("== ablation: remove one feature group at a time ==")
    full = ar["best_score"]
    abl = [{"removed": "nothing (full model)", "cv_mae": full, "delta": 0.0, "n_features": len(FEATURES)}]
    for g, cols in GROUPS.items():
        keep = [f for f in FEATURES if f not in cols]
        v = cv_mae(xgb(**ar["best"]), keep)
        abl.append({"removed": g, "cv_mae": v, "delta": v - full, "n_features": len(keep)})
        print(f"  without {g:<9} CV MAE ${v:.3f}  (Δ {v - full:+.3f})")

    best = xgb(**ar["best"]).fit(Xtr, ytr)
    pte = best.predict(Xte)  # the test window is scored exactly once
    oof = np.full(len(tr), np.nan)
    for a, b in tscv.split(Xtr):
        oof[b] = xgb(**ar["best"]).fit(Xtr.iloc[a], ytr[a]).predict(Xtr.iloc[b])
    resid = np.abs(ytr - oof)[~np.isnan(oof)]
    q80 = float(np.quantile(resid, 0.8))
    base_meter = LinearRegression().fit(Xtr[["distance"]], ytr).predict(Xte[["distance"]])
    test = {"mae": float(np.mean(np.abs(yte - pte))), "rmse": float(np.sqrt(np.mean((yte - pte) ** 2))), "r2": float(1 - np.sum((yte - pte) ** 2) / np.sum((yte - yte.mean()) ** 2)),
            "mape": float(np.mean(np.abs(yte - pte) / yte)), "within_1usd": float(np.mean(np.abs(yte - pte) <= 1)), "within_2usd": float(np.mean(np.abs(yte - pte) <= 2)),
            "baseline_median_mae": float(np.mean(np.abs(yte - np.median(ytr)))), "baseline_meter_mae": float(np.mean(np.abs(yte - base_meter))), "q80": q80,
            "coverage80": float(np.mean(np.abs(yte - pte) <= q80)), "n_train": len(tr), "n_test": len(te)}
    print("TEST:", {k: round(v, 3) for k, v in test.items()})

    contrib = best.get_booster().predict(xgboost.DMatrix(Xte), pred_contribs=True)
    shap_global = sorted(zip(FEATURES, np.abs(contrib[:, :-1]).mean(0).tolist()), key=lambda kv: -kv[1])
    sl = df.iloc[te].assign(err=np.abs(yte - pte), pred=pte)
    slices = {"by_borough": [[b, int(len(d)), float(d.err.mean()), float(d.fare.mean())] for b, d in sl.groupby("pickup_borough")],
              "by_hour": sl.groupby(sl.pickup.dt.hour).err.mean().reindex(range(24)).round(3).tolist(),
              "by_distance": [[lab, int(len(d)), float(d.err.mean()), float((d.err / d.fare).mean())] for lab, d in sl.groupby(pd.cut(sl.distance, [0, 1, 2, 5, 10, 50], labels=["<1 mi", "1–2", "2–5", "5–10", "10+"]), observed=True)]}
    drift = sorted([{"feature": f, "psi": psi(Xtr[f].to_numpy(float), Xte[f].to_numpy(float))} for f in FEATURES if Xtr[f].nunique() > 2], key=lambda r: -r["psi"])
    drift.append({"feature": "fare (target)", "psi": psi(ytr, yte)})

    # ---- EDA + spatial summaries (descriptive, computed on the cleaned full month)
    d = df.assign(hour=df.pickup.dt.hour, dow=df.pickup.dt.dayofweek, mins=(df.dropoff - df.pickup).dt.total_seconds() / 60, tip_pct=100 * df.tip / df.fare)
    num = ["distance", "fare", "tip", "mins"]
    iqr = []
    for c in num:
        q1, q3 = d[c].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        iqr.append({"col": c, "q1": float(q1), "median": float(d[c].median()), "q3": float(q3), "mean": float(d[c].mean()), "lo": float(lo), "hi": float(hi), "outliers": int(((d[c] < lo) | (d[c] > hi)).sum()),
                    "hist": np.histogram(d[c].clip(upper=d[c].quantile(0.99)), bins=24)[0].tolist(), "edges": np.histogram(d[c].clip(upper=d[c].quantile(0.99)), bins=24)[1].round(2).tolist()})
    beta = np.polyfit(d.distance, d.fare, 1)
    s = d.sample(700, random_state=SEED)
    zagg = d.groupby("pickup_zone").agg(trips=("fare", "size"), fare=("fare", "mean"), tip_pct=("tip_pct", "mean"), dist=("distance", "mean"))
    P = np.array([Z[z]["c"] for z in d.pickup_zone])
    km = KMeans(6, n_init=10, random_state=SEED).fit(P)
    cl = d.assign(c=km.labels_).groupby("c").agg(trips=("fare", "size"), fare=("fare", "mean"), dist=("distance", "mean"), tip_pct=("tip_pct", "mean"), top=("pickup_zone", lambda x: x.mode().iloc[0]))
    eda = {"iqr": iqr, "ols": {"slope": float(beta[0]), "intercept": float(beta[1]), "r": float(np.corrcoef(d.distance, d.fare)[0, 1])},
           "scatter": [{"x": float(a), "y": float(b)} for a, b in zip(s.distance, s.fare)],
           "heat_trips": d.pivot_table(index="dow", columns="hour", values="fare", aggfunc="size").reindex(index=range(7), columns=range(24)).fillna(0).values.tolist(),
           "heat_speed": (d.assign(sp=d.distance / (d.mins / 60).clip(lower=0.02)).pivot_table(index="dow", columns="hour", values="sp", aggfunc="median").reindex(index=range(7), columns=range(24)).round(1)).values.tolist(),
           "borough": [[b, int(len(x)), float(len(x) / len(d)), float(x.fare.mean()), float(x.distance.median()), float(x.tip_pct.mean())] for b, x in d.groupby("pickup_borough")],
           "corr_cols": ["distance", "fare", "tip", "minutes", "passengers", "hour"], "pearson": d[["distance", "fare", "tip", "mins", "passengers", "hour"]].corr().round(3).values.tolist(),
           "spearman": d[["distance", "fare", "tip", "mins", "passengers", "hour"]].corr(method="spearman").round(3).values.tolist(),
           "payment": d.payment.fillna("unknown").value_counts().to_dict(), "color": d.color.value_counts().to_dict(), "tip_by_payment": d.groupby(d.payment.fillna("unknown")).tip_pct.mean().round(2).to_dict(),
           "daily": [[str(k), int(v)] for k, v in d.groupby(d.pickup.dt.date).size().items()]}
    spatial = {"trips": {z: int(v) for z, v in zagg.trips.items()}, "fare": {z: float(v) for z, v in zagg.fare[zagg.trips >= 5].items()}, "tip_pct": {z: float(v) for z, v in zagg.tip_pct[zagg.trips >= 5].items()},
               "centers": km.cluster_centers_.round(4).tolist(), "clusters": [[int(c), int(r.trips), float(r.fare), float(r.dist), float(r.tip_pct), r.top] for c, r in cl.iterrows()],
               "top_zones": [[z, int(r.trips), float(r.fare), float(r.tip_pct)] for z, r in zagg.sort_values("trips", ascending=False).head(12).iterrows()]}

    data_hash = hashlib.sha256((ROOT / "data" / "taxis.csv").read_bytes()).hexdigest()[:12]
    version = hashlib.sha1(json.dumps([data_hash, ar["best"], FEATURES], sort_keys=True).encode()).hexdigest()[:10]
    registry = {"version": version, "data_sha256": data_hash, "algorithm": "XGBoost reg:absoluteerror", "params": ar["best"], "n_features": len(FEATURES), "trained_rows": len(tr),
                "train_window": [str(df.pickup.iloc[tr[0]].date()), str(df.pickup.iloc[tr[-1]].date())], "test_window": [str(df.pickup.iloc[te[0]].date()), str(df.pickup.iloc[te[-1]].date())],
                "libs": {"xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    joblib.dump({"model": best, "q80": q80, "features": FEATURES, "version": version}, ART / "model.joblib", compress=3)
    save_json(ART / "metrics.json", {"quality": q, "clean_rows": len(df), "leaderboard": sorted(board, key=lambda r: r["cv_mae"]), "autoresearch": ar, "ablation": abl, "test": test,
                                     "shap_global": shap_global, "shap_base": float(contrib[0, -1]), "slices": slices, "drift": drift, "eda": eda, "spatial": spatial, "registry": registry,
                                     "groups": GROUPS, "pred_vs_actual": [{"x": float(a), "y": float(b)} for a, b in zip(yte[:600], pte[:600])], "secs": round(time.time() - t0, 1)})
    print(f"model {version} · done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
