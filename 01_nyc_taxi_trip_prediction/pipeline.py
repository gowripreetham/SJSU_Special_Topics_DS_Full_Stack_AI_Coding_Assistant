"""Project 01 — NYC Taxi Trip Duration: CRISP-DM training pipeline.

Data: a 6.3k-trip sample of the Kaggle "NYC Taxi Trip Duration" competition (Jan–Jun 2016, real GPS coordinates).
Target: log1p(trip_duration seconds)  ->  metric RMSLE (the competition metric).
Run:  python pipeline.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hill_climb, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42
JFK, LGA = (-73.7781, 40.6413), (-73.8740, 40.7769)
BBOX = dict(lon=(-74.3, -73.6), lat=(40.5, 41.0))


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load_clean() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(ROOT / "data" / "nyc_taxi_trips_sample.csv", parse_dates=["pickup_datetime"])
    n0 = len(df)
    rep = {"raw_rows": n0, "missing_cells": int(df.isna().sum().sum()), "duplicate_ids": int(df.id.duplicated().sum())}
    ok_d = df.trip_duration.between(60, 3 * 3600)
    ok_g = (df.pickup_longitude.between(*BBOX["lon"]) & df.dropoff_longitude.between(*BBOX["lon"])
            & df.pickup_latitude.between(*BBOX["lat"]) & df.dropoff_latitude.between(*BBOX["lat"]))
    ok_p = df.passenger_count.between(1, 6)
    d = haversine(df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude)
    ok_s = (d / (df.trip_duration / 3600)).between(0.5, 100) & (d > 0.05)  # implied straight-line speed sanity
    rep.update(dropped_duration=int((~ok_d).sum()), dropped_out_of_nyc=int((~ok_g).sum()),
               dropped_passengers=int((~ok_p).sum()), dropped_speed=int((~ok_s & ok_d & ok_g).sum()))
    df = df[ok_d & ok_g & ok_p & ok_s].reset_index(drop=True)
    rep["clean_rows"] = len(df)
    return df, rep


def features(df: pd.DataFrame, km: KMeans | None = None) -> pd.DataFrame:
    """Only information known at pickup time — no dropoff timestamp, no duration-derived columns."""
    X = pd.DataFrame(index=df.index)
    plon, plat, dlon, dlat = (df[c].to_numpy() for c in ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"])
    X["dist_km"] = haversine(plon, plat, dlon, dlat)
    X["manhattan_km"] = haversine(plon, plat, dlon, plat) + haversine(dlon, plat, dlon, dlat)
    X["log_dist"] = np.log1p(X.dist_km)
    X["bearing"] = np.degrees(np.arctan2(np.sin(np.radians(dlon - plon)) * np.cos(np.radians(dlat)),
                                         np.cos(np.radians(plat)) * np.sin(np.radians(dlat)) - np.sin(np.radians(plat)) * np.cos(np.radians(dlat)) * np.cos(np.radians(dlon - plon))))
    for n, v in [("pickup_lon", plon), ("pickup_lat", plat), ("dropoff_lon", dlon), ("dropoff_lat", dlat)]:
        X[n] = v
    th = np.radians(29)  # Manhattan's street grid is rotated ~29° from true north
    X["p_rot"], X["d_rot"] = plon * np.cos(th) + plat * np.sin(th), dlon * np.cos(th) + dlat * np.sin(th)
    t = df.pickup_datetime.dt
    X["hour"], X["dow"], X["month"] = t.hour + t.minute / 60, t.dayofweek, t.month
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * X.hour / 24), np.cos(2 * np.pi * X.hour / 24)
    X["is_weekend"] = (X.dow >= 5).astype(int)
    X["rush"] = ((X.dow < 5) & (X.hour.between(7, 10) | X.hour.between(16, 19))).astype(int)
    X["passengers"] = df.passenger_count.to_numpy()
    X["vendor"] = df.vendor_id.to_numpy()
    X["jfk"] = np.minimum(haversine(plon, plat, *JFK), haversine(dlon, dlat, *JFK))
    X["lga"] = np.minimum(haversine(plon, plat, *LGA), haversine(dlon, dlat, *LGA))
    if km is not None:
        X["p_cluster"] = km.predict(np.c_[plon, plat])
        X["d_cluster"] = km.predict(np.c_[dlon, dlat])
    return X


def rmsle(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))  # y, p already in log1p space


def main() -> None:
    t0 = time.time()
    df, quality = load_clean()
    y = np.log1p(df.trip_duration.to_numpy())
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
    # spatial clusters are *fit on the training rows only* (no test-set information leaks into features)
    km = KMeans(12, n_init=5, random_state=SEED).fit(np.r_[df.loc[tr, ["pickup_longitude", "pickup_latitude"]].to_numpy(),
                                                             df.loc[tr, ["dropoff_longitude", "dropoff_latitude"]].to_numpy()])
    X = features(df, km)
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    cv = KFold(5, shuffle=True, random_state=SEED)

    def xgb(**kw):
        base = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=3, reg_lambda=1.0, random_state=SEED, n_jobs=2, tree_method="hist")
        return XGBRegressor(**{**base, **kw})

    print("== model tournament (5-fold CV on train) ==")
    cands = {
        "Median baseline": None,
        "Ridge (scaled)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12))),
        "Random Forest": RandomForestRegressor(300, min_samples_leaf=3, n_jobs=2, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED),
        "XGBoost (default)": xgb(),
    }
    board, oof = [], {}
    for name, m in cands.items():
        t1 = time.time()
        p = np.full_like(ytr, np.median(ytr)) if m is None else cross_val_predict(m, Xtr, ytr, cv=cv)
        oof[name] = p
        board.append({"model": name, "cv_rmsle": rmsle(ytr, p), "cv_mae_min": float(np.mean(np.abs(np.expm1(ytr) - np.expm1(p))) / 60), "secs": round(time.time() - t1, 1)})
        print(f"  {name:<24} RMSLE {board[-1]['cv_rmsle']:.4f}")

    print("== AutoResearch hill-climb on XGBoost (3-fold CV, train only) ==")
    cv3 = KFold(3, shuffle=True, random_state=SEED)
    space = {"max_depth": [4, 5, 6, 7, 8, 10], "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1], "n_estimators": [300, 500, 800, 1200],
             "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "colsample_bytree": [0.5, 0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5, 10, 20],
             "reg_lambda": [0.5, 1.0, 3.0, 10.0], "gamma": [0, 0.05, 0.2]}
    start = dict(max_depth=6, learning_rate=0.05, n_estimators=400, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, gamma=0)
    space["n_estimators"].append(400)
    ar = hill_climb(lambda c: rmsle(ytr, cross_val_predict(xgb(**c), Xtr, ytr, cv=cv3)), space, start, n_iter=24, seed=SEED, maximize=False)

    best = xgb(**ar["best"])
    p_oof = cross_val_predict(best, Xtr, ytr, cv=cv)
    board.append({"model": "XGBoost (AutoResearch-tuned)", "cv_rmsle": rmsle(ytr, p_oof), "cv_mae_min": float(np.mean(np.abs(np.expm1(ytr) - np.expm1(p_oof))) / 60), "secs": None})
    best.fit(Xtr, ytr)
    pte = best.predict(Xte)  # the test set is touched exactly once, here
    # split-conformal interval from out-of-fold residuals (log space) -> honest 80% band
    q = float(np.quantile(np.abs(ytr - p_oof), 0.80))
    cover = float(np.mean(np.abs(yte - pte) <= q))
    test = {"rmsle": rmsle(yte, pte), "mae_min": float(np.mean(np.abs(np.expm1(yte) - np.expm1(pte))) / 60),
            "median_ape": float(np.median(np.abs(np.expm1(yte) - np.expm1(pte)) / np.expm1(yte))),
            "r2_log": float(1 - np.sum((yte - pte) ** 2) / np.sum((yte - yte.mean()) ** 2)),
            "baseline_rmsle": rmsle(yte, np.full_like(yte, np.median(ytr))), "conformal_q80": q, "coverage80": cover, "n_test": len(te), "n_train": len(tr)}
    print("TEST:", {k: round(v, 4) for k, v in test.items()})

    pi = permutation_importance(best, Xte, yte, n_repeats=5, random_state=SEED, scoring="neg_root_mean_squared_error")
    imp = sorted(zip(X.columns, pi.importances_mean), key=lambda kv: -kv[1])

    # ---- EDA summaries for the dashboard
    d = df.assign(dist=X.dist_km, hour=df.pickup_datetime.dt.hour, dow=df.pickup_datetime.dt.dayofweek, mins=df.trip_duration / 60)
    d["speed"] = d.dist / (d.mins / 60)
    sp = d.pivot_table(index="dow", columns="hour", values="speed", aggfunc="median").reindex(index=range(7), columns=range(24))
    rs = np.random.RandomState(SEED)
    samp = d.sample(900, random_state=SEED)
    res_idx = rs.choice(len(te), 600, replace=False)
    eda = {"dur_hist": np.histogram(np.clip(d.mins, 0, 60), bins=30)[0].tolist(), "dur_edges": np.linspace(0, 60, 31).tolist(),
           "log_hist": np.histogram(y, bins=30)[0].tolist(), "log_edges": np.histogram(y, bins=30)[1].tolist(),
           "speed_heat": sp.round(1).values.tolist(), "trips_by_hour": d.groupby("hour").size().reindex(range(24), fill_value=0).tolist(),
           "dur_by_hour": d.groupby("hour").mins.median().reindex(range(24)).round(2).tolist(),
           "trips_by_dow": d.groupby("dow").size().tolist(), "dur_by_month": d.groupby(d.pickup_datetime.dt.month).mins.median().round(2).to_dict(),
           "scatter": [{"x": round(a, 2), "y": round(b, 1)} for a, b in zip(samp.dist, samp.mins)],
           "pickups": [[round(a, 4), round(b, 4)] for a, b in zip(d.pickup_longitude[:2500], d.pickup_latitude[:2500])],
           "desc": d[["mins", "dist", "speed", "passenger_count"]].describe().round(2).reset_index().values.tolist(),
           "corr_cols": ["duration", "dist_km", "hour", "dow", "passengers", "jfk", "rush"],
           "corr": np.corrcoef(np.c_[y, X[["dist_km", "hour", "dow", "passengers", "jfk", "rush"]].to_numpy()].T).round(3).tolist(),
           "resid": [{"x": round(float(np.expm1(pte[i]) / 60), 2), "y": round(float((np.expm1(yte[i]) - np.expm1(pte[i])) / 60), 2)} for i in res_idx],
           "pred_vs_actual": [{"x": round(float(np.expm1(yte[i]) / 60), 2), "y": round(float(np.expm1(pte[i]) / 60), 2)} for i in res_idx],
           "resid_hist": np.histogram(yte - pte, bins=30, range=(-1.5, 1.5))[0].tolist()}

    joblib.dump({"model": best, "kmeans": km, "q80": q, "columns": list(X.columns)}, ART / "model.joblib", compress=3)
    save_json(ART / "metrics.json", {"quality": quality, "leaderboard": sorted(board, key=lambda r: r["cv_rmsle"]), "autoresearch": ar,
                                     "test": test, "importance": [[k, float(v)] for k, v in imp], "eda": eda, "features": list(X.columns),
                                     "date_range": [str(df.pickup_datetime.min()), str(df.pickup_datetime.max())], "train_secs": round(time.time() - t0, 1)})
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
