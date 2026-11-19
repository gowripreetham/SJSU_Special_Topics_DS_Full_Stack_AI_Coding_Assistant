"""Project 03 — Customer segmentation on the Kaggle "Mall Customers" dataset (200 customers).

Compares K-Means, Gaussian Mixtures, Ward agglomerative and DBSCAN with internal validity indices,
runs an AutoResearch hill-climb over (algorithm, k, feature set, scaler), and stress-tests the winner
with bootstrap stability (adjusted Rand index).   Run:  python pipeline.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hill_climb, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42
FEATURE_SETS = {"income+spend": ["income", "spend"], "age+income+spend": ["age", "income", "spend"],
                "age+spend": ["age", "spend"], "all+gender": ["age", "income", "spend", "male"]}
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "Mall_Customers.csv")
    df.columns = ["id", "gender", "age", "income", "spend"]
    df["male"] = (df.gender == "Male").astype(int)
    return df


def fit_labels(algo: str, k: int, Z: np.ndarray, seed: int = SEED):
    if algo == "kmeans":
        m = KMeans(k, n_init=10, random_state=seed).fit(Z)
        return m.labels_, m
    if algo == "gmm":
        m = GaussianMixture(k, n_init=3, random_state=seed).fit(Z)
        return m.predict(Z), m
    m = AgglomerativeClustering(k, linkage="ward").fit(Z)
    return m.labels_, m


def persona(row: pd.Series, g: pd.Series) -> tuple[str, str]:
    """Rule-based naming from the centroid relative to the population mean."""
    hi_i, lo_i = row.income > g.income + 12, row.income < g.income - 12
    hi_s, lo_s = row.spend > g.spend + 12, row.spend < g.spend - 12
    if hi_i and hi_s:
        return "Premium Enthusiasts", "VIP programme, early access, premium bundles"
    if hi_i and lo_s:
        return "Affluent but Careful", "Value-proof messaging, quality guarantees, targeted luxury offers"
    if lo_i and hi_s:
        return "Young Trend Spenders", "Flash sales, buy-now-pay-later, social campaigns"
    if lo_i and lo_s:
        return "Budget Essentials", "Discount coupons, loyalty points on staples"
    return ("Mainstream Seniors" if row.age > 48 else "Mainstream Young Adults" if row.age < 32 else "Mainstream Middle"), "Seasonal promotions and cross-sell of popular categories"


def main() -> None:
    t0 = time.time()
    df = load()

    def Z_of(fs: str, sc: str) -> np.ndarray:
        return SCALERS[sc]().fit_transform(df[FEATURE_SETS[fs]])

    # --- k sweep on the classic 2-D income × spend view
    Z2 = Z_of("income+spend", "standard")
    sweep = []
    for k in range(2, 11):
        lab, m = fit_labels("kmeans", k, Z2)
        gm = GaussianMixture(k, n_init=3, random_state=SEED).fit(Z2)
        sweep.append({"k": k, "inertia": float(m.inertia_), "silhouette": float(silhouette_score(Z2, lab)), "calinski": float(calinski_harabasz_score(Z2, lab)),
                      "davies": float(davies_bouldin_score(Z2, lab)), "gmm_bic": float(gm.bic(Z2)),
                      "ward_silhouette": float(silhouette_score(Z2, fit_labels("ward", k, Z2)[0]))})
    # --- DBSCAN grid (density view; label -1 = noise)
    dbs = []
    for eps in [0.25, 0.3, 0.35, 0.4, 0.5]:
        for ms in [3, 5, 8]:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z2)
            n = len(set(lab)) - (1 if -1 in lab else 0)
            core = lab != -1
            dbs.append({"eps": eps, "min_samples": ms, "clusters": n, "noise_pct": float(100 * (~core).mean()),
                        "silhouette": float(silhouette_score(Z2[core], lab[core])) if n > 1 else None})

    # --- AutoResearch: objective = silhouette penalised by instability (1 - bootstrap ARI)
    rs = np.random.RandomState(SEED)
    boots = [rs.choice(len(df), len(df), replace=True) for _ in range(8)]

    def stability(algo: str, k: int, Z: np.ndarray, ref: np.ndarray) -> float:
        s = []
        for i, b in enumerate(boots):
            lab_b, m = fit_labels(algo if algo != "ward" else "kmeans", k, Z[b], seed=i)
            s.append(adjusted_rand_score(ref, m.predict(Z)))
        return float(np.mean(s))

    def objective(c: dict) -> float:
        Z = Z_of(c["features"], c["scaler"])
        lab, _ = fit_labels(c["algo"], c["k"], Z)
        if min(np.bincount(lab)) < 8:  # reject degenerate micro-clusters
            return -1.0
        return float(silhouette_score(Z, lab)) * (0.5 + 0.5 * stability(c["algo"], c["k"], Z, lab))

    ar = hill_climb(objective, {"algo": ["kmeans", "gmm", "ward"], "k": list(range(2, 10)), "features": list(FEATURE_SETS), "scaler": list(SCALERS)},
                    {"algo": "kmeans", "k": 3, "features": "age+income+spend", "scaler": "standard"}, n_iter=28, seed=SEED)
    best = ar["best"]
    scaler = SCALERS[best["scaler"]]().fit(df[FEATURE_SETS[best["features"]]])
    Z = scaler.transform(df[FEATURE_SETS[best["features"]]])
    labels, model = fit_labels(best["algo"], best["k"], Z)
    if best["algo"] == "ward":  # serve with K-Means centroids initialised from the Ward partition
        model = KMeans(best["k"], init=np.array([Z[labels == j].mean(0) for j in range(best["k"])]), n_init=1, random_state=SEED).fit(Z)
        labels = model.labels_
    stab = stability(best["algo"], best["k"], Z, labels)
    sil = silhouette_samples(Z, labels)

    df["cluster"] = labels
    g = df[["age", "income", "spend"]].mean()
    prof = []
    for c, d in df.groupby("cluster"):
        name, action = persona(d[["age", "income", "spend"]].mean(), g)
        prof.append({"cluster": int(c), "persona": name, "action": action, "n": len(d), "share": len(d) / len(df), "age": d.age.mean(), "income": d.income.mean(),
                     "spend": d.spend.mean(), "male_pct": 100 * d.male.mean(), "silhouette": float(sil[labels == c].mean())})
    names = {}
    for p in sorted(prof, key=lambda r: -r["n"]):  # keep persona names unique
        base, i = p["persona"], 2
        while p["persona"] in names.values():
            p["persona"] = f"{base} {i}"
            i += 1
        names[p["cluster"]] = p["persona"]

    pca = PCA(2, random_state=SEED).fit(StandardScaler().fit_transform(df[["age", "income", "spend"]]))
    xy = pca.transform(StandardScaler().fit_transform(df[["age", "income", "spend"]]))
    joblib.dump({"scaler": scaler, "model": model, "features": FEATURE_SETS[best["features"]], "names": names}, ART / "model.joblib")
    save_json(ART / "metrics.json", {
        "n": len(df), "sweep": sweep, "dbscan": dbs, "autoresearch": ar, "best": best, "profiles": sorted(prof, key=lambda r: r["cluster"]),
        "final": {"silhouette": float(silhouette_score(Z, labels)), "calinski": float(calinski_harabasz_score(Z, labels)), "davies": float(davies_bouldin_score(Z, labels)), "stability_ari": stab},
        "points": [{"age": int(a), "income": int(i), "spend": int(s), "c": int(c), "px": round(float(x), 3), "py": round(float(y), 3), "sil": round(float(v), 3)}
                   for a, i, s, c, (x, y), v in zip(df.age, df.income, df.spend, labels, xy, sil)],
        "pca_var": pca.explained_variance_ratio_.tolist(), "pca_load": pca.components_.round(3).tolist(),
        "hist": {c: np.histogram(df[c], bins=14)[0].tolist() for c in ["age", "income", "spend"]},
        "hist_edges": {c: np.histogram(df[c], bins=14)[1].round(1).tolist() for c in ["age", "income", "spend"]},
        "gender": df.gender.value_counts().to_dict(), "corr": df[["age", "income", "spend"]].corr().round(3).values.tolist(),
        "desc": df[["age", "income", "spend"]].describe().round(1).reset_index().values.tolist(), "secs": round(time.time() - t0, 1)})
    print("best:", best, "| silhouette", round(silhouette_score(Z, labels), 3), "| stability ARI", round(stab, 3), "|", [p["persona"] for p in prof])


if __name__ == "__main__":
    main()
