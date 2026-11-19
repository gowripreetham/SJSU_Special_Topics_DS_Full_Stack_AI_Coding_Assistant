"""Project 06 — Anomaly detection on the Kaggle "Credit Card Fraud" dataset (284,807 transactions, 0.17% fraud).

Six unsupervised detectors are trained WITHOUT labels and compared against a supervised reference model.
Labels are used only (a) on the validation split for model selection / thresholding and (b) on the test split for the final score.
Data: the ADBench copy of the dataset (V1..V28 PCA features + Amount, min-max scaled); downloaded on first run.
Run:  python pipeline.py
"""
from __future__ import annotations
import sys
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hill_climb, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42
URL = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical/13_fraud.npz"
FEATS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load() -> tuple[np.ndarray, np.ndarray]:
    f = ROOT / "data" / "cache" / "13_fraud.npz"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        print("downloading dataset (46 MB)…", flush=True)
        urllib.request.urlretrieve(URL, f)
    d = np.load(f, allow_pickle=True)
    return d["X"].astype(np.float32), d["y"].astype(int)


class HBOS:
    """Histogram-Based Outlier Score: sum over features of -log(density) from per-feature histograms."""

    def __init__(self, bins: int = 30):
        self.bins = bins

    def fit(self, X):
        self.edges, self.logd = [], []
        for j in range(X.shape[1]):
            h, e = np.histogram(X[:, j], bins=self.bins, density=True)
            self.edges.append(e)
            self.logd.append(-np.log(h + 1e-4))
        return self

    def score(self, X):
        s = np.zeros(len(X))
        for j, (e, ld) in enumerate(zip(self.edges, self.logd)):
            s += ld[np.clip(np.searchsorted(e, X[:, j]) - 1, 0, len(ld) - 1)]
        return s


class Detectors:
    """All detectors expose score(X): higher = more anomalous. Fit on unlabeled training data only."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {"if_trees": 200, "if_samples": 256, "pca_k": 10, "ae_hidden": 8, "lof_k": 20}

    def fit(self, X: np.ndarray):
        c, rs = self.cfg, np.random.RandomState(SEED)
        self.scaler = StandardScaler().fit(X)  # audit-ok: X is the training split — main() calls Detectors(...).fit(Xtr)
        Z = self.scaler.transform(X)
        sub = Z[rs.choice(len(Z), min(20000, len(Z)), replace=False)]
        self.iforest = IsolationForest(n_estimators=c["if_trees"], max_samples=c["if_samples"], random_state=SEED, n_jobs=2).fit(Z)
        self.pca = PCA(c["pca_k"], random_state=SEED).fit(Z)  # audit-ok: Z = scaled training split
        self.mcd = MinCovDet(random_state=SEED, support_fraction=0.9).fit(sub)
        self.hbos = HBOS().fit(Z)
        self.lof = LocalOutlierFactor(n_neighbors=c["lof_k"], novelty=True, n_jobs=2).fit(sub)
        ae_x = Z[rs.choice(len(Z), min(60000, len(Z)), replace=False)]
        self.ae = MLPRegressor(hidden_layer_sizes=(16, c["ae_hidden"], 16), activation="tanh", max_iter=25, early_stopping=True, random_state=SEED).fit(ae_x, ae_x)
        return self

    def scores(self, X: np.ndarray, only: list[str] | None = None) -> dict[str, np.ndarray]:
        Z = self.scaler.transform(X)
        fns = {"Isolation Forest": lambda: -self.iforest.score_samples(Z),
               "PCA reconstruction": lambda: ((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2).sum(1),
               "Robust Mahalanobis (MCD)": lambda: np.log1p(self.mcd.mahalanobis(Z)),  # log: monotone, tames the heavy tail
               "HBOS": lambda: self.hbos.score(Z),
               "Local Outlier Factor": lambda: -self.lof.score_samples(Z),
               "Autoencoder": lambda: ((Z - self.ae.predict(Z)) ** 2).sum(1)}
        return {k: f() for k, f in fns.items() if only is None or k in only}


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    top = np.argsort(-s)
    k = int(y.sum())
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)), "recall_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
            "precision_at_k": float(y[top[:k]].mean()), "precision_at_100": float(y[top[:100]].mean())}


def rank01(s: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map scores to [0,1] by their percentile within a reference (training) score sample."""
    return np.searchsorted(np.sort(ref), s) / len(ref)


def main() -> None:
    t0 = time.time()
    X, y = load()
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=SEED)
    print(f"train {len(Xtr):,} / val {len(Xva):,} / test {len(Xte):,}; fraud rate {y.mean():.4%}")

    print("== AutoResearch on detector hyper-parameters (validation PR-AUC of the rank-ensemble) ==")
    rs = np.random.RandomState(SEED)
    small = Xtr[rs.choice(len(Xtr), 40000, replace=False)]

    def evaluate(c: dict) -> float:
        d = Detectors({k: c[k] for k in ["if_trees", "if_samples", "pca_k", "ae_hidden", "lof_k"]})
        d.scaler = StandardScaler().fit(small)
        Z, Zv = d.scaler.transform(small), d.scaler.transform(Xva)
        parts = []
        if "if" in c["members"]:
            m = IsolationForest(n_estimators=c["if_trees"], max_samples=c["if_samples"], random_state=SEED, n_jobs=2).fit(Z)
            parts.append(rankdata(-m.score_samples(Zv)))
        if "pca" in c["members"]:
            p = PCA(c["pca_k"], random_state=SEED).fit(Z)  # audit-ok: Z = scaled 40k training subsample; validation rows are only transformed
            parts.append(rankdata(((Zv - p.inverse_transform(p.transform(Zv))) ** 2).sum(1)))
        if "hbos" in c["members"]:
            parts.append(rankdata(HBOS().fit(Z).score(Zv)))
        return float(average_precision_score(yva, np.mean(parts, axis=0)))

    ar = hill_climb(evaluate, {"if_trees": [100, 200, 400], "if_samples": [128, 256, 512, 1024, 4096], "pca_k": [3, 5, 8, 10, 15, 20], "ae_hidden": [4, 8, 12], "lof_k": [10, 20, 40],
                               "members": ["if", "if+pca", "if+hbos", "pca+hbos", "if+pca+hbos", "pca", "hbos"]},
                    {"if_trees": 100, "if_samples": 256, "pca_k": 10, "ae_hidden": 8, "lof_k": 20, "members": "if"}, n_iter=18, seed=SEED)
    best = ar["best"]

    print("== fitting all detectors on the full training split (labels unused) ==")
    det = Detectors({k: best[k] for k in ["if_trees", "if_samples", "pca_k", "ae_hidden", "lof_k"]}).fit(Xtr)
    sva, ste = det.scores(Xva), det.scores(Xte)
    str_ref = det.scores(Xtr[rs.choice(len(Xtr), 30000, replace=False)])
    val_rank = sorted(sva, key=lambda k: -average_precision_score(yva, sva[k]))
    members = val_rank[:3]  # ensemble = top-3 detectors by VALIDATION PR-AUC
    ens = lambda S: np.mean([rank01(S[k], str_ref[k]) for k in members], axis=0)  # noqa: E731
    sva["Rank ensemble (top-3)"], ste["Rank ensemble (top-3)"] = ens(sva), ens(ste)

    print("== supervised reference (uses labels) ==")
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=float((ytr == 0).sum() / ytr.sum()) ** 0.5,
                        random_state=SEED, n_jobs=2, tree_method="hist", eval_metric="aucpr").fit(Xtr, ytr)
    sva["XGBoost (supervised reference)"], ste["XGBoost (supervised reference)"] = xgb.predict_proba(Xva)[:, 1], xgb.predict_proba(Xte)[:, 1]

    sva["Random scores (baseline)"], ste["Random scores (baseline)"] = rs.rand(len(yva)), rs.rand(len(yte))  # PR-AUC of a coin flip = prevalence
    board = [{"model": k, "supervised": k.startswith("XGBoost"), "val": metrics(yva, sva[k]), "test": metrics(yte, ste[k])} for k in ste]
    board.sort(key=lambda r: -r["test"]["pr_auc"])
    for r in board:
        print(f"  {r['model']:<32} test PR-AUC {r['test']['pr_auc']:.4f}  ROC-AUC {r['test']['roc_auc']:.4f}")

    # operating point: alert budget of 0.5% of transactions, threshold chosen on validation, reported on test
    champion = max((r for r in board if not r["supervised"] and "baseline" not in r["model"]), key=lambda r: r["val"]["pr_auc"])["model"]
    thr = float(np.quantile(sva[champion], 0.995))
    flag = ste[champion] >= thr
    op = {"model": champion, "threshold": thr, "alert_rate": float(flag.mean()), "tp": int((flag & (yte == 1)).sum()), "fp": int((flag & (yte == 0)).sum()),
          "fn": int((~flag & (yte == 1)).sum()), "tn": int((~flag & (yte == 0)).sum())}
    op["precision"], op["recall"] = op["tp"] / max(1, op["tp"] + op["fp"]), op["tp"] / max(1, op["tp"] + op["fn"])

    curves = {}
    for k in [champion, "Isolation Forest", "XGBoost (supervised reference)"]:
        pr, rc, _ = precision_recall_curve(yte, ste[k])
        fpr, tpr, _ = roc_curve(yte, ste[k])
        ip, ir = np.linspace(0, len(pr) - 1, 120).astype(int), np.linspace(0, len(fpr) - 1, 120).astype(int)
        curves[k] = {"pr": [[float(rc[i]), float(pr[i])] for i in ip], "roc": [[float(fpr[i]), float(tpr[i])] for i in ir]}

    # demo transactions for the live scorer: 40 frauds + 160 normals from the TEST split
    idx = np.r_[rs.choice(np.where(yte == 1)[0], 40, replace=False), rs.choice(np.where(yte == 0)[0], 160, replace=False)]
    rs.shuffle(idx)
    np.savez_compressed(ROOT / "data" / "demo_transactions.npz", X=Xte[idx], y=yte[idx])
    Zs = det.scaler.transform(Xtr[rs.choice(len(Xtr), 20000, replace=False)])
    sc = ste[champion]
    joblib.dump({"det": det, "members": members, "ref": {k: np.sort(v)[::30] for k, v in str_ref.items()}, "threshold": thr, "champion": champion, "xgb": xgb}, ART / "model.joblib", compress=3)
    Zte = det.scaler.transform(Xte)
    save_json(ART / "metrics.json", {
        "n": len(X), "n_fraud": int(y.sum()), "rate": float(y.mean()), "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte), "test_fraud": int(yte.sum())},
        "leaderboard": board, "autoresearch": ar, "members": members, "champion": champion, "operating_point": op, "curves": curves, "features": FEATS,
        "score_hist": {"normal": np.histogram(sc[yte == 0], bins=40, range=(float(sc.min()), float(sc.max())))[0].tolist(), "fraud": np.histogram(sc[yte == 1], bins=40, range=(float(sc.min()), float(sc.max())))[0].tolist(),
                       "edges": np.linspace(float(sc.min()), float(sc.max()), 41).tolist()},
        "feature_shift": sorted([[f, float(abs(Zte[yte == 1][:, j].mean() - Zte[yte == 0][:, j].mean()))] for j, f in enumerate(FEATS)], key=lambda kv: -kv[1]),
        "pca_scatter": {"normal": det.pca.transform(Zte[yte == 0][:1500])[:, :2].round(2).tolist(), "fraud": det.pca.transform(Zte[yte == 1])[:, :2].round(2).tolist()},
        "amount": {"normal": np.histogram(Xte[yte == 0][:, -1], bins=30, range=(0, 0.05))[0].tolist(), "fraud": np.histogram(Xte[yte == 1][:, -1], bins=30, range=(0, 0.05))[0].tolist()},
        "secs": round(time.time() - t0, 1)})
    print("operating point:", op, f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
