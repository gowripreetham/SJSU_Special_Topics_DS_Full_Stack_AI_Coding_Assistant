"""Project 10 — CRISP-DM Master's curriculum on the Adult census dataset (48,842 people).

One dataset, seven analytical lenses, each a phase of the course:
  1 EDA & preprocessing · 2 clustering · 3 anomaly detection · 4 supervised learning (+ fairness slice)
  5 association rules · 6 sub-linear similarity search with cosine LSH · 7 synthesis
Run:  python pipeline.py
"""
from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42
NUM = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT = ["workclass", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "adult.csv.gz")
    df["y"] = (df.income == ">50K").astype(int)
    return df


def preprocessor() -> ColumnTransformer:
    """fnlwgt (a sampling weight) and `education` (duplicate of education_num) are deliberately excluded."""
    return ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
                              ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="Unknown"), OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)), CAT)])


class CosineLSH:
    """Random-hyperplane LSH for cosine similarity (Charikar, 2002).

    Each of `n_tables` hash tables maps a vector to an `n_bits`-bit signature: bit i = sign(rᵢ · x). Two vectors agree on a bit
    with probability 1 − θ/π, so near neighbours collide in at least one table with high probability, and a query inspects only
    the colliding candidates instead of all N rows.
    """

    def __init__(self, n_bits: int = 12, n_tables: int = 8, seed: int = SEED):
        self.n_bits, self.n_tables, self.seed = n_bits, n_tables, seed

    def fit(self, X: np.ndarray) -> "CosineLSH":
        rs = np.random.RandomState(self.seed)
        self.X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        self.planes = rs.randn(self.n_tables, self.n_bits, X.shape[1])
        self.pow2 = 1 << np.arange(self.n_bits)
        self.tables: list[dict[int, np.ndarray]] = []
        for t in range(self.n_tables):
            keys = ((self.X @ self.planes[t].T) > 0) @ self.pow2
            order = np.argsort(keys, kind="stable")
            uniq, start = np.unique(keys[order], return_index=True)
            self.tables.append({int(k): order[s:e] for k, s, e in zip(uniq, start, list(start[1:]) + [len(order)])})
        return self

    def candidates(self, q: np.ndarray) -> np.ndarray:
        q = q / (np.linalg.norm(q) + 1e-12)
        keys = ((self.planes @ q) > 0) @ self.pow2
        parts = [self.tables[t].get(int(k)) for t, k in enumerate(keys)]
        parts = [p for p in parts if p is not None]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=int)

    def query(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray, int]:
        c = self.candidates(q)
        if not len(c):
            return c, np.empty(0), 0
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = self.X[c] @ qn
        top = np.argsort(-sims)[:k]
        return c[top], sims[top], len(c)

    def exact(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        sims = self.X @ (q / (np.linalg.norm(q) + 1e-12))
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        return top, sims[top]


def main() -> None:
    t0 = time.time()
    df = load()
    out: dict = {"n": len(df), "pos_rate": float(df.y.mean())}
    rs = np.random.RandomState(SEED)

    # ---------------------------------------------------------------- Phase 1: EDA
    rate = lambda c, n=10: [[str(k), float(v), int(cnt)] for k, v, cnt in df.groupby(c).y.agg(["mean", "size"]).sort_values("mean", ascending=False).head(n).reset_index().itertuples(index=False)]  # noqa: E731
    out["eda"] = {
        "missing": [[c, int(df[c].isna().sum()), float(df[c].isna().mean())] for c in df.columns if df[c].isna().any()],
        "age_hist": np.histogram(df.age, bins=range(15, 95, 5))[0].tolist(), "age_hist_pos": np.histogram(df.age[df.y == 1], bins=range(15, 95, 5))[0].tolist(),
        "hours_hist": np.histogram(df.hours_per_week, bins=range(0, 105, 5))[0].tolist(),
        "by_education": [[str(k), float(v), int(n)] for k, v, n in df.groupby("education_num").y.agg(["mean", "size"]).reset_index().itertuples(index=False)],
        "edu_names": df.groupby("education_num").education.first().to_dict(),
        "by_occupation": rate("occupation", 14), "by_marital": rate("marital_status"), "by_sex": rate("sex"), "by_race": rate("race"),
        "capital_gain_zero": float((df.capital_gain == 0).mean()), "corr": df[NUM + ["y"]].corr().round(3).values.tolist(), "corr_cols": NUM + ["income>50K"],
        "desc": df[NUM].describe().round(1).reset_index().values.tolist(), "dup_rows": int(df.duplicated().sum()),
        "leak_note": "education vs education_num are 1:1 duplicates; fnlwgt is a survey weight, not a personal attribute"}

    # ---------------------------------------------------------------- split FIRST; everything learned below uses train only
    tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=SEED)
    pre = preprocessor().fit(tr)
    Ztr, Zte = pre.transform(tr), pre.transform(te)
    out["split"] = {"train": len(tr), "test": len(te), "n_features_encoded": int(Ztr.shape[1])}

    # ---------------------------------------------------------------- Phase 2: clustering
    sub = rs.choice(len(Ztr), 12000, replace=False)
    Zs, trs = Ztr[sub], tr.iloc[sub]
    sweep = []
    for k in range(2, 9):
        km = KMeans(k, n_init=4, random_state=SEED).fit(Zs)
        sweep.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(Zs[:4000], km.labels_[:4000]))})
    k_best = 5
    km = KMeans(k_best, n_init=8, random_state=SEED).fit(Zs)
    pca = PCA(2, random_state=SEED).fit(Zs)  # audit-ok: Zs is a 12k subsample of the *training* matrix, used only for a 2-D plot
    xy = pca.transform(Zs[:2500])
    prof = []
    for c in range(k_best):
        d = trs[km.labels_ == c]
        prof.append({"cluster": c, "n": len(d), "share": len(d) / len(trs), "age": float(d.age.mean()), "hours": float(d.hours_per_week.mean()), "edu": float(d.education_num.mean()),
                     "pct_high_income": float(100 * d.y.mean()), "pct_male": float(100 * (d.sex == "Male").mean()), "top_marital": d.marital_status.mode().iloc[0], "top_occupation": d.occupation.mode().iloc[0]})
    out["cluster"] = {"sweep": sweep, "k": k_best, "profiles": prof, "scatter": [[round(float(a), 2), round(float(b), 2), int(c)] for (a, b), c in zip(xy, km.labels_[:2500])], "pca_var": pca.explained_variance_ratio_.tolist()}

    # ---------------------------------------------------------------- Phase 3: anomaly detection
    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=2).fit(pre.named_transformers_["num"].transform(tr[NUM]))
    s_all = -iso.score_samples(pre.named_transformers_["num"].transform(tr[NUM]))
    top = np.argsort(-s_all)[:15]
    thr = np.quantile(s_all, 0.99)
    out["anomaly"] = {"hist": np.histogram(s_all, bins=40)[0].tolist(), "edges": np.histogram(s_all, bins=40)[1].round(3).tolist(), "threshold": float(thr),
                      "top": [[int(tr.age.iloc[i]), tr.education.iloc[i], int(tr.capital_gain.iloc[i]), int(tr.capital_loss.iloc[i]), int(tr.hours_per_week.iloc[i]), str(tr.occupation.iloc[i]), tr.income.iloc[i], round(float(s_all[i]), 3)] for i in top],
                      "rate_high_income_among_outliers": float(tr.y[s_all >= thr].mean()), "rate_overall": float(tr.y.mean()), "n_cap_99999": int((df.capital_gain == 99999).sum()), "n_hours_99": int((df.hours_per_week == 99).sum())}

    # ---------------------------------------------------------------- Phase 4: supervised
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cands = {"Majority class": None, "Logistic regression": LogisticRegression(max_iter=1000, C=1.0), "Random forest": RandomForestClassifier(200, min_samples_leaf=3, n_jobs=2, random_state=SEED),
             "Hist gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=SEED)}
    board, fitted = [], {}
    for name, m in cands.items():
        if m is None:
            board.append({"model": name, "cv_auc": 0.5, "test_auc": 0.5, "test_acc": float(1 - te.y.mean()), "test_f1": 0.0, "test_ap": float(te.y.mean())})
            continue
        pipe = Pipeline([("pre", preprocessor()), ("clf", m)])  # preprocessing is re-fit inside every CV fold
        cva = float(cross_val_score(pipe, tr, tr.y, cv=cv, scoring="roc_auc").mean())
        pipe.fit(tr, tr.y)
        p = pipe.predict_proba(te)[:, 1]
        board.append({"model": name, "cv_auc": cva, "test_auc": float(roc_auc_score(te.y, p)), "test_acc": float(accuracy_score(te.y, p >= 0.5)), "test_f1": float(f1_score(te.y, p >= 0.5)), "test_ap": float(average_precision_score(te.y, p))})
        fitted[name] = (pipe, p)
        print(f"  {name:<24} CV AUC {cva:.4f}  test AUC {board[-1]['test_auc']:.4f}", flush=True)
    best_name = max((b for b in board if b["model"] in fitted), key=lambda b: b["cv_auc"])["model"]  # chosen by CV, not by test
    best, p = fitted[best_name]
    fpr, tpr, _ = roc_curve(te.y, p)
    ii = np.linspace(0, len(fpr) - 1, 100).astype(int)
    pi = permutation_importance(best, te.sample(4000, random_state=SEED), te.y.loc[te.sample(4000, random_state=SEED).index], n_repeats=3, random_state=SEED, scoring="roc_auc")
    pred = p >= 0.5
    fair = []
    for col in ["sex", "race"]:
        for g, d in te.assign(pred=pred).groupby(col):
            if len(d) < 200:
                continue
            fair.append({"attr": col, "group": str(g), "n": len(d), "base_rate": float(d.y.mean()), "selection_rate": float(d.pred.mean()), "tpr": float(d.pred[d.y == 1].mean()), "fpr": float(d.pred[d.y == 0].mean()),
                         "precision": float(d.y[d.pred].mean()) if d.pred.any() else None})
    sr = {f["group"]: f["selection_rate"] for f in fair if f["attr"] == "sex"}
    out["supervised"] = {"leaderboard": sorted(board, key=lambda b: -b["cv_auc"]), "best": best_name, "roc": [[float(fpr[i]), float(tpr[i])] for i in ii],
                         "importance": sorted([[c, float(v)] for c, v in zip(te.columns, pi.importances_mean) if c in NUM + CAT], key=lambda kv: -kv[1]),
                         "fairness": fair, "disparate_impact_sex": float(min(sr.values()) / max(sr.values())), "confusion": [[int((pred & (te.y == 1)).sum()), int((~pred & (te.y == 1)).sum())], [int((pred & (te.y == 0)).sum()), int((~pred & (te.y == 0)).sum())]]}
    joblib.dump(best, ART / "model.joblib", compress=3)

    # ---------------------------------------------------------------- Phase 5: association rules
    b = pd.DataFrame({"age": pd.cut(tr.age, [0, 25, 35, 50, 65, 120], labels=["age≤25", "age 26-35", "age 36-50", "age 51-65", "age 65+"]).astype(str),
                      "hours": pd.cut(tr.hours_per_week, [0, 34, 40, 50, 120], labels=["part-time", "full-time 35-40h", "overtime 41-50h", "50h+"]).astype(str),
                      "edu": pd.cut(tr.education_num, [0, 8, 9, 12, 13, 16], labels=["no HS diploma", "HS grad", "some college", "bachelors", "postgraduate"]).astype(str),
                      "marital": tr.marital_status.astype(str), "occupation": tr.occupation.fillna("Unknown").astype(str), "sex": tr.sex.astype(str),
                      "gain": np.where(tr.capital_gain > 0, "has capital gain", "no capital gain"), "income": tr.income.astype(str)})
    oh = pd.get_dummies(b, prefix="", prefix_sep="").astype(bool)
    fi = fpgrowth(oh, min_support=0.02, use_colnames=True, max_len=4)
    rules = association_rules(fi, metric="confidence", min_threshold=0.5)
    rules = rules[rules.consequents.apply(lambda s: s == frozenset({">50K"}))].sort_values("lift", ascending=False)
    low = association_rules(fi, metric="confidence", min_threshold=0.97)
    low = low[low.consequents.apply(lambda s: s == frozenset({"<=50K"}))].sort_values("support", ascending=False)
    fmt = lambda r: [[", ".join(sorted(a)), round(float(s), 4), round(float(c), 3), round(float(l), 2)] for a, s, c, l in zip(r.antecedents, r.support, r.confidence, r.lift)]  # noqa: E731
    out["rules"] = {"n_itemsets": len(fi), "n_rules_high": len(rules), "high": fmt(rules.head(15)), "low": fmt(low.head(8)), "min_support": 0.02}

    # ---------------------------------------------------------------- Phase 6: LSH
    N = 20000
    idx = rs.choice(len(Ztr), N, replace=False)
    Xl = Ztr[idx].astype(np.float32)
    queries = Zte[rs.choice(len(Zte), 300, replace=False)].astype(np.float32)
    study = []
    for bits in [6, 8, 10, 12, 14, 16]:
        for tables in [4, 8, 16]:
            lsh = CosineLSH(bits, tables).fit(Xl)
            rec, cand, t_l, t_e = [], [], 0.0, 0.0
            for q in queries:
                t1 = time.perf_counter(); got, _, nc = lsh.query(q, 10); t_l += time.perf_counter() - t1
                t1 = time.perf_counter(); ex, _ = lsh.exact(q, 10); t_e += time.perf_counter() - t1
                rec.append(len(set(got) & set(ex)) / 10)
                cand.append(nc / N)
            study.append({"bits": bits, "tables": tables, "recall": float(np.mean(rec)), "candidates": float(np.mean(cand)), "speedup": float(t_e / t_l), "lsh_ms": 1000 * t_l / len(queries), "exact_ms": 1000 * t_e / len(queries)})
    ok = [s for s in study if s["recall"] >= 0.9]
    chosen = min(ok, key=lambda s: s["candidates"]) if ok else max(study, key=lambda s: s["recall"])
    out["lsh"] = {"n": N, "dim": int(Xl.shape[1]), "study": study, "chosen": chosen, "n_queries": len(queries)}
    np.savez_compressed(ART / "lsh_data.npz", X=Xl.astype(np.float16), idx=idx)
    tr.iloc[idx][["age", "education", "occupation", "marital_status", "sex", "hours_per_week", "income"]].to_csv(ART / "lsh_people.csv.gz", index=False, compression="gzip")
    joblib.dump(pre, ART / "preprocessor.joblib", compress=3)

    out["options"] = {c: sorted(df[c].dropna().unique().tolist()) for c in CAT}
    out["secs"] = round(time.time() - t0, 1)
    save_json(ART / "metrics.json", out)
    print("best:", best_name, "| LSH chosen:", chosen, f"| done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
