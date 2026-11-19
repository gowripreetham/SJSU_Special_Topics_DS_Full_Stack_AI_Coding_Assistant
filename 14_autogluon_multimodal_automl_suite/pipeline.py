"""Project 14 — AutoGluon capability suite: five studies that go beyond `fit()` on a clean table.

  A  Text + tabular fusion      Titanic with free-text Name/Ticket/Cabin: does AutoGluon's automatic text featurisation help?
  B  Probabilistic forecasting   AutoGluon quantile regression (P10/P50/P90) on lag features of Mauna Loa CO2
  C  AutoEDA & covariate drift   KS / total-variation tests + adversarial validation, on a random and a deliberately shifted split
  D  Knowledge distillation      compress the stacked ensemble into one small LightGBM: fidelity, accuracy, speed-up
  E  Calibration                 reliability of the ensemble's probabilities

Scope note (honest improvisation): the reference project used Chronos and vision-language models, which need multi-GB
downloads from Hugging Face and a GPU. This suite keeps to capabilities that run offline on a 2-core CPU, and measures them.
Run:  python pipeline.py      (≈ 6 minutes)
"""
from __future__ import annotations
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
MODELS = ROOT / "data" / "cache" / "ag_models"
SEED = 42
FAST = bool(os.environ.get("AG_FAST"))
T = (lambda s: max(8, s // 3)) if FAST else (lambda s: s)
QL = [0.1, 0.5, 0.9]


def fit(tr: pd.DataFrame, label: str, name: str, time_limit: int, metric: str | None = None, keep: bool = False, **kw):
    from autogluon.tabular import TabularPredictor
    path = MODELS / name
    shutil.rmtree(path, ignore_errors=True)
    p = TabularPredictor(label=label, eval_metric=metric, path=str(path), verbosity=0, **{k: kw.pop(k) for k in ("problem_type", "quantile_levels") if k in kw}).fit(tr, time_limit=T(time_limit), **kw)
    return p


def titanic() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "titanic_full.csv").drop(columns=["PassengerId"])


def adult() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "adult.csv.gz").drop(columns=["fnlwgt", "education"])
    return df.assign(income=(df.income == ">50K").astype(int))


def co2_frame() -> pd.DataFrame:
    """Supervised frame for one-step-ahead forecasting: every feature is a lag (≥ 1 month old) or a calendar term."""
    s = pd.read_csv(ROOT / "data" / "mauna_loa_co2.csv")
    y = s.co2.to_numpy()
    d = pd.DataFrame({"month": s.Month, "level": y, "diff": np.r_[np.nan, np.diff(y)]})
    for l in [1, 2, 3, 6, 11, 12, 13, 24]:
        d[f"lag_{l}"] = d["diff"].shift(l)
    d["roll_12"] = d["diff"].shift(1).rolling(12).mean()
    m = pd.to_datetime(d.month).dt.month
    d["m_sin"], d["m_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    d["prev_level"] = d.level.shift(1)
    return d.dropna().reset_index(drop=True)


def pinball(y, q, tau):
    e = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def main() -> None:
    t0 = time.time()
    out: dict = {}

    # ================================================================= A. text + tabular
    print("== A. text + tabular fusion (Titanic) ==", flush=True)
    df = titanic()
    variants = {"Numeric + categorical only": lambda d: d.drop(columns=["Name", "Ticket", "Cabin"]),
                "+ raw text columns (AutoGluon featurises Name / Ticket / Cabin)": lambda d: d,
                "+ hand-engineered Title only": lambda d: d.drop(columns=["Name", "Ticket", "Cabin"]).assign(Title=d.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip())}
    rows, ftypes, imp = [], {}, []
    for seed in ([0] if FAST else [0, 1, 2]):
        tr, te = train_test_split(df, test_size=0.3, stratify=df.Survived, random_state=seed)
        for name, fn in variants.items():
            p = fit(fn(tr), "Survived", "titanic", 20, "roc_auc")
            pr = p.predict_proba(fn(te))[1]
            rows.append({"variant": name, "seed": seed, "auc": float(roc_auc_score(te.Survived, pr)), "acc": float(accuracy_score(te.Survived, pr >= 0.5)), "n_models": len(p.model_names())})
            if seed == 0 and "raw text" in name:
                fm = p.feature_metadata
                sp = fm.get_type_map_special()
                counts: dict = {}
                for _, tags in sp.items():
                    for tg in tags:
                        counts[tg] = counts.get(tg, 0) + 1
                ftypes = {"raw_in": {k: str(v) for k, v in p.feature_metadata_in.type_map_raw.items()}, "special_in": {k: list(v) for k, v in p.feature_metadata_in.get_type_map_special().items() if v},
                          "generated_by_tag": counts, "n_generated": len(fm.get_features()), "examples": [f for f in fm.get_features() if f.startswith(("Name", "__nlp__"))][:14]}
                imp = [[k, float(v)] for k, v in p.feature_importance(fn(te), silent=True, num_shuffle_sets=3)["importance"].head(10).items()]
            print(f"   seed {seed}  {name[:38]:<38} AUC {rows[-1]['auc']:.4f}", flush=True)
    agg = pd.DataFrame(rows).groupby("variant", sort=False).agg(auc=("auc", "mean"), lo=("auc", "min"), hi=("auc", "max"), acc=("acc", "mean")).reset_index()
    out["text"] = {"runs": rows, "summary": agg.to_dict("records"), "feature_types": ftypes, "importance": imp, "n": len(df), "titles": df.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip().value_counts().head(8).to_dict(),
                   "survival_by_title": df.assign(t=df.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip()).groupby("t").Survived.agg(["mean", "size"]).query("size >= 6").sort_values("mean").round(3).reset_index().values.tolist()}

    # ================================================================= B. probabilistic forecasting
    print("== B. quantile forecasting (CO2) ==", flush=True)
    d = co2_frame()
    cut = int(np.where(d.month >= "1997-01")[0][0])
    feats = [c for c in d.columns if c.startswith(("lag_", "roll_", "m_"))]
    tr, te = d.iloc[:cut], d.iloc[cut:]
    p = fit(tr[feats + ["diff"]], "diff", "co2", 45, problem_type="quantile", quantile_levels=QL)
    Q = np.sort(p.predict(te[feats]).to_numpy(), axis=1)
    # seasonal-naive quantile baseline: last year's change ± empirical quantiles of its historical error (train only)
    err = (tr["diff"] - tr["lag_12"]).to_numpy()
    Qb = te["lag_12"].to_numpy()[:, None] + np.quantile(err, QL)[None, :]
    y = te["diff"].to_numpy()
    sc = lambda Qx: {"pinball": float(np.mean([pinball(y, Qx[:, i], q) for i, q in enumerate(QL)])), "coverage80": float(np.mean((y >= Qx[:, 0]) & (y <= Qx[:, 2]))), "width": float(np.mean(Qx[:, 2] - Qx[:, 0])),  # noqa: E731
                         "mae_p50": float(np.mean(np.abs(y - Qx[:, 1])))}
    lbq = p.leaderboard(te[feats + ["diff"]], silent=True)
    out["forecast"] = {"autogluon": sc(Q), "baseline": sc(Qb), "months": te.month.tolist(), "actual": te.level.round(3).tolist(), "p10": (te.prev_level + Q[:, 0]).round(3).tolist(), "p50": (te.prev_level + Q[:, 1]).round(3).tolist(),
                       "p90": (te.prev_level + Q[:, 2]).round(3).tolist(), "diff_actual": y.round(3).tolist(), "diff_q": Q.round(3).tolist(), "n_train": len(tr), "n_test": len(te), "features": feats,
                       "leaderboard": [[r.model, float(r.score_test), float(r.score_val), float(r.fit_time)] for r in lbq.itertuples()], "best": p.model_best}
    print("   AutoGluon", {k: round(v, 4) for k, v in out["forecast"]["autogluon"].items()}, "\n   baseline ", {k: round(v, 4) for k, v in out["forecast"]["baseline"].items()}, flush=True)

    # ================================================================= C. AutoEDA + drift
    print("== C. drift detection + adversarial validation (Adult) ==", flush=True)
    A = adult().sample(6000 if FAST else 20000, random_state=SEED).reset_index(drop=True)
    num = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    cat = ["workclass", "marital_status", "occupation", "relationship", "race", "sex"]
    rs = np.random.RandomState(SEED)
    scen = {"Random split (no drift expected)": rs.rand(len(A)) < 0.7, "Shifted split (train: age < 40, test: age ≥ 40)": (A.age < 40).to_numpy()}
    out["drift"] = {}
    for name, mask in scen.items():
        tr, te = A[mask], A[~mask]
        tests = []
        for c in num:
            ks = stats.ks_2samp(tr[c], te[c])
            tests.append({"feature": c, "type": "numeric", "stat": float(ks.statistic), "p": float(ks.pvalue), "test": "Kolmogorov–Smirnov D"})
        for c in cat:
            a, b = tr[c].fillna("NA").value_counts(normalize=True), te[c].fillna("NA").value_counts(normalize=True)
            tv = float(0.5 * a.subtract(b, fill_value=0).abs().sum())
            ct = pd.crosstab(pd.concat([tr[c], te[c]]).fillna("NA"), np.r_[np.zeros(len(tr)), np.ones(len(te))])
            tests.append({"feature": c, "type": "categorical", "stat": tv, "p": float(stats.chi2_contingency(ct)[1]), "test": "total-variation distance"})
        # adversarial validation: can a model tell train rows from test rows? (label and target removed)
        adv = pd.concat([tr.drop(columns="income").assign(is_test=0), te.drop(columns="income").assign(is_test=1)]).sample(frac=1, random_state=SEED)
        a_tr, a_te = train_test_split(adv, test_size=0.3, stratify=adv.is_test, random_state=SEED)
        pa = fit(a_tr, "is_test", "adv", 15, "roc_auc", hyperparameters={"GBM": {}})
        adv_auc = float(roc_auc_score(a_te.is_test, pa.predict_proba(a_te)[1]))
        adv_imp = [[k, float(v)] for k, v in pa.feature_importance(a_te.sample(min(2000, len(a_te)), random_state=SEED), silent=True, num_shuffle_sets=2)["importance"].head(6).items()]
        # consequence: how does a model trained on `tr` do on `te`, versus on a held-out slice of its own distribution?
        t_in, v_in = train_test_split(tr, test_size=0.25, stratify=tr.income, random_state=SEED)
        pm = fit(t_in, "income", "driftmodel", 15, "roc_auc", hyperparameters={"GBM": {}})
        out["drift"][name] = {"tests": sorted(tests, key=lambda r: -r["stat"]), "adv_auc": adv_auc, "adv_importance": adv_imp, "n_train": len(tr), "n_test": len(te),
                              "auc_in_dist": float(roc_auc_score(v_in.income, pm.predict_proba(v_in)[1])), "auc_on_test": float(roc_auc_score(te.income, pm.predict_proba(te)[1])),
                              "acc_in_dist": float(accuracy_score(v_in.income, pm.predict(v_in))), "acc_on_test": float(accuracy_score(te.income, pm.predict(te))),
                              "pos_rate_train": float(tr.income.mean()), "pos_rate_test": float(te.income.mean())}
        print(f"   {name[:30]:<30} adversarial AUC {adv_auc:.3f}  model AUC in-dist {out['drift'][name]['auc_in_dist']:.3f} → on test {out['drift'][name]['auc_on_test']:.3f}", flush=True)

    # ================================================================= D. distillation + E. calibration
    print("== D. distillation / E. calibration (Adult) ==", flush=True)
    tr, te = train_test_split(A, test_size=0.3, stratify=A.income, random_state=SEED)
    tr, dis = train_test_split(tr, test_size=0.5, stratify=tr.income, random_state=SEED)  # the student learns on rows the teacher never trained on
    teacher = fit(tr, "income", "teacher", 60, "roc_auc", num_bag_folds=4, num_stack_levels=1, hyperparameters={"GBM": {}, "XGB": {}, "RF": {}, "XT": {}}, dynamic_stacking=False)
    Xd, Xt = dis.drop(columns="income"), te.drop(columns="income")
    soft = teacher.predict_proba(Xd)[1].to_numpy()
    t1 = time.perf_counter(); pt = teacher.predict_proba(Xt)[1].to_numpy(); teach_ms = (time.perf_counter() - t1) * 1000

    def enc(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in [c for c in X.columns if X[c].dtype == object]:
            X[c] = pd.Categorical(X[c].fillna("NA"), categories=sorted(A[c].fillna("NA").unique()))
        return X

    kw = dict(n_estimators=120, num_leaves=15, learning_rate=0.08, random_state=SEED, n_jobs=1, verbose=-1)
    logit = np.log(np.clip(soft, 1e-4, 1 - 1e-4) / (1 - np.clip(soft, 1e-4, 1 - 1e-4)))
    s_soft = LGBMRegressor(**kw).fit(enc(Xd), logit)           # distilled: regress the teacher's logit
    s_hard = LGBMClassifier(**kw).fit(enc(Xd), dis.income)     # same student, trained on hard labels
    t1 = time.perf_counter(); ps = 1 / (1 + np.exp(-s_soft.predict(enc(Xt)))); stud_ms = (time.perf_counter() - t1) * 1000
    ph = s_hard.predict_proba(enc(Xt))[:, 1]
    lbt = teacher.leaderboard(te, silent=True)
    out["distill"] = {"teacher": {"auc": float(roc_auc_score(te.income, pt)), "acc": float(accuracy_score(te.income, pt >= 0.5)), "ms": teach_ms, "n_models": len(lbt), "best": teacher.model_best},
                      "student_soft": {"auc": float(roc_auc_score(te.income, ps)), "acc": float(accuracy_score(te.income, ps >= 0.5)), "ms": stud_ms, "fidelity": float(np.mean((ps >= 0.5) == (pt >= 0.5))), "corr": float(np.corrcoef(ps, pt)[0, 1])},
                      "student_hard": {"auc": float(roc_auc_score(te.income, ph)), "acc": float(accuracy_score(te.income, ph >= 0.5)), "fidelity": float(np.mean((ph >= 0.5) == (pt >= 0.5))), "corr": float(np.corrcoef(ph, pt)[0, 1])},
                      "speedup": teach_ms / stud_ms, "n_test": len(te), "n_distill": len(dis), "n_teacher_train": len(tr),
                      "scatter": [{"x": float(a), "y": float(b)} for a, b in zip(pt[:500], ps[:500])],
                      "teacher_lb": [[r.model, float(r.score_test), int(r.stack_level), float(r.pred_time_test)] for r in lbt.itertuples()]}
    ft, mp = calibration_curve(te.income, pt, n_bins=10, strategy="quantile")
    fs, ms_ = calibration_curve(te.income, ps, n_bins=10, strategy="quantile")
    out["calibration"] = {"teacher": [[float(a), float(b)] for a, b in zip(mp, ft)], "student": [[float(a), float(b)] for a, b in zip(ms_, fs)], "brier_teacher": float(np.mean((pt - te.income) ** 2)),
                          "brier_student": float(np.mean((ps - te.income) ** 2)), "ece_teacher": float(np.mean(np.abs(ft - mp))), "ece_student": float(np.mean(np.abs(fs - ms_))), "base_brier": float(te.income.mean() * (1 - te.income.mean()))}
    print("   teacher", {k: round(v, 4) for k, v in out["distill"]["teacher"].items() if isinstance(v, float)}, "\n   student", {k: round(v, 4) for k, v in out["distill"]["student_soft"].items()}, f"\n   speed-up {out['distill']['speedup']:.0f}×", flush=True)

    shutil.rmtree(MODELS, ignore_errors=True)
    import autogluon.tabular
    out["versions"] = {"autogluon": autogluon.tabular.__version__}
    out["secs"] = round(time.time() - t0, 1)
    save_json(ART / "metrics.json", out)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
