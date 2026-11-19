"""Project 07 — AutoML with AutoGluon on three task types, with honest baselines.

  binary classification   Adult census income (>50K?)       48,842 rows
  multiclass              Handwritten digits (0-9)           1,797 rows
  regression              NYC taxi fare (seaborn `taxis`)    6,433 rows

For each task: AutoGluon `TabularPredictor.fit` under a time budget vs a tuned-by-default scikit-learn baseline, both scored
on the same untouched test split. Then two studies on Adult: (1) multi-layer stack ensembling (bagging + stacking), and
(2) a time-budget sweep — the AutoML analogue of AutoResearch hill-climbing: how much accuracy does each extra second buy?
Run:  python pipeline.py        (≈ 8 minutes on 2 CPU cores)
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
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
MODELS = ROOT / "data" / "cache" / "ag_models"
SEED = 42
FAST = bool(os.environ.get("AG_FAST"))
BUDGET = 20 if FAST else 60


def adult() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "adult.csv.gz")


def taxi() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "taxis.csv", parse_dates=["pickup"]).dropna()
    df["hour"], df["dow"] = df.pickup.dt.hour, df.pickup.dt.dayofweek
    # only what is known when the ride is booked: no tip, tolls, total, drop-off time (those would leak the fare)
    return df[["passengers", "distance", "hour", "dow", "color", "payment", "pickup_borough", "dropoff_borough", "pickup_zone", "dropoff_zone", "fare"]]


def digits() -> pd.DataFrame:
    d = load_digits(as_frame=True)
    return d.frame.rename(columns={"target": "digit"})


def baseline(kind: str, tr: pd.DataFrame, te: pd.DataFrame, label: str) -> tuple[float, float]:
    """A strong, zero-tuning scikit-learn baseline: histogram gradient boosting with native categorical support."""
    Xtr, Xte = tr.drop(columns=[label]).copy(), te.drop(columns=[label]).copy()
    cat = [c for c in Xtr.columns if Xtr[c].dtype == object]
    for c in cat:
        cats = pd.Categorical(Xtr[c]).categories
        Xtr[c], Xte[c] = pd.Categorical(Xtr[c], categories=cats), pd.Categorical(Xte[c], categories=cats)
    t0 = time.time()
    if kind == "regression":
        m = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED).fit(Xtr, tr[label])
        return -float(root_mean_squared_error(te[label], m.predict(Xte))), time.time() - t0
    m = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=SEED).fit(Xtr, tr[label])
    if kind == "binary":
        return float(roc_auc_score(te[label], m.predict_proba(Xte)[:, 1])), time.time() - t0
    return float(accuracy_score(te[label], m.predict(Xte))), time.time() - t0


def fit_ag(tr: pd.DataFrame, label: str, metric: str, name: str, time_limit: int, **kw):
    from autogluon.tabular import TabularPredictor
    path = MODELS / name
    shutil.rmtree(path, ignore_errors=True)
    t0 = time.time()
    p = TabularPredictor(label=label, eval_metric=metric, path=str(path), verbosity=0).fit(tr, time_limit=time_limit, **kw)
    return p, time.time() - t0


def board(p, te: pd.DataFrame) -> list[dict]:
    lb = p.leaderboard(te, silent=True)
    return [{"model": r.model, "score_test": float(r.score_test), "score_val": float(r.score_val), "fit_time": float(r.fit_time), "pred_time": float(r.pred_time_test),
             "stack_level": int(r.stack_level)} for r in lb.itertuples()]


def main() -> None:
    t0 = time.time()
    out: dict = {"tasks": {}, "budget": BUDGET}
    specs = [("binary", "Adult census income", adult(), "income", "roc_auc", "ROC-AUC"), ("multiclass", "Handwritten digits", digits(), "digit", "accuracy", "accuracy"),
             ("regression", "NYC taxi fare", taxi(), "fare", "root_mean_squared_error", "−RMSE ($)")]
    splits = {}
    for kind, title, df, label, metric, mname in specs:
        tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[label] if kind != "regression" else None)
        if kind == "binary":
            tr, te = tr.assign(income=(tr.income == ">50K").astype(int)), te.assign(income=(te.income == ">50K").astype(int))
        splits[kind] = (tr, te)
        print(f"== {title}: train {len(tr):,} test {len(te):,} ==", flush=True)
        b, bt = baseline(kind, tr, te, label)
        p, ft = fit_ag(tr, label, metric, kind, BUDGET, presets="medium_quality")
        lb = board(p, te)
        best = p.model_best
        imp = p.feature_importance(te.sample(min(1500, len(te)), random_state=SEED), silent=True, num_shuffle_sets=3, subsample_size=1500).head(12)
        info = {"title": title, "kind": kind, "label": label, "metric": mname, "n_train": len(tr), "n_test": len(te), "n_features": df.shape[1] - 1, "baseline": b, "baseline_secs": bt,
                "ag_score": next(r["score_test"] for r in lb if r["model"] == best), "ag_secs": ft, "best_model": best, "leaderboard": lb,
                "importance": [[k, float(v)] for k, v in imp["importance"].items()], "problem_type": p.problem_type,
                "feature_types": {k: len(v) for k, v in p.feature_metadata.to_dict(inverse=True).items()} if hasattr(p.feature_metadata, "to_dict") else {}}
        if kind == "regression":
            pr = p.predict(te)
            info["r2"] = float(r2_score(te[label], pr))
            s = te.assign(pred=pr).sample(500, random_state=SEED)
            info["scatter"] = [{"x": float(a), "y": float(c)} for a, c in zip(s[label], s.pred)]
        if kind == "multiclass":
            pr = p.predict(te)
            info["confusion"] = pd.crosstab(te[label], pr).reindex(index=range(10), columns=range(10), fill_value=0).values.tolist()
        out["tasks"][kind] = info
        print(f"   baseline {b:.4f} ({bt:.1f}s)  AutoGluon {info['ag_score']:.4f} ({ft:.0f}s, best={best})", flush=True)

    tr, te = splits["binary"]
    print("== study 1: multi-layer stack ensembling on Adult ==", flush=True)
    p, ft = fit_ag(tr.sample(8000 if FAST else 16000, random_state=SEED), "income", "roc_auc", "binary_stack", 40 if FAST else 150, num_bag_folds=4, num_stack_levels=1,
                   hyperparameters={"GBM": {}, "XGB": {}, "RF": {}, "XT": {}, "LR": {}}, dynamic_stacking=False)
    lb = board(p, te)
    w = {}
    try:
        w = {k: float(v) for k, v in p.info()["model_info"][p.model_best]["children_info"]["S1F1"]["model_weights"].items()}
    except Exception:  # audit-ok: ensemble weights are optional display metadata; their layout differs across AutoGluon versions
        pass
    out["stack"] = {"leaderboard": lb, "best": p.model_best, "secs": ft, "weights": w, "edges": [[a, b] for a, b in p._trainer.model_graph.edges()] if hasattr(p, "_trainer") else []}

    print("== study 2: time-budget sweep on Adult ==", flush=True)
    sweep = []
    for tl in ([5, 10] if FAST else [5, 10, 20, 40]):
        p2, ft2 = fit_ag(tr, "income", "roc_auc", "binary_sweep", tl, presets="medium_quality")
        lb2 = board(p2, te)
        sweep.append({"budget": tl, "secs": ft2, "score": next(r["score_test"] for r in lb2 if r["model"] == p2.model_best), "n_models": len(lb2), "best": p2.model_best})
        print(f"   budget {tl:>3}s → {sweep[-1]['score']:.4f} with {len(lb2)} models", flush=True)
    sweep.append({"budget": BUDGET, "secs": out["tasks"]["binary"]["ag_secs"], "score": out["tasks"]["binary"]["ag_score"], "n_models": len(out["tasks"]["binary"]["leaderboard"]), "best": out["tasks"]["binary"]["best_model"]})
    out["sweep"] = sweep
    shutil.rmtree(MODELS / "binary_sweep", ignore_errors=True)
    shutil.rmtree(MODELS / "binary_stack", ignore_errors=True)

    import autogluon.tabular
    out["versions"] = {"autogluon": autogluon.tabular.__version__}
    out["adult_options"] = {c: sorted(adult()[c].dropna().unique().tolist()) for c in ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex"]}
    out["secs"] = round(time.time() - t0, 1)
    save_json(ART / "metrics.json", out)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
