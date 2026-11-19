"""Project 07 — AutoML with AutoGluon: three task types, stack-ensembling anatomy, time-budget study and a live AutoML run."""
from __future__ import annotations
import shutil
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
_s: dict = {}
SPEC = {"binary": ("income", "roc_auc"), "multiclass": ("digit", "accuracy"), "regression": ("fare", "root_mean_squared_error")}


def M() -> dict:
    if "m" not in _s:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _s["m"]


class Live(BaseModel):
    task: str = "multiclass"
    budget: int = Field(15, ge=5, le=60)


def fam(name: str) -> str:
    for k, v in [("WeightedEnsemble", "ensemble"), ("LightGBM", "LightGBM"), ("XGBoost", "XGBoost"), ("RandomForest", "random forest"), ("ExtraTrees", "extra trees"), ("NeuralNet", "neural net"), ("KNeighbors", "k-NN"), ("LinearModel", "linear")]:
        if name.startswith(k):
            return v
    return "other"


def lb_table(lb: list[dict], metric: str) -> dict:
    return TABLE("Leaderboard (every model AutoGluon trained)", ["model", "family", "stack level", f"test {metric}", f"validation {metric}", "fit s", "predict s"],
                 [[r["model"], PILL(fam(r["model"]), "info" if r["stack_level"] > 1 else ""), r["stack_level"], r["score_test"], r["score_val"], round(r["fit_time"], 1), round(r["pred_time"], 2)] for r in lb], digits=4, maxh=330)


def task_blocks(k: str) -> list[dict]:
    t = M()["tasks"][k]
    lb = t["leaderboard"]
    win = t["ag_score"] - t["baseline"]
    blocks = [KPIS(K("Rows (train / test)", f"{t['n_train']:,} / {t['n_test']:,}"), K("Features", t["n_features"]), K(f"AutoGluon {t['metric']}", t["ag_score"], digits=4, hint=f"best: {t['best_model']}, {t['ag_secs']:.0f} s"),
                   K(f"sklearn baseline {t['metric']}", t["baseline"], digits=4, hint=f"HistGradientBoosting defaults, {t['baseline_secs']:.1f} s"), K("AutoML advantage", f"{win:+.4f}", delta="wins" if win > 0.0005 else "ties" if abs(win) <= 0.0005 else "loses", delta_good=win > -0.0005)),
              GRID(2, CHART("hbar", f"Test {t['metric']} by model", [S("score", [r["score_test"] for r in lb], colors=["3" if r["model"] == t["best_model"] else "1" for r in lb]), ], [r["model"] for r in lb], height=300,
                         xmin=min(r["score_test"] for r in lb) - 0.02 * abs(min(r["score_test"] for r in lb)), note="Green = the model AutoGluon selected using validation data only."),
                   CHART("hbar", "Permutation feature importance (AutoGluon)", [S("importance", [v for _, v in t["importance"]], color="7")], [n for n, _ in t["importance"]], height=300)), lb_table(lb, t["metric"])]
    if k == "regression":
        blocks.insert(2, CHART("scatter", f"Predicted vs actual fare (R² = {t['r2']:.3f})", [S("trip", t["scatter"], r=2.5), S("_ideal", [{"x": 0, "y": 0}, {"x": 80, "y": 80}], line=True, color="--axis")], x="actual $", y="predicted $", height=280))
    if k == "multiclass":
        blocks.insert(2, HEAT("Confusion matrix (rows = true digit)", list(range(10)), list(range(10)), t["confusion"], digits=0, labw=30))
    return blocks


def p_tasks() -> list[dict]:
    m = M()
    return [H("Three tasks, one API", f"`TabularPredictor(label=…).fit(train, time_limit={m['budget']})` — AutoGluon {m['versions']['autogluon']} infers the problem type, engineers features, trains a zoo of models and ensembles them. Each result is compared with a zero-tuning scikit-learn baseline on the same test split."),
            TABS(*[(f"{t['title']} ({k})", task_blocks(k)) for k, t in m["tasks"].items()]),
            CALLOUT("info", "The honest scoreboard", "Under a 60-second, 2-core budget AutoGluon **wins** on digits, **ties** on Adult and **loses** on taxi fares to a default gradient-boosting model. AutoML is a strong, low-effort baseline — not magic. Its advantage grows with budget (see *Time-budget study*) and with stacking (see *Stacking anatomy*).")]


def p_stack() -> list[dict]:
    s = M()["stack"]
    lb = s["leaderboard"]
    levels = sorted({r["stack_level"] for r in lb})
    return [H("Stacking anatomy", "With `num_bag_folds=4, num_stack_levels=1` AutoGluon builds a multi-layer stack: bagged base models (L1) → stacker models that also see L1's out-of-fold predictions (L2) → a weighted ensemble (L3)."),
            KPIS(K("Models trained", len(lb)), K("Layers", len(levels)), K("Final model", s["best"]), K("Fit time", f"{s['secs']:.0f} s"), K("Dependency edges", len(s["edges"]))),
            GRID(2, CHART("hbar", "Test ROC-AUC by model, coloured by stack level", [S("ROC-AUC", [r["score_test"] for r in lb], colors=[str(r["stack_level"]) for r in lb])], [r["model"] for r in lb], height=380, xmin=min(r["score_test"] for r in lb) - 0.005,
                          note="Blue = level 1 (bagged base models), orange = level 2 (stackers), green = level 3 (final weighted ensemble)."),
                 CHART("doughnut", f"Weights inside {s['best']} (Caruana greedy ensemble selection)", [S("weight", list(s["weights"].values()))], list(s["weights"]), height=380) if s["weights"] else MD("Ensemble weights were not exposed by this AutoGluon version.")),
            lb_table(lb, "ROC-AUC"),
            MD("""**Why stacking does not leak.** Each base model is trained with k-fold *bagging*; the predictions passed to the next layer are **out-of-fold** — every row is predicted by a model that never saw it. The final layer is Caruana's greedy forward selection: repeatedly add (with replacement) the model that most improves validation score, and use the selection counts as weights.

**What we measured.** On this 16k-row sample the L2 stackers and the L3 ensemble score within ±0.002 ROC-AUC of the best single bagged model — stacking gives robustness more than raw gain on a dataset this simple.""", title="How to read this")]


def p_budget() -> list[dict]:
    m = M()
    sw = sorted(m["sweep"], key=lambda r: r["budget"])
    base = m["tasks"]["binary"]["baseline"]
    return [H("Time-budget study", "The AutoML counterpart of hill-climbing: instead of proposing hyper-parameters by hand, buy more search with more seconds. Adult census, medium-quality preset."),
            CHART("line", "Test ROC-AUC vs time budget", [S("AutoGluon", [r["score"] for r in sw], points=True), S("sklearn HistGB baseline (0.5 s)", [base] * len(sw), dash=True, color="--muted")], [f"{r['budget']} s" for r in sw], x="time_limit", y="ROC-AUC", height=300),
            TABLE("Sweep", ["time_limit (s)", "actual fit s", "models trained", "best model", "test ROC-AUC"], [[r["budget"], round(r["secs"], 1), r["n_models"], r["best"], r["score"]] for r in sw], digits=4),
            CALLOUT("warn", "Not monotone", "More time does not always mean a better score: with tiny budgets AutoGluon truncates training of individual models differently from run to run, so 20 s scored below 10 s here. The curve only stabilises once the budget covers a full LightGBM fit.")]


def p_live() -> list[dict]:
    return [H("Live AutoML run", "Launch a real AutoGluon fit from the browser. The request blocks for roughly the chosen budget, then shows the leaderboard it produced."),
            FORM("api/automl", [F("task", "Task", "select", "multiclass", options=[["multiclass", "Digits — multiclass"], ["binary", "Adult income — binary (8k-row sample)"], ["regression", "Taxi fare — regression"]]), F("budget", "Time budget (seconds)", "range", 15, min=5, max=45, step=5)],
                 submit="Run AutoGluon now")]


CRISPDM = [
    ("Business understanding", "How good a model can we get with *no* manual tuning, and what does each extra second of compute buy? AutoML is judged against a cheap, competent baseline — not against nothing."),
    ("Data understanding", "Three public datasets covering the three tabular problem types: Adult (binary, mixed types, missing values), digits (10 classes, 64 pixel features), NYC taxi fares (regression, high-cardinality zones)."),
    ("Data preparation", "Deliberately minimal — AutoGluon's feature generator handles types, missing values and categoricals. The one manual step is removing post-trip columns (tip, tolls, total, drop-off time) from the taxi data so the fare cannot leak."),
    ("Modeling", "`TabularPredictor.fit` with a time limit; LightGBM, XGBoost, random forests, extra trees, k-NN, linear and neural models as available; weighted ensembling; an explicit bagging + stacking run."),
    ("Evaluation", "A stratified 20% test split per task, untouched by AutoGluon's internal validation. Leaderboards, permutation importance, a budget sweep, and the baseline comparison."),
    ("Deployment", "`POST api/automl` runs a fit on demand. In production one would call `predictor.clone_for_deployment()` and `persist()` to serve only the winning ensemble."),
]

app = make_app(__file__, "07", "AutoML with AutoGluon", "Binary · multiclass · regression · stack ensembling · time-budget study · live fit", [
    ("tasks", "Three tasks", "App", p_tasks), ("live", "Live AutoML run", "App", p_live),
    ("stack", "Stacking anatomy", "Data science", p_stack), ("budget", "Time-budget study", "Data science", p_budget),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/automl")
def automl(l: Live) -> dict:
    from autogluon.tabular import TabularPredictor
    from sklearn.model_selection import train_test_split
    if l.task not in SPEC:
        from fastapi import HTTPException
        raise HTTPException(422, "unknown task")
    label, metric = SPEC[l.task]
    df = {"binary": lambda: pipeline.adult().sample(8000, random_state=0).assign(income=lambda d: (d.income == ">50K").astype(int)), "multiclass": pipeline.digits, "regression": pipeline.taxi}[l.task]()
    tr, te = train_test_split(df, test_size=0.25, random_state=0)
    path = tempfile.mkdtemp(prefix="ag_live_")
    t0 = time.time()
    try:
        p = TabularPredictor(label=label, eval_metric=metric, path=path, verbosity=0).fit(tr, time_limit=l.budget, presets="medium_quality")
        lb = pipeline.board(p, te)
        best = p.model_best
    finally:
        shutil.rmtree(path, ignore_errors=True)
    secs = time.time() - t0
    return {"best": best, "blocks": [KPIS(K("Wall time", f"{secs:.1f} s"), K("Models trained", len(lb)), K("Selected model", best), K(f"Test {metric}", next(r["score_test"] for r in lb if r["model"] == best), digits=4)),
                                     CHART("hbar", "Models trained in this run", [S("test score", [r["score_test"] for r in lb], colors=["3" if r["model"] == best else "1" for r in lb])], [r["model"] for r in lb], height=max(180, 30 * len(lb)),
                                           xmin=min(r["score_test"] for r in lb) - 0.02 * abs(min(r["score_test"] for r in lb))), lb_table(lb, metric)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8007)
