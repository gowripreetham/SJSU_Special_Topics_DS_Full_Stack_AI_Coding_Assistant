"""Project 14 — AutoGluon capability suite: text+tabular fusion, quantile forecasting, drift detection, distillation, calibration."""
from __future__ import annotations
import re
import sys
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
_s: dict = {}


def M() -> dict:
    if "m" not in _s:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _s["m"]


def predictor():
    """A small text+tabular predictor for the live demo, trained on first use (~15 s) and cached on disk."""
    if "p" not in _s:
        from autogluon.tabular import TabularPredictor
        path = pipeline.MODELS / "titanic_live"
        try:
            _s["p"] = TabularPredictor.load(str(path), verbosity=0)
        except Exception:
            _s["p"] = pipeline.fit(pipeline.titanic(), "Survived", "titanic_live", 15, "roc_auc")
    return _s["p"]


class Passenger(BaseModel):
    Name: str = Field("Dawson, Mr. Jack", max_length=120)
    Sex: str = "male"
    Age: float = Field(20, ge=0, le=90)
    Pclass: int = Field(3, ge=1, le=3)
    Fare: float = Field(8.0, ge=0, le=600)
    SibSp: int = Field(0, ge=0, le=8)
    Parch: int = Field(0, ge=0, le=8)
    Embarked: str = "S"


def p_text() -> list[dict]:
    t = M()["text"]
    sm = t["summary"]
    ft = t["feature_types"]
    return [H("A · Text + tabular fusion", f"Titanic ({t['n']} passengers) with its free-text columns left in. AutoGluon detects `Name` as text and expands it into n-gram and text-statistics features automatically."),
            GRID(2, CHART("bar", "Test ROC-AUC (mean of 3 random splits)", [S("mean AUC", [r["auc"] for r in sm], colors=["--axis", "3", "1"])], ["tabular only", "+ raw text (auto)", "+ hand-made Title"], ymin=0.8, height=280,
                          note=" · ".join(f"**{n}**: {r['lo']:.3f}–{r['hi']:.3f}" for n, r in zip(["tabular", "auto text", "Title"], sm)) + " (min–max across splits)"),
                 TABLE("Every run", ["variant", "split seed", "ROC-AUC", "accuracy", "models"], [[r["variant"].split("(")[0], r["seed"], r["auc"], r["acc"], r["n_models"]] for r in t["runs"]], digits=4)),
            GRID(2, TABLE("What AutoGluon generated from 11 raw columns", ["feature tag", "features"], [[{"code": k}, v] for k, v in sorted(ft["generated_by_tag"].items(), key=lambda kv: -kv[1])] + [["total model features", ft["n_generated"]]],
                          sub="Examples: " + ", ".join(f"`{e}`" for e in ft["examples"][:8])),
                 CHART("hbar", "Permutation importance with text features in play", [S("importance", [v for _, v in t["importance"]], color="7")], [k for k, _ in t["importance"]], height=300)),
            TABLE("Why the name matters: survival by honorific", ["title", "survival rate", "passengers"], [[a, b, int(c)] for a, b, c in t["survival_by_title"]], sub="`Master` (boys) and `Mrs`/`Miss` survived far more often than `Mr` — information that is *only* in the text."),
            CALLOUT("info", "Reading the result", "Automatic text featurisation lifts mean AUC by about two points and, on these splits, does at least as well as a hand-engineered `Title` feature — with zero feature code. With 891 rows the split-to-split range is as large as the effect, which is why three splits are shown rather than one."),
            FORM("api/passenger", [F("Name", "Passenger name (free text — try “Master”, “Mrs.”, “Dr.”, “Rev.”)", "text", "Dawson, Mr. Jack", wide=True), F("Sex", "Sex", "select", "male", options=["male", "female"]), F("Age", "Age", "range", 20, min=1, max=80, step=1),
                                   F("Pclass", "Class", "select", 3, options=[1, 2, 3]), F("Fare", "Fare (£)", "number", 8)], title="Live: would this passenger have survived? (first call trains a 15-second model)", submit="Predict")]


def p_forecast() -> list[dict]:
    f = M()["forecast"]
    a, b = f["autogluon"], f["baseline"]
    return [H("B · Probabilistic forecasting with quantile regression", f"`TabularPredictor(problem_type='quantile', quantile_levels=[0.1, 0.5, 0.9])` on lag features of monthly Mauna Loa CO₂. Trained on {f['n_train']} months up to 1996, tested one step ahead on {f['n_test']} months (1997–2001)."),
            KPIS(K("Pinball loss", a["pinball"], digits=4, hint=f"seasonal-naive baseline {b['pinball']:.4f}"), K("Improvement", f"{100 * (1 - a['pinball'] / b['pinball']):.0f}%"), K("P10–P90 coverage", f"{100 * a['coverage80']:.0f}%", hint="target 80%"),
                 K("Band width (ppm)", a["width"], digits=3, hint=f"baseline {b['width']:.3f}"), K("Median MAE (ppm)", a["mae_p50"], digits=3)),
            CHART("line", "One-step-ahead forecast band vs observed CO₂", [S("observed", f["actual"], color="--ink", width=2.5), S("P50", f["p50"], color="1", dash=True), S("_p10", f["p10"], color="1", width=0), S("P10–P90 band", f["p90"], color="1", width=0, fill="-1")], f["months"], y="ppm", height=320),
            GRID(2, CHART("line", "Monthly change: forecast band vs actual", [S("actual Δ", f["diff_actual"], color="--ink"), S("_lo", [q[0] for q in f["diff_q"]], color="2", width=0), S("P10–P90", [q[2] for q in f["diff_q"]], color="2", width=0, fill="-1")], f["months"], y="ppm / month", height=260),
                 TABLE(f"Quantile models trained (best: {f['best']})", ["model", "test pinball (neg.)", "val pinball (neg.)", "fit s"], f["leaderboard"], digits=4)),
            CALLOUT("warn", "Calibration gap", f"The band is sharper than the baseline but covers {100 * a['coverage80']:.0f}% instead of 80% — slightly over-confident. A conformal adjustment on a validation window would widen it to nominal coverage. Every feature is a lag of at least one month, so nothing from the forecast month leaks in.")]


def drift_blocks(name: str) -> list[dict]:
    d = M()["drift"][name]
    sev = lambda r: PILL("drift", "crit") if (r["stat"] > 0.2) else PILL("minor", "warn") if r["stat"] > 0.05 else PILL("stable", "good")  # noqa: E731
    return [KPIS(K("Adversarial-validation AUC", d["adv_auc"], digits=3, hint="0.5 = train and test are indistinguishable"), K("Model AUC on its own distribution", d["auc_in_dist"], digits=3), K("Model AUC on the test population", d["auc_on_test"], digits=3,
                   delta=f"{d['auc_on_test'] - d['auc_in_dist']:+.3f}", delta_good=d["auc_on_test"] >= d["auc_in_dist"] - 0.01), K("Positive rate train → test", f"{100 * d['pos_rate_train']:.0f}% → {100 * d['pos_rate_test']:.0f}%")),
            GRID(2, TABLE("Per-feature drift tests", ["feature", "type", "statistic", "test", "p-value", "status"], [[r["feature"], r["type"], r["stat"], r["test"], r["p"], sev(r)] for r in d["tests"]], digits=3),
                 CHART("hbar", "What gives the split away? (importance in the adversarial classifier)", [S("importance", [v for _, v in d["adv_importance"]], color="8")], [k for k, _ in d["adv_importance"]], height=280))]


def p_drift() -> list[dict]:
    return [H("C · AutoEDA & covariate drift", "Before trusting a validation score, ask whether the data you will predict on looks like the data you trained on. Two detectors: per-feature statistical tests, and *adversarial validation* — train AutoGluon to tell train rows from test rows."),
            TABS(*[(k, drift_blocks(k)) for k in M()["drift"]]),
            CALLOUT("info", "Why both detectors", "Univariate tests miss shifts that only show up in feature *combinations*; the adversarial classifier catches those and its feature importance says where to look. In the shifted scenario it is perfect (AUC 1.0) because age alone separates the sets — and the income model loses about four AUC points on the older population it never saw.")]


def p_distill() -> list[dict]:
    m = M()
    d, c = m["distill"], m["calibration"]
    t, s, h = d["teacher"], d["student_soft"], d["student_hard"]
    return [H("D · Distillation & E · calibration", f"A {t['n_models']}-model bagged + stacked AutoGluon ensemble (teacher) is compressed into one 120-tree LightGBM (student) trained to reproduce the teacher's logits on {d['n_distill']:,} rows the teacher never saw."),
            KPIS(K("Inference speed-up", f"{d['speedup']:.0f}×", hint=f"{t['ms']:.0f} ms → {s['ms']:.0f} ms for {d['n_test']:,} rows"), K("Fidelity", f"{100 * s['fidelity']:.1f}%", hint="student agrees with teacher's decision"), K("Teacher ROC-AUC", t["auc"], digits=4),
                 K("Distilled student ROC-AUC", s["auc"], digits=4, hint=f"same student on hard labels: {h['auc']:.4f}"), K("AUC given up", f"{t['auc'] - s['auc']:.4f}")),
            GRID(2, CHART("scatter", "Student vs teacher probability (500 test rows)", [S("person", d["scatter"], r=2.5), S("_ideal", [{"x": 0, "y": 0}, {"x": 1, "y": 1}], line=True, color="--axis")], x="teacher P(>50K)", y="student P(>50K)", height=300, sub=f"Correlation {s['corr']:.3f}."),
                 CHART("scatter", "Reliability diagram", [S("teacher (ensemble)", [{"x": a, "y": b} for a, b in c["teacher"]], line=True, extra={"pointRadius": 3}), S("distilled student", [{"x": a, "y": b} for a, b in c["student"]], line=True, extra={"pointRadius": 3}),
                                                      S("_perfect", [{"x": 0, "y": 0}, {"x": 1, "y": 1}], line=True, color="--axis")], x="mean predicted probability", y="observed frequency", height=300,
                       sub=f"Brier: teacher {c['brier_teacher']:.4f}, student {c['brier_student']:.4f} (no-skill {c['base_brier']:.4f}). ECE: {c['ece_teacher']:.3f} / {c['ece_student']:.3f}.")),
            GRID(2, TABLE("Three ways to get a small model", ["model", "ROC-AUC", "accuracy", "agrees with teacher", "corr. with teacher"], [["Teacher: stacked ensemble", t["auc"], t["acc"], "—", "—"], ["Student distilled from teacher logits", s["auc"], s["acc"], f"{100 * s['fidelity']:.1f}%", round(s["corr"], 3)],
                                                                                                                                                  ["Same student trained on hard labels", h["auc"], h["acc"], f"{100 * h['fidelity']:.1f}%", round(h["corr"], 3)]], digits=4),
                 TABLE("Inside the teacher", ["model", "test ROC-AUC", "stack level", "predict s"], d["teacher_lb"], digits=4, maxh=260)),
            CALLOUT("info", "What distillation bought", "The student keeps almost all of the teacher's accuracy at a fraction of the latency and with one model file instead of a directory of dozens. Soft targets transfer the teacher's *ranking* (higher correlation and fidelity than hard-label training), which is the point when the student must stand in for the ensemble.")]


CRISPDM = [
    ("Business understanding", "AutoML is being considered as the default modelling path. Before adopting it we want evidence on four practical questions: messy text columns, uncertainty estimates, data drift, and serving cost."),
    ("Data understanding", "Titanic with free text (891 rows), Mauna Loa CO₂ (526 months), Adult census (20k-row sample). Each is small enough to re-run in minutes and well-known enough that results can be sanity-checked."),
    ("Data preparation", "Almost none by design — the studies test what AutoGluon does on its own. Exceptions: lag features for forecasting (strictly past values) and removal of a sampling-weight column from Adult."),
    ("Modeling", "AutoGluon TabularPredictor in classification and quantile modes; bagging + stacking for the teacher; LightGBM students; an adversarial train-vs-test classifier."),
    ("Evaluation", "Held-out splits throughout (three seeds for the small Titanic data, chronological for CO₂). Pinball loss and coverage for quantiles; KS/TV and adversarial AUC for drift; fidelity, AUC and latency for distillation; reliability curves."),
    ("Deployment", "The distilled student is the deployment candidate: one model, ~12× faster. `POST api/passenger` demonstrates text+tabular inference live."),
]


def p_overview() -> list[dict]:
    m = M()
    t, f, d = m["text"]["summary"], m["forecast"], m["distill"]
    dr = m["drift"]
    k2 = list(dr)[1]
    return [H("AutoGluon capability suite", f"Five measured studies with AutoGluon {m['versions']['autogluon']}, all runnable offline on a 2-core CPU in about {m['secs'] / 60:.0f} minutes."),
            TABLE("Scoreboard", ["study", "question", "headline result"], [
                ["A · Text + tabular", "Does automatic text featurisation help?", f"ROC-AUC {t[0]['auc']:.3f} → {t[1]['auc']:.3f} when raw text columns are kept"],
                ["B · Quantile forecasting", "Can a tabular AutoML forecast with uncertainty?", f"pinball loss {100 * (1 - f['autogluon']['pinball'] / f['baseline']['pinball']):.0f}% below seasonal-naive; coverage {100 * f['autogluon']['coverage80']:.0f}% vs 80% nominal"],
                ["C · Drift detection", "Would we notice a population shift?", f"adversarial AUC {list(dr.values())[0]['adv_auc']:.2f} (random split) vs {dr[k2]['adv_auc']:.2f} (shifted); model AUC drops {dr[k2]['auc_in_dist'] - dr[k2]['auc_on_test']:.3f}"],
                ["D · Distillation", "Can the ensemble be made cheap to serve?", f"{d['speedup']:.0f}× faster, {100 * d['student_soft']['fidelity']:.1f}% decision fidelity, −{d['teacher']['auc'] - d['student_soft']['auc']:.4f} AUC"],
                ["E · Calibration", "Are the probabilities usable as probabilities?", f"ECE {m['calibration']['ece_teacher']:.3f} (teacher), {m['calibration']['ece_student']:.3f} (student)"]]),
            CALLOUT("warn", "Scope, stated plainly", "The reference project showcased Chronos (a pretrained time-series transformer) and vision-language fusion. Those need multi-gigabyte model downloads and a GPU, so they are **not** reproduced here. In their place: quantile forecasting through AutoGluon's tabular engine, and text+tabular fusion through its built-in n-gram featuriser — both measured rather than asserted.")]


app = make_app(__file__, "14", "AutoGluon Capability Suite", "Text+tabular fusion · quantile forecasting · drift detection · distillation · calibration", [
    ("overview", "Overview", "Suite", p_overview), ("text", "A · Text + tabular", "Suite", p_text), ("forecast", "B · Quantile forecasting", "Suite", p_forecast), ("drift", "C · Drift detection", "Suite", p_drift), ("distill", "D/E · Distillation & calibration", "Suite", p_distill),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/passenger")
def passenger(p: Passenger) -> dict:
    t0 = time.time()
    pr = predictor()
    base = {"Ticket": "A/5 21171", "Cabin": None, **p.model_dump()}
    row = pd.DataFrame([base])
    prob = float(pr.predict_proba(row)[1].iloc[0])
    title = (re.search(r",\s*([^\.]+)\.", p.Name) or [None, "—"])[1]
    alts = []
    for alt in ["Mr", "Mrs", "Miss", "Master", "Dr", "Rev"]:
        nm = re.sub(r",\s*[^\.]+\.", f", {alt}.", p.Name) if "," in p.Name and "." in p.Name else f"Doe, {alt}. Sam"
        alts.append((alt, float(pr.predict_proba(pd.DataFrame([{**base, "Name": nm}]))[1].iloc[0])))
    return {"probability": prob, "blocks": [KPIS(K("P(survived)", f"{100 * prob:.0f}%"), K("Honorific parsed from text", title), K("Model", pr.model_best), K("Latency", f"{(time.time() - t0) * 1000:.0f} ms")),
                                             CHART("bar", "Same passenger, only the honorific in the name changed", [S("P(survived)", [v for _, v in alts], colors=["3" if a == title else "1" for a, _ in alts])], [a for a, _ in alts], ymin=0, ymax=1, height=230,
                                                   note="Sex, age, class and fare are held fixed — any difference comes purely from the text column.")]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8014)
