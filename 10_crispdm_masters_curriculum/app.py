"""Project 10 — CRISP-DM Master's Platform: seven analytical phases on one dataset, each with a lesson, live results and a quiz."""
from __future__ import annotations
import inspect
import sys
import time
from pathlib import Path

import numpy as np
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


def clf():
    if "clf" not in _s:
        _s["clf"] = load_or_train(ROOT / "artifacts" / "model.joblib", pipeline.main)
    return _s["clf"]


def lsh():
    if "lsh" not in _s:
        d = np.load(ROOT / "artifacts" / "lsh_data.npz")
        c = M()["lsh"]["chosen"]
        _s["lsh"] = pipeline.CosineLSH(c["bits"], c["tables"]).fit(d["X"].astype(np.float32))
        _s["people"] = pd.read_csv(ROOT / "artifacts" / "lsh_people.csv.gz")
        _s["pre"] = load_or_train(ROOT / "artifacts" / "preprocessor.joblib", pipeline.main)
    return _s["lsh"]


def q(text, options, answer, why):
    return {"q": text, "options": options, "answer": answer, "why": why}


class Person(BaseModel):
    age: int = Field(38, ge=17, le=90)
    education_num: int = Field(13, ge=1, le=16)
    hours_per_week: int = Field(45, ge=1, le=99)
    capital_gain: int = Field(0, ge=0, le=99999)
    capital_loss: int = Field(0, ge=0, le=5000)
    workclass: str = "Private"
    marital_status: str = "Married-civ-spouse"
    occupation: str = "Exec-managerial"
    relationship: str = "Husband"
    race: str = "White"
    sex: str = "Male"
    native_country: str = "United-States"


def person_fields(m: dict, d: Person | None = None) -> list[dict]:
    d = d or Person()
    o = m["options"]
    return [F("age", "Age", "range", d.age, min=17, max=90, step=1), F("education_num", "Education (years, 9=HS, 13=BSc)", "range", d.education_num, min=1, max=16, step=1),
            F("hours_per_week", "Hours / week", "range", d.hours_per_week, min=1, max=99, step=1), F("capital_gain", "Capital gain ($)", "number", d.capital_gain),
            F("occupation", "Occupation", "select", d.occupation, options=o["occupation"]), F("marital_status", "Marital status", "select", d.marital_status, options=o["marital_status"]),
            F("relationship", "Relationship", "select", d.relationship, options=o["relationship"]), F("workclass", "Work class", "select", d.workclass, options=o["workclass"]),
            F("sex", "Sex", "select", d.sex, options=o["sex"]), F("race", "Race", "select", d.race, options=o["race"])]


# ----------------------------------------------------------------------------- phases
def p_overview() -> list[dict]:
    m = M()
    return [H("CRISP-DM, end to end", "One real dataset — 48,842 people from the 1994 US census — examined through seven analytical lenses."),
            KPIS(K("People", m["n"]), K("Earn > $50K", f"{100 * m['pos_rate']:.1f}%"), K("Train / test", f"{m['split']['train']:,} / {m['split']['test']:,}"), K("Encoded features", m["split"]["n_features_encoded"]), K("Full pipeline", f"{m['secs']:.0f} s")),
            MD("""
| Phase | Question | Technique | CRISP-DM stage |
|---|---|---|---|
| 1 · EDA & preprocessing | What is in the data, and what must not be used? | Profiling, missingness, leakage review | Data understanding / preparation |
| 2 · Clustering | Are there natural groups of people? | K-Means, silhouette, PCA | Modeling (unsupervised) |
| 3 · Anomaly detection | Which records are strange? | Isolation Forest | Data understanding / modeling |
| 4 · Supervised learning | Can we predict income — and is the model fair? | LogReg → RF → gradient boosting, fairness slice | Modeling / evaluation |
| 5 · Association rules | Which attribute combinations go with high income? | FP-Growth, lift | Modeling (descriptive) |
| 6 · Sub-linear search | How do we find similar people without scanning everyone? | Cosine LSH | Deployment |
| 7 · Synthesis | What did we learn, and what would we ship? | — | Evaluation / deployment |

**One rule runs through every phase:** the 20% test set was split off *before* anything was fit. Scalers, encoders, clusters, detectors and rules all learn from training rows only.""")]


def p1() -> list[dict]:
    e = M()["eda"]
    ages = [f"{a}-{a + 4}" for a in range(15, 90, 5)]
    edu = e["by_education"]
    names = e["edu_names"]
    return [H("Phase 1 · EDA & preprocessing", "Look before you model. Half of data-science mistakes are visible in a profile."),
            GRID(2, CHART("bar", "Age distribution", [S("everyone", e["age_hist"]), S("earning > $50K", e["age_hist_pos"], color="2")], ages, x="age", y="people", sub="High earners are concentrated between 35 and 55."),
                 CHART("bar", "Share earning > $50K by education", [S("share", [r[1] for r in edu], color="3")], [names.get(str(int(float(r[0]))), r[0]) for r in edu], y="share > $50K", sub="A near-monotone relationship — the strongest single signal.")),
            GRID(2, CHART("hbar", "Share > $50K by occupation", [S("share", [r[1] for r in e["by_occupation"]])], [r[0] for r in e["by_occupation"]], height=380),
                 HEAT("Correlation (numeric features + target)", e["corr_cols"], e["corr_cols"], e["corr"], diverging=True, min=-1, max=1, labw=110)),
            GRID(2, TABLE("Missing values", ["column", "missing", "share"], [[c, n, f"{100 * s:.1f}%"] for c, n, s in e["missing"]], sub="Missing is itself informative (unknown occupation ⇒ often not working), so it becomes an explicit 'Unknown' category rather than being dropped."),
                 MD(f"""**Decisions taken from the profile**
- `education` and `education_num` are 1:1 duplicates → keep the ordinal one.
- `fnlwgt` is a census *sampling weight*, not a property of the person → excluded from features.
- `capital_gain` is zero for {100 * e['capital_gain_zero']:.0f}% of people and capped at 99,999 → heavy tail, flagged again in Phase 3.
- {e['dup_rows']} exact duplicate rows exist; kept (different people can share coarse attributes).
- Preprocessing = median-impute + standardise numerics; 'Unknown'-impute + one-hot categoricals (levels with < 50 rows pooled). It lives in a `Pipeline`, so it is **re-fit inside every CV fold**.""", title="Preprocessing plan")),
            QUIZ("Phase 1 quiz", [q("Why exclude `fnlwgt` from the model?", ["It has missing values", "It is a survey weight describing the sample design, not the person", "It is categorical", "It is the target"], 1, "Using it would teach the model about how the census was sampled."),
                                  q("Where must the scaler be fit?", ["On all rows before splitting", "On the training rows only (inside the pipeline)", "On the test rows", "It does not matter"], 1, "Fitting on all rows leaks test-set means and variances into training."),
                                  q("7% of `occupation` is missing. Best first move?", ["Drop those rows", "Impute the mode silently", "Treat missing as its own category and check whether it is informative", "Drop the column"], 2, "Here missingness correlates with the target, so it carries signal.")])]


def p2() -> list[dict]:
    c = M()["cluster"]
    sw = c["sweep"]
    return [H("Phase 2 · Clustering", "No labels: do people fall into natural groups?"),
            GRID(2, CHART("line", "Silhouette by k", [S("silhouette", [r["silhouette"] for r in sw], points=True)], [r["k"] for r in sw], x="k", y="silhouette", sub="Low values everywhere — census attributes form a continuum, not tight blobs. That is a finding, not a failure."),
                 CHART("line", "Elbow (inertia)", [S("inertia", [r["inertia"] for r in sw], points=True, color="2")], [r["k"] for r in sw], x="k", y="within-cluster SSE")),
            CHART("scatter", f"K-Means (k = {c['k']}) in PCA space", [S(f"cluster {k}", [{"x": a, "y": b} for a, b, cc in c["scatter"] if cc == k], r=2.5) for k in range(c["k"])], x=f"PC1 ({100 * c['pca_var'][0]:.0f}%)", y=f"PC2 ({100 * c['pca_var'][1]:.0f}%)", height=340),
            TABLE("Cluster profiles — income was NOT used to form the clusters", ["#", "people", "share", "avg age", "avg hours", "avg edu yrs", "% > $50K", "% male", "typical marital status", "typical occupation"],
                  [[p["cluster"], p["n"], f"{100 * p['share']:.0f}%", round(p["age"], 1), round(p["hours"], 1), round(p["edu"], 1), round(p["pct_high_income"], 1), round(p["pct_male"]), p["top_marital"], p["top_occupation"]] for p in c["profiles"]]),
            CALLOUT("info", "What the clusters reveal", "Although income never entered the algorithm, the share of high earners differs several-fold between clusters — marital status and hours structure the population in a way that lines up with income."),
            QUIZ("Phase 2 quiz", [q("The best silhouette is only ~0.2. What does that mean?", ["The code is broken", "Clusters overlap heavily; treat segments as a useful summary, not natural kinds", "k is too small", "K-Means cannot run on one-hot data"], 1, "Silhouette near 0 = points sit between clusters."),
                                  q("Why standardise before K-Means?", ["It is required by scikit-learn", "Euclidean distance would otherwise be dominated by large-unit features like capital_gain", "To make clusters equal in size", "To remove outliers"], 1, "K-Means minimises squared Euclidean distance."),
                                  q("Clusters differ strongly in % > $50K, though income was not an input. This shows…", ["Leakage", "The features carry income-related structure", "Overfitting", "Nothing"], 1, "Unsupervised structure can align with a held-out label; no label was seen.")])]


def p3() -> list[dict]:
    a = M()["anomaly"]
    mids = [f"{(x + y) / 2:.2f}" for x, y in zip(a["edges"], a["edges"][1:])]
    return [H("Phase 3 · Anomaly detection", "Isolation Forest on the five numeric features: unusual records are isolated by few random splits."),
            KPIS(K("Records with capital_gain = 99,999", a["n_cap_99999"], hint="a top-coded value, not a real amount"), K("Records reporting 99 h/week", a["n_hours_99"]),
                 K("High earners among top-1% outliers", f"{100 * a['rate_high_income_among_outliers']:.0f}%", hint=f"vs {100 * a['rate_overall']:.0f}% overall")),
            CHART("bar", "Anomaly score distribution (training set)", [S("people", a["hist"], color="7")], mids, x="anomaly score (higher = stranger)", y="people (log)", logy=True, height=240),
            TABLE("The 15 most anomalous people", ["age", "education", "capital gain", "capital loss", "hours/wk", "occupation", "income", "score"], a["top"]),
            CALLOUT("warn", "Outliers are not noise here", "The strangest records are overwhelmingly *rich* people with capital gains — exactly the cases the classifier most needs. Deleting outliers by reflex would have removed real signal. The one true data-quality issue is the 99,999 top-code."),
            QUIZ("Phase 3 quiz", [q("How does an Isolation Forest score a point?", ["Distance to the nearest centroid", "Average path length needed to isolate it with random splits", "Reconstruction error", "Density of its neighbourhood"], 1, "Anomalies are few and different, so they are separated near the root."),
                                  q("An outlier detector flags mostly high-income people. You should…", ["Delete them before training", "Investigate: they may be legitimate, informative extremes", "Cap every feature at the median", "Ignore the finding"], 1, "Whether an outlier is an error is a domain question."),
                                  q("capital_gain = 99999 for 244 people most likely means…", ["They all earned exactly that", "The value is top-coded (censored) by the census", "A parsing bug", "Missing data"], 1, "Public-use files cap extreme values for privacy.")])]


def p4() -> list[dict]:
    m = M()
    s = m["supervised"]
    lb = s["leaderboard"]
    fair = s["fairness"]
    return [H("Phase 4 · Supervised learning", "Predict whether income exceeds $50K — then ask who the model fails."),
            GRID(2, TABLE("Model tournament", ["model", "5-fold CV AUC", "test AUC", "test accuracy", "test F1", "test PR-AUC"], [[r["model"], r["cv_auc"], r["test_auc"], r["test_acc"], r["test_f1"], r["test_ap"]] for r in lb], digits=4,
                          sub=f"**{s['best']}** was selected by cross-validation; the test column was read once, afterwards."),
                 CHART("scatter", "ROC curve (test)", [S(s["best"], [{"x": a, "y": b} for a, b in s["roc"]], line=True), S("_chance", [{"x": 0, "y": 0}, {"x": 1, "y": 1}], line=True, color="--axis")], x="false-positive rate", y="true-positive rate", height=280)),
            GRID(2, CHART("hbar", "Permutation importance (drop in test AUC)", [S("importance", [v for _, v in s["importance"]])], [k for k, _ in s["importance"]], height=330),
                 HEAT("Confusion matrix @ 0.5", ["pred > 50K", "pred ≤ 50K"], ["actual > 50K", "actual ≤ 50K"], s["confusion"], digits=0, labw=100)),
            TABLE("Fairness slice (test set, threshold 0.5)", ["attribute", "group", "n", "base rate", "selection rate", "TPR (recall)", "FPR", "precision"],
                  [[f["attr"], f["group"], f["n"], f["base_rate"], f["selection_rate"], f["tpr"], f["fpr"], f["precision"]] for f in fair], digits=3),
            CALLOUT("warn", f"Disparate-impact ratio by sex: {s['disparate_impact_sex']:.2f} (the '80% rule' asks for ≥ 0.80)",
                    "The model selects women as high-income far less often than men. Most of that mirrors a real base-rate gap in 1994 data — but recall is also lower for women, meaning qualified women are missed more often. A model like this must not be used for decisions about individuals without mitigation (threshold adjustment per group, reweighing) and review."),
            FORM("api/predict", person_fields(m), title="Live prediction", submit="Predict income", auto=True, run_on_load=True),
            QUIZ("Phase 4 quiz", [q("Why is the model chosen by CV score rather than test score?", ["CV is faster", "Choosing on the test set turns it into training data and inflates the reported score", "Test AUC is always lower", "No reason"], 1, "The test set must stay untouched until the final report."),
                                  q("Accuracy is 87% and the majority class is 76%. The honest reading is…", ["87% is excellent", "The lift over a do-nothing baseline is 11 points; look at AUC / F1", "The model is useless", "The data is leaked"], 1, "Always compare with the trivial baseline."),
                                  q("Equal accuracy across groups guarantees fairness.", ["True", "False — error *types* (TPR/FPR) can still differ sharply between groups"], 1, "Equalised odds compares TPR and FPR per group.")])]


def p5() -> list[dict]:
    r = M()["rules"]
    return [H("Phase 5 · Association rule mining", f"FP-Growth over discretised attributes (support ≥ {r['min_support']}): which combinations travel with high income?"),
            KPIS(K("Frequent itemsets", r["n_itemsets"]), K("Rules ⇒ income > 50K (conf ≥ 0.5)", r["n_rules_high"])),
            TABLE("Top rules predicting  > $50K, by lift", ["if a person is…", "support", "confidence", "lift"], r["high"]),
            TABLE("Highest-support rules predicting  ≤ $50K (confidence ≥ 0.97)", ["if a person is…", "support", "confidence", "lift"], r["low"]),
            CALLOUT("info", "Rules vs the classifier", "Rules are less accurate than gradient boosting but *readable*: 'married + bachelor's + 41–50 hours' is something a policy analyst can discuss. They also confirm what permutation importance said in Phase 4: marital status, education and capital gains dominate."),
            QUIZ("Phase 5 quiz", [q("Lift = 3 for {A} ⇒ {>50K} means…", ["A causes high income", "High income is 3× as common among A as in the population", "Confidence is 3%", "Support is 3%"], 1, "Lift = confidence / base rate; it is association, not causation."),
                                  q("Why discretise age and hours first?", ["FP-Growth needs binary items", "To make the model faster only", "To remove outliers", "It is optional"], 0, "Itemset mining works on presence/absence of items."),
                                  q("A rule has confidence 0.98 for ≤50K. Impressive?", ["Yes", "Not by itself — the base rate is already 0.76, so lift is only ~1.3"], 1, "Always read confidence against the consequent's base rate.")])]


def p6() -> list[dict]:
    m = M()
    L = m["lsh"]
    st = L["study"]
    c = L["chosen"]
    lsh()
    return [H("Phase 6 · Sub-linear search with LSH", "Find the most similar people without comparing against everyone: random-hyperplane hashing for cosine similarity."),
            KPIS(K("Indexed people", L["n"]), K("Dimensions", L["dim"]), K("Chosen config", f"{c['bits']} bits × {c['tables']} tables"), K("Recall@10", f"{100 * c['recall']:.0f}%", hint="vs exact search"),
                 K("Rows inspected per query", f"{100 * c['candidates']:.1f}%", hint="instead of 100%"), K("Measured speed-up", f"{c['speedup']:.1f}×")),
            GRID(2, CHART("scatter", "The LSH trade-off: recall vs work", [S(f"{t} tables", [{"x": 100 * s["candidates"], "y": s["recall"]} for s in st if s["tables"] == t], line=True, extra={"pointRadius": 4}) for t in [4, 8, 16]], x="% of rows inspected (log)", y="recall@10", logx=True, height=300,
                          sub="More bits ⇒ smaller buckets ⇒ less work but lower recall. More tables buy the recall back."),
                 TABLE("Parameter study", ["bits", "tables", "recall@10", "% inspected", "speed-up", "LSH ms", "exact ms"], [[s["bits"], s["tables"], round(s["recall"], 3), round(100 * s["candidates"], 2), round(s["speedup"], 2), round(s["lsh_ms"], 3), round(s["exact_ms"], 3)] for s in st], maxh=300)),
            CALLOUT("info", "Honest scale note", f"At N = {L['n']:,} a vectorised brute-force scan takes ~0.4 ms, so the wall-clock gain is modest. The point is the *scaling law*: brute force is O(N·d) per query while LSH touches a fixed fraction of N — at 20 million rows that is the difference between 0.4 s and a few milliseconds."),
            FORM("api/similar", person_fields(m), title="Find similar people (LSH vs exact)", submit="Search", run_on_load=True),
            CODE("CosineLSH — pipeline.py", inspect.getsource(pipeline.CosineLSH)),
            QUIZ("Phase 6 quiz", [q("Two vectors at angle θ agree on one random-hyperplane bit with probability…", ["cos θ", "1 − θ/π", "θ/π", "0.5"], 1, "A random hyperplane separates them with probability θ/π."),
                                  q("Increasing bits per table (tables fixed) will…", ["Raise recall, raise work", "Lower recall, lower work", "Change nothing", "Raise both precision and recall"], 1, "Longer signatures collide less often."),
                                  q("LSH returns *approximate* neighbours. When is that acceptable?", ["Never", "When a small recall loss is worth a large latency gain — recommendation, dedup, retrieval"], 1, "Most retrieval systems re-rank LSH/ANN candidates exactly.")])]


def p7() -> list[dict]:
    m = M()
    s, L = m["supervised"], m["lsh"]["chosen"]
    best = next(r for r in s["leaderboard"] if r["model"] == s["best"])
    return [H("Phase 7 · Synthesis", "What the seven lenses add up to."),
            MD(f"""
## Findings
1. **Signal**: education, marital status, age, hours and capital gains explain most of the variation in income (Phases 1, 4, 5 agree independently).
2. **Structure**: the population is a continuum, not discrete tribes (silhouette ≈ 0.2), yet a 5-segment summary still separates high-income rates several-fold.
3. **Extremes matter**: the 'outliers' are mostly wealthy people with capital gains — informative, not erroneous. One real artefact: the 99,999 top-code.
4. **Prediction**: {s['best']} reaches **test ROC-AUC {best['test_auc']:.3f}** (CV {best['cv_auc']:.3f}), accuracy {100 * best['test_acc']:.1f}% against a 76.1% majority baseline.
5. **Fairness**: disparate-impact ratio by sex is **{s['disparate_impact_sex']:.2f}** — far below the 0.80 guideline. Not deployable for individual decisions as is.
6. **Scale**: cosine LSH ({L['bits']} bits × {L['tables']} tables) keeps {100 * L['recall']:.0f}% recall while inspecting {100 * L['candidates']:.0f}% of rows.

## What I would ship
- The **explanatory artefacts** (profiles, rules, importance) for policy analysis — yes.
- The **classifier** as an aggregate estimator (e.g. small-area income statistics) — yes, with monitoring.
- The classifier for **decisions about a person** (credit, hiring) — no, not without bias mitigation and legal review.

## What I would do next
Time-aware validation on a newer census wave, calibration, per-group threshold analysis, and SHAP for individual explanations.

## Method audit
| Check | Status |
|---|---|
| Test set split before any fitting | ✓ |
| Preprocessing inside the CV pipeline | ✓ |
| Model chosen by CV, test read once | ✓ |
| Sampling weight and duplicate column removed | ✓ |
| Baselines reported alongside every metric | ✓ |
| Fairness slice reported, not hidden | ✓ |
| Fixed seeds (42) | ✓ |"""),
            QUIZ("Final quiz", [q("Which CRISP-DM phase is most often skipped and most often fatal?", ["Modeling", "Business & data understanding", "Deployment", "None"], 1, "A model answering the wrong question on misunderstood data cannot be rescued by tuning."),
                                q("CRISP-DM is a strictly linear waterfall.", ["True", "False — arrows go backwards: evaluation routinely sends you back to data preparation"], 1, "The cycle diagram has feedback loops by design.")])]


app = make_app(__file__, "10", "CRISP-DM Master's Platform", "Adult census · 7 phases: EDA, clustering, anomalies, supervised + fairness, rules, LSH, synthesis", [
    ("overview", "Overview", "Course", p_overview),
    ("eda", "1 · EDA & preprocessing", "Phases", p1), ("clustering", "2 · Clustering", "Phases", p2), ("anomaly", "3 · Anomaly detection", "Phases", p3),
    ("supervised", "4 · Supervised learning", "Phases", p4), ("rules", "5 · Association rules", "Phases", p5), ("lsh", "6 · LSH search", "Phases", p6), ("synthesis", "7 · Synthesis", "Phases", p7),
])


@app.post("/api/predict")
def predict(p: Person) -> dict:
    row = pd.DataFrame([p.model_dump()])
    prob = float(clf().predict_proba(row)[0, 1])
    sweeps = {}
    for col, vals in [("education_num", list(range(5, 17))), ("hours_per_week", list(range(10, 81, 5))), ("age", list(range(20, 71, 5)))]:
        sweeps[col] = (vals, clf().predict_proba(pd.concat([row] * len(vals)).assign(**{col: vals}))[:, 1].round(4).tolist())
    return {"probability": prob, "blocks": [KPIS(K("P(income > $50K)", f"{100 * prob:.1f}%"), K("Decision @ 0.5", "> $50K" if prob >= 0.5 else "≤ $50K"), K("Population base rate", f"{100 * M()['pos_rate']:.1f}%")),
                                             GRID(3, *[CHART("line", f"What if {c.replace('_', ' ')} changed?", [S("P(> $50K)", v[1], color=col)], v[0], height=200, ymin=0, ymax=1) for (c, v), col in zip(sweeps.items(), ["1", "2", "3"])])]}


@app.post("/api/similar")
def similar(p: Person) -> dict:
    L = lsh()
    qv = _s["pre"].transform(pd.DataFrame([p.model_dump()]))[0].astype(np.float32)
    t0 = time.perf_counter(); idx, sims, n_c = L.query(qv, 8); t_l = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); ex, _ = L.exact(qv, 8); t_e = (time.perf_counter() - t0) * 1000
    ppl = _s["people"]
    rows = [[*[ppl.iloc[i][c].item() if hasattr(ppl.iloc[i][c], "item") else ppl.iloc[i][c] for c in ppl.columns], round(float(s), 3), PILL("also in exact top-8", "good") if i in set(ex) else PILL("approximate", "warn")] for i, s in zip(idx, sims)]
    return {"blocks": [KPIS(K("Candidates inspected", f"{n_c:,} of {len(L.X):,}", hint=f"{100 * n_c / len(L.X):.1f}%"), K("Overlap with exact top-8", f"{len(set(idx) & set(ex))} / 8"), K("LSH query", f"{t_l:.2f} ms"), K("Exact scan", f"{t_e:.2f} ms"),
                            K("Share earning > $50K among neighbours", f"{100 * np.mean([r[6] == '>50K' for r in rows]):.0f}%" if rows else "—")),
                       TABLE("Most similar people in the index", [*ppl.columns, "cosine sim", "check"], rows)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)
