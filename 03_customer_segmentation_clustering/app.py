"""Project 03 — Customer Intelligence Clustering: segment explorer, live assignment API, DS admin dashboard."""
from __future__ import annotations
import sys
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


def model() -> dict:
    if "model" not in _s:
        _s["model"] = load_or_train(ROOT / "artifacts" / "model.joblib", pipeline.main)
    return _s["model"]


class Customer(BaseModel):
    age: int = Field(30, ge=18, le=90)
    income: float = Field(60, ge=0, le=200, description="annual income, k$")
    spend: float = Field(50, ge=1, le=100, description="spending score 1-100")
    gender: str = "Female"


def by_cluster(key_x: str, key_y: str) -> list[dict]:
    m = M()
    return [S(p["persona"], [{"x": q[key_x], "y": q[key_y]} for q in m["points"] if q["c"] == p["cluster"]], r=4.5) for p in m["profiles"]]


def p_explorer() -> list[dict]:
    m = M()
    f = m["final"]
    return [H("Segment explorer", f"{m['n']} mall customers partitioned into {len(m['profiles'])} actionable segments."),
            KPIS(K("Segments", len(m["profiles"])), K("Silhouette", f["silhouette"], digits=3, hint="−1…1, higher = tighter"),
                 K("Bootstrap stability (ARI)", f["stability_ari"], digits=3, hint="1.0 = identical clusters on resampled data"), K("Davies–Bouldin", f["davies"], digits=3, hint="lower is better")),
            GRID(2, CHART("scatter", "Annual income × spending score", by_cluster("income", "spend"), x="annual income (k$)", y="spending score", height=380),
                 CHART("scatter", "PCA projection (age, income, spend)", by_cluster("px", "py"), x=f"PC1 ({100 * m['pca_var'][0]:.0f}% var)", y=f"PC2 ({100 * m['pca_var'][1]:.0f}% var)", height=380)),
            TABLE("Segment profiles & recommended marketing action", ["#", "persona", "customers", "share", "avg age", "avg income k$", "avg spend", "% male", "silhouette", "recommended action"],
                  [[p["cluster"], p["persona"], p["n"], f"{100 * p['share']:.0f}%", round(p["age"], 1), round(p["income"], 1), round(p["spend"], 1), round(p["male_pct"]), round(p["silhouette"], 3), p["action"]] for p in m["profiles"]]),
            FORM("api/assign", [F("age", "Age", "range", 29, min=18, max=70, step=1), F("income", "Annual income (k$)", "range", 85, min=15, max=140, step=1),
                                F("spend", "Spending score", "range", 80, min=1, max=100, step=1), F("gender", "Gender", "select", "Female", options=["Female", "Male"])],
                 title="Which segment is this customer?", submit="Assign segment", auto=True, run_on_load=True)]


def p_selection() -> list[dict]:
    sw = M()["sweep"]
    ks = [r["k"] for r in sw]
    return [H("Choosing k and the algorithm", "Four internal validity views of the same data. No single index is trusted on its own."),
            GRID(2, CHART("line", "Elbow — K-Means inertia", [S("inertia", [r["inertia"] for r in sw], points=True)], ks, x="k", y="within-cluster SSE"),
                 CHART("line", "Silhouette by k", [S("K-Means", [r["silhouette"] for r in sw], points=True), S("Ward", [r["ward_silhouette"] for r in sw], points=True)], ks, x="k", y="mean silhouette")),
            GRID(2, CHART("line", "Gaussian mixture BIC (lower is better)", [S("BIC", [r["gmm_bic"] for r in sw], points=True, color="7")], ks, x="components", y="BIC"),
                 CHART("line", "Davies–Bouldin (lower is better)", [S("DB index", [r["davies"] for r in sw], points=True, color="2")], ks, x="k", y="DB index")),
            TABLE("DBSCAN density grid (income × spend, standardised)", ["eps", "min_samples", "clusters found", "noise %", "silhouette (core pts)"],
                  [[r["eps"], r["min_samples"], r["clusters"], round(r["noise_pct"], 1), r["silhouette"]] for r in M()["dbscan"]], digits=3, maxh=300,
                  sub="DBSCAN needs no k but is sensitive to eps; on 200 points it either merges segments or labels many customers as noise.")]


def p_ar() -> list[dict]:
    return [H("AutoResearch", "Search over algorithm × k × feature set × scaler. Objective = silhouette × (0.5 + 0.5 · bootstrap ARI): compact **and** reproducible clusters, micro-clusters (<8 customers) rejected."),
            *ar_blocks(M()["autoresearch"], "objective")]


def p_admin() -> list[dict]:
    m = M()
    pts = sorted(m["points"], key=lambda q: (q["c"], -q["sil"]))
    names = {p["cluster"]: p["persona"] for p in m["profiles"]}
    he = m["hist_edges"]
    return [H("Data-science admin dashboard"),
            GRID(3, *[CHART("bar", t, [S("customers", m["hist"][c], color=col)], [f"{(a + b) / 2:.0f}" for a, b in zip(he[c], he[c][1:])], height=200)
                      for c, t, col in [("age", "Age", "1"), ("income", "Annual income (k$)", "3"), ("spend", "Spending score", "2")]]),
            GRID(2, CHART("bar", "Silhouette plot (every customer, grouped by segment)", [S(names[c], [q["sil"] if q["c"] == c else None for q in pts]) for c in sorted(names)], [""] * len(pts), stacked=True, y="silhouette", height=300,
                       sub="Bars below zero are customers closer to a neighbouring segment — candidates for manual review."),
                 HEAT("PCA loadings", ["age", "income", "spend"], ["PC1", "PC2"], m["pca_load"], diverging=True, min=-1, max=1, labw=50, sub="PC1 contrasts age with spending; PC2 is mostly income.")),
            GRID(2, HEAT("Correlation", ["age", "income", "spend"], ["age", "income", "spend"], m["corr"], diverging=True, min=-1, max=1, labw=60),
                 TABLE("Descriptive statistics", ["stat", "age", "income", "spend"], m["desc"])),
            MD(f"""- **Final model**: `{m['best']['algo']}` with k={m['best']['k']} on `{m['best']['features']}` ({m['best']['scaler']} scaling)
- **Validity**: silhouette {m['final']['silhouette']:.3f} · Calinski–Harabasz {m['final']['calinski']:.0f} · Davies–Bouldin {m['final']['davies']:.3f}
- **Stability**: mean adjusted Rand index {m['final']['stability_ari']:.3f} over 8 bootstrap refits
- **Caveats**: 200 customers from one mall; spending score is a proprietary 1–100 index; segments are descriptive, not causal
- **Leakage note**: unsupervised — there is no label to leak. The scaler and centroids are fit once and frozen for serving.""", title="Model card")]


CRISPDM = [
    ("Business understanding", "Marketing wants a handful of customer groups it can *name* and *act on*. Success = well-separated, stable segments with a clear action per segment."),
    ("Data understanding", "Kaggle *Mall Customers*: 200 customers × age, gender, annual income, spending score. No missing values; income and spend are uncorrelated (r≈0.01), which is exactly why 2-D structure appears."),
    ("Data preparation", "Rename columns, encode gender, compare three scalers. Feature subsets are treated as a hyper-parameter rather than assumed."),
    ("Modeling", "K-Means, Gaussian mixtures, Ward linkage and DBSCAN; k swept 2–10; AutoResearch hill-climb over algorithm × k × features × scaler."),
    ("Evaluation", "Silhouette, Calinski–Harabasz, Davies–Bouldin, GMM BIC and bootstrap stability (ARI). Per-customer silhouette plot exposes borderline assignments."),
    ("Deployment", "`POST api/assign` returns the segment, persona, distance to every centroid and a marketing action for any new customer; the explorer page calls it live."),
]

app = make_app(__file__, "03", "Customer Intelligence Clustering", "K-Means · GMM · Ward · DBSCAN · stability-aware AutoResearch", [
    ("explorer", "Segment explorer", "App", p_explorer),
    ("selection", "Model selection", "Data science", p_selection),
    ("autoresearch", "AutoResearch", "Data science", p_ar),
    ("admin", "Admin dashboard", "Data science", p_admin),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/assign")
def assign(c: Customer) -> dict:
    mm, m = model(), M()
    row = pd.DataFrame([{"age": c.age, "income": c.income, "spend": c.spend, "male": int(c.gender == "Male")}])[mm["features"]]
    z = mm["scaler"].transform(row)
    cl = int(mm["model"].predict(z)[0])
    cents = mm["model"].cluster_centers_ if hasattr(mm["model"], "cluster_centers_") else mm["model"].means_
    d = np.linalg.norm(cents - z, axis=1)
    conf = np.exp(-d ** 2)
    conf = conf / conf.sum()
    prof = {p["cluster"]: p for p in m["profiles"]}
    return {"cluster": cl, "persona": prof[cl]["persona"],
            "blocks": [KPIS(K("Segment", prof[cl]["persona"], hint=f"cluster {cl} · {prof[cl]['n']} similar customers"), K("Soft assignment", f"{100 * conf[cl]:.0f}%", hint="RBF weight on centroid distance")),
                       CALLOUT("good", "Recommended action", prof[cl]["action"]),
                       CHART("hbar", "Distance to each segment centroid (standardised units)", [S("distance", d.tolist(), colors=["3" if i == cl else "--axis" for i in range(len(d))])], [prof[i]["persona"] for i in range(len(d))], height=220)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)
