"""Project 06 — Anomaly Threat Intelligence: live transaction scorer + detector leaderboard + DS admin dashboard."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
sys.modules.setdefault("pipeline", pipeline)  # so joblib can un-pickle pipeline.Detectors / pipeline.HBOS
_s: dict = {}


def M() -> dict:
    if "m" not in _s:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _s["m"]


def model() -> dict:
    if "model" not in _s:
        import __main__
        for cls in ("Detectors", "HBOS"):  # artifacts pickled when pipeline.py ran as a script reference __main__.<cls>
            if not hasattr(__main__, cls):
                setattr(__main__, cls, getattr(pipeline, cls))
        _s["model"] = load_or_train(ROOT / "artifacts" / "model.joblib", pipeline.main)
        _s["demo"] = np.load(ROOT / "data" / "demo_transactions.npz")
    return _s["model"]


def score_rows(X: np.ndarray) -> dict[str, np.ndarray]:
    mm = model()
    raw = mm["det"].scores(X)
    pct = {k: np.searchsorted(mm["ref"][k], v) / len(mm["ref"][k]) for k, v in raw.items()}
    pct["Rank ensemble (top-3)"] = np.mean([pct[k] for k in mm["members"]], axis=0)
    pct["_champion_raw"] = raw[mm["champion"]]
    pct["_xgb"] = mm["xgb"].predict_proba(X)[:, 1]
    return pct


class Tx(BaseModel):
    index: int = Field(0, ge=0, le=199)


def p_scorer() -> list[dict]:
    m = M()
    model()
    y = _s["demo"]["y"]
    op = m["operating_point"]
    return [H("Threat scorer", "200 real transactions held out from training. Pick one: every detector scores it, and the page explains which features look abnormal."),
            KPIS(K("Champion detector", m["champion"], hint="chosen on validation, no labels used to train"), K("Test PR-AUC", next(r for r in m["leaderboard"] if r["model"] == m["champion"])["test"]["pr_auc"], digits=3, hint=f"random = {m['rate']:.4f}"),
                 K("Recall at 0.5% alert budget", f"{100 * op['recall']:.0f}%", hint=f"precision {100 * op['precision']:.0f}%"), K("Fraud prevalence", f"{100 * m['rate']:.2f}%")),
            FORM("api/score", [F("index", "Transaction", "select", int(np.where(y == 1)[0][0]), options=[[i, f"#{i:03d}"] for i in range(len(y))])], title="Score one transaction", submit="Score", auto=True, run_on_load=True),
            FORM("api/scan", [], title="Scan all 200 demo transactions (ground truth revealed afterwards)", submit="Run scan")]


def p_board() -> list[dict]:
    m = M()
    lb = m["leaderboard"]
    return [H("Detector leaderboard", f"Trained on {m['split']['train']:,} unlabeled transactions; scored once on {m['split']['test']:,} test transactions containing {m['split']['test_fraud']} frauds."),
            CHART("hbar", "Test PR-AUC (average precision)", [S("PR-AUC", [r["test"]["pr_auc"] for r in lb], colors=["--axis" if r["supervised"] or "baseline" in r["model"] else "1" for r in lb])], [r["model"] for r in lb], height=300,
                  note="Grey = the supervised reference that sees fraud labels, and the random-score baseline. Blue = unsupervised detectors that never see a label during training."),
            TABLE("All metrics", ["detector", "type", "val PR-AUC", "test PR-AUC", "test ROC-AUC", "recall @ 1% FPR", "precision @ top-100"],
                  [[r["model"], PILL("supervised", "warn") if r["supervised"] else PILL("baseline") if "baseline" in r["model"] else PILL("unsupervised", "info"), r["val"]["pr_auc"], r["test"]["pr_auc"], r["test"]["roc_auc"], r["test"]["recall_at_1pct_fpr"], r["test"]["precision_at_100"]] for r in lb], digits=4),
            CALLOUT("info", "Why ROC-AUC flatters everyone here", "Six of seven detectors exceed 0.94 ROC-AUC, yet their PR-AUC ranges from 0.08 to 0.62. With 0.17% positives, a 1% false-positive rate is still ~570 false alarms per 99 frauds — only the precision–recall view shows that."),
            MD("## AutoResearch\nA hill-climb over isolation-forest size, PCA rank and ensemble membership, scored by **validation** PR-AUC on a 40k training subsample.", card=False),
            *ar_blocks(m["autoresearch"], "val PR-AUC")]


def p_admin() -> list[dict]:
    m = M()
    op, sh, c = m["operating_point"], m["score_hist"], m["curves"]
    mids = [f"{(a + b) / 2:.1f}" for a, b in zip(sh["edges"], sh["edges"][1:])]
    fs = m["feature_shift"][:12]
    return [H("Data-science admin dashboard"),
            GRID(2, CHART("scatter", "Precision–recall (test)", [S(k, [{"x": a, "y": b} for a, b in v["pr"]], line=True) for k, v in c.items()], x="recall", y="precision", height=300),
                 CHART("scatter", "ROC (test)", [S(k, [{"x": a, "y": b} for a, b in v["roc"]], line=True) for k, v in c.items()], x="false-positive rate", y="true-positive rate", height=300, logx=False)),
            GRID(2, CHART("bar", f"{m['champion']} score — normal transactions", [S("normal", sh["normal"])], mids, x="anomaly score", y="count (log)", logy=True, height=240),
                 CHART("bar", "…and frauds (same axis)", [S("fraud", sh["fraud"], color="8")], mids, x="anomaly score", y="count", height=240)),
            GRID(2, HEAT(f"Confusion matrix at the 0.5% alert budget (threshold {op['threshold']:.2f}, set on validation)", ["flagged", "passed"], ["fraud", "normal"], [[op["tp"], op["fn"]], [op["fp"], op["tn"]]], digits=0, labw=60,
                         sub=f"Analysts review {100 * op['alert_rate']:.2f}% of traffic and catch {100 * op['recall']:.0f}% of fraud; {100 * op['precision']:.0f}% of alerts are real."),
                 CHART("hbar", "Which features separate fraud? |Δ mean| in σ units", [S("shift", [v for _, v in fs], color="7")], [k for k, _ in fs], height=300)),
            CHART("scatter", "PCA view of the test set", [S("normal (sample)", [{"x": a, "y": b} for a, b in m["pca_scatter"]["normal"]], r=2, color="--axis"), S("fraud", [{"x": a, "y": b} for a, b in m["pca_scatter"]["fraud"]], r=4, color="8")], x="PC1", y="PC2", height=320),
            MD(f"""- **Data**: {m['n']:,} card transactions, {m['n_fraud']} frauds ({100 * m['rate']:.3f}%). Features V1–V28 are anonymised PCA components plus Amount.
- **Split**: stratified 60/20/20. No timestamp is available in this copy, so the split is random — in production use a time-based split.
- **Label hygiene**: detectors are fit on unlabeled training rows. Validation labels pick the champion, ensemble members and threshold. Test labels are used once.
- **Why the ensemble loses to its best member**: rank-averaging helps when members are comparably good; here Mahalanobis is far ahead, so averaging dilutes it. Reported as measured.
- **Runtime**: {m['secs']} s on 2 CPU cores.""", title="Method notes")]


CRISPDM = [
    ("Business understanding", "Card fraud is rare (0.17%) and labels arrive late. The fraud desk can review a fixed share of transactions, so success = fraud caught **within an alert budget**, not accuracy."),
    ("Data understanding", "284,807 European card transactions over two days; 492 frauds. Features are PCA-anonymised. Fraud shifts a handful of components (V14, V17, V12…) by several standard deviations."),
    ("Data preparation", "Stratified 60/20/20 split; StandardScaler fit on training rows only; a 20k subsample for the O(n²) detectors (LOF, MCD)."),
    ("Modeling", "Isolation Forest, PCA reconstruction error, robust Mahalanobis distance (MCD), HBOS, Local Outlier Factor, an MLP autoencoder, a rank ensemble — plus supervised XGBoost as an upper-bound reference."),
    ("Evaluation", "PR-AUC as the headline (ROC-AUC shown to demonstrate why it misleads), recall at 1% FPR, precision@100, and a concrete operating point chosen on validation and reported on test."),
    ("Deployment", "`POST api/score` returns every detector's percentile score, the verdict at the calibrated threshold and the most abnormal features for one transaction."),
]

app = make_app(__file__, "06", "Anomaly Threat Intelligence", "Credit-card fraud · 6 unsupervised detectors vs a supervised reference · PR-AUC first", [
    ("scorer", "Threat scorer", "App", p_scorer),
    ("leaderboard", "Leaderboard & AutoResearch", "Data science", p_board),
    ("admin", "Admin dashboard", "Data science", p_admin),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/score")
def score(t: Tx) -> dict:
    mm = model()
    x, truth = _s["demo"]["X"][[t.index]], int(_s["demo"]["y"][t.index])
    sc = score_rows(x)
    flagged = bool(sc["_champion_raw"][0] >= mm["threshold"])
    z = mm["det"].scaler.transform(x)[0]
    top = np.argsort(-np.abs(z))[:8]
    names = [k for k in sc if not k.startswith("_")]
    return {"flagged": flagged, "truth": truth, "blocks": [
        KPIS(K("Verdict", "⚠ FLAGGED" if flagged else "✓ passed", hint=f"{mm['champion']} vs threshold {mm['threshold']:.2f}"), K("Champion score", float(sc["_champion_raw"][0]), digits=2),
             K("Supervised reference P(fraud)", f"{100 * float(sc['_xgb'][0]):.1f}%"), K("Ground truth", "FRAUD" if truth else "legitimate", hint="hidden from every unsupervised detector")),
        GRID(2, CHART("hbar", "Anomaly percentile by detector (1.0 = more extreme than all training data)", [S("percentile", [float(sc[k][0]) for k in names], colors=["8" if sc[k][0] > 0.995 else "4" if sc[k][0] > 0.95 else "1" for k in names])], names, xmin=0, xmax=1, height=270),
             CHART("hbar", "Most abnormal features (z-score vs training data)", [S("z", [float(z[j]) for j in top], colors=["2" if z[j] > 0 else "1" for j in top])], [pipeline.FEATS[j] for j in top], height=270))]}


@app.post("/api/scan")
def scan() -> dict:
    mm = model()
    X, y = _s["demo"]["X"], _s["demo"]["y"]
    sc = score_rows(X)
    raw = sc["_champion_raw"]
    flag = raw >= mm["threshold"]
    order = np.argsort(-raw)
    tp, fp, fn = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())
    return {"blocks": [KPIS(K("Transactions scanned", len(y)), K("Alerts raised", int(flag.sum())), K("Frauds caught", f"{tp} / {int(y.sum())}"), K("False alarms", fp), K("Missed frauds", fn)),
                       CALLOUT("info", "Note on this sample", "The demo set is enriched to 20% fraud so there is something to see; real prevalence is 0.17%, where the same threshold gives the precision shown on the admin page."),
                       TABLE("Ranked by anomaly score", ["rank", "transaction", "score", "decision", "truth", "outcome"],
                             [[i + 1, f"#{j:03d}", round(float(raw[j]), 2), PILL("flag", "crit") if flag[j] else PILL("pass"), "fraud" if y[j] else "legit",
                               PILL("caught", "good") if flag[j] and y[j] else PILL("false alarm", "warn") if flag[j] else PILL("missed", "crit") if y[j] else ""] for i, j in enumerate(order[:60])], maxh=420)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8006)
