"""Project 11 — Enterprise DS Audit: a static (AST) audit of every other project in this repository, run live.

Nothing on these pages is hard-coded: each page load re-parses the current source files with common/audit.py, so the
report cannot drift from the code. Findings that were reviewed and accepted carry an inline `# audit-ok: reason` waiver
in the source, and are listed — with their reasons — rather than hidden.
"""
from __future__ import annotations
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403
from common.audit import DIMENSIONS, PENALTY, audit_project, audit_source  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# What the auditor is told about each project: which files hold the data-science logic, and which special rules apply.
CONFIG: dict[str, dict] = {
    "01": dict(files=["pipeline.py"], forbidden=("dropoff_datetime", "trip_duration", "dropoff")),
    "02": dict(files=["pipeline.py"]),
    "03": dict(files=["pipeline.py"], supervised=False),
    "04": dict(files=["pipeline.py"]),
    "05": dict(files=["skills.py"]),
    "06": dict(files=["pipeline.py"], imbalanced=True),
    "07": dict(files=["pipeline.py"], forbidden=("tip", "tolls", "total", "dropoff")),
    "09": dict(files=["engine.py"]),
    "10": dict(files=["pipeline.py"]),
    "12": dict(files=["pipeline.py"], temporal=True),
    "13": dict(files=["pipeline.py"], temporal=True, forbidden=("tip", "tolls", "total", "dropoff", "payment", "fare")),
    "14": dict(files=["pipeline.py"]),
    "15": dict(files=["pipeline.py"], temporal=True),
}
NOT_AUDITED = {"00": "CRUD application — no modelling code", "08": "client-side teaching simulators — no training or evaluation", "11": "this auditor"}
SEV_KIND = {"fail": "crit", "warn": "warn", "info": "info", "waived": ""}


def proj_dir(nn: str) -> Path:
    return next(p for p in REPO.iterdir() if p.is_dir() and p.name.startswith(nn + "_"))


def run_all() -> dict[str, dict]:
    out = {}
    for nn, cfg in CONFIG.items():
        c = dict(cfg)
        d = proj_dir(nn)
        r = audit_project(d, c.pop("files"), **c)
        r["dir"], r["flags"] = d.name, ", ".join(k if v is True else f"{k}={len(v)} cols" for k, v in c.items() if v) or "default"
        out[nn] = r
    return out


def all_findings(res: dict[str, dict]) -> list[dict]:
    return [{**f, "project": nn, "file": fl["path"]} for nn, r in res.items() for fl in r["files"] for f in fl["findings"]]


class Filter(BaseModel):
    project: str = "all"
    severity: str = "all"


class Snippet(BaseModel):
    code: str = Field(max_length=20000)
    temporal: bool = False
    imbalanced: bool = False


def p_scorecard() -> list[dict]:
    res = run_all()
    F_ = all_findings(res)
    n = {s: sum(f["severity"] == s for f in F_) for s in ["fail", "warn", "info", "waived"]}
    overall = sum(r["overall"] for r in res.values()) / len(res)
    return [H("Portfolio audit scorecard", f"{len(res)} projects, {sum(r['loc'] for r in res.values()):,} lines of data-science code, re-audited on every page load."),
            KPIS(K("Portfolio score", overall, digits=1, hint="mean of project scores"), K("Checks passed", sum(r["checks_passed"] for r in res.values())), K("Failures", n["fail"]), K("Warnings", n["warn"]), K("Info notes", n["info"]), K("Reviewed & waived", n["waived"], hint="accepted with a written reason")),
            HEAT("Score by project × audit dimension (100 = no open finding)", DIMENSIONS, [f"{nn} {r['dir'][3:].replace('_', ' ')[:30]}" for nn, r in res.items()], [[r["dimensions"][d] for d in DIMENSIONS] for r in res.values()], digits=0, min=50, max=100, labw=250),
            TABLE("Per-project summary", ["#", "project", "grade", "score", "LOC", "passed", "fail", "warn", "info", "waived", "rule set"],
                  [[nn, r["dir"][3:], PILL(r["grade"], "good" if r["grade"].startswith("A") else "warn"), round(r["overall"], 1), r["loc"], r["checks_passed"], r["counts"]["fail"], r["counts"]["warn"], r["counts"]["info"], r["counts"]["waived"], r["flags"]] for nn, r in res.items()]),
            TABLE("Not in scope", ["#", "reason"], [[k, v] for k, v in NOT_AUDITED.items()]),
            CALLOUT("info", "How to read the grade", "Each dimension starts at 100 and loses 25 per failure, 10 per warning, 3 per info note. Waived findings cost nothing but stay visible. A static audit raises questions for a human; it does not prove the absence of leakage — Project 15 adds *executable* leakage tests for that reason.")]


def p_findings() -> list[dict]:
    return [H("Findings explorer", "Every open finding and every waiver, with file, line and the offending code."),
            FORM("api/findings", [F("project", "Project", "select", "all", options=["all"] + list(CONFIG)), F("severity", "Severity", "select", "all", options=["all", "fail", "warn", "info", "waived"])], submit="Filter", auto=True, run_on_load=True)]


def p_evidence() -> list[dict]:
    res = run_all()
    rows = [[nn, {"code": p["rule"]}, p["dimension"], p["message"]] for nn, r in res.items() for fl in r["files"] for p in fl["passes"]]
    by_rule: dict[str, int] = {}
    for r_ in rows:
        by_rule[r_[1]["code"]] = by_rule.get(r_[1]["code"], 0) + 1
    return [H("Positive evidence", "What the auditor verified, not just what it flagged."),
            CHART("hbar", "Projects passing each rule", [S("projects", list(by_rule.values()), color="3")], list(by_rule), height=360), TABLE(None, ["project", "rule", "dimension", "evidence"], rows, maxh=520)]


RULES = [("LEAK-FIT", "Leakage", "warn", "A scaler / imputer / encoder / PCA / selector is `.fit()` on a variable that is not recognisably the training split, in a file that also evaluates on held-out data."),
         ("LEAK-TARGET", "Leakage", "fail", "A feature-building function mentions a column that only exists after the outcome is known (configured per project)."),
         ("EVAL-TEST-IN-SEARCH", "Evaluation integrity", "fail", "The objective passed to a hyper-parameter search reads a test-set variable."),
         ("EVAL-HOLDOUT", "Evaluation integrity", "warn", "No split or cross-validation anywhere in a supervised pipeline."),
         ("EVAL-TEMPORAL", "Evaluation integrity", "fail", "Random `train_test_split` or shuffled K-fold in a project declared as time-ordered."),
         ("EVAL-BASELINE", "Evaluation integrity", "warn", "No naive baseline is computed next to the models."),
         ("REPRO-SEED", "Reproducibility", "warn", "A stochastic estimator or splitter is created without `random_state`."),
         ("REPRO-GLOBAL", "Reproducibility", "info", "No named seed constant."),
         ("METRIC-IMBALANCE", "Metric alignment", "fail", "A project declared imbalanced computes no PR-AUC / F1 / ROC-AUC."),
         ("METRIC-ACC-ONLY", "Metric alignment", "info", "Accuracy is the only classification metric."),
         ("HYG-*", "Code hygiene", "info–fail", "Bare or silent `except`, `eval`/`exec`, blanket `warnings.filterwarnings('ignore')`."),
         ("DOC-*", "Documentation", "info–warn", "Missing module docstring; few documented public functions.")]

LEAKY = '''import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")
X, y = df.drop(columns="y"), df.y
X = StandardScaler().fit_transform(X)            # scaled before the split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

def evaluate(params):
    m = RandomForestClassifier(**params).fit(X_train, y_train)
    return accuracy_score(y_test, m.predict(X_test))   # tuning on the test set

best = max(({"n_estimators": n} for n in (50, 100, 200)), key=evaluate)
search = GridSearchCV(RandomForestClassifier(), {"max_depth": [3, 5]}, scoring=evaluate)
try:
    print(accuracy_score(y_test, RandomForestClassifier(**best).fit(X_train, y_train).predict(X_test)))
except:
    pass
'''


def p_method() -> list[dict]:
    return [H("Audit method", "A dozen AST rules aimed at the usual ways a result gets inflated."),
            TABLE("Rule catalogue", ["rule", "dimension", "severity", "what triggers it"], [[{"code": a}, b, c, d] for a, b, c, d in RULES]),
            MD(f"""**Scoring.** Per dimension: 100 − {PENALTY['fail']}·fails − {PENALTY['warn']}·warnings − {PENALTY['info']}·infos (floor 0). Project score = mean of the six dimensions.

**Waivers.** A reviewer who judges a finding acceptable adds `# audit-ok: <reason>` on that line. The finding then costs no points but is listed with its reason. Example reasons in this repository: *"unsupervised EDA projection, no label involved"*, *"deliberate leakage demonstration"*.

**Limits of static analysis.** The auditor reads names and structure; it cannot follow data through arbitrary code. It will miss leakage hidden behind indirection and will sometimes flag safe code (hence waivers). It complements, not replaces: (1) the executable leakage tests in Project 15, (2) the pipeline-inside-CV discipline used throughout, and (3) review by a person.""", title="Scoring, waivers and limits"),
            MD("""The first full run of this auditor over the repository reported **0 failures and 18 warnings**. Each was reviewed:

| Outcome | Warnings | Examples |
|---|---|---|
| **Fixed in code** | 3 | unseeded `KMeans` refit (03) · no naive baseline in the anomaly leaderboard → random-score baseline added (06) · no baseline model in FlowForge → `baseline_prior` option added (09) |
| **Auditor was wrong → rule improved** | 4 | manual and chronological hold-outs (`idx[:n]`, `folds()`) were not recognised as splits (02, 04) · seeds passed through `**kwargs` were not visible (01) |
| **Waived with a written reason** | 11 | unsupervised projections used only for plots · detector classes whose `fit(X)` receives the training split · the *deliberate* leakage demo in the Skills Lab · FlowForge's intentionally leaky `impute` node |

Two info-level notes were also waived (optional metadata lookups wrapped in `try/except`), for 13 waivers in total, and one info note was fixed (a named `SEED` in Project 12). Remaining open items are info-level (blanket warning filters, undocumented `main()` functions) and are left visible on purpose.""", title="Audit log — what the first run found and what was done"),
            CODE("The LEAK-FIT rule, as implemented (common/audit.py)", "\n".join((REPO / "common" / "audit.py").read_text().split("# ---- R-LEAK-1")[1].split("# ---- R-LEAK-2")[0].splitlines()[:16]))]


def p_live() -> list[dict]:
    return [H("Audit your own code", "Paste any training script. The same rules run on it instantly — nothing is executed, only parsed. The default snippet contains five classic mistakes."),
            FORM("api/audit", [F("code", "Python source", "textarea", LEAKY, rows=18), F("temporal", "Data is time-ordered", "checkbox", False), F("imbalanced", "Target is imbalanced", "checkbox", True)], submit="Audit this code", run_on_load=True)]


def findings_table(fs: list[dict], with_project: bool = True) -> dict:
    cols = (["project", "file"] if with_project else []) + ["severity", "rule", "dimension", "line", "finding", "code / waiver reason"]
    rows = [([f["project"], f["file"]] if with_project else []) + [PILL(f["severity"], SEV_KIND[f["severity"]]), {"code": f["rule"]}, f["dimension"], f["line"], f["message"], ("waived — " + f["waiver"]) if f["waiver"] else {"code": f["code"]}] for f in fs]
    return TABLE(f"{len(rows)} finding(s)", cols, rows or [["—"] * len(cols)], maxh=560)


app = make_app(__file__, "11", "Enterprise DS Audit", "Live static audit of every project: leakage · evaluation integrity · reproducibility · metrics", [
    ("scorecard", "Scorecard", "Audit", p_scorecard), ("findings", "Findings explorer", "Audit", p_findings), ("evidence", "Positive evidence", "Audit", p_evidence), ("live", "Audit your own code", "Audit", p_live),
    ("method", "Audit method", "Docs", p_method),
])


@app.post("/api/findings")
def findings(f: Filter) -> dict:
    fs = [x for x in all_findings(run_all()) if (f.project == "all" or x["project"] == f.project) and (f.severity == "all" or x["severity"] == f.severity)]
    order = {"fail": 0, "warn": 1, "info": 2, "waived": 3}
    return {"count": len(fs), "blocks": [findings_table(sorted(fs, key=lambda x: (order[x["severity"]], x["project"], x["line"])))]}


@app.post("/api/audit")
def audit(s: Snippet) -> dict:
    from common.audit import score
    rep = audit_source(s.code, "<pasted>", temporal=s.temporal, imbalanced=s.imbalanced)
    sc = score([rep])
    fs = [{**f.__dict__} for f in rep.findings]
    return {"grade": sc["grade"], "blocks": [KPIS(K("Grade", sc["grade"]), K("Score", sc["overall"], digits=1), K("Failures", sc["counts"]["fail"]), K("Warnings", sc["counts"]["warn"]), K("Checks passed", sc["checks_passed"])),
                                             GRID(2, CHART("hbar", "Score by dimension", [S("score", list(sc["dimensions"].values()), colors=["8" if v < 70 else "4" if v < 90 else "3" for v in sc["dimensions"].values()])], list(sc["dimensions"]), xmin=0, xmax=100, height=230),
                                                  TABLE("Passed", ["rule", "evidence"], [[{"code": p.rule}, p.message] for p in rep.passes] or [["—", "nothing passed"]])), findings_table(fs, with_project=False)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8011)
