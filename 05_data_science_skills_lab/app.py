"""Project 05 — Data Science Skills Mastery Lab: 28 executable skills, each rendered as a visual dashboard."""
from __future__ import annotations
import sys
import time
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

skills = load_local(__file__, "skills")
SK = skills.SKILLS
PHASES = ["Data understanding", "Data preparation", "Modeling", "Evaluation", "Analytics"]


class Run(BaseModel):
    skill: str


def execute(sid: str) -> dict:
    if sid not in SK:
        raise HTTPException(404, f"unknown skill {sid}")
    s, t0 = SK[sid], time.time()
    blocks = s.run()  # always executed live — nothing is cached
    ms = (time.time() - t0) * 1000
    return {"skill": sid, "ms": ms, "blocks": [MD(f"### {s.name}\n{s.blurb}\n\n`dataset: {s.dataset}` · `phase: {s.phase}` · executed live in **{ms:.0f} ms**", card=False), *blocks]}


def p_lab() -> list[dict]:
    return [H("Skills lab", f"{len(SK)} data-science skills. Pick one and it executes live on a real dataset — the result is a dashboard, never raw JSON."),
            FORM("api/run", [F("skill", "Skill", "select", "leakage", options=[[s.id, f"{s.phase} — {s.name}"] for p in PHASES for s in SK.values() if s.phase == p], wide=True)],
                 submit="Execute skill live", auto=True, run_on_load=True)]


def p_catalog() -> list[dict]:
    return [H("Skill catalog", "Every skill, the dataset it runs on, and the CRISP-DM phase it belongs to."),
            KPIS(*[K(p, sum(1 for s in SK.values() if s.phase == p), hint="skills") for p in PHASES]),
            TABLE(None, ["phase", "skill", "dataset", "what it demonstrates"], [[PILL(s.phase, "info"), f"**{s.name}**", s.dataset, s.blurb] for p in PHASES for s in SK.values() if s.phase == p])]


def p_bench() -> list[dict]:
    return [H("Run-all benchmark", "Executes every skill end-to-end and reports status and latency — the lab's own regression test."),
            FORM("api/run_all", [], submit="Run all skills now", title="Benchmark")]


CRISPDM = [
    ("Business understanding", "Skills are useless without a question. Each skill card states the decision it informs (e.g. *which threshold minimises cost?*)."),
    ("Data understanding", "Profiling, missingness mechanisms (MCAR vs MAR), outlier rules, collinearity (VIF) and normality checks."),
    ("Data preparation", "Imputation, encoding (incl. leak-free target encoding), scaling, feature selection, PCA and class-imbalance handling."),
    ("Modeling", "Baselines first, regularisation paths, randomised search, CV strategy for grouped data, learning curves and stacking."),
    ("Evaluation", "ROC/PR, calibration, cost-sensitive thresholds, permutation importance, partial dependence — and a leakage demonstration on pure noise."),
    ("Deployment", "Every skill is an API call: `POST api/run {\"skill\": \"calibration\"}` returns the rendered analysis, so skills can be embedded in other tools."),
]

app = make_app(__file__, "05", "DS Skills Mastery Lab", "28 executable data-science skills on real datasets, rendered visually", [
    ("lab", "Skills lab", "App", p_lab),
    ("catalog", "Skill catalog", "App", p_catalog),
    ("bench", "Run-all benchmark", "App", p_bench),
    ("crispdm", "CRISP-DM map", "Docs", lambda: [H("Skills mapped to CRISP-DM"), CRISP(CRISPDM)]),
])


@app.post("/api/run")
def run(r: Run) -> dict:
    return execute(r.skill)


@app.post("/api/run_all")
def run_all() -> dict:
    rows, ok = [], 0
    for s in SK.values():
        t0 = time.time()
        try:
            n = len(s.run())
            ok += 1
            rows.append([s.name, s.phase, PILL("pass", "good"), round((time.time() - t0) * 1000), n])
        except Exception as e:
            rows.append([s.name, s.phase, PILL(f"fail: {type(e).__name__}", "crit"), round((time.time() - t0) * 1000), 0])
    tot = sum(x[3] for x in rows)
    return {"passed": ok, "total": len(rows), "blocks": [KPIS(K("Skills passed", f"{ok} / {len(rows)}"), K("Total runtime", f"{tot / 1000:.1f} s"), K("Slowest", max(rows, key=lambda x: x[3])[0])),
                                                           CHART("hbar", "Latency per skill (ms)", [S("ms", [x[3] for x in rows])], [x[0] for x in rows], height=620),
                                                           TABLE("Results", ["skill", "phase", "status", "ms", "blocks rendered"], rows)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
