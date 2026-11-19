"""Project 09 — FlowForge DAG Engine: a visual, type-checked pipeline builder for data-science workflows."""
from __future__ import annotations
import inspect
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

E = load_local(__file__, "engine")
engine = E.Engine()
STATUS_KIND = {"ok": "good", "cached": "info", "failed": "crit", "skipped": "warn"}


def spec_json() -> dict:
    return {k: {"kind": s.kind, "label": s.label, "inputs": list(s.inputs), "output": s.output, "variadic": s.variadic, "params": [asdict(p) for p in s.params]} for k, s in E.SPECS.items()}


def result_blocks(r: dict[str, Any], ran: bool) -> list[dict]:
    b: list[dict] = []
    for e in r["errors"]:
        b.append(CALLOUT("crit", "Validation error — nothing was executed", e))
    for w in r["warnings"]:
        b.append(CALLOUT("warn", "Lint warning", w))
    if r["errors"]:
        return b
    if not ran:
        return b + [CALLOUT("good", "Graph is valid", f"{len(r['levels'])} execution levels: " + " → ".join("[" + ", ".join(lv) + "]" for lv in r["levels"]))]
    info = r["nodes"]
    n_c, n_f = sum(i["status"] == "cached" for i in info.values()), sum(i["status"] == "failed" for i in info.values())
    b.append(KPIS(K("Run status", "success" if r["ok"] else "partial failure"), K("Wall time", f"{r['total_ms']:.0f} ms"), K("Nodes", len(info)), K("Parallel levels", len(r["levels"])),
                  K("Cache hits", n_c, hint="unchanged nodes are not re-run"), K("Failed / skipped", f"{n_f} / {sum(i['status'] == 'skipped' for i in info.values())}")))
    reports = [v for k, v in r.get("outputs", {}).items()]
    final = max(reports, key=lambda rep: len(rep.rows)) if reports else None
    order = [n for lv in r["levels"] for n in lv]
    left = TABLE("Execution trace (topological order)", ["level", "node", "status", "ms", "output / error"],
                 [[li + 1, n, PILL(info[n]["status"], STATUS_KIND[info[n]["status"]]), round(info[n]["ms"], 1), info[n].get("error") or info[n].get("summary", "")] for li, lv in enumerate(r["levels"]) for n in lv], maxh=360)
    if final:
        right = TABLE("Final report", ["model", "accuracy", "F1", "ROC-AUC", "test rows", "fit ms"], [[x["model"], x["accuracy"], x["f1"], x["roc_auc"], x["n_test"], x["fit_ms"]] for x in final.rows], digits=4)
        b.append(GRID(2, left, right))
    else:
        b.append(left)
    b.append(CHART("hbar", "Time per node (ms)", [S("ms", [round(info[n]["ms"], 1) for n in order], colors=[{"ok": "--good", "cached": "1", "failed": "--crit", "skipped": "--warn"}[info[n]["status"]] for n in order])], order, height=max(180, 26 * len(order))))
    return b


def p_builder() -> list[dict]:
    return [H("Pipeline builder", "Each box is a typed operation. Edges are type-checked, the graph is scheduled with Kahn's algorithm, independent branches run in parallel, and unchanged nodes are served from a content-addressed cache."),
            CUSTOM("dag", {"specs": spec_json(), "templates": E.TEMPLATES})]


def p_catalog() -> list[dict]:
    return [H("Node catalog", "The registry the validator checks edges against."),
            TABLE(None, ["kind", "label", "inputs", "output", "parameters", "notes"],
                  [[{"code": s.kind}, s.label, ", ".join(s.inputs) + (" …" if s.variadic else "") or "—", PILL(s.output, "info"), ", ".join(f"{p.name}={p.default}" for p in s.params) or "—",
                    "fits statistics → must not precede a split" if s.learns_from_data else ""] for s in E.SPECS.values()])]


def p_runs() -> list[dict]:
    rs = engine.runs[-40:]
    if not rs:
        return [H("Run history"), CALLOUT("info", "No runs yet", "Run a pipeline in the builder.")]
    return [H("Run history", "In-memory log of this server session."),
            KPIS(K("Runs", len(engine.runs)), K("Cached results held", len(engine.cache)), K("Success rate", f"{100 * sum(r['ok'] for r in engine.runs) / len(engine.runs):.0f}%")),
            CHART("bar", "Wall time per run (ms)", [S("ms", [round(r["ms"], 1) for r in rs])], [r["at"] for r in rs], height=220),
            TABLE(None, ["time", "nodes", "status", "ms", "cache hits", "failed"], [[r["at"], r["nodes"], PILL("ok", "good") if r["ok"] else PILL("failed", "crit"), round(r["ms"], 1), r["cached"], r["failed"]] for r in reversed(rs)])]


DESIGN = """
## Why a DAG engine?
Real data-science work is a graph, not a script: one dataset fans out into several models whose reports fan back in.
Encoding that graph explicitly buys **parallelism**, **caching**, **failure isolation** and — the part most tools skip — **static checks before anything runs**.

## Type discipline (the “Matt Pocock skills”, ported to Python)
| TypeScript idea | Here |
|---|---|
| Branded types | `NodeId = NewType("NodeId", str)` |
| Discriminated unions | `Result = Ok[T] | Err`, narrowed on `.ok` |
| Literal unions, exhaustive | `PortType = Literal["Frame","Split","Model","Report"]` |
| Declarative, typed config | `NodeSpec(kind, inputs, output, params, run)` registry |
| `tsc --strict` | `mypy --strict engine.py` → *no issues found* |

## Scheduling
Kahn's algorithm peels off zero-in-degree nodes level by level. Leftover nodes ⇒ a cycle (reported by name). Every level is submitted to a thread pool.

## Caching
`key(node) = sha1(kind, params, sorted(parent keys))` — a Merkle hash. Change one parameter and only that node and its descendants re-run.

## Leakage lint
Nodes declare `learns_from_data`. If such a node can reach a `split`, the validator warns that test rows influenced a fitted statistic. Try the *Leaky pipeline* template.

## Failure isolation
A raising node is marked `failed`; its descendants are `skipped`; unrelated branches still finish. Try the *failure isolation* template.
"""

app = make_app(__file__, "09", "FlowForge DAG Engine", "Typed pipeline graphs · Kahn scheduling · parallel execution · Merkle caching · leakage lint", [
    ("builder", "Pipeline builder", "App", p_builder),
    ("catalog", "Node catalog", "App", p_catalog),
    ("runs", "Run history", "App", p_runs),
    ("design", "Design & types", "Docs", lambda: [H("Design & type system"), MD(DESIGN), CODE("Kahn's algorithm — engine.py", inspect.getsource(E.kahn)), CODE("Static validation + leakage lint — engine.py", inspect.getsource(E.validate))]),
])


@app.post("/api/validate")
def validate(g: dict[str, Any]) -> dict:
    graph = E.graph_from_json(g)
    errors, warnings = E.validate(graph)
    k = E.kahn(graph)
    r = {"errors": errors, "warnings": warnings, "levels": k.value if k.ok else [], "nodes": {}}
    return {"ok": not errors, "errors": errors, "warnings": warnings, "nodes": {}, "blocks": result_blocks(r, ran=False)}


@app.post("/api/run")
def run(g: dict[str, Any]) -> dict:
    r = engine.run(E.graph_from_json(g))
    return {"ok": r["ok"], "errors": r["errors"], "warnings": r["warnings"], "nodes": r["nodes"], "levels": r["levels"], "total_ms": r["total_ms"], "blocks": result_blocks(r, ran=True)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8009)
