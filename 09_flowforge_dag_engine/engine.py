"""FlowForge — a strictly-typed DAG engine for data-science pipelines.

The reference project demonstrated Matt Pocock-style TypeScript discipline. This port keeps the same ideas in typed Python
(checked with `mypy --strict`):

  * branded identifiers      NodeId = NewType(...)            -> ids cannot be confused with arbitrary strings
  * discriminated unions     Ok[T] | Err, matched on `.ok`    -> errors are values, not exceptions
  * exhaustive literal types PortType / Status = Literal[...] -> the checker rejects unknown states
  * a declarative registry   NodeSpec: typed ports + params   -> edges are type-checked before anything runs

Scheduling is Kahn's algorithm: it yields a topological order, detects cycles, and groups nodes into levels whose members
have no mutual dependencies and can therefore execute in parallel.
"""
from __future__ import annotations
import hashlib
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Literal, NewType, TypeVar, Union

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NodeId = NewType("NodeId", str)
PortType = Literal["Frame", "Split", "Model", "Report"]
Status = Literal["pending", "ok", "cached", "failed", "skipped"]
T = TypeVar("T")
DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T
    ok: Literal[True] = True


@dataclass(frozen=True)
class Err:
    error: str
    ok: Literal[False] = False


Result = Union[Ok[T], Err]


@dataclass(frozen=True)
class Param:
    name: str
    kind: Literal["text", "number", "select"]
    default: Any
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class NodeSpec:
    kind: str
    label: str
    inputs: tuple[PortType, ...]
    output: PortType
    params: tuple[Param, ...]
    run: Callable[[list[Any], dict[str, Any]], Any]
    variadic: bool = False          # accepts 1..n inputs of inputs[0]
    learns_from_data: bool = False  # fits statistics on its input (matters for the leakage lint)


@dataclass
class Node:
    id: NodeId
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class Edge:
    src: NodeId
    dst: NodeId


@dataclass
class Graph:
    nodes: dict[NodeId, Node]
    edges: list[Edge]

    def parents(self, n: NodeId) -> list[NodeId]:
        return [e.src for e in self.edges if e.dst == n]

    def children(self, n: NodeId) -> list[NodeId]:
        return [e.dst for e in self.edges if e.src == n]


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target: str


@dataclass
class Fitted:
    name: str
    model: Any
    split: Split
    fit_ms: float


@dataclass
class Report:
    rows: list[dict[str, Any]]


# ----------------------------------------------------------------------------- node implementations
def _source(_: list[Any], p: dict[str, Any]) -> pd.DataFrame:
    if p["dataset"] == "titanic":
        return pd.read_csv(DATA / "titanic.csv")[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    from sklearn import datasets
    d = datasets.load_breast_cancer(as_frame=True) if p["dataset"] == "breast_cancer" else datasets.load_wine(as_frame=True)
    df: pd.DataFrame = d.frame
    if p["dataset"] == "wine":
        df = df.assign(target=(df.target == 0).astype(int))
    return df


def _filter(i: list[Any], p: dict[str, Any]) -> pd.DataFrame:
    df: pd.DataFrame = i[0]
    col, op, raw = str(p["column"]), str(p["op"]), p["value"]
    if col not in df.columns:
        raise ValueError(f"filter: unknown column '{col}' (have {list(df.columns)})")
    val: Any = float(raw) if df[col].dtype.kind in "if" else str(raw)
    mask = {">": df[col] > val, "<": df[col] < val, "==": df[col] == val, "!=": df[col] != val}[op]
    return df[mask]


def _select(i: list[Any], p: dict[str, Any]) -> pd.DataFrame:
    df: pd.DataFrame = i[0]
    cols = [c.strip() for c in str(p["drop"]).split(",") if c.strip()]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"drop_columns: unknown columns {missing}")
    return df.drop(columns=cols)


def _impute(i: list[Any], p: dict[str, Any]) -> pd.DataFrame:
    df: pd.DataFrame = i[0].copy()
    num = df.select_dtypes("number").columns
    df[num] = SimpleImputer(strategy=str(p["strategy"])).fit_transform(df[num])  # audit-ok: intentionally leaky node — FlowForge's own lint flags it at run time when placed before a split
    for c in df.columns.difference(num):
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def _encode(i: list[Any], _: dict[str, Any]) -> pd.DataFrame:
    return pd.get_dummies(i[0], drop_first=True).astype(float)


def _split(i: list[Any], p: dict[str, Any]) -> Split:
    df: pd.DataFrame = i[0]
    t = str(p["target"])
    if t not in df.columns:
        raise ValueError(f"split: target '{t}' not in columns")
    X, y = df.drop(columns=[t]), df[t].astype(int)
    if X.select_dtypes(exclude="number").shape[1]:
        raise ValueError("split: non-numeric feature columns remain — add an 'encode' node upstream")
    a, b, c, d = train_test_split(X, y, test_size=float(p["test_size"]), random_state=int(p["seed"]), stratify=y)
    return Split(a, b, c, d, t)


def _train(i: list[Any], p: dict[str, Any]) -> Fitted:
    s: Split = i[0]
    est = {"baseline_prior": DummyClassifier(strategy="prior"), "logistic": LogisticRegression(max_iter=2000), "random_forest": RandomForestClassifier(200, random_state=0),
           "gradient_boosting": GradientBoostingClassifier(random_state=0)}[str(p["model"])]
    # imputer + scaler live INSIDE the pipeline, so they are fit on the training split only
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    t0 = time.perf_counter()
    pipe.fit(s.X_train, s.y_train)
    return Fitted(str(p["model"]), pipe, s, (time.perf_counter() - t0) * 1000)


def _evaluate(i: list[Any], _: dict[str, Any]) -> Report:
    f: Fitted = i[0]
    proba = f.model.predict_proba(f.split.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(f.split.y_test, pred)
    return Report([{"model": f.name, "accuracy": float(accuracy_score(f.split.y_test, pred)), "f1": float(f1_score(f.split.y_test, pred)),
                    "roc_auc": float(roc_auc_score(f.split.y_test, proba)), "n_test": int(len(pred)), "fit_ms": round(f.fit_ms, 1), "confusion": cm.tolist()}])


def _compare(i: list[Any], _: dict[str, Any]) -> Report:
    rows = [r for rep in i for r in rep.rows]
    return Report(sorted(rows, key=lambda r: -float(r["roc_auc"])))


SPECS: dict[str, NodeSpec] = {s.kind: s for s in [
    NodeSpec("source", "Load dataset", (), "Frame", (Param("dataset", "select", "titanic", ("titanic", "breast_cancer", "wine")),), _source),
    NodeSpec("filter", "Filter rows", ("Frame",), "Frame", (Param("column", "text", "fare"), Param("op", "select", "<", (">", "<", "==", "!=")), Param("value", "text", "300")), _filter),
    NodeSpec("drop_columns", "Drop columns", ("Frame",), "Frame", (Param("drop", "text", "embarked"),), _select),
    NodeSpec("impute", "Impute missing", ("Frame",), "Frame", (Param("strategy", "select", "median", ("median", "mean", "most_frequent")),), _impute, learns_from_data=True),
    NodeSpec("encode", "One-hot encode", ("Frame",), "Frame", (), _encode),
    NodeSpec("split", "Train/test split", ("Frame",), "Split", (Param("target", "text", "survived"), Param("test_size", "number", 0.25), Param("seed", "number", 42)), _split),
    NodeSpec("train", "Train model", ("Split",), "Model", (Param("model", "select", "logistic", ("baseline_prior", "logistic", "random_forest", "gradient_boosting")),), _train),
    NodeSpec("evaluate", "Evaluate", ("Model",), "Report", (), _evaluate),
    NodeSpec("compare", "Compare reports", ("Report",), "Report", (), _compare, variadic=True),
]}


# ----------------------------------------------------------------------------- scheduling & validation
def kahn(g: Graph) -> Result[list[list[NodeId]]]:
    """Topological levels via Kahn's algorithm. Err if the graph has a cycle."""
    indeg = {n: 0 for n in g.nodes}
    for e in g.edges:
        indeg[e.dst] += 1
    frontier = deque(sorted(n for n, d in indeg.items() if d == 0))
    levels: list[list[NodeId]] = []
    seen = 0
    while frontier:
        level = list(frontier)
        frontier.clear()
        levels.append(level)
        seen += len(level)
        for n in level:
            for c in g.children(n):
                indeg[c] -= 1
                if indeg[c] == 0:
                    frontier.append(c)
    if seen != len(g.nodes):
        return Err("cycle detected among nodes: " + ", ".join(sorted(n for n, d in indeg.items() if d > 0)))
    return Ok(levels)


def validate(g: Graph) -> tuple[list[str], list[str]]:
    """Static checks before anything runs. Returns (errors, lint warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    for e in g.edges:
        if e.src not in g.nodes or e.dst not in g.nodes:
            errors.append(f"edge {e.src}→{e.dst} references a missing node")
    for n in g.nodes.values():
        spec = SPECS.get(n.kind)
        if spec is None:
            errors.append(f"{n.id}: unknown node kind '{n.kind}'")
            continue
        ps = [p for p in g.parents(n.id) if p in g.nodes]
        if spec.variadic:
            if not ps:
                errors.append(f"{n.id} ({spec.label}): needs at least one input")
        elif len(ps) != len(spec.inputs):
            errors.append(f"{n.id} ({spec.label}): expects {len(spec.inputs)} input(s), has {len(ps)}")
        for p in ps:
            want = spec.inputs[0] if spec.inputs else None
            got = SPECS[g.nodes[p].kind].output if g.nodes[p].kind in SPECS else None
            if want and got and want != got:
                errors.append(f"type error on {p}→{n.id}: {spec.label} expects {want} but {SPECS[g.nodes[p].kind].label} produces {got}")
    order = kahn(g)
    if not order.ok:
        errors.append(order.error)
        return errors, warnings

    # leakage lint: a node that learns statistics from its input must not sit upstream of the train/test split
    def reaches_split(n: NodeId, seen: set[NodeId]) -> bool:
        return any(g.nodes[c].kind == "split" or (c not in seen and reaches_split(c, seen | {c})) for c in g.children(n))
    for n in g.nodes.values():
        if n.kind in SPECS and SPECS[n.kind].learns_from_data and reaches_split(n.id, {n.id}):
            warnings.append(f"LEAKAGE: '{n.id}' ({SPECS[n.kind].label}) fits statistics on rows that later become the test set. Remove it — the train node already imputes inside its pipeline, on training rows only.")
    if not any(n.kind in ("evaluate", "compare") for n in g.nodes.values()):
        warnings.append("pipeline has no evaluate/compare node — nothing will be measured")
    return errors, warnings


class Engine:
    """Executes a validated graph level by level with a thread pool, content-addressed caching and failure isolation."""

    def __init__(self, workers: int = 4) -> None:
        self.cache: dict[str, Any] = {}
        self.workers = workers
        self.runs: list[dict[str, Any]] = []

    def _key(self, g: Graph, n: NodeId, keys: dict[NodeId, str]) -> str:
        node = g.nodes[n]
        payload = json.dumps([node.kind, node.params, sorted(keys[p] for p in g.parents(n))], sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    def run(self, g: Graph) -> dict[str, Any]:
        t0 = time.perf_counter()
        errors, warnings = validate(g)
        if errors:
            return {"ok": False, "errors": errors, "warnings": warnings, "nodes": {}, "levels": [], "total_ms": 0.0}
        order = kahn(g)
        assert order.ok
        levels = order.value
        keys: dict[NodeId, str] = {}
        out: dict[NodeId, Any] = {}
        info: dict[str, dict[str, Any]] = {}

        def exec_node(n: NodeId) -> None:
            parents = sorted(g.parents(n))
            if any(info[p]["status"] in ("failed", "skipped") for p in parents):
                info[n] = {"status": "skipped", "ms": 0.0, "error": "upstream failure", "key": ""}
                return
            keys[n] = self._key(g, n, keys)
            t1 = time.perf_counter()
            if keys[n] in self.cache:
                out[n] = self.cache[keys[n]]
                info[n] = {"status": "cached", "ms": (time.perf_counter() - t1) * 1000, "error": None, "key": keys[n]}
                return
            spec, node = SPECS[g.nodes[n].kind], g.nodes[n]
            params = {p.name: node.params.get(p.name, p.default) for p in spec.params}
            try:
                out[n] = spec.run([out[p] for p in parents], params)
                self.cache[keys[n]] = out[n]
                info[n] = {"status": "ok", "ms": (time.perf_counter() - t1) * 1000, "error": None, "key": keys[n]}
            except Exception as e:  # a failing node must not take the whole run down
                info[n] = {"status": "failed", "ms": (time.perf_counter() - t1) * 1000, "error": f"{type(e).__name__}: {e}", "key": keys[n]}

        with ThreadPoolExecutor(self.workers) as pool:
            for level in levels:
                list(pool.map(exec_node, level))
        for n, v in out.items():
            info[n]["summary"] = summarize(v)
        res = {"ok": all(i["status"] in ("ok", "cached") for i in info.values()), "errors": [], "warnings": warnings, "nodes": info,
               "levels": levels, "total_ms": (time.perf_counter() - t0) * 1000, "outputs": {n: v for n, v in out.items() if isinstance(v, Report)}}
        self.runs.append({"at": time.strftime("%H:%M:%S"), "nodes": len(g.nodes), "ok": res["ok"], "ms": res["total_ms"],
                          "cached": sum(i["status"] == "cached" for i in info.values()), "failed": sum(i["status"] == "failed" for i in info.values())})
        return res


def summarize(v: Any) -> str:
    if isinstance(v, pd.DataFrame):
        return f"Frame {v.shape[0]:,} × {v.shape[1]} · {int(v.isna().sum().sum())} missing"
    if isinstance(v, Split):
        return f"Split train {len(v.X_train)} / test {len(v.X_test)} · {v.X_train.shape[1]} features"
    if isinstance(v, Fitted):
        return f"Model {v.name} · fit {v.fit_ms:.0f} ms"
    if isinstance(v, Report):
        return "Report " + ", ".join(f"{r['model']} AUC {r['roc_auc']:.3f}" for r in v.rows)
    return type(v).__name__


def graph_from_json(d: dict[str, Any]) -> Graph:
    nodes = {NodeId(str(n["id"])): Node(NodeId(str(n["id"])), str(n["kind"]), dict(n.get("params", {})), float(n.get("x", 0)), float(n.get("y", 0))) for n in d.get("nodes", [])}
    return Graph(nodes, [Edge(NodeId(str(e["src"])), NodeId(str(e["dst"]))) for e in d.get("edges", [])])


def _n(i: str, k: str, x: int, y: int, **p: Any) -> dict[str, Any]:
    return {"id": i, "kind": k, "x": x, "y": y, "params": p}


TEMPLATES: dict[str, dict[str, Any]] = {
    "Titanic — baseline + 3-model tournament (clean)": {
        "nodes": [_n("load", "source", 20, 140, dataset="titanic"), _n("no_outliers", "filter", 190, 140, column="fare", op="<", value="300"), _n("encode", "encode", 360, 140),
                  _n("split", "split", 530, 140, target="survived", test_size=0.25, seed=42),
                  _n("base", "train", 700, 5, model="baseline_prior"), _n("lr", "train", 700, 95, model="logistic"), _n("rf", "train", 700, 185, model="random_forest"), _n("gb", "train", 700, 275, model="gradient_boosting"),
                  _n("eval_base", "evaluate", 870, 5), _n("eval_lr", "evaluate", 870, 95), _n("eval_rf", "evaluate", 870, 185), _n("eval_gb", "evaluate", 870, 275), _n("leaderboard", "compare", 1040, 140)],
        "edges": [{"src": a, "dst": b} for a, b in [("load", "no_outliers"), ("no_outliers", "encode"), ("encode", "split"), ("split", "base"), ("base", "eval_base"), ("eval_base", "leaderboard"), ("split", "lr"), ("split", "rf"), ("split", "gb"),
                                                    ("lr", "eval_lr"), ("rf", "eval_rf"), ("gb", "eval_gb"), ("eval_lr", "leaderboard"), ("eval_rf", "leaderboard"), ("eval_gb", "leaderboard")]]},
    "Leaky pipeline — impute before split (lint demo)": {
        "nodes": [_n("load", "source", 20, 120, dataset="titanic"), _n("impute_all", "impute", 200, 120, strategy="mean"), _n("encode", "encode", 380, 120),
                  _n("split", "split", 560, 120, target="survived", test_size=0.25, seed=42), _n("rf", "train", 740, 120, model="random_forest"), _n("eval", "evaluate", 920, 120)],
        "edges": [{"src": a, "dst": b} for a, b in [("load", "impute_all"), ("impute_all", "encode"), ("encode", "split"), ("split", "rf"), ("rf", "eval")]]},
    "Broken — type error + cycle (validator demo)": {
        "nodes": [_n("load", "source", 20, 120, dataset="wine"), _n("train", "train", 240, 60, model="logistic"), _n("a", "filter", 240, 200, column="alcohol", op=">", value="12"),
                  _n("b", "drop_columns", 460, 200, drop="ash")],
        "edges": [{"src": "load", "dst": "train"}, {"src": "load", "dst": "a"}, {"src": "a", "dst": "b"}, {"src": "b", "dst": "a"}]},
    "Breast cancer — failure isolation demo": {
        "nodes": [_n("load", "source", 20, 150, dataset="breast_cancer"), _n("split", "split", 220, 150, target="target", test_size=0.3, seed=7),
                  _n("lr", "train", 420, 60, model="logistic"), _n("eval_lr", "evaluate", 620, 60), _n("bad_filter", "filter", 220, 290, column="no_such_column", op=">", value="1"),
                  _n("split2", "split", 420, 290, target="target", test_size=0.3, seed=7), _n("rf", "train", 620, 290, model="random_forest"), _n("eval_rf", "evaluate", 820, 290), _n("board", "compare", 1000, 150)],
        "edges": [{"src": a, "dst": b} for a, b in [("load", "split"), ("split", "lr"), ("lr", "eval_lr"), ("load", "bad_filter"), ("bad_filter", "split2"), ("split2", "rf"), ("rf", "eval_rf"), ("eval_lr", "board")]]},
}
