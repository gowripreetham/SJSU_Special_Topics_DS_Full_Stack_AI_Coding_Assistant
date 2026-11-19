"""Static data-science code auditor (AST based). Used by Project 11 (portfolio audit) and Project 13 (code-auditor workbench).

It does not run any code. It parses Python source and applies rules that target the classic ways a data-science result gets
quietly inflated: leakage, test-set reuse, shuffled time, unseeded randomness, flattering metrics, missing baselines.

Static analysis is heuristic: a finding means "a human should look here", not "this is certainly a bug". A line can be
waived with an inline comment  `# audit-ok: <reason>`  — the waiver and its reason are shown in the report, never hidden.
"""
from __future__ import annotations
import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

DIMENSIONS = ["Leakage", "Evaluation integrity", "Reproducibility", "Metric alignment", "Code hygiene", "Documentation"]
PENALTY = {"fail": 25, "warn": 10, "info": 3, "waived": 0, "pass": 0}

PREPROCESSORS = {"StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer", "KNNImputer", "OneHotEncoder", "OrdinalEncoder", "PCA", "SelectKBest",
                 "TargetEncoder", "PolynomialFeatures", "QuantileTransformer", "PowerTransformer", "TfidfVectorizer", "CountVectorizer"}
NEEDS_SEED = {"train_test_split", "RandomForestClassifier", "RandomForestRegressor", "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier",
              "HistGradientBoostingRegressor", "KMeans", "IsolationForest", "XGBClassifier", "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "MLPRegressor", "MLPClassifier",
              "GaussianMixture", "MinCovDet", "ShuffleSplit", "StratifiedShuffleSplit", "DecisionTreeClassifier", "ExtraTreesClassifier", "RandomizedSearchCV"}
SHUFFLING_CV = {"KFold", "StratifiedKFold"}
SEARCHERS = {"hill_climb", "GridSearchCV", "RandomizedSearchCV"}
TRAIN_LIKE = re.compile(r"(?<![A-Za-z])(tr|train|Xtr|X_train|ytr|y_train|trva|fold|small|sub|boot)(?![a-df-z])", re.I)
TEST_LIKE = re.compile(r"(?<![A-Za-z_])(te|test|Xte|yte|X_test|y_test|te_ex)(?![A-Za-z_])")
SPLITTERS = {"train_test_split", "TimeSeriesSplit", "KFold", "StratifiedKFold", "GroupKFold", "cross_val_score", "cross_val_predict", "cross_validate"}
ROBUST_METRICS = ("average_precision_score", "precision_recall_curve", "f1_score", "roc_auc", "roc_auc_score")


@dataclass
class Finding:
    rule: str
    dimension: str
    severity: str  # fail | warn | info | waived | pass
    line: int
    message: str
    code: str = ""
    waiver: str = ""


@dataclass
class FileReport:
    path: str
    loc: int
    findings: list[Finding] = field(default_factory=list)
    passes: list[Finding] = field(default_factory=list)


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    return ""


def _root_ctor(node: ast.AST) -> str:
    """For `StandardScaler().fit_transform(x)` return 'StandardScaler'."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return _name(node) if isinstance(node, ast.Call) else ""


def audit_source(src: str, path: str = "<snippet>", temporal: bool = False, imbalanced: bool = False, forbidden: tuple[str, ...] = (), supervised: bool = True) -> FileReport:
    lines = src.splitlines()
    rep = FileReport(path, len([l for l in lines if l.strip() and not l.strip().startswith("#")]))
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rep.findings.append(Finding("SYNTAX", "Code hygiene", "fail", e.lineno or 1, f"file does not parse: {e.msg}"))
        return rep

    def add(rule: str, dim: str, sev: str, line: int, msg: str) -> None:
        code = lines[line - 1].strip() if 0 < line <= len(lines) else ""
        m = re.search(r"#\s*audit-ok:\s*(.+)$", lines[line - 1]) if 0 < line <= len(lines) else None
        if m and sev != "pass":
            rep.findings.append(Finding(rule, dim, "waived", line, msg, code[:160], m.group(1).strip()))
        elif sev == "pass":
            rep.passes.append(Finding(rule, dim, "pass", line, msg))
        else:
            rep.findings.append(Finding(rule, dim, sev, line, msg, code[:160]))

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    names = {_name(c.func) for c in calls}
    seg = lambda n: ast.get_source_segment(src, n) or ""  # noqa: E731

    # ---- variables bound to a preprocessor instance, e.g.  sc = StandardScaler()
    bound: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and _name(n.value.func) in PREPROCESSORS:
            for t in n.targets:
                if isinstance(t, ast.Name):
                    bound[t.id] = _name(n.value.func)

    # ---- R-LEAK-1: preprocessor fit on data that is not clearly the training split
    has_split = bool(names & SPLITTERS) or bool(re.search(r"iloc\[:|idx\[:|\[:n\]|\[n:\]|HELD_OUT|folds\(", src))  # incl. manual / chronological splits
    n_leak = 0
    for c in calls:
        if isinstance(c.func, ast.Attribute) and c.func.attr in ("fit", "fit_transform") and c.args:
            ctor = _root_ctor(c.func.value) if isinstance(c.func.value, ast.Call) else bound.get(_name(c.func.value), "")
            if ctor in PREPROCESSORS:
                arg = seg(c.args[0])
                if not TRAIN_LIKE.search(arg) and supervised and has_split:
                    n_leak += 1
                    add("LEAK-FIT", "Leakage", "warn", c.lineno, f"{ctor}.{c.func.attr}({arg[:40]}) — fitted on data not clearly limited to the training split; put it in a Pipeline or fit on train only")
    if supervised and has_split and not n_leak:
        add("LEAK-FIT", "Leakage", "pass", 0, "no preprocessor is fitted outside the training split")

    # ---- R-LEAK-2: forbidden (target-derived) columns inside feature construction
    if forbidden:
        hit = False
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and re.search(r"feature", n.name)]:
            for s in [x for x in ast.walk(fn) if isinstance(x, ast.Constant) and isinstance(x.value, str)]:
                if s.value in forbidden:
                    hit = True
                    add("LEAK-TARGET", "Leakage", "fail", s.lineno, f"feature function `{fn.name}` references post-outcome column '{s.value}'")
        if not hit:
            add("LEAK-TARGET", "Leakage", "pass", 0, f"feature construction never touches post-outcome columns ({', '.join(forbidden)})")

    # ---- R-EVAL-1: test data referenced inside a model-selection loop
    sel_hit = False
    for c in calls:
        if _name(c.func) in SEARCHERS:
            for a in list(c.args[:1]) + [k.value for k in c.keywords if k.arg in ("evaluate", "scoring")]:
                body = seg(a)
                if isinstance(a, ast.Name):  # a named function: inspect its definition
                    for fn in ast.walk(tree):
                        if isinstance(fn, ast.FunctionDef) and fn.name == a.id:
                            body = seg(fn)
                m = TEST_LIKE.search(body)
                if m:
                    sel_hit = True
                    add("EVAL-TEST-IN-SEARCH", "Evaluation integrity", "fail", c.lineno, f"model selection ({_name(c.func)}) reads test data (`{m.group(0)}`) — the test set must be scored once, after selection")
    if names & SEARCHERS and not sel_hit:
        add("EVAL-TEST-IN-SEARCH", "Evaluation integrity", "pass", 0, "hyper-parameter search never reads the test split")

    # ---- R-EVAL-2: some held-out evaluation exists
    if supervised:
        if has_split:
            add("EVAL-HOLDOUT", "Evaluation integrity", "pass", 0, "a held-out split or cross-validation is used")
        else:
            add("EVAL-HOLDOUT", "Evaluation integrity", "warn", 1, "no train/test split or cross-validation found — results would be in-sample")

    # ---- R-EVAL-3: temporal ordering
    if temporal:
        bad = False
        for c in calls:
            kw = {k.arg: seg(k.value) for k in c.keywords}
            if _name(c.func) == "train_test_split" and kw.get("shuffle") != "False":
                bad = True
                add("EVAL-TEMPORAL", "Evaluation integrity", "fail", c.lineno, "random train_test_split on time-ordered data — future rows leak into training; split chronologically")
            if _name(c.func) in SHUFFLING_CV and kw.get("shuffle") == "True":
                bad = True
                add("EVAL-TEMPORAL", "Evaluation integrity", "fail", c.lineno, "shuffled K-fold on time-ordered data; use TimeSeriesSplit / rolling origin")
        if not bad:
            add("EVAL-TEMPORAL", "Evaluation integrity", "pass", 0, "no shuffled split on time-ordered data")

    # ---- R-EVAL-4: baseline
    if supervised:
        if re.search(r"aseline|[Mm]ajority|[Nn]aive|popularity|Drift", src):
            add("EVAL-BASELINE", "Evaluation integrity", "pass", 0, "a naive baseline is reported next to the models")
        else:
            add("EVAL-BASELINE", "Evaluation integrity", "warn", 1, "no naive baseline found — metrics are uninterpretable without one")

    # ---- R-REPRO: seeds
    unseeded = 0
    for c in calls:
        nm = _name(c.func)
        kws = {k.arg for k in c.keywords}
        if None in kws:  # f(**cfg): arguments are not statically visible, give the benefit of the doubt
            continue
        if (nm in NEEDS_SEED and not ({"random_state", "seed"} & kws)) or (nm in SHUFFLING_CV and any(k.arg == "shuffle" and seg(k.value) == "True" for k in c.keywords) and "random_state" not in kws):
            if nm == "train_test_split" and any(k.arg == "shuffle" and seg(k.value) == "False" for k in c.keywords):
                continue
            unseeded += 1
            add("REPRO-SEED", "Reproducibility", "warn", c.lineno, f"{nm}(...) without random_state — results change between runs")
    if not unseeded:
        add("REPRO-SEED", "Reproducibility", "pass", 0, "every stochastic estimator / splitter is seeded")
    if re.search(r"^\s*SEED\s*=|manual_seed\(|RandomState\(|default_rng\(", src, re.M):
        add("REPRO-GLOBAL", "Reproducibility", "pass", 0, "a named global seed is defined")
    elif names & NEEDS_SEED:
        add("REPRO-GLOBAL", "Reproducibility", "info", 1, "no named SEED constant; seeds are scattered literals")

    # ---- R-METRIC: imbalanced targets need PR-aware metrics
    if imbalanced:
        if any(m in src for m in ROBUST_METRICS):
            add("METRIC-IMBALANCE", "Metric alignment", "pass", 0, "imbalance-aware metrics (PR-AUC / F1 / ROC-AUC) are computed")
        else:
            add("METRIC-IMBALANCE", "Metric alignment", "fail", 1, "imbalanced target but only accuracy-style metrics found")
    if supervised and "accuracy_score" in src and not any(m in src for m in ROBUST_METRICS) and not imbalanced:
        add("METRIC-ACC-ONLY", "Metric alignment", "info", 1, "accuracy is the only classification metric — add a threshold-free one")
    elif supervised:
        add("METRIC-ACC-ONLY", "Metric alignment", "pass", 0, "metrics go beyond plain accuracy")

    # ---- R-HYGIENE
    hyg = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.ExceptHandler) and n.type is None:
            hyg += 1
            add("HYG-BARE-EXCEPT", "Code hygiene", "warn", n.lineno, "bare `except:` swallows every error including KeyboardInterrupt")
        if isinstance(n, ast.ExceptHandler) and n.type is not None and len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
            hyg += 1
            add("HYG-SILENT-EXCEPT", "Code hygiene", "info", n.lineno, "exception silently ignored (`pass`) — a failed step would go unnoticed")
        if isinstance(n, ast.Call) and _name(n.func) in ("eval", "exec") and isinstance(n.func, ast.Name):
            hyg += 1
            add("HYG-EVAL", "Code hygiene", "fail", n.lineno, f"{_name(n.func)}() on dynamic input")
        if isinstance(n, ast.Call) and _name(n.func) == "filterwarnings" and n.args and seg(n.args[0]).strip("'\"") == "ignore" and len(n.args) + len(n.keywords) == 1:
            hyg += 1
            add("HYG-WARNINGS", "Code hygiene", "info", n.lineno, "all warnings globally silenced — convergence warnings would be hidden")
    if not hyg:
        add("HYG", "Code hygiene", "pass", 0, "no bare excepts, eval/exec or blanket warning filters")

    # ---- R-DOC
    doc = ast.get_docstring(tree) or ""
    if len(doc) >= 80:
        add("DOC-MODULE", "Documentation", "pass", 0, "module docstring explains purpose and method")
    else:
        add("DOC-MODULE", "Documentation", "warn", 1, "missing or very short module docstring")
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and not n.name.startswith("_") and len(n.body) > 6]
    if fns:
        share = sum(bool(ast.get_docstring(f)) for f in fns) / len(fns)
        if share < 0.3:
            add("DOC-FUNCS", "Documentation", "info", fns[0].lineno, f"only {100 * share:.0f}% of non-trivial public functions have docstrings")
        else:
            add("DOC-FUNCS", "Documentation", "pass", 0, f"{100 * share:.0f}% of non-trivial public functions are documented")
    return rep


def score(reports: list[FileReport]) -> dict:
    dims = {d: 100 for d in DIMENSIONS}
    counts = {"fail": 0, "warn": 0, "info": 0, "waived": 0}
    for r in reports:
        for f in r.findings:
            dims[f.dimension] = max(0, dims[f.dimension] - PENALTY[f.severity])
            counts[f.severity] += 1
    overall = sum(dims.values()) / len(dims)
    grade = "A+" if overall >= 97 else "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 70 else "D" if overall >= 60 else "F"
    return {"dimensions": dims, "overall": overall, "grade": grade, "counts": counts, "checks_passed": sum(len(r.passes) for r in reports)}


def audit_project(root: Path, files: list[str], **cfg) -> dict:
    reports = [audit_source((root / f).read_text(), f, **cfg) for f in files if (root / f).exists()]
    return {**score(reports), "files": [{"path": r.path, "loc": r.loc, "findings": [asdict(f) for f in r.findings], "passes": [asdict(p) for p in r.passes]} for r in reports],
            "loc": sum(r.loc for r in reports)}
