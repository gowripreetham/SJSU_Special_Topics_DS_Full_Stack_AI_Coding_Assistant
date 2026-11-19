"""Tiny builders for the JSON 'blocks' the front-end kit (common/kit/kit.js) knows how to render."""
from __future__ import annotations
from typing import Any

__all__ = ["H", "MD", "K", "KPIS", "S", "CHART", "TABLE", "HEAT", "GRID", "CALLOUT", "CODE", "F", "FORM",
           "QUIZ", "STEPS", "TABS", "CUSTOM", "HTML", "PILL", "df_table", "hist_chart", "CRISP"]


def H(text: str, sub: str | None = None) -> dict: return {"t": "h", "text": text, "sub": sub}
def MD(text: str, title: str | None = None, **kw) -> dict: return {"t": "md", "text": text, "title": title, **kw}
def HTML(html: str, title: str | None = None, **kw) -> dict: return {"t": "html", "html": html, "title": title, **kw}
def K(label: str, value: Any, hint: str | None = None, digits: int | None = None, **kw) -> dict:
    return {"label": label, "value": value, "hint": hint, "digits": digits, **kw}
def KPIS(*items: dict) -> dict: return {"t": "kpis", "items": list(items)}
def S(name: str, data: list, **kw) -> dict: return {"name": name, "data": data, **kw}
def CHART(kind: str, title: str, series: list[dict], labels: list | None = None, **kw) -> dict:
    return {"t": "chart", "kind": kind, "title": title, "labels": labels, "series": series, **kw}
def TABLE(title: str | None, cols: list[str], rows: list[list], **kw) -> dict:
    return {"t": "table", "title": title, "cols": cols, "rows": rows, **kw}
def HEAT(title: str, xs: list, ys: list, z: list[list], **kw) -> dict:
    return {"t": "heatmap", "title": title, "xs": [str(x) for x in xs], "ys": [str(y) for y in ys], "z": z, **kw}
def GRID(cols: int, *items: dict) -> dict: return {"t": "grid", "cols": cols, "items": list(items)}
def CALLOUT(kind: str, title: str, text: str = "") -> dict: return {"t": "callout", "kind": kind, "title": title, "text": text}
def CODE(title: str, text: str) -> dict: return {"t": "code", "title": title, "text": text}
def F(name: str, label: str, type: str = "number", value: Any = None, **kw) -> dict:
    return {"name": name, "label": label, "type": type, "value": value, **kw}
def FORM(action: str, fields: list[dict], title: str | None = None, submit: str = "Run", **kw) -> dict:
    return {"t": "form", "action": action, "fields": fields, "title": title, "submit": submit, **kw}
def QUIZ(title: str, questions: list[dict], **kw) -> dict: return {"t": "quiz", "title": title, "questions": questions, **kw}
def STEPS(items: list[dict]) -> dict: return {"t": "steps", "items": items}
def TABS(*tabs: tuple[str, list[dict]]) -> dict: return {"t": "tabs", "tabs": [{"title": t, "blocks": b} for t, b in tabs]}
def CUSTOM(name: str, props: dict | None = None, title: str | None = None, **kw) -> dict:
    return {"t": "custom", "name": name, "props": props or {}, "title": title, **kw}
def PILL(text: str, kind: str = "") -> dict: return {"pill": text, "kind": kind}


def df_table(df, title: str | None = None, n: int = 50, **kw) -> dict:
    """Render the first n rows of a pandas DataFrame."""
    d = df.head(n)
    rows = [[None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else v) for v in r]
            for r in d.itertuples(index=False, name=None)]
    return TABLE(title, [str(c) for c in d.columns], rows, **kw)


def hist_chart(values, title: str, bins: int = 30, x: str | None = None, color: str | None = None, **kw) -> dict:
    import numpy as np
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    counts, edges = np.histogram(v, bins=bins)
    labels = [f"{(edges[i] + edges[i + 1]) / 2:.3g}" for i in range(len(counts))]
    s = S("count", counts.tolist())
    if color:
        s["color"] = color
    return CHART("bar", title, [s], labels, x=x, y="count", **kw)


def CRISP(phases: list[tuple[str, str]]) -> dict:
    """The six CRISP-DM phases as a numbered timeline. phases = [(title, markdown), ...]."""
    return STEPS([{"title": t, "text": m, "status": "done"} for t, m in phases])
