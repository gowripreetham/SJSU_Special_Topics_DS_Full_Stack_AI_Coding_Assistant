"""App factory: every project is a FastAPI app that serves the shared kit shell + JSON page blocks."""
from __future__ import annotations
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

KIT_DIR = Path(__file__).parent / "kit"

SHELL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<link rel="icon" href="data:,"><link rel="stylesheet" href="kit/kit.css"><script src="kit/vendor/chart.umd.js"></script><script src="kit/kit.js"></script>
</head><body><div id="shell"><nav id="side"></nav><main id="main"><div class="loading">Loading…</div></main></div></body></html>"""


def clean(o: Any) -> Any:
    """Make numpy / pandas / NaN values JSON-safe."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=1))


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def load_local(file: str, name: str):
    """Import a sibling module (e.g. pipeline.py) under a unique name so 16 projects can coexist in one process."""
    d = Path(file).resolve().parent
    uniq = f"proj_{d.name}_{name}"
    if uniq in sys.modules:
        return sys.modules[uniq]
    spec = importlib.util.spec_from_file_location(uniq, d / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[uniq] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class Safe(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False).encode("utf-8")


def make_app(file: str, number: str, title: str, subtitle: str,
             pages: list[tuple[str, str, str, Callable[[], list[dict]]]]) -> FastAPI:
    """pages = [(id, title, group, builder)]; builder() returns a list of blocks."""
    root = Path(file).resolve().parent
    app = FastAPI(title=title, default_response_class=Safe)
    app.state.meta = {"number": number, "title": title, "subtitle": subtitle, "hub": False,
                      "custom_js": (root / "web" / "custom.js").exists(),
                      "pages": [{"id": p[0], "title": p[1], "group": p[2]} for p in pages]}
    builders = {p[0]: p[3] for p in pages}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return SHELL.replace("__TITLE__", title)

    @app.get("/api/meta")
    def meta() -> dict:
        return app.state.meta

    @app.get("/api/page/{pid}")
    def page(pid: str) -> dict:
        if pid not in builders:
            raise HTTPException(404, f"unknown page {pid}")
        try:
            return {"blocks": builders[pid]()}
        except Exception as e:  # surface the error in the UI instead of a blank page
            traceback.print_exc()
            return {"blocks": [{"t": "callout", "kind": "crit", "err": True, "title": f"Page failed: {type(e).__name__}",
                                "text": f"`{e}`\n\nRun `python pipeline.py` in this project to (re)build artifacts."}]}

    app.mount("/kit", StaticFiles(directory=KIT_DIR), name="kit")
    if (root / "web").exists():
        app.mount("/web", StaticFiles(directory=root / "web"), name="web")
    return app


def load_or_train(path: Path, train: Callable[[], Any]) -> Any:
    """Load a joblib artifact; if it is missing or was pickled by an incompatible library version, retrain."""
    import joblib
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[artifacts] {path.name}: {type(e).__name__} — rebuilding by running the pipeline…", flush=True)
        train()
        return joblib.load(path)
