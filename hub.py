"""Hub: one process that serves all 16 projects.   python run.py   ->   http://127.0.0.1:8000

Each project is an independent FastAPI app (see <project>/app.py); the hub mounts them under /p/<nn>/.
"""
from __future__ import annotations
import importlib.util
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.server import KIT_DIR, Safe  # noqa: E402

app = FastAPI(title="DS Full-Stack Portfolio", default_response_class=Safe)
PROJECTS: list[dict] = []

for d in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit() and (p / "app.py").exists()):
    nn = d.name[:2]
    try:
        spec = importlib.util.spec_from_file_location(f"proj_{d.name}_app", d / "app.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        sys.modules[spec.name] = mod  # type: ignore[union-attr]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        mod.app.state.meta["hub"] = True
        app.mount(f"/p/{nn}", mod.app)
        m = mod.app.state.meta
        PROJECTS.append({"number": nn, "dir": d.name, "title": m["title"], "subtitle": m["subtitle"], "pages": len(m["pages"]), "ok": True})
    except Exception as e:  # one broken project must not take the portfolio down
        traceback.print_exc()
        PROJECTS.append({"number": nn, "dir": d.name, "title": d.name, "subtitle": f"failed to load: {e}", "pages": 0, "ok": False})

app.mount("/kit", StaticFiles(directory=KIT_DIR), name="kit")


@app.get("/api/projects")
def projects() -> list[dict]:
    return PROJECTS


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    cards = "".join(
        f'<a class="pcard" href="p/{p["number"]}/"><div class="num">PROJECT {p["number"]}</div><div class="name">{p["title"]}</div>'
        f'<div class="sub">{p["subtitle"]}</div><div class="foot">{p["pages"]} pages →</div></a>' for p in PROJECTS)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DS Full-Stack Portfolio</title><link rel="icon" href="data:,"><link rel="stylesheet" href="kit/kit.css"><style>
.wrap{{max-width:1180px;margin:0 auto;padding:44px 28px}}h1{{font-size:30px;margin:0 0 6px;letter-spacing:-.02em}}
.lead{{color:var(--ink2);max-width:760px;margin:0 0 28px}}.pg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(262px,1fr));gap:16px}}
.pcard{{display:block;text-decoration:none;color:inherit;background:var(--surface);border:1px solid var(--grid);border-radius:12px;padding:16px 18px;transition:.15s}}
.pcard:hover{{border-color:var(--accent);transform:translateY(-2px)}}.pcard .num{{font:600 11px var(--mono);color:var(--muted);letter-spacing:.08em}}
.pcard .name{{font-weight:700;font-size:16px;margin:3px 0 4px}}.pcard .sub{{color:var(--ink2);font-size:12.5px;min-height:38px}}.pcard .foot{{margin-top:10px;color:var(--accent);font-size:12.5px;font-weight:600}}
</style></head><body><div class="wrap"><h1>Data Science Full-Stack Portfolio</h1>
<p class="lead">16 end-to-end data science systems built with an AI coding assistant for SJSU Special Topics — each one a FastAPI back end,
a real training pipeline with CRISP-DM documentation, and an interactive dashboard. Pick a project.</p><div class="pg">{cards}</div></div></body></html>"""
