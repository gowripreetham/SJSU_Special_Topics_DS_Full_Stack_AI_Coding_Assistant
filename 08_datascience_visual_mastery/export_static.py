"""Export this project as a static, GitHub-Pages-ready site in <repo>/docs/visual-mastery/.

The kit runtime reads pages from an embedded `window.KIT_STATIC` object instead of calling the API.
    python export_static.py        then enable GitHub Pages on the /docs folder.
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import app as project  # noqa: E402

OUT = ROOT.parent / "docs" / "visual-mastery"


def main() -> None:
    c = TestClient(project.app)
    meta = c.get("/api/meta").json()
    static = {"api/meta": meta, **{f"api/page/{p['id']}": c.get(f"/api/page/{p['id']}").json() for p in meta["pages"]}}
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "kit" / "vendor").mkdir(parents=True)
    (OUT / "web").mkdir()
    kit = ROOT.parent / "common" / "kit"
    for f in ["kit.css", "kit.js"]:
        shutil.copy(kit / f, OUT / "kit" / f)
    shutil.copy(kit / "vendor" / "chart.umd.js", OUT / "kit" / "vendor" / "chart.umd.js")
    shutil.copy(ROOT / "web" / "custom.js", OUT / "web" / "custom.js")
    html = c.get("/").text.replace('<script src="kit/kit.js">', f"<script>window.KIT_STATIC = {json.dumps(static)};</script>\n<script src=\"kit/kit.js\">")
    (OUT / "index.html").write_text(html)
    (OUT.parent / ".nojekyll").write_text("")
    print(f"static site written to {OUT} ({len(static) - 1} pages)")


if __name__ == "__main__":
    main()
