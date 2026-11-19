"""Browser test + screenshot tour (Playwright). Opens every page of every project through the hub,
fails on console errors / failed page loads, and saves PNGs to <project>/screenshots/.

    python scripts/shots.py            # all projects
    python scripts/shots.py 03 12      # only some
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765


def main() -> int:
    only = set(sys.argv[1:])
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "hub:app", "--port", str(PORT), "--log-level", "warning"], cwd=ROOT)
    try:
        for _ in range(240):
            try:
                projects = httpx.get(f"http://127.0.0.1:{PORT}/api/projects", timeout=2).json()
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("hub did not start")
        bad = 0
        with sync_playwright() as p:
            try:
                br = p.chromium.launch(channel="chrome")
            except Exception:
                br = p.chromium.launch()
            pg = br.new_page(viewport={"width": 1440, "height": 900})
            errs: list[str] = []
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.on("pageerror", lambda e: errs.append(str(e)))
            if not only:
                pg.goto(f"http://127.0.0.1:{PORT}/")
                pg.wait_for_selector(".pcard")
                (ROOT / "docs").mkdir(exist_ok=True)
                pg.screenshot(path=ROOT / "docs" / "hub.png", full_page=True)
            for pr in projects:
                if only and pr["number"] not in only:
                    continue
                out = ROOT / pr["dir"] / "screenshots"
                out.mkdir(exist_ok=True)
                meta = httpx.get(f"http://127.0.0.1:{PORT}/p/{pr['number']}/api/meta").json()
                for i, page in enumerate(meta["pages"]):
                    errs.clear()
                    pg.goto(f"http://127.0.0.1:{PORT}/p/{pr['number']}/#{page['id']}")
                    pg.reload()
                    try:
                        pg.wait_for_selector("body[data-ready='1']", timeout=60000)
                        pg.wait_for_load_state("networkidle")
                        pg.wait_for_timeout(900)
                        failed = pg.locator(".kit-error").count()
                    except Exception as e:
                        failed, _ = 1, errs.append(f"timeout {e}")
                    pg.screenshot(path=out / f"{i + 1:02d}_{page['id']}.png", full_page=True)
                    status = "FAIL" if (failed or errs) else "ok"
                    bad += status == "FAIL"
                    print(f"{status:4} {pr['number']} {page['id']:<14} {'; '.join(errs)[:160]}")
            br.close()
        print("ALL PAGES OK" if not bad else f"{bad} PAGE(S) FAILED")
        return 1 if bad else 0
    finally:
        srv.terminate()


if __name__ == "__main__":
    sys.exit(main())
