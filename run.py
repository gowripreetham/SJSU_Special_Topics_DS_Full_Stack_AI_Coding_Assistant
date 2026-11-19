"""Launcher.

    python run.py                 # hub with all 16 projects on http://127.0.0.1:8000
    python run.py 03              # just project 03 on http://127.0.0.1:8003
    python run.py train           # re-run every training pipeline (rebuilds artifacts/)
    python run.py train 03 06     # re-run selected pipelines
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def proj(nn: str) -> Path:
    m = [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(nn + "_")]
    if not m:
        raise SystemExit(f"no project {nn}")
    return m[0]


def main() -> None:
    a = sys.argv[1:]
    if a and a[0] == "train":
        todo = [proj(n) for n in a[1:]] or sorted(p for p in ROOT.iterdir() if (p / "pipeline.py").exists())
        for d in todo:
            print(f"\n=== training {d.name} ===", flush=True)
            subprocess.run([sys.executable, "pipeline.py"], cwd=d, check=True)
        return
    import uvicorn
    if a:
        d = proj(a[0])
        sys.path.insert(0, str(d))
        print(f"→ {d.name} on http://127.0.0.1:{8000 + int(a[0])}")
        uvicorn.run("app:app", app_dir=str(d), host="127.0.0.1", port=8000 + int(a[0]))
    else:
        print("→ hub on http://127.0.0.1:8000")
        uvicorn.run("hub:app", app_dir=str(ROOT), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
