"""Project 00 — Zenith Task Workspace: FastAPI + SQLite + Server-Sent Events, with a no-build SPA front end."""
from __future__ import annotations
import asyncio
import json
import re
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "todo.db"
STATUSES = ["todo", "doing", "done"]
PRIORITIES = ["low", "medium", "high", "urgent"]


# ----------------------------------------------------------------------------- storage
def db() -> sqlite3.Connection:
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, notes TEXT DEFAULT '',
        status TEXT DEFAULT 'todo', priority TEXT DEFAULT 'medium', tags TEXT DEFAULT '[]',
        due TEXT, created TEXT NOT NULL, completed TEXT, position REAL DEFAULT 0)""")
    return con


def row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["tags"] = json.loads(d["tags"] or "[]")
    return d


def seed() -> None:
    with db() as con:
        if con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]:
            return
        today = date.today()
        demo = [("Record 2-minute walkthrough videos", "doing", "urgent", ["assignment"], 1, 0, None),
                ("Train NanoLlama checkpoint", "done", "high", ["ml"], -2, 6, 2),
                ("Run leakage audit across all projects", "done", "high", ["audit"], -1, 5, 1),
                ("Tune XGBoost for taxi duration", "done", "medium", ["ml"], -3, 7, 3),
                ("Write CRISP-DM report for clustering", "doing", "medium", ["docs"], 2, 3, None),
                ("Add PSI drift monitor", "todo", "medium", ["mlops"], 4, 2, None),
                ("Read PatchTST paper", "todo", "low", ["reading"], 7, 1, None),
                ("Fix flaky SSE reconnect", "todo", "high", ["bug"], 0, 1, None),
                ("Publish repo README with screenshots", "todo", "high", ["assignment", "docs"], 1, 0, None),
                ("Benchmark FP-Growth vs Apriori", "done", "low", ["ml"], -4, 8, 4),
                ("Grocery run", "done", "low", ["personal"], -5, 9, 5),
                ("Email professor the repo link", "todo", "urgent", ["assignment"], 1, 0, None)]
        for i, (t, s, p, tags, due, age, done_ago) in enumerate(demo):
            con.execute("INSERT INTO tasks(title,status,priority,tags,due,created,completed,position) VALUES(?,?,?,?,?,?,?,?)",
                        (t, s, p, json.dumps(tags), str(today + timedelta(days=due)),
                         (datetime.now() - timedelta(days=age)).isoformat(timespec="seconds"),
                         (datetime.now() - timedelta(days=done_ago)).isoformat(timespec="seconds") if done_ago is not None else None, i))


# ----------------------------------------------------------------------------- quick-add NLP
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_quick(text: str, today: date | None = None) -> dict:
    """Natural-language quick add:  'Submit report tomorrow #school !high'  ->  structured task."""
    today = today or date.today()
    tags = re.findall(r"#([\w-]+)", text)
    pr = re.search(r"!(low|medium|high|urgent)\b", text, re.I)
    due = None
    low = text.lower()
    if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", low):
        due = m.group(1)
    elif re.search(r"\btoday\b", low):
        due = str(today)
    elif re.search(r"\btomorrow\b", low):
        due = str(today + timedelta(days=1))
    elif m := re.search(r"\bin (\d+) days?\b", low):
        due = str(today + timedelta(days=int(m.group(1))))
    elif re.search(r"\bnext week\b", low):
        due = str(today + timedelta(days=7))
    else:
        for i, wd in enumerate(WEEKDAYS):
            if re.search(rf"\b(on |next )?{wd}\b", low):
                due = str(today + timedelta(days=(i - today.weekday()) % 7 or 7))
                break
    title = re.sub(r"#[\w-]+|!(low|medium|high|urgent)\b|\b(today|tomorrow|next week|in \d+ days?|\d{4}-\d{2}-\d{2})\b|\b(on |next )?("
                   + "|".join(WEEKDAYS) + r")\b", "", text, flags=re.I)
    return {"title": re.sub(r"\s+", " ", title).strip(" ,-") or text.strip(), "tags": tags,
            "priority": pr.group(1).lower() if pr else "medium", "due": due}


# ----------------------------------------------------------------------------- live sync (SSE)
subscribers: set[asyncio.Queue] = set()


def publish(kind: str, payload: dict) -> None:
    for q in list(subscribers):
        q.put_nowait({"kind": kind, **payload})


class TaskIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    notes: str = ""
    status: str = "todo"
    priority: str = "medium"
    tags: list[str] = []
    due: str | None = None


class TaskPatch(BaseModel):
    title: str | None = None
    notes: str | None = None
    status: str | None = None
    priority: str | None = None
    tags: list[str] | None = None
    due: str | None = None
    position: float | None = None


class Quick(BaseModel):
    text: str


# ----------------------------------------------------------------------------- pages
def analytics() -> list[dict]:
    with db() as con:
        ts = [row(r) for r in con.execute("SELECT * FROM tasks")]
    today = date.today()
    done = [t for t in ts if t["status"] == "done"]
    overdue = [t for t in ts if t["status"] != "done" and t["due"] and t["due"] < str(today)]
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    comp = [sum(1 for t in done if (t["completed"] or "")[:10] == str(d)) for d in days]
    created = [sum(1 for t in ts if t["created"][:10] == str(d)) for d in days]
    open_n, burn = [], 0
    for d in days:  # open tasks at end of each day
        open_n.append(sum(1 for t in ts if t["created"][:10] <= str(d) and not (t["completed"] and t["completed"][:10] <= str(d))))
    cycle = [(datetime.fromisoformat(t["completed"]) - datetime.fromisoformat(t["created"])).total_seconds() / 86400 for t in done if t["completed"]]
    tagc: dict[str, int] = {}
    for t in ts:
        for g in t["tags"]:
            tagc[g] = tagc.get(g, 0) + 1
    tg = sorted(tagc.items(), key=lambda kv: -kv[1])[:8]
    return [H("Productivity analytics", "Computed live from the SQLite store — complete a task on the board and these numbers move."),
            KPIS(K("Total tasks", len(ts)), K("Completion rate", f"{100 * len(done) / max(1, len(ts)):.0f}%"),
                 K("Overdue", len(overdue), hint="open and past due"), K("Avg cycle time", f"{sum(cycle) / max(1, len(cycle)):.1f} d", hint="created → done")),
            GRID(2, CHART("bar", "Throughput — last 14 days", [S("completed", comp, color="3"), S("created", created, color="1")], [d.strftime("%b %d") for d in days], y="tasks"),
                 CHART("area", "Open tasks (burn-down)", [S("open", open_n)], [d.strftime("%b %d") for d in days], y="open tasks")),
            GRID(2, CHART("hbar", "Open tasks by priority", [S("open", [sum(1 for t in ts if t["priority"] == p and t["status"] != "done") for p in PRIORITIES])], PRIORITIES),
                 CHART("hbar", "Most used tags", [S("tasks", [v for _, v in tg], color="7")], [k for k, _ in tg]))]


DESIGN = """
## Goals
A fast, keyboard-friendly task workspace that stays in sync across every open tab — the "hello world" of full-stack
done properly: real persistence, real-time updates, validation, and analytics.

## Architecture
| Layer | Choice | Why |
|---|---|---|
| API | **FastAPI** + Pydantic models | typed request validation, auto OpenAPI docs at `/docs` |
| Storage | **SQLite** (`data/todo.db`) | zero-config, transactional, survives restarts |
| Realtime | **Server-Sent Events** (`GET api/events`) | one-way push is all a todo list needs; simpler than WebSockets, auto-reconnects |
| Front end | Vanilla JS module on the shared DS Kit | no build step — `python app.py` is the whole toolchain |

## Data model
`tasks(id, title, notes, status ∈ {todo, doing, done}, priority ∈ {low, medium, high, urgent}, tags JSON, due, created, completed, position)`

`position` is a float so drag-and-drop re-ordering is a single-row update (insert between neighbours = average of their positions).

## REST surface
- `GET api/tasks` · `POST api/tasks` · `PATCH api/tasks/{id}` · `DELETE api/tasks/{id}`
- `POST api/quick` — natural-language quick add (`"Submit report friday #school !high"`)
- `GET api/events` — SSE stream of `created / updated / deleted` events

## UX decisions
- **Kanban + list views** over the same data; drag cards between columns.
- **Quick add** parses `#tags`, `!priority`, and dates (`today`, `tomorrow`, `friday`, `in 3 days`, `2026-12-01`).
- Overdue tasks are flagged with an icon *and* a label (never colour alone).
- Optimistic UI: the card moves immediately; the SSE echo reconciles other tabs.

## What I added beyond the reference prompt
Natural-language quick add, live multi-tab sync, and the analytics page (throughput, burn-down, cycle time).
"""

app = make_app(__file__, "00", "Zenith Task Workspace", "Full-stack reactive todo app · FastAPI · SQLite · SSE", [
    ("board", "Task board", "Workspace", lambda: [H("Task board", "Drag cards between columns · quick-add understands `#tags`, `!priority` and dates like *tomorrow* or *friday* · open a second tab to watch live sync."), CUSTOM("todo", card=False)]),
    ("analytics", "Analytics", "Workspace", analytics),
    ("design", "Design doc", "Docs", lambda: [H("Design document"), MD(DESIGN)]),
])
seed()


@app.get("/api/tasks")
def list_tasks() -> list[dict]:
    with db() as con:
        return [row(r) for r in con.execute("SELECT * FROM tasks ORDER BY position, id")]


def _insert(t: TaskIn) -> dict:
    if t.status not in STATUSES or t.priority not in PRIORITIES:
        raise HTTPException(422, "invalid status or priority")
    with db() as con:
        pos = con.execute("SELECT COALESCE(MAX(position),0)+1 FROM tasks").fetchone()[0]
        cur = con.execute("INSERT INTO tasks(title,notes,status,priority,tags,due,created,position) VALUES(?,?,?,?,?,?,?,?)",
                          (t.title.strip(), t.notes, t.status, t.priority, json.dumps(t.tags), t.due, datetime.now().isoformat(timespec="seconds"), pos))
        out = row(con.execute("SELECT * FROM tasks WHERE id=?", (cur.lastrowid,)).fetchone())
    publish("created", {"task": out})
    return out


@app.post("/api/tasks")
def create(t: TaskIn) -> dict:
    return _insert(t)


@app.post("/api/quick")
def quick(q: Quick) -> dict:
    if not q.text.strip():
        raise HTTPException(422, "empty task")
    return _insert(TaskIn(**parse_quick(q.text)))


@app.patch("/api/tasks/{tid}")
def update(tid: int, p: TaskPatch) -> dict:
    ch = p.model_dump(exclude_unset=True)
    if "status" in ch and ch["status"] not in STATUSES or "priority" in ch and ch["priority"] not in PRIORITIES:
        raise HTTPException(422, "invalid status or priority")
    with db() as con:
        cur = con.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if not cur:
            raise HTTPException(404, "task not found")
        if "status" in ch:
            ch["completed"] = datetime.now().isoformat(timespec="seconds") if ch["status"] == "done" else None
        if "tags" in ch:
            ch["tags"] = json.dumps(ch["tags"])
        if ch:
            con.execute(f"UPDATE tasks SET {', '.join(k + '=?' for k in ch)} WHERE id=?", (*ch.values(), tid))
        out = row(con.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchone())
    publish("updated", {"task": out})
    return out


@app.delete("/api/tasks/{tid}")
def delete(tid: int) -> dict:
    with db() as con:
        if not con.execute("DELETE FROM tasks WHERE id=?", (tid,)).rowcount:
            raise HTTPException(404, "task not found")
    publish("deleted", {"id": tid})
    return {"ok": True}


@app.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue()
    subscribers.add(q)

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while not await request.is_disconnected():
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=10)
                    yield f"data: {json.dumps(ev)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            subscribers.discard(q)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
