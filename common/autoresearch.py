"""AutoResearch: a transparent greedy hill-climber over a discrete config space.

Inspired by the 'autoresearch' loop (propose one change -> run experiment -> keep only if the
validation metric improves). Every trial is logged, so the dashboard can show the full research
trajectory rather than only the winner. The evaluate() callback must only ever see training /
validation data -- the held-out test set is scored exactly once, after the climb is over.
"""
from __future__ import annotations
import random
import time
from typing import Any, Callable

from .blocks import CHART, S, TABLE, GRID, PILL, KPIS, K


def hill_climb(evaluate: Callable[[dict], float], space: dict[str, list], start: dict,
               n_iter: int = 20, seed: int = 42, maximize: bool = True, verbose: bool = True) -> dict:
    rng = random.Random(seed)
    best, t0 = dict(start), time.time()
    best_score = float(evaluate(best))
    history = [{"iter": 0, "change": "baseline", "config": dict(best), "score": best_score,
                "accepted": True, "best": best_score, "secs": round(time.time() - t0, 2)}]
    seen = {tuple(sorted((k, str(v)) for k, v in best.items()))}
    for i in range(1, n_iter + 1):
        cand = None
        for _ in range(50):  # propose a single-parameter mutation we have not tried yet
            k = rng.choice(list(space))
            opts = [v for v in space[k] if v != best[k]]
            if not opts:
                continue
            c = dict(best)
            c[k] = rng.choice(opts)
            key = tuple(sorted((kk, str(v)) for kk, v in c.items()))
            if key not in seen:
                seen.add(key)
                cand, change = c, f"{k}: {best[k]} → {c[k]}"
                break
        if cand is None:
            break
        t1 = time.time()
        try:
            score = float(evaluate(cand))
        except Exception as e:  # a failed experiment is still an experiment
            score, change = float("-inf") if maximize else float("inf"), change + f" (failed: {type(e).__name__})"
        ok = score > best_score if maximize else score < best_score
        if ok:
            best, best_score = cand, score
        history.append({"iter": i, "change": change, "config": cand, "score": score, "accepted": ok,
                        "best": best_score, "secs": round(time.time() - t1, 2)})
        if verbose:
            print(f"  [autoresearch] {i:02d} {'KEEP   ' if ok else 'discard'} {score:.5f}  best={best_score:.5f}  {change}")
    return {"best": best, "best_score": best_score, "history": history, "maximize": maximize,
            "n_trials": len(history), "n_accepted": sum(h["accepted"] for h in history)}


def ar_blocks(ar: dict, metric: str) -> list[dict]:
    """Dashboard blocks for an AutoResearch run."""
    h = ar["history"]
    fin = [x for x in h if x["score"] is not None and abs(x["score"]) != float("inf")]
    return [
        KPIS(K("Experiments run", ar["n_trials"]), K("Accepted (kept)", ar["n_accepted"]),
             K(f"Baseline {metric}", h[0]["score"], digits=4), K(f"Best {metric}", ar["best_score"], digits=4,
               hint="validation only — test set untouched")),
        GRID(2,
             CHART("line", "Hill-climbing trajectory", [
                 S("best so far", [x["best"] for x in h], stepped=True),
                 S("trial score", [x["score"] if x in fin else None for x in h], color="2", points=True, width=1, dash=True)],
                 [x["iter"] for x in h], x="experiment #", y=metric,
                 sub="Each experiment mutates one hyper-parameter; it is kept only if validation improves."),
             TABLE("Winning configuration", ["parameter", "value"], [[k, str(v)] for k, v in ar["best"].items()])),
        TABLE("Experiment log", ["#", "change proposed", metric, "decision", "best so far", "secs"],
              [[x["iter"], x["change"], x["score"] if x in fin else None,
                PILL("keep", "good") if x["accepted"] else PILL("discard"), x["best"], x["secs"]] for x in h], maxh=340, digits=4),
    ]
