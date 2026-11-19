"""Project 04 — Market Basket Pattern Mining: live rule-based recommender, rule explorer, network graph, DS dashboard."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
pipeline = load_local(__file__, "pipeline")
_s: dict = {}


def M() -> dict:
    if "m" not in _s:
        if not (ROOT / "artifacts" / "metrics.json").exists():
            pipeline.main()
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
    return _s["m"]


def rules() -> pd.DataFrame:
    if "r" not in _s:
        try:
            _s["r"] = pd.read_pickle(ROOT / "artifacts" / "rules.pkl")
        except Exception:
            pipeline.main()
            _s["r"] = pd.read_pickle(ROOT / "artifacts" / "rules.pkl")
    return _s["r"]


class Basket(BaseModel):
    items: list[str] = []


class RuleQuery(BaseModel):
    contains: str = ""
    min_lift: float = 1.0
    min_conf: float = 0.0
    sort: str = "lift"


def p_reco() -> list[dict]:
    m = M()
    t = m["test"]
    return [H("Basket recommender", "Rules mined with FP-Growth fire on whatever is in the basket; each suggestion shows the rule that produced it."),
            KPIS(K("Baskets", m["n_tx"]), K("Distinct items", m["n_items"]), K("Rules in the book", m["n_rules"]),
                 K("Held-out hit-rate@3", f"{100 * t['rules']['hit_rate']:.1f}%", hint=f"popularity baseline {100 * t['popularity']['hit_rate']:.1f}%")),
            CUSTOM("basket", {"items": [k for k, _ in m["item_freq"]] + [i for i in ["butter", "curd", "domestic eggs", "whipped/sour cream", "frozen vegetables", "beef", "chicken", "onions", "hamburger meat", "sugar", "flour", "herbs", "red/blush wine", "liquor"] if i not in dict(m["item_freq"])],
                             "start": ["yogurt", "root vegetables"]}, title="Build a basket")]


def p_rules() -> list[dict]:
    m = M()
    return [H("Rule explorer", f"{m['n_rules']} rules at support ≥ {m['best']['min_support']}, confidence ≥ {m['best']['min_conf']}, lift ≥ {m['best']['min_lift']}."),
            FORM("api/rules", [F("contains", "Item contains", "text", "", placeholder="e.g. yogurt"), F("min_lift", "Min lift", "range", 1.5, min=1, max=4, step=0.1),
                               F("min_conf", "Min confidence", "range", 0.2, min=0, max=0.7, step=0.05), F("sort", "Sort by", "select", "lift", options=["lift", "confidence", "support", "conviction", "leverage"])],
                 title="Filter the rule book", submit="Search rules", auto=True, run_on_load=True),
            CHART("scatter", "Support × confidence of every rule", [S("lift < 2", [p for p in m["scatter"] if p["lift"] < 2], r=3, color="--axis"), S("lift 2–3", [p for p in m["scatter"] if 2 <= p["lift"] < 3], r=3.5, color="1"), S("lift ≥ 3", [p for p in m["scatter"] if p["lift"] >= 3], r=4.5, color="2")],
                  x="support", y="confidence", height=340, sub="High-lift rules live at low support — interesting but rare; high-confidence rules mostly predict *whole milk*.")]


def p_graph() -> list[dict]:
    m = M()
    return [H("Rule network", "Top 45 rules by lift. Node size = number of rules touching the item; arrow darkness and width = lift."),
            CUSTOM("rulegraph", {"edges": m["edges"]}),
            HEAT("Pairwise lift among the 14 most frequent items", m["lift_items"], m["lift_items"], m["lift_matrix"], digits=1, labw=120, sub="Lift > 1 means the pair co-occurs more often than independence predicts.")]


def p_bench() -> list[dict]:
    m = M()
    b = m["bench"]
    t = m["test"]
    return [H("Algorithms & held-out evaluation"),
            GRID(2, CHART("line", "Apriori vs FP-Growth runtime", [S("Apriori", [r["apriori_s"] for r in b], points=True), S("FP-Growth", [r["fpgrowth_s"] for r in b], points=True)], [str(r["min_support"]) for r in b], x="min support (decreasing →)", y="seconds",
                       sub="Both return identical itemsets (asserted in the pipeline). Apriori wins at high support; FP-Growth scales better as the candidate space explodes."),
                 CHART("bar", "Frequent itemsets found", [S("itemsets", [r["itemsets"] for r in b], color="3")], [str(r["min_support"]) for r in b], x="min support", y="itemsets")),
            MD(f"""## Does the rule book actually predict anything?
Most association-mining demos stop at "here are rules with high lift". This one asks the harder question: *hide one item from an unseen basket — do the rules recover it?*

- Split: **{m['split']['train']:,} train / {m['split']['val']:,} validation / {m['split']['test']:,} test** baskets.
- AutoResearch tunes `min_support`, `min_conf`, `min_lift` and the ranking metric on **validation** hit-rate@3.
- The winning thresholds are refit on train+validation and scored **once** on the test baskets.

| Recommender | Test hit-rate@3 | Baskets with a rule firing |
|---|---|---|
| Popularity only (top-3 items) | {100 * t['popularity']['hit_rate']:.1f}% | — |
| Rules only | {100 * t['rules_only']['hit_rate']:.1f}% | {100 * t['rules_only']['coverage']:.0f}% |
| **Rules + popularity back-fill** | **{100 * t['rules']['hit_rate']:.1f}%** | {100 * t['rules']['coverage']:.0f}% |

**Honest reading:** the rule book beats a popularity baseline by only a small margin. Grocery baskets are dominated by a few staples (whole milk is in 26% of baskets), so "recommend milk" is hard to beat on hit-rate. The value of the rules is *explanatory* (which products travel together, for shelf placement and bundles) more than predictive.""", card=True),
            *ar_blocks(m["autoresearch"], "val hit-rate@3")]


def p_admin() -> list[dict]:
    m = M()
    return [H("Data-science admin dashboard"),
            KPIS(K("Avg basket size", m["avg_size"], digits=2), K("Matrix density", f"{100 * m['density']:.2f}%", hint="share of 1s in the basket × item matrix"),
                 K("Frequent pairs (sup ≥ 1%)", m["n_pairs"]), K("Pipeline runtime", f"{m['secs']} s")),
            GRID(2, CHART("hbar", "Top 25 items by support", [S("support", [v for _, v in m["item_freq"]])], [k for k, _ in m["item_freq"]], height=520),
                 CHART("bar", "Basket size distribution", [S("baskets", [v for _, v in m["size_hist"]], color="7")], [k for k, _ in m["size_hist"]], x="items in basket", y="baskets", height=520, sub="A long tail: 22% of baskets hold a single item and can never fire a rule.")),
            MD("""- **Metrics**: support P(A∪B), confidence P(B|A), lift = confidence / P(B), leverage = P(A∪B) − P(A)P(B), conviction = (1 − P(B)) / (1 − confidence)
- **Leakage control**: thresholds chosen on validation baskets; the test baskets are never seen while mining or tuning
- **Reproducibility**: fixed seed 42 for the split and the hidden-item draw
- **Limits**: one month of one store (1990s German grocer); no prices, timestamps or customer ids — so no sequence or personalisation""", title="Method notes")]


CRISPDM = [
    ("Business understanding", "A grocer wants to know which products are bought together — for shelf layout, bundles and next-item suggestions at checkout."),
    ("Data understanding", "9,835 baskets × 169 items, average 4.4 items per basket, 2.6% dense. Whole milk, other vegetables and rolls/buns dominate."),
    ("Data preparation", "Parse the ragged CSV, de-duplicate items within a basket, one-hot encode with a fixed column order shared by train / validation / test."),
    ("Modeling", "Apriori and FP-Growth benchmarked at six support levels (identical outputs asserted). Rules scored with five interestingness measures."),
    ("Evaluation", "Leave-one-out hit-rate@3 on held-out baskets versus a popularity baseline; thresholds tuned by AutoResearch on a validation split only."),
    ("Deployment", "`POST api/recommend` fires the rule book on any basket and returns the explaining rule; `POST api/rules` queries the book."),
]

app = make_app(__file__, "04", "Market Basket Pattern Mining", "Apriori vs FP-Growth · rule network · held-out recommender evaluation", [
    ("recommender", "Basket recommender", "App", p_reco),
    ("rules", "Rule explorer", "App", p_rules),
    ("graph", "Rule network", "App", p_graph),
    ("bench", "Algorithms & evaluation", "Data science", p_bench),
    ("admin", "Admin dashboard", "Data science", p_admin),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/recommend")
def recommend(b: Basket) -> dict:
    m = M()
    items = set(b.items) & set(m["items"])
    recs = pipeline.recommend(rules(), items, k=5, score=m["best"]["rank_by"])
    if not items:
        return {"recommendations": [], "blocks": [CALLOUT("info", "Empty basket", "Add at least one product.")]}
    if not recs:
        return {"recommendations": [], "blocks": [CALLOUT("warn", "No rule fires for this basket", f"Fallback to most popular items: **{', '.join(k for k, _ in m['item_freq'][:3])}**.")]}
    return {"recommendations": [r[0] for r in recs],
            "blocks": [TABLE(f"Customers who bought {{{', '.join(sorted(items))}}} also buy…", ["suggested item", m["best"]["rank_by"], "because of rule"],
                             [[c, {"bar": s if m["best"]["rank_by"] == "confidence" else min(1, s / 4), "text": f"{s:.2f}"}, why] for c, s, why in recs])]}


@app.post("/api/rules")
def query_rules(q: RuleQuery) -> dict:
    r = rules()
    r = r[(r.lift >= q.min_lift) & (r.confidence >= q.min_conf)]
    if q.contains.strip():
        s = q.contains.strip().lower()
        r = r[r.antecedents.apply(lambda a: any(s in x for x in a)) | r.consequent.str.contains(s, regex=False)]
    r = r.sort_values(q.sort if q.sort in r.columns else "lift", ascending=False).head(60)
    rows = [[", ".join(sorted(a)), c, round(s, 4), round(cf, 3), round(l, 2), round(lv, 4), round(cv, 2) if cv != float("inf") else None]
            for a, c, s, cf, l, lv, cv in zip(r.antecedents, r.consequent, r.support, r.confidence, r.lift, r.leverage, r.conviction)]
    return {"count": len(rows), "blocks": [TABLE(f"{len(rows)} rules shown", ["if basket has", "then also", "support", "confidence", "lift", "leverage", "conviction"], rows, maxh=420)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8004)
