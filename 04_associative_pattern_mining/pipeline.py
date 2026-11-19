"""Project 04 — Market-basket pattern mining on the Kaggle/UCI "Groceries" dataset (9,835 baskets, 169 items).

Apriori vs FP-Growth benchmark, full rule metrics (support, confidence, lift, leverage, conviction),
and — unusually for association mining — a *held-out* evaluation: rules mined on 80% of baskets are scored
as a next-item recommender on the other 20% (leave-one-out hit-rate@3). AutoResearch tunes thresholds on a
validation split.   Run:  python pipeline.py
"""
from __future__ import annotations
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hill_climb, save_json  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42


def load() -> list[list[str]]:
    return [sorted(set(x.strip() for x in line.split(",") if x.strip())) for line in (ROOT / "data" / "groceries.csv").read_text().splitlines() if line.strip()]


def onehot(tx: list[list[str]], cols: list[str] | None = None) -> pd.DataFrame:
    te = TransactionEncoder()
    if cols is None:
        return pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
    te.columns_ = cols
    te.columns_mapping_ = {c: i for i, c in enumerate(cols)}
    return pd.DataFrame(te.transform(tx), columns=cols)


def mine(df: pd.DataFrame, min_support: float, min_conf: float, min_lift: float, max_len: int = 3) -> pd.DataFrame:
    fi = fpgrowth(df, min_support=min_support, use_colnames=True, max_len=max_len)
    if fi.empty:
        return pd.DataFrame()
    r = association_rules(fi, metric="confidence", min_threshold=min_conf)
    r = r[(r.lift >= min_lift) & (r.consequents.apply(len) == 1)].copy()
    r["consequent"] = r.consequents.apply(lambda s: next(iter(s)))
    return r.sort_values(["confidence", "lift"], ascending=False).reset_index(drop=True)


def recommend(rules: pd.DataFrame, basket: set[str], k: int = 3, score: str = "confidence") -> list[tuple[str, float, str]]:
    """Fire every rule whose antecedent is contained in the basket; rank unseen consequents by best score."""
    best: dict[str, tuple[float, str]] = {}
    for ante, cons, s, lift in zip(rules.antecedents, rules.consequent, rules[score], rules.lift):
        if cons not in basket and ante <= basket and s > best.get(cons, (0, ""))[0]:
            best[cons] = (float(s), f"{{{', '.join(sorted(ante))}}} → {cons} (lift {lift:.2f})")
    return [(c, s, why) for c, (s, why) in sorted(best.items(), key=lambda kv: -kv[1][0])[:k]]


def hit_rate(rules: pd.DataFrame, tx: list[list[str]], k: int = 3, score: str = "confidence", popular: list[str] | None = None) -> dict:
    """Leave-one-out: hide one random item of each basket (size>=2) and check whether it is recommended."""
    rs = np.random.RandomState(SEED)
    hits = n = covered = 0
    for b in tx:
        if len(b) < 2:
            continue
        hidden = b[rs.randint(len(b))]
        rest = set(b) - {hidden}
        recs = [c for c, _, _ in recommend(rules, rest, k, score)] if len(rules) else []
        covered += bool(recs)
        if popular:  # back-fill with most popular items so every basket gets k suggestions
            recs += [p for p in popular if p not in rest and p not in recs][: k - len(recs)]
        hits += hidden in recs
        n += 1
    return {"hit_rate": hits / n, "coverage": covered / n, "n": n}


def main() -> None:
    t0 = time.time()
    tx = load()
    rs = np.random.RandomState(SEED)
    idx = rs.permutation(len(tx))
    n_tr, n_va = int(0.6 * len(tx)), int(0.2 * len(tx))
    tr, va, te = [tx[i] for i in idx[:n_tr]], [tx[i] for i in idx[n_tr:n_tr + n_va]], [tx[i] for i in idx[n_tr + n_va:]]
    df_all = onehot(tx)
    cols = list(df_all.columns)
    df_tr = onehot(tr, cols)
    freq = df_all.mean().sort_values(ascending=False)
    popular_tr = df_tr.mean().sort_values(ascending=False).index.tolist()

    print("== Apriori vs FP-Growth benchmark ==")
    bench = []
    for ms in [0.05, 0.03, 0.02, 0.01, 0.007, 0.005]:
        t1 = time.time(); a = apriori(df_all, min_support=ms, use_colnames=True, max_len=4); ta = time.time() - t1
        t1 = time.time(); f = fpgrowth(df_all, min_support=ms, use_colnames=True, max_len=4); tf = time.time() - t1
        assert len(a) == len(f), "algorithms must agree on the frequent itemsets"
        bench.append({"min_support": ms, "itemsets": len(f), "apriori_s": round(ta, 3), "fpgrowth_s": round(tf, 3)})
        print(f"  support≥{ms:<6} itemsets={len(f):<5} apriori={ta:.2f}s fpgrowth={tf:.2f}s")

    print("== AutoResearch: tune thresholds for validation hit-rate@3 ==")
    pop_base = hit_rate(pd.DataFrame(), va, popular=popular_tr)
    ar = hill_climb(lambda c: hit_rate(mine(df_tr, c["min_support"], c["min_conf"], c["min_lift"]), va, score=c["rank_by"], popular=popular_tr)["hit_rate"],
                    {"min_support": [0.002, 0.003, 0.005, 0.008, 0.01, 0.02], "min_conf": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4], "min_lift": [1.0, 1.1, 1.3, 1.5, 2.0], "rank_by": ["confidence", "lift"]},
                    {"min_support": 0.01, "min_conf": 0.3, "min_lift": 1.0, "rank_by": "confidence"}, n_iter=22, seed=SEED)
    b = ar["best"]
    # refit on train+validation with the chosen thresholds, then score the untouched test baskets once
    df_trva = onehot(tr + va, cols)
    rules = mine(df_trva, b["min_support"], b["min_conf"], b["min_lift"])
    pop_trva = df_trva.mean().sort_values(ascending=False).index.tolist()
    test = {"rules": hit_rate(rules, te, score=b["rank_by"], popular=pop_trva), "rules_only": hit_rate(rules, te, score=b["rank_by"]),
            "popularity": hit_rate(pd.DataFrame(), te, popular=pop_trva), "val_popularity": pop_base["hit_rate"]}
    print("TEST hit@3:", {k: round(v["hit_rate"], 4) if isinstance(v, dict) else round(v, 4) for k, v in test.items()})

    # final rule book for the UI is mined on all data with the chosen thresholds
    final = mine(df_all, b["min_support"], b["min_conf"], b["min_lift"])
    final["antecedent"] = final.antecedents.apply(lambda s: ", ".join(sorted(s)))
    rule_rows = final[["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction"]].replace([np.inf], np.nan)
    top = final.sort_values("lift", ascending=False).head(45)
    edges = [{"a": sorted(a), "c": c, "lift": float(l), "conf": float(cf)} for a, c, l, cf in zip(top.antecedents, top.consequent, top.lift, top.confidence)]
    sizes = Counter(len(t) for t in tx)
    pair = fpgrowth(df_all, min_support=0.01, use_colnames=True, max_len=2)
    top_items = freq.index[:14].tolist()
    co = [[float((df_all[a] & df_all[b]).mean() / (freq[a] * freq[b])) if a != b else None for b in top_items] for a in top_items]
    final.drop(columns=["antecedent"]).to_pickle(ART / "rules.pkl")
    save_json(ART / "metrics.json", {
        "n_tx": len(tx), "n_items": len(cols), "avg_size": float(np.mean([len(t) for t in tx])), "density": float(df_all.values.mean()),
        "item_freq": [[k, float(v)] for k, v in freq.head(25).items()], "size_hist": [[k, sizes[k]] for k in sorted(sizes) if k <= 20],
        "bench": bench, "autoresearch": ar, "best": b, "test": test, "n_rules": len(final), "n_pairs": int((pair.itemsets.apply(len) == 2).sum()),
        "rules": rule_rows.round(4).values.tolist()[:400], "edges": edges, "items": cols, "lift_items": top_items, "lift_matrix": co,
        "scatter": [{"x": round(float(s), 4), "y": round(float(c), 3), "lift": round(float(l), 2)} for s, c, l in zip(final.support, final.confidence, final.lift)][:1500],
        "split": {"train": len(tr), "val": len(va), "test": len(te)}, "secs": round(time.time() - t0, 1)})
    print(f"{len(final)} rules, done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
