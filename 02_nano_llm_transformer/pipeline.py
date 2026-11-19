"""Project 02 — NanoLlama training pipeline (CPU-only, ~20 minutes on 2 cores).

Stage 0  AutoResearch ablation   — short runs toggling RoPE / SwiGLU / RMSNorm / width / lr, judged on validation loss
Stage 1  Pretraining             — next-character prediction on TinyShakespeare + the concept notes
Stage 2  Supervised fine-tuning  — chat-formatted Q&A; loss only on the assistant's reply
Evaluation: validation perplexity (held-out text) and exact-match on question phrasings never seen in SFT.
Run:  python pipeline.py            (set NANO_FAST=1 for a 1-minute smoke run)
"""
from __future__ import annotations
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import save_json  # noqa: E402
from model import CharTokenizer, Config, NanoLlama  # noqa: E402
from sft_data import CONCEPTS, HELD_OUT_TEMPLATES, TRAIN_TEMPLATES  # noqa: E402

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 1337
FAST = bool(os.environ.get("NANO_FAST"))
torch.set_num_threads(2)


def lr_at(step: int, total: int, peak: float, warmup: int = 100) -> float:
    """Linear warm-up then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))


def batch(data: torch.Tensor, ctx: int, bs: int, g: torch.Generator):
    ix = torch.randint(len(data) - ctx - 1, (bs,), generator=g)
    return torch.stack([data[i:i + ctx] for i in ix]), torch.stack([data[i + 1:i + ctx + 1] for i in ix])


@torch.no_grad()
def eval_loss(model, data, ctx, n=12, bs=32) -> float:
    model.eval()
    g = torch.Generator().manual_seed(0)  # fixed evaluation batches -> comparable across runs
    out = float(np.mean([model(*batch(data, ctx, bs, g))[1].item() for _ in range(n)]))
    model.train()
    return out


def train_lm(model, train, val, steps, peak_lr, bs=32, log_every=100, tag="pretrain"):
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    g, curve, t0 = torch.Generator().manual_seed(SEED), [], time.time()
    for s in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s, steps, peak_lr)
        x, y = batch(train, model.c.ctx, bs, g)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % log_every == 0 or s == steps - 1:
            v = eval_loss(model, val, model.c.ctx)
            curve.append({"step": s, "train": round(loss.item(), 4), "val": round(v, 4), "lr": round(lr_at(s, steps, peak_lr), 6)})
            print(f"  [{tag}] step {s:5d}  train {loss.item():.3f}  val {v:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return curve


def sft_examples(tok: CharTokenizer, templates: list[str]) -> list[tuple[list[int], int, str, str]]:
    out = []
    for c, a in CONCEPTS.items():
        for t in templates:
            q = t.format(c=c)
            ids = tok.chat(q, a)
            out.append((ids, len(tok.chat(q)), q, a))
    return out


def train_sft(model, tok, examples, steps, peak_lr, bs=24):
    """Loss is masked to the assistant's reply: the model is not rewarded for predicting the user's question."""
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.01)
    rs, curve, t0 = np.random.RandomState(SEED), [], time.time()
    L = max(len(e[0]) for e in examples)
    assert L <= model.c.ctx, f"SFT example of {L} tokens exceeds context {model.c.ctx}"
    for s in range(steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(s, steps, peak_lr, warmup=50)
        x = torch.full((bs, L - 1), tok.end, dtype=torch.long)
        y = torch.full((bs, L - 1), -100, dtype=torch.long)
        for b, j in enumerate(rs.randint(len(examples), size=bs)):
            ids, n_prompt, _, _ = examples[j]
            t = torch.tensor(ids)
            x[b, :len(ids) - 1] = t[:-1]
            y[b, n_prompt - 1:len(ids) - 1] = t[n_prompt:]
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 100 == 0 or s == steps - 1:
            curve.append({"step": s, "loss": round(loss.item(), 4)})
            print(f"  [sft] step {s:5d}  loss {loss.item():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return curve


@torch.no_grad()
def answer(model, tok, question: str, temperature=0.0, top_k=20, top_p=1.0, max_new=150) -> tuple[str, list[float]]:
    model.eval()
    q = question.strip().lower()[:60]
    ids = torch.tensor([tok.chat(q)])
    out, probs = model.generate(ids, max_new, temperature=temperature, top_k=top_k, top_p=top_p, stop=tok.end)
    return tok.decode(out[0, ids.shape[1]:].tolist()), probs


def char_sim(a: str, b: str) -> float:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> None:
    t0 = time.time()
    torch.manual_seed(SEED)
    shakespeare = (ROOT / "data" / "tinyshakespeare.txt").read_text()
    notes = "\n".join(f"{c}: {a}" for c, a in CONCEPTS.items()) + "\n"
    templates = TRAIN_TEMPLATES + HELD_OUT_TEMPLATES
    chars = sorted(set(shakespeare + notes + "".join(templates)))
    tok = CharTokenizer(chars)
    enc = lambda s: torch.tensor(tok.encode(s), dtype=torch.long)  # noqa: E731
    n = int(0.9 * len(shakespeare))
    train, val = torch.cat([enc(shakespeare[:n])] + [enc(notes)] * 20), enc(shakespeare[n:])  # validation = unseen final 10% of the plays

    # ---------------- Stage 0: AutoResearch ablation (short budget per experiment)
    abl_steps = 40 if FAST else 150

    def experiment(c: dict) -> float:
        torch.manual_seed(SEED)
        m = NanoLlama(Config(vocab_size=tok.vocab_size, d_model=c["d_model"], n_layer=c["n_layer"], n_head=4, ctx=128, dropout=c["dropout"],
                             use_rope=c["rope"], use_swiglu=c["swiglu"], use_rmsnorm=c["rmsnorm"]))
        opt = torch.optim.AdamW(m.parameters(), lr=c["lr"], betas=(0.9, 0.95), weight_decay=0.1)
        g = torch.Generator().manual_seed(SEED)
        for s in range(abl_steps):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(s, abl_steps, c["lr"], warmup=30)
            _, loss = m(*batch(train, 128, 24, g))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        return eval_loss(m, val, 128, n=6)

    print("== Stage 0: AutoResearch ablation ==")
    # A fixed research agenda (instead of random mutations) so that every primitive is guaranteed to be tested once.
    agenda = [("rope", True), ("rmsnorm", True), ("swiglu", True), ("lr", 2e-3), ("lr", 3e-3), ("dropout", 0.0), ("d_model", 128), ("n_layer", 4)]
    best = {"rope": False, "swiglu": False, "rmsnorm": False, "d_model": 96, "n_layer": 3, "lr": 1e-3, "dropout": 0.1}
    t1 = time.time()
    best_score = experiment(best)
    hist = [{"iter": 0, "change": "baseline: learned positions + GELU MLP + LayerNorm", "config": dict(best), "score": best_score, "accepted": True, "best": best_score, "secs": round(time.time() - t1, 1)}]
    for i, (k, v) in enumerate(agenda[:2] if FAST else agenda, 1):
        cand, t1 = {**best, k: v}, time.time()
        sc = experiment(cand)
        ok = sc < best_score
        change = f"{k}: {best[k]} → {v}"
        if ok:
            best, best_score = cand, sc
        hist.append({"iter": i, "change": change, "config": cand, "score": sc, "accepted": ok, "best": best_score, "secs": round(time.time() - t1, 1)})
        print(f"  [ablation] {i:02d} {'KEEP   ' if ok else 'discard'} {sc:.4f}  best={best_score:.4f}  {change}", flush=True)
    ar = {"best": best, "best_score": best_score, "history": hist, "maximize": False, "n_trials": len(hist), "n_accepted": sum(h["accepted"] for h in hist)}
    b = best

    # ---------------- Stage 1: pretraining with the winning primitives, at full width/depth
    cfg = Config(vocab_size=tok.vocab_size, d_model=128, n_layer=4, n_head=4, ctx=192, dropout=0.05, use_rope=True, use_swiglu=True, use_rmsnorm=True)  # the Llama-style block is the design brief; the ablation above reports what each piece bought at this scale
    model = NanoLlama(cfg)
    print(f"== Stage 1: pretraining ({model.n_params():,} parameters) ==")
    pre_curve = train_lm(model, train, val, 80 if FAST else 1600, peak_lr=min(b["lr"], 2e-3), bs=24)
    val_loss = eval_loss(model, val, cfg.ctx, n=30)
    model.eval()
    sample_pre = tok.decode(model.generate(torch.tensor([tok.encode("ROMEO:\n")]), 300, temperature=0.8, top_k=40)[0][0].tolist())

    # ---------------- Stage 2: supervised fine-tuning on chat-formatted Q&A
    print("== Stage 2: supervised fine-tuning ==")
    model.train()
    tr_ex, te_ex = sft_examples(tok, TRAIN_TEMPLATES), sft_examples(tok, HELD_OUT_TEMPLATES)
    sft_curve = train_sft(model, tok, tr_ex, 60 if FAST else 1300, peak_lr=1e-3, bs=16)
    val_after = eval_loss(model, val, cfg.ctx, n=30)

    def score(examples):
        rows = []
        for _, _, q, a in examples:
            got, _ = answer(model, tok, q)
            rows.append({"q": q, "expected": a, "got": got, "exact": got.strip() == a, "sim": char_sim(got.strip(), a)})
        return rows

    seen, unseen = score(tr_ex[::7]), score(te_ex)
    evalr = {"seen_exact": float(np.mean([r["exact"] for r in seen])), "unseen_exact": float(np.mean([r["exact"] for r in unseen])),
             "unseen_sim": float(np.mean([r["sim"] for r in unseen])), "n_seen": len(seen), "n_unseen": len(unseen)}
    ood, _ = answer(model, tok, "who won the world cup?")
    print("EVAL:", evalr)

    torch.save({"state": model.state_dict(), "config": cfg.to_dict(), "chars": chars}, ART / "nanollama.pt")
    groups = {"token embedding (tied with head)": model.tok.weight.numel(), "attention (qkv + out)": sum(p.numel() for bl in model.blocks for p in bl.att.parameters()),
              "feed-forward": sum(p.numel() for bl in model.blocks for p in bl.ff.parameters()), "norms": sum(p.numel() for n_, p in model.named_parameters() if ".n1." in n_ or ".n2." in n_ or n_.startswith("norm"))}
    save_json(ART / "metrics.json", {
        "config": cfg.to_dict(), "n_params": model.n_params(), "param_groups": groups, "vocab": tok.itos, "corpus_chars": len(shakespeare), "train_tokens": len(train), "val_tokens": len(val),
        "autoresearch": ar, "pretrain_curve": pre_curve, "sft_curve": sft_curve, "val_loss": val_loss, "val_ppl": math.exp(val_loss), "val_loss_after_sft": val_after,
        "uniform_ppl": tok.vocab_size, "sample_pretrain": sample_pre, "eval": evalr, "unseen_rows": unseen, "ood_example": {"q": "who won the world cup?", "got": ood},
        "n_concepts": len(CONCEPTS), "n_sft_train": len(tr_ex), "n_sft_test": len(te_ex), "train_templates": TRAIN_TEMPLATES, "held_out_templates": HELD_OUT_TEMPLATES,
        "torch": torch.__version__, "fast_mode": FAST, "secs": round(time.time() - t0, 1)})
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
