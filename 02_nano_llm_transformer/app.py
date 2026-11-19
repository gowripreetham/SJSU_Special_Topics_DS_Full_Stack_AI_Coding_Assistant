"""Project 02 — NanoLlama: chat studio, completion playground and a training/architecture dashboard for a from-scratch LLM."""
from __future__ import annotations
import inspect
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

ROOT = Path(__file__).resolve().parent
_s: dict = {}


def M() -> dict:
    if "m" not in _s:
        _s["m"] = load_json(ROOT / "artifacts" / "metrics.json")
        if _s["m"] is None:
            raise RuntimeError("no trained checkpoint yet — run `python pipeline.py` in 02_nano_llm_transformer (≈20 min on CPU)")
    return _s["m"]


def LM():
    """Lazy-load torch + checkpoint on first use so the hub starts instantly."""
    if "lm" not in _s:
        import torch
        mod, pipe = load_local(__file__, "model"), load_local(__file__, "pipeline")
        ck = torch.load(ROOT / "artifacts" / "nanollama.pt", map_location="cpu")
        tok = mod.CharTokenizer(ck["chars"])
        model = mod.NanoLlama(mod.Config(**ck["config"]))
        model.load_state_dict(ck["state"])
        model.eval()
        _s["lm"] = (model, tok, pipe, torch, mod)
    return _s["lm"]


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=200)
    temperature: float = Field(0.0, ge=0, le=2)
    top_k: int = Field(20, ge=1, le=100)


class Complete(BaseModel):
    prompt: str = Field("ROMEO:\n", max_length=300)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int = Field(40, ge=1, le=100)
    top_p: float = Field(1.0, ge=0.1, le=1)
    max_new: int = Field(300, ge=10, le=600)


class Att(BaseModel):
    text: str = Field("what is overfitting?", max_length=40)
    layer: int = Field(3, ge=0, le=7)


def p_chat() -> list[dict]:
    m = M()
    return [H("Chat studio", "A decoder-only transformer with RoPE, RMSNorm and SwiGLU — written, trained and served from scratch, no pretrained weights."),
            CUSTOM("chat", {"params": m["n_params"], "n": m["n_concepts"], "suggestions": ["what is overfitting?", "explain rope", "help me understand data leakage", "define crisp-dm", "what is meant by attention?", "tell me about shap values", "who won the world cup?"]})]


def p_play() -> list[dict]:
    return [H("Completion playground", "Raw next-character generation from the same weights. Start a line of Shakespeare and vary the sampling knobs."),
            FORM("api/complete", [F("prompt", "Prompt", "textarea", "KING HENRY:\nOnce more unto the", rows=3), F("temperature", "Temperature", "range", 0.8, min=0, max=1.5, step=0.1),
                                  F("top_k", "Top-k", "range", 40, min=1, max=80, step=1), F("top_p", "Top-p (nucleus)", "range", 1.0, min=0.5, max=1, step=0.05), F("max_new", "New tokens", "range", 300, min=50, max=600, step=50)],
                 submit="Generate", run_on_load=True),
            CALLOUT("info", "What to expect", "After SFT the model's Shakespeare is rustier than right after pretraining (see *catastrophic forgetting* on the Training page). It produces play-shaped text with real words and speaker turns — not meaning. That is what 0.8M parameters buys.")]


def p_train() -> list[dict]:
    m = M()
    pc, sc = m["pretrain_curve"], m["sft_curve"]
    return [H("Training dashboard", f"Trained on CPU in {m['secs'] / 60:.0f} min with PyTorch {m['torch']}. Every number below was measured by the pipeline."),
            KPIS(K("Parameters", m["n_params"]), K("Validation perplexity", m["val_ppl"], digits=2, hint=f"uniform guessing = {m['uniform_ppl']}"), K("Val loss (nats/char)", m["val_loss"], digits=3),
                 K("Unseen-phrasing exact match", f"{100 * m['eval']['unseen_exact']:.0f}%", hint=f"{m['eval']['n_unseen']} held-out questions")),
            GRID(2, CHART("line", "Stage 1 — pretraining loss", [S("train (batch)", [p["train"] for p in pc]), S("validation (held-out 10% of the plays)", [p["val"] for p in pc], color="2")], [p["step"] for p in pc], x="step", y="cross-entropy (nats)"),
                 CHART("line", "Stage 2 — SFT loss (assistant tokens only)", [S("train", [p["loss"] for p in sc], color="3")], [p["step"] for p in sc], x="step", y="cross-entropy (nats)")),
            GRID(2, CHART("line", "Learning-rate schedule", [S("lr", [p["lr"] for p in pc], color="7")], [p["step"] for p in pc], x="step", y="learning rate", height=220, sub="Linear warm-up, cosine decay to 10%."),
                 CALLOUT("warn", "Catastrophic forgetting, measured", f"Validation loss on Shakespeare went from **{m['val_loss']:.3f}** after pretraining to **{m['val_loss_after_sft']:.3f}** after fine-tuning on Q&A. SFT specialises the model and costs general ability — the reason real systems use low learning rates, LoRA or data mixing.")),
            MD("## Stage 0 — AutoResearch ablation\nBefore the long run, short experiments toggle each primitive from a classic GPT-2-style baseline (learned positions, GELU MLP, LayerNorm). A change is kept only if validation loss drops.", card=False),
            *ar_blocks(m["autoresearch"], "val loss"),
            MD(f"**Sample right after pretraining** (temperature 0.8):\n```\n{m['sample_pretrain'][:420]}\n```", title="What the pretrained model writes")]


def p_arch() -> list[dict]:
    m = M()
    c = m["config"]
    mod = load_local(__file__, "model")
    g = m["param_groups"]
    return [H("Architecture", "Llama-style block: pre-norm → attention with rotary positions → pre-norm → gated feed-forward, with residual connections."),
            GRID(2, TABLE("Configuration", ["hyper-parameter", "value"], [[k, str(v)] for k, v in c.items()]),
                 CHART("doughnut", "Where the parameters live", [S("parameters", list(g.values()))], list(g), height=280)),
            TABS(("RoPE", [CODE("Rotary position embedding", inspect.getsource(mod.rope_cache) + "\n" + inspect.getsource(mod.apply_rope)), MD("Positions are injected by **rotating** each (even, odd) channel pair of the query and key by `position × frequency`. The dot product of two rotated vectors depends only on their *relative* offset, so attention generalises across positions and no positional parameters are learned.", card=False)]),
                 ("RMSNorm", [CODE("RMSNorm", inspect.getsource(mod.RMSNorm)), MD("LayerNorm without the mean subtraction and bias: divide by the root-mean-square and apply a learned gain. Fewer operations, same stabilising effect.", card=False)]),
                 ("SwiGLU", [CODE("Gated feed-forward", inspect.getsource(mod.FeedForward)), MD("`W2( SiLU(W1·x) ⊙ W3·x )` — one branch gates the other. The hidden size is scaled by ⅔ so the parameter count matches a standard 4× MLP.", card=False)]),
                 ("Sampling", [CODE("Temperature · top-k · top-p", inspect.getsource(mod.NanoLlama.generate))])),
            FORM("api/attention", [F("text", "Text (≤ 40 chars)", "text", "what is overfitting?"), F("layer", "Layer", "select", c["n_layer"] - 1, options=list(range(c["n_layer"])))], title="Attention explorer", submit="Show attention", run_on_load=True),
            TABLE("Tokenizer vocabulary", ["id", "token"], [[i, {"code": repr(t)}] for i, t in enumerate(m["vocab"])], maxh=220)]


def p_eval() -> list[dict]:
    m = M()
    e = m["eval"]
    rows = sorted(m["unseen_rows"], key=lambda r: r["sim"])
    return [H("Evaluation", "Does it generalise, or just memorise prompts? Fine-tuning used 7 question phrasings per concept; these 3 phrasings were held out."),
            KPIS(K("Seen phrasings — exact match", f"{100 * e['seen_exact']:.0f}%", hint=f"n = {e['n_seen']}"), K("Unseen phrasings — exact match", f"{100 * e['unseen_exact']:.0f}%", hint=f"n = {e['n_unseen']}"),
                 K("Unseen — character similarity", f"{100 * e['unseen_sim']:.0f}%"), K("Concepts", m["n_concepts"])),
            MD(f"- **Training phrasings**: {', '.join('`' + t + '`' for t in m['train_templates'])}\n- **Held-out phrasings**: {', '.join('`' + t + '`' for t in m['held_out_templates'])}\n- **Out-of-scope probe**: *“{m['ood_example']['q']}”* → “{m['ood_example']['got'][:110]}” — the model has no world knowledge and always answers with one of its 56 memorised explanations. The chat UI flags most such replies using the joint probability of the first 12 characters (how sure it was about *which* answer to give); that heuristic is imperfect and some off-topic questions still slip through.", title="Protocol"),
            TABLE("Held-out questions, worst first", ["question (unseen phrasing)", "model reply", "match", "similarity"], [[r["q"], r["got"][:140], PILL("exact", "good") if r["exact"] else PILL("differs", "warn"), round(r["sim"], 2)] for r in rows], maxh=460)]


CRISPDM = [
    ("Business understanding", "Goal: a chatbot that explains data-science concepts, built *entirely from scratch* and trainable on a laptop CPU — to understand every moving part of an LLM, not to compete with one."),
    ("Data understanding", "Pretraining corpus: TinyShakespeare (1.1 MB of plays) plus concept notes. SFT corpus: 56 concepts × 7 question phrasings in a `<|user|>…<|assistant|>…<|end|>` format. 99-symbol character vocabulary."),
    ("Data preparation", "Character tokenizer with 3 special chat tokens; last 10% of the plays held out for validation; SFT loss masked to the assistant's reply; 3 phrasings per concept held out for testing."),
    ("Modeling", "Decoder-only transformer (4 layers, 4 heads, width 128, context 192) with RoPE, RMSNorm, SwiGLU and weight tying. AdamW, warm-up + cosine decay, gradient clipping. AutoResearch ablation picks the primitives."),
    ("Evaluation", "Validation perplexity vs the uniform baseline, exact-match and similarity on unseen phrasings, a measured catastrophic-forgetting delta, and an out-of-scope probe."),
    ("Deployment", "FastAPI: `POST api/chat`, `POST api/complete`, `POST api/attention`. Weights load lazily on first request; replies carry confidence and tokens/second."),
]

app = make_app(__file__, "02", "NanoLlama LLM", "From-scratch transformer · RoPE · RMSNorm · SwiGLU · pretrain + SFT on CPU", [
    ("chat", "Chat studio", "App", p_chat),
    ("playground", "Completion playground", "App", p_play),
    ("training", "Training dashboard", "Data science", p_train),
    ("architecture", "Architecture", "Data science", p_arch),
    ("evaluation", "Evaluation", "Data science", p_eval),
    ("crispdm", "CRISP-DM report", "Docs", lambda: [H("CRISP-DM report"), CRISP(CRISPDM)]),
])


@app.post("/api/chat")
def chat(c: Chat) -> dict:
    model, tok, pipe, torch, _ = LM()
    t0 = time.time()
    torch.manual_seed(0)
    reply, probs = pipe.answer(model, tok, c.message, temperature=c.temperature, top_k=c.top_k)
    dt = time.time() - t0
    # Answer-selection confidence: joint probability of the first 12 characters. Once the model has committed to an answer the
    # remaining characters are near-certain, so the *mean* token probability is ~0.99 even for wrong answers and is useless here.
    conf = 1.0
    for pr in probs[:12]:
        conf *= pr
    return {"reply": reply.strip() or "(empty reply)", "confidence": conf, "tokens": len(probs), "ms": dt * 1000, "tok_per_s": len(probs) / max(dt, 1e-6)}


@app.post("/api/complete")
def complete(c: Complete) -> dict:
    model, tok, _, torch, _ = LM()
    t0 = time.time()
    ids = torch.tensor([tok.encode(c.prompt) or [tok.unk]])
    out, probs = model.generate(ids, c.max_new, temperature=c.temperature, top_k=c.top_k, top_p=c.top_p)
    dt = time.time() - t0
    text = tok.decode(out[0].tolist())
    return {"text": text, "blocks": [KPIS(K("Tokens generated", len(probs)), K("Speed", f"{len(probs) / dt:.0f} tok/s"), K("Mean token probability", sum(probs) / len(probs), digits=3)),
                                     CODE("Completion", text)]}


@app.post("/api/attention")
def attention(a: Att) -> dict:
    model, tok, _, torch, _ = LM()
    ids = torch.tensor([tok.chat(a.text.lower())])
    layer = min(a.layer, model.c.n_layer - 1)
    with torch.no_grad():
        model(ids, keep_att=True)
    att = model.blocks[layer].att.last_att[0].mean(0).numpy()  # average over heads -> (T, T)
    labs = [("⟨u⟩" if i == tok.user else "⟨a⟩" if i == tok.assistant else tok.itos[i].replace(" ", "␣")) for i in ids[0].tolist()]
    return {"blocks": [HEAT(f"Layer {layer} attention (mean over {model.c.n_head} heads) — row attends to column", [f"{t}" for t in labs], [f"{t} " for t in labs], att.round(3).tolist(), show=False, labw=30, cellw=10,
                            sub="Causal mask: each character can only look left, so the upper triangle is empty. ⟨u⟩/⟨a⟩ are the chat role tokens.")]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
