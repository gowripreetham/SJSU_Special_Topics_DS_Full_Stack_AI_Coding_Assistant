"""NanoLlama — a from-scratch decoder-only transformer with modern (Llama-style) primitives, small enough for a laptop CPU.

Primitives implemented here, not imported:
  * RoPE   — rotary position embeddings applied to queries/keys (Su et al., 2021)
  * RMSNorm — pre-normalisation without mean-centering (Zhang & Sennrich, 2019)
  * SwiGLU — gated feed-forward  W2( silu(W1 x) * W3 x )  (Shazeer, 2020)
  * weight tying between the token embedding and the output head
Each primitive can be switched off (use_rope / use_swiglu / use_rmsnorm) so the ablation study is one flag away.
"""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 96
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    ctx: int = 192
    dropout: float = 0.05
    use_rope: bool = True
    use_swiglu: bool = True
    use_rmsnorm: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return self.g * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_cache(ctx: int, dim: int) -> torch.Tensor:
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    ang = torch.outer(torch.arange(ctx).float(), freqs)
    return torch.stack([ang.cos(), ang.sin()], dim=-1)  # (ctx, dim/2, 2)


def apply_rope(x: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of channels by a position-dependent angle. x: (B, H, T, D)."""
    T = x.shape[2]
    xr = x.reshape(*x.shape[:-1], -1, 2)
    cos, sin = rc[:T, :, 0], rc[:T, :, 1]
    out = torch.stack([xr[..., 0] * cos - xr[..., 1] * sin, xr[..., 0] * sin + xr[..., 1] * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.out = nn.Linear(c.d_model, c.d_model, bias=False)
        self.drop = nn.Dropout(c.dropout)
        self.last_att: torch.Tensor | None = None

    def forward(self, x, rc, keep_att: bool = False):
        B, T, D = x.shape
        h = self.c.n_head
        q, k, v = (t.view(B, T, h, D // h).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        if self.c.use_rope:
            q, k = apply_rope(q, rc), apply_rope(k, rc)
        if keep_att:  # explicit path so the dashboard can visualise attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(D // h)
            att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1), float("-inf")).softmax(-1)
            self.last_att = att.detach()
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.c.dropout if self.training else 0.0)
        return self.drop(self.out(y.transpose(1, 2).reshape(B, T, D)))


class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.swiglu = c.use_swiglu
        hid = int(8 * c.d_model / 3) if c.use_swiglu else 4 * c.d_model  # same parameter budget either way
        self.w1 = nn.Linear(c.d_model, hid, bias=False)
        self.w2 = nn.Linear(hid, c.d_model, bias=False)
        self.w3 = nn.Linear(c.d_model, hid, bias=False) if c.use_swiglu else None
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w3(x) if self.swiglu else F.gelu(self.w1(x))
        return self.drop(self.w2(h))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        norm = RMSNorm if c.use_rmsnorm else nn.LayerNorm
        self.n1, self.att, self.n2, self.ff = norm(c.d_model), Attention(c), norm(c.d_model), FeedForward(c)

    def forward(self, x, rc, keep_att=False):
        x = x + self.att(self.n1(x), rc, keep_att)
        return x + self.ff(self.n2(x))


class NanoLlama(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = None if c.use_rope else nn.Embedding(c.ctx, c.d_model)  # learned absolute positions = the ablation baseline
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = (RMSNorm if c.use_rmsnorm else nn.LayerNorm)(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.register_buffer("rc", rope_cache(c.ctx, c.d_model // c.n_head), persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, keep_att=False):
        x = self.tok(idx)
        if self.pos is not None:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x, self.rc, keep_att)
        logits = self.head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new: int, temperature: float = 0.8, top_k: int | None = 40, top_p: float = 1.0, stop: int | None = None):
        """Autoregressive sampling with temperature, top-k and nucleus (top-p) filtering. Returns (tokens, per-token probabilities)."""
        probs_out = []
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.c.ctx:])
            logits = logits[:, -1, :]
            if temperature <= 1e-4:
                p = F.softmax(logits, -1)
                nxt = p.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                p = F.softmax(logits, -1)
                if top_p < 1.0:
                    sp, si = torch.sort(p, descending=True)
                    cut = sp.cumsum(-1) - sp > top_p
                    sp = sp.masked_fill(cut, 0.0)
                    p = torch.zeros_like(p).scatter(-1, si, sp / sp.sum(-1, keepdim=True))
                nxt = torch.multinomial(p, 1)
            probs_out.append(float(p[0, nxt[0, 0]]))
            idx = torch.cat([idx, nxt], dim=1)
            if stop is not None and int(nxt[0, 0]) == stop:
                break
        return idx, probs_out


class CharTokenizer:
    """Character-level tokenizer with three special tokens for the chat format."""
    SPECIALS = ["<|user|>", "<|assistant|>", "<|end|>"]

    def __init__(self, chars: list[str]):
        self.itos = self.SPECIALS + chars
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.user, self.assistant, self.end = 0, 1, 2
        self.unk = self.stoi.get(" ", 3)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(ch, self.unk) for ch in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i >= len(self.SPECIALS))

    def chat(self, question: str, answer: str | None = None) -> list[int]:
        ids = [self.user] + self.encode(question) + [self.assistant]
        return ids if answer is None else ids + self.encode(answer) + [self.end]
