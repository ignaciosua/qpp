#!/usr/bin/env python3
"""H2.1: Train a GPT-2 12M from scratch with GumbelQPP attention on WikiText-2.

Replaces all attention Q/K/V/O projections with LowRankGumbelQPPLinear.
Compares against an identical dense baseline.

Model: GPT-2, d=128, 6 layers, 8 heads (~12M params, ~48 MB)
Data: WikiText-2 (downloaded locally as data/wikitext2_train.txt)
Time: ~1-2h on RTX 4060 Ti 16GB

ponytail: minimum viable real-data experiment. Upgrade path: BPE tokenizer,
larger model (GPT-2 124M), Sinkhorn normalization, shared ordering.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpp.gumbel_qpp import LowRankGumbelQPPLinear

# ═══════════════════════════════════════
# GPT-2 style transformer
# ═══════════════════════════════════════

class GPT2Attention(nn.Module):
    def __init__(self, d_model, n_head, use_qpp=False, qpp_anchors=32, qpp_row_block=16, qpp_perm_rank=16):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head

        if use_qpp:
            kw = {"anchors": qpp_anchors, "row_block": qpp_row_block, "bias": True, "perm_rank": qpp_perm_rank}
            self.q_proj = LowRankGumbelQPPLinear(d_model, d_model, **kw)
            self.k_proj = LowRankGumbelQPPLinear(d_model, d_model, **kw)
            self.v_proj = LowRankGumbelQPPLinear(d_model, d_model, **kw)
            self.out_proj = LowRankGumbelQPPLinear(d_model, d_model, **kw)
        else:
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_k)
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_head, use_qpp=False, qpp_anchors=32, qpp_row_block=16, qpp_perm_rank=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GPT2Attention(d_model, n_head, use_qpp, qpp_anchors, qpp_row_block, qpp_perm_rank)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=8, n_layers=6, max_seq=256,
                 use_qpp=False, qpp_anchors=32, qpp_row_block=16, qpp_perm_rank=16):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, n_head, use_qpp, qpp_anchors, qpp_row_block, qpp_perm_rank)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


# ═══════════════════════════════════════
# Data: WikiText-2 local file
# ═══════════════════════════════════════

def load_wikitext2_local(max_chars=5_000_000):
    """Load WikiText-2 from local file (downloaded from pytorch/examples)."""
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "data" / "wikitext2_train.txt"
    if not path.exists():
        # Auto-download
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading WikiText-2 from {url}...", flush=True)
        data = urllib.request.urlopen(url).read()
        path.write_bytes(data)
        print(f"Saved {len(data)} bytes to {path}", flush=True)
    text = path.read_text(encoding="utf-8")
    return text[:max_chars]


def build_char_level(text, seq_len):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    num_seqs = len(ids) // seq_len
    ids = ids[: num_seqs * seq_len]
    return ids.view(-1, seq_len), stoi, itos


# ═══════════════════════════════════════
# Training
# ═══════════════════════════════════════

def train_one(use_qpp, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = "QPP" if use_qpp else "DENSE"
    print(f"\n{'='*60}", flush=True)
    print(f"Training GPT-2 12M — {label}", flush=True)
    print(f"{'='*60}", flush=True)

    print("Loading WikiText-2...", flush=True)
    text = load_wikitext2_local(config["max_chars"])
    data_t, stoi, itos = build_char_level(text, config["seq_len"])
    data_t = data_t.to(device)
    vocab_size = len(stoi)
    print(f"  Vocab: {vocab_size} chars | Data: {data_t.shape[0]} seqs of {config['seq_len']} tokens", flush=True)

    model = GPT2Model(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_head=config["n_head"],
        n_layers=config["n_layers"],
        max_seq=config["seq_len"],
        use_qpp=use_qpp,
        qpp_anchors=config["qpp_anchors"],
        qpp_row_block=config["qpp_row_block"],
        qpp_perm_rank=config["qpp_perm_rank"],
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} | Trainable: {trainable:,}", flush=True)

    if use_qpp:
        order_params = 0
        for block in model.blocks:
            for pn in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                p = getattr(block.attn, pn)
                if hasattr(p, "order_U"):
                    order_params += p.order_U.numel() + p.order_V.numel()
        anchor_params = 0
        for block in model.blocks:
            for pn in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                p = getattr(block.attn, pn)
                if hasattr(p, "anchor_values"):
                    anchor_params += p.anchor_values.numel()
        dense_equiv = config["d_model"] * config["d_model"] * 4 * config["n_layers"]
        print(f"  Order params: {order_params:,} | Anchor params: {anchor_params:,} | Dense equiv: {dense_equiv:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["max_steps"])

    bs = config["batch_size"]
    losses = []
    perm_changes = []
    prev_hard_perms = []

    t0 = time.perf_counter()

    for step in range(config["max_steps"]):
        # Temperature annealing
        # Exponential annealing: temp = temp_min + (temp_init - temp_min) * exp(-step / tau)
        tau = config["temp_anneal_steps"] / 3  # faster decay
        temp = config["temp_min"] + (config["temp_init"] - config["temp_min"]) * math.exp(-step / max(1, tau))
        if use_qpp:
            for block in model.blocks:
                for pn in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    getattr(block.attn, pn).temperature.fill_(temp)

        idx = torch.randint(0, data_t.shape[0] - 1, (bs,))
        x = data_t[idx]
        y = data_t[idx]

        logits = model(x)[:, :-1, :].contiguous()
        targets = y[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if use_qpp and step % 100 == 0:
            current_perms = []
            for block in model.blocks:
                for pn in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    try:
                        current_perms.append(getattr(block.attn, pn).hard_permutation().cpu())
                    except Exception:
                        pass
            if prev_hard_perms and current_perms:
                changes = sum(
                    (c != p).sum().item() for c, p in zip(current_perms, prev_hard_perms)
                    if c.shape == p.shape
                )
                perm_changes.append(changes)
            prev_hard_perms = current_perms

        if step % 200 == 0 or step == config["max_steps"] - 1:
            elapsed = time.perf_counter() - t0
            ppl = math.exp(min(loss.item(), 10))
            perm_str = f" perm_changes={perm_changes[-1]}" if perm_changes else ""
            print(
                f"  [{label}] step {step:5d}/{config['max_steps']} | loss={loss.item():.4f} ppl={ppl:.2f} "
                f"temp={temp:.3f} lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s{perm_str}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    final_ppl = math.exp(min(losses[-1], 10))
    perm_final = perm_changes[-1] if perm_changes else None
    perm_converged = perm_final is not None and len(perm_changes) >= 2 and perm_final < perm_changes[0] * 0.3

    result = {
        "use_qpp": use_qpp,
        "total_params": total,
        "final_loss": losses[-1],
        "final_ppl": final_ppl,
        "elapsed_s": elapsed,
        "perm_converged": perm_converged,
        "final_perm_changes": perm_final,
    }
    print(
        f"\n  [{label}] DONE | ppl={final_ppl:.2f} | {elapsed:.0f}s | "
        f"perm_conv={'YES' if perm_converged else 'N/A' if perm_final is None else 'NO'}",
        flush=True,
    )
    return result


def main():
    config = dict(
        d_model=128, n_head=8, n_layers=6, seq_len=256,
        qpp_anchors=32, qpp_row_block=16, qpp_perm_rank=16,
        batch_size=16, max_steps=5000, max_chars=5_000_000,
        lr=5e-4, temp_init=2.0, temp_min=0.1, temp_anneal_steps=800,
    )

    print("GPT-2 12M — GumbelQPP from scratch on WikiText-2", flush=True)
    print(
        f"Config: d={config['d_model']} L={config['n_layers']} h={config['n_head']} "
        f"K={config['qpp_anchors']} B={config['qpp_row_block']} R={config['qpp_perm_rank']}",
        flush=True,
    )

    dense = train_one(use_qpp=False, config=config)
    qpp = train_one(use_qpp=True, config=config)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON: Dense vs QPP from scratch on WikiText-2")
    print("=" * 70)
    print(f"  Dense:  ppl={dense['final_ppl']:.2f} | {dense['elapsed_s']:.0f}s | params={dense['total_params']:,}")
    print(f"  QPP:    ppl={qpp['final_ppl']:.2f} | {qpp['elapsed_s']:.0f}s | params={qpp['total_params']:,}")
    ratio = qpp['final_ppl'] / dense['final_ppl']
    print(f"  PPL ratio (QPP/Dense): {ratio:.3f}")
    if ratio < 1.05:
        print(f"  [OK] QPP matches dense within 5% PPL!")
    elif ratio < 1.15:
        print(f"  [WARN] QPP within 15% - needs more training / hyperparam tuning")
    else:
        print(f"  [FAIL] QPP significantly worse - investigate")

    if qpp['perm_converged']:
        print(f"  [OK] Permutation CONVERGED (final changes: {qpp['final_perm_changes']})")
    elif qpp['final_perm_changes'] is not None:
        print(f"  [WARN] Permutation NOT converged (final changes: {qpp['final_perm_changes']})")

    print(f"\n  github.com/ignaciosua/qpp")


if __name__ == "__main__":
    main()
