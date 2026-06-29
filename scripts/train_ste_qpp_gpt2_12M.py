#!/usr/bin/env python3
"""H2.1b: Train GPT-2 12M from scratch with STE QPP on WikiText-2.

Uses STEQPPLinear with periodic hard reordering instead of Gumbel-Softmax.
No temperature annealing. No soft permutation. Just train anchors → reorder → repeat.

Compares Dense vs STE-QPP.
If this converges, we bypass the Gumbel plateau entirely.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpp.ste_qpp import STEQPPLinear, STEAttention


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_head, use_ste=False, qpp_anchors=32, qpp_row_block=16, reorder_every=200):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        if use_ste:
            self.attn = STEAttention(d_model, n_head, qpp_anchors, qpp_row_block, reorder_every)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        if isinstance(self.attn, STEAttention):
            x = x + self.attn(self.ln1(x))
        else:
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
            x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

    def step_ordering(self):
        if isinstance(self.attn, STEAttention):
            return self.attn.step_all()
        return []


class GPT2Model(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=8, n_layers=6, max_seq=256,
                 use_ste=False, qpp_anchors=32, qpp_row_block=16, reorder_every=200):
        super().__init__()
        self.use_ste = use_ste
        self.reorder_every = reorder_every
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, n_head, use_ste, qpp_anchors, qpp_row_block, reorder_every)
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

    def step_all_orderings(self):
        triggered = 0
        for block in self.blocks:
            results = block.step_ordering()
            triggered += sum(1 for r in results if r)
        return triggered


def load_wikitext2(max_chars=5_000_000):
    repo_root = Path(__file__).resolve().parent.parent
    path = repo_root / "data" / "wikitext2_train.txt"
    if not path.exists():
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading WikiText-2...", flush=True)
        data = urllib.request.urlopen(url).read()
        path.write_bytes(data)
    return path.read_text(encoding="utf-8")[:max_chars]


def build_char_level(text, seq_len):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    num_seqs = len(ids) // seq_len
    ids = ids[: num_seqs * seq_len]
    return ids.view(-1, seq_len), len(stoi)


def train_one(use_ste, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = "STE-QPP" if use_ste else "DENSE"
    print(f"\n{'='*60}", flush=True)
    print(f"Training GPT-2 12M — {label}", flush=True)
    print(f"{'='*60}", flush=True)

    text = load_wikitext2(config["max_chars"])
    data_t, vocab_size = build_char_level(text, config["seq_len"])
    data_t = data_t.to(device)
    print(f"  Vocab: {vocab_size} | Data: {data_t.shape[0]} seqs of {config['seq_len']}", flush=True)

    model = GPT2Model(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_head=config["n_head"],
        n_layers=config["n_layers"],
        max_seq=config["seq_len"],
        use_ste=use_ste,
        qpp_anchors=config["qpp_anchors"],
        qpp_row_block=config["qpp_row_block"],
        reorder_every=config["reorder_every"],
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total:,} | Trainable: {trainable:,}", flush=True)

    if use_ste:
        order_bytes = 0
        anchor_params = 0
        for block in model.blocks:
            for pn in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                p = getattr(block.attn, pn)
                order_bytes += p.orders_i16.numel() * 2
                anchor_params += p.anchor_values.numel()
        print(f"  Order storage: {order_bytes:,} B | Anchor params: {anchor_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["max_steps"])

    bs = config["batch_size"]
    losses = []
    reorders_triggered = 0
    t0 = time.perf_counter()

    for step in range(config["max_steps"]):
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

        # STE: periodic reordering
        if use_ste:
            triggered = model.step_all_orderings()
            reorders_triggered += triggered

        losses.append(loss.item())

        if step % 200 == 0 or step == config["max_steps"] - 1:
            elapsed = time.perf_counter() - t0
            ppl = math.exp(min(loss.item(), 10))
            ro = f" reorders={reorders_triggered}" if use_ste else ""
            print(
                f"  [{label}] step {step:5d}/{config['max_steps']} | loss={loss.item():.4f} ppl={ppl:.2f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s{ro}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    final_ppl = math.exp(min(losses[-1], 10))
    result = {
        "use_ste": use_ste,
        "total_params": total,
        "final_loss": losses[-1],
        "final_ppl": final_ppl,
        "elapsed_s": elapsed,
        "reorders_triggered": reorders_triggered,
    }
    print(f"\n  [{label}] DONE | ppl={final_ppl:.2f} | {elapsed:.0f}s | {reorders_triggered} reorders", flush=True)
    return result


def main():
    config = dict(
        d_model=128, n_head=8, n_layers=6, seq_len=256,
        qpp_anchors=32, qpp_row_block=16, reorder_every=200,
        batch_size=16, max_steps=3000, max_chars=5_000_000,
        lr=5e-4,
    )

    print("GPT-2 12M — STE QPP from scratch on WikiText-2", flush=True)
    print(f"Config: d={config['d_model']} L={config['n_layers']} h={config['n_head']} "
          f"K={config['qpp_anchors']} reorder_every={config['reorder_every']}", flush=True)

    dense = train_one(use_ste=False, config=config)
    ste = train_one(use_ste=True, config=config)

    print("\n" + "=" * 70)
    print("FINAL: Dense vs STE-QPP on WikiText-2")
    print("=" * 70)
    print(f"  Dense:   ppl={dense['final_ppl']:.2f} | {dense['elapsed_s']:.0f}s | params={dense['total_params']:,}")
    print(f"  STE-QPP: ppl={ste['final_ppl']:.2f} | {ste['elapsed_s']:.0f}s | params={ste['total_params']:,}")
    ratio = ste['final_ppl'] / dense['final_ppl']
    print(f"  PPL ratio (STE-QPP/Dense): {ratio:.3f}")

    if ratio < 1.05:
        print(f"  ✅ STE-QPP MATCHES dense within 5% PPL!")
    elif ratio < 1.15:
        print(f"  ⚠️  Within 15% — needs tuning")
    elif ste['final_ppl'] < 20:
        print(f"  ⚠️  QPP is learning but behind dense. More steps or larger anchors needed.")
    else:
        print(f"  ❌ QPP not learning. Investigate.")

    print(f"\n  github.com/ignaciosua/qpp")


if __name__ == "__main__":
    main()
