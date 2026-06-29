#!/usr/bin/env python3
"""Quick proof-of-concept: train a tiny transformer from scratch using GumbelQPPLinear.

Replaces attention Q/K/V/O projections with Gumbel-Softmax QPP layers.
Compares FullRank vs LowRank variants at d_model=64 and d_model=256.

ponytail: absolute minimum viable experiment. Upgrade path: larger model,
real dataset (WikiText-2), Sinkhorn normalization, temperature annealing schedule.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpp.gumbel_qpp import GumbelQPPLinear, LowRankGumbelQPPLinear


class GumbelQPPAttention(nn.Module):
    """Multi-head attention where Q/K/V/O are GumbelQPPLinear."""

    def __init__(self, d_model: int, n_head: int, anchors: int, row_block: int,
                 use_lowrank: bool = False, perm_rank: int = 16):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head

        cls = LowRankGumbelQPPLinear if use_lowrank else GumbelQPPLinear
        cls_kwargs = {"anchors": anchors, "row_block": row_block, "bias": False}
        if use_lowrank:
            cls_kwargs["perm_rank"] = perm_rank

        self.q_proj = cls(d_model, d_model, **cls_kwargs)
        self.k_proj = cls(d_model, d_model, **cls_kwargs)
        self.v_proj = cls(d_model, d_model, **cls_kwargs)
        self.out_proj = cls(d_model, d_model, **cls_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_k)
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_head=4, n_layers=2,
                 anchors=16, row_block=8, max_seq_len=128,
                 use_lowrank=False, perm_rank=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": GumbelQPPAttention(d_model, n_head, anchors, row_block,
                                           use_lowrank, perm_rank),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model),
                ),
            })
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        for layer in self.layers:
            h = h + layer["attn"](layer["ln1"](h))
            h = h + layer["mlp"](layer["ln2"](h))
        return self.head(self.ln_final(h))


def make_toy_corpus(seq_len=128, num_seqs=200):
    chars = "abcdefghijklmnopqrstuvwxyz "
    vocab = {c: i for i, c in enumerate(chars)}
    needed = seq_len * num_seqs
    repeats = (needed // len(chars)) + 2
    text = (chars * repeats)[:needed]
    ids = torch.tensor([vocab[c] for c in text], dtype=torch.long)
    return ids.reshape(-1, seq_len), len(vocab)


def _get_temperature_fn(step, num_batches, temp_init, temp_min, temp_anneal_steps):
    """ponytail: simple piecewise linear annealing."""
    if step < temp_anneal_steps:
        frac = step / max(1, temp_anneal_steps)
        return temp_min + (temp_init - temp_min) * (1 - frac)
    return temp_min


def train_one(config: dict) -> dict:
    """Train one config, return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = config["label"]

    data, vocab_size = make_toy_corpus(config["seq_len"], num_seqs=config["batch_size"] * 10)

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        n_head=config["n_head"],
        n_layers=config["n_layers"],
        anchors=config["anchors"],
        row_block=config["row_block"],
        max_seq_len=config["seq_len"],
        use_lowrank=config.get("use_lowrank", False),
        perm_rank=config.get("perm_rank", 16),
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    order_params = 0
    for layer in model.layers:
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(layer["attn"], proj_name)
            if hasattr(proj, "order_U"):
                order_params += proj.order_U.numel() + proj.order_V.numel()
            elif proj.order_logits is not None:
                order_params += proj.order_logits.numel()

    bs = config["batch_size"]
    num_batches = config["num_batches"]
    temp_init = config["temp_init"]
    temp_min = config["temp_min"]
    temp_anneal_steps = config["temp_anneal_steps"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_batches)

    losses = []
    perm_changes = []
    prev_hard_perms = []

    t0 = time.perf_counter()

    for step in range(num_batches):
        temp = _get_temperature_fn(step, num_batches, temp_init, temp_min, temp_anneal_steps)
        for layer in model.layers:
            for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                getattr(layer["attn"], proj_name).temperature.fill_(temp)

        batch_idx = (step * bs) % (data.shape[0] - bs)
        x = data[batch_idx: batch_idx + bs].to(device)
        y = data[batch_idx: batch_idx + bs].to(device)

        logits = model(x)[:, :-1, :].contiguous()
        targets = y[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if step % 20 == 0:
            current_perms = []
            for layer in model.layers:
                for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    proj = getattr(layer["attn"], proj_name)
                    try:
                        current_perms.append(proj.hard_permutation().cpu())
                    except Exception:
                        pass
            if prev_hard_perms and current_perms:
                changes = sum(
                    (c != p).sum().item()
                    for c, p in zip(current_perms, prev_hard_perms)
                    if c.shape == p.shape
                )
                perm_changes.append(changes)
            prev_hard_perms = current_perms

        if step % 50 == 0 or step == num_batches - 1:
            print(f"  [{label}] step {step:3d}/{num_batches} loss={loss.item():.4f} temp={temp:.3f}", flush=True)

    elapsed = time.perf_counter() - t0
    baseline = -math.log(1.0 / vocab_size)
    loss_drop = losses[0] - losses[-1]
    final_perm_changes = perm_changes[-1] if perm_changes else 999
    perm_converged = (
        final_perm_changes < (perm_changes[0] * 0.3) if perm_changes and perm_changes[0] > 0 else False
    )

    return {
        "label": label,
        "total_params": total,
        "trainable_params": trainable,
        "order_params": order_params,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_drop": loss_drop,
        "baseline_loss": baseline,
        "learning": loss_drop > 0.5,
        "final_perm_changes": final_perm_changes,
        "perm_converged": perm_converged,
        "elapsed_s": elapsed,
        "steps": num_batches,
    }


def compare():
    print("=" * 70, flush=True)
    print("FULL-RANK vs LOW-RANK GUMBELQPP COMPARISON", flush=True)
    print("=" * 70, flush=True)

    base_config = dict(
        seq_len=64, batch_size=8, num_batches=200,
        n_layers=2, anchors=16, row_block=8,
        lr=5e-4, temp_init=1.0, temp_min=0.3, temp_anneal_steps=150,
    )

    configs = [
        {**base_config, "d_model": 64, "n_head": 4, "use_lowrank": False, "label": "FullRank d=64"},
        {**base_config, "d_model": 64, "n_head": 4, "use_lowrank": True, "perm_rank": 16,
         "label": "LowRank d=64 r=16"},
        {**base_config, "d_model": 256, "n_head": 8, "use_lowrank": True, "perm_rank": 16,
         "row_block": 32, "label": "LowRank d=256 r=16"},
    ]

    results = []
    for cfg in configs:
        print(f"\n--- {cfg['label']} ---", flush=True)
        r = train_one(cfg)
        results.append(r)
        print(f"  Params: {r['total_params']:,} | Order params: {r['order_params']:,}", flush=True)
        print(f"  Loss: {r['loss_initial']:.3f} -> {r['loss_final']:.3f} (baseline={r['baseline_loss']:.2f})", flush=True)
        print(f"  Learning: {'YES' if r['learning'] else 'NO'} | "
              f"Perm converged: {'YES' if r['perm_converged'] else 'WARN'} ({r['final_perm_changes']})", flush=True)
        print(f"  Time: {r['elapsed_s']:.1f}s", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("SUMMARY TABLE", flush=True)
    print("=" * 70, flush=True)
    hdr = f"{'Config':<22s} {'Params':>10s} {'Order':>10s} {'Loss0':>7s} {'LossN':>7s} {'Drop':>7s} {'Learn':>6s} {'Perm':>6s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        print(
            f"{r['label']:<22s} {r['total_params']:>10,d} {r['order_params']:>10,d} "
            f"{r['loss_initial']:>7.3f} {r['loss_final']:>7.3f} {r['loss_drop']:>7.3f} "
            f"{'YES' if r['learning'] else 'NO':>6s} "
            f"{'YES' if r['perm_converged'] else 'WARN':>6s}",
            flush=True,
        )

    if len(results) >= 2:
        fr = results[0]
        lr = results[1]
        print("\n--- LowRank vs FullRank at d=64 ---", flush=True)
        print(f"  FullRank loss: {fr['loss_initial']:.3f} -> {fr['loss_final']:.3f} (drop: {fr['loss_drop']:.3f})", flush=True)
        print(f"  LowRank  loss: {lr['loss_initial']:.3f} -> {lr['loss_final']:.3f} (drop: {lr['loss_drop']:.3f})", flush=True)
        ratio = lr['loss_drop'] / max(fr['loss_drop'], 1e-9)
        print(f"  LowRank/FullRank drop ratio: {ratio:.2f} {'OK' if ratio > 0.7 else 'DEGRADED'}", flush=True)
        mem_ratio = lr['order_params'] / max(fr['order_params'], 1)
        print(f"  Order memory: {lr['order_params']:,} vs {fr['order_params']:,} ({1/mem_ratio:.1f}x {'SAVINGS' if mem_ratio < 1 else 'MORE'})", flush=True)

    if len(results) >= 3:
        r256 = results[2]
        print(f"\n--- Scale check: LowRank at d=256 ---", flush=True)
        print(f"  Loss: {r256['loss_initial']:.3f} -> {r256['loss_final']:.3f} (drop: {r256['loss_drop']:.3f})", flush=True)
        if r256['learning']:
            print(f"  SCALES to d=256", flush=True)
            order256 = r256['order_params']
            order768 = order256 * (768/256) * (768/256)
            print(f"  Order params d=256: {order256:,} | Projected d=768: {order768:,.0f} ({order768/1e6:.1f}M)", flush=True)
        else:
            print(f"  FAILS at d=256", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    compare()
