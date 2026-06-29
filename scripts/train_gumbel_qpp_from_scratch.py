#!/usr/bin/env python3
"""Quick proof-of-concept: train a tiny transformer from scratch using GumbelQPPLinear.

Replaces attention Q/K/V/O projections with Gumbel-Softmax QPP layers.
Runs on CPU in ~3 minutes. Checks:
  1. Does the model learn at all? (loss drops?)
  2. Does the learned ordering converge? (permutation stabilizes?)
  3. Does a quantile-like curve emerge in the anchors?

ponytail: absolute minimum viable experiment. Upgrade path: larger model,
real dataset (WikiText-2), Sinkhorn normalization, temperature annealing schedule.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent to path so we can import qpp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpp.gumbel_qpp import GumbelQPPLinear

# ═══════════════════════════════════════════════════════
# Tiny Transformer with GumbelQPP attention projections
# ═══════════════════════════════════════════════════════


class GumbelQPPAttention(nn.Module):
    """Multi-head attention where Q/K/V/O are GumbelQPPLinear."""

    def __init__(self, d_model: int, n_head: int, anchors: int, row_block: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.q_proj = GumbelQPPLinear(d_model, d_model, anchors, row_block, bias=False)
        self.k_proj = GumbelQPPLinear(d_model, d_model, anchors, row_block, bias=False)
        self.v_proj = GumbelQPPLinear(d_model, d_model, anchors, row_block, bias=False)
        self.out_proj = GumbelQPPLinear(d_model, d_model, anchors, row_block, bias=False)

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
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 2,
        anchors: int = 16,
        row_block: int = 8,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": GumbelQPPAttention(d_model, n_head, anchors, row_block),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos_ids)
        for layer in self.layers:
            h = h + layer["attn"](layer["ln1"](h))
            h = h + layer["mlp"](layer["ln2"](h))
        return self.head(self.ln_final(h))


# ═══════════════════════════════════════════════════════
# Toy data: repeated alphabet (character-level)
# ═══════════════════════════════════════════════════════

def make_toy_corpus(seq_len: int = 128, num_seqs: int = 200) -> tuple[torch.Tensor, int]:
    """Repeat 'abcdefghijklmnopqrstuvwxyz ' so the model has SOMETHING to learn."""
    chars = "abcdefghijklmnopqrstuvwxyz "
    vocab = {c: i for i, c in enumerate(chars)}
    needed = seq_len * num_seqs
    repeats = (needed // len(chars)) + 2
    text = (chars * repeats)[:needed]
    ids = torch.tensor([vocab[c] for c in text], dtype=torch.long)
    return ids.reshape(-1, seq_len), len(vocab)


# ═══════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Hyperparams (kept tiny for fast CPU run)
    D_MODEL = 64
    N_HEAD = 4
    N_LAYERS = 2
    ANCHORS = 16
    ROW_BLOCK = 8
    SEQ_LEN = 64
    BATCH_SIZE = 8
    NUM_BATCHES = 200
    LR = 5e-4
    TEMP_INIT = 1.0
    TEMP_MIN = 0.3
    TEMP_ANNEAL_STEPS = 150

    data, vocab_size = make_toy_corpus(SEQ_LEN, num_seqs=BATCH_SIZE * 10)

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        anchors=ANCHORS,
        row_block=ROW_BLOCK,
        max_seq_len=SEQ_LEN,
    ).to(device)

    # Count params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}", flush=True)

    # Compare: dense equivalent model
    dense_total = vocab_size * D_MODEL  # embed
    dense_total += SEQ_LEN * D_MODEL    # pos
    for _ in range(N_LAYERS):
        dense_total += D_MODEL * D_MODEL * 4  # Q/K/V/O
        dense_total += D_MODEL * (D_MODEL * 4) * 2  # MLP FF
    print(f"Dense model would be ~{dense_total:,} params for Q/K/V/O", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_BATCHES)

    losses: list[float] = []
    perm_changes: list[int] = []
    prev_hard_perms: list[torch.Tensor] = []

    t0 = time.perf_counter()

    for step in range(NUM_BATCHES):
        # Anneal temperature
        temp = max(TEMP_MIN, TEMP_INIT * (TEMP_ANNEAL_STEPS / max(1, step + 1)))
        for layer in model.layers:
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                getattr(layer["attn"], proj).temperature.fill_(temp)

        # Get batch
        batch_idx = (step * BATCH_SIZE) % (data.shape[0] - BATCH_SIZE)
        x = data[batch_idx: batch_idx + BATCH_SIZE].to(device)
        y = data[batch_idx: batch_idx + BATCH_SIZE].to(device)  # next-token pred
        # ponytail: shift targets left by 1 for causal LM
        logits = model(x)[:, :-1, :].contiguous()
        targets = y[:, 1:].contiguous()

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients (Gumbel-Softmax can be spiky)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        # Track permutation changes
        if step % 20 == 0:
            current_perms = []
            for layer in model.layers:
                for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                    proj = getattr(layer["attn"], proj_name)
                    current_perms.append(proj.hard_permutation().cpu())

            if prev_hard_perms:
                changes = sum(
                    (c != p).sum().item() for c, p in zip(current_perms, prev_hard_perms)
                )
                perm_changes.append(changes)
            prev_hard_perms = current_perms

        if step % 25 == 0 or step == NUM_BATCHES - 1:
            baseline_loss = -math.log(1.0 / vocab_size)  # random baseline
            perm_stable = perm_changes[-1] if perm_changes else "N/A"
            print(
                f"[{step:3d}/{NUM_BATCHES}] loss={loss.item():.4f} "
                f"(baseline={baseline_loss:.2f}) temp={temp:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.2e} "
                f"perm_changes={perm_stable}",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    print(f"\nTraining done in {elapsed:.1f}s ({elapsed/NUM_BATCHES:.2f}s/batch)", flush=True)

    # ═══════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════

    # 1. Loss curve
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(losses)
    ax.set_title("Training Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy")
    ax.axhline(-math.log(1.0 / vocab_size), color="red", linestyle="--", label="Random baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Permutation stability
    if perm_changes:
        ax = axes[0, 1]
        ax.plot(range(20, NUM_BATCHES, 20), perm_changes)
        ax.set_title("Permutation Changes (↓ = converging)")
        ax.set_xlabel("Step")
        ax.set_ylabel("# columns changed")
        ax.grid(True, alpha=0.3)

    # 3. Learned anchors vs sorted weights (first Q-proj of first layer)
    ax = axes[0, 2]
    first_q = model.layers[0]["attn"].q_proj
    weight, bias = first_q.get_dense_weight()
    w_np = weight.detach().cpu().numpy()
    # Sort the dense weight rows and plot a few
    for i in range(0, min(5, D_MODEL), 1):
        sorted_row = np.sort(w_np[i])
        ax.plot(sorted_row, alpha=0.6, linewidth=0.8, label=f"row {i}" if i == 0 else "")
    ax.set_title("Learned Weights (sorted by value)")
    ax.set_xlabel("Sorted column index")
    ax.set_ylabel("Weight value")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. Anchor curve: plot anchors in sorted order
    ax = axes[1, 0]
    anchor_vals = first_q.anchor_values.detach().cpu().numpy()
    for i in range(0, min(5, D_MODEL), 1):
        # Interpolate anchors as smooth curve
        basis = first_q.basis.detach().cpu().numpy()
        curve = basis @ anchor_vals[i]
        ax.plot(curve, alpha=0.6, linewidth=0.8)
    ax.set_title(f"Reconstructed Curves from {ANCHORS} Anchors")
    ax.set_xlabel("Column index (in sorted order)")
    ax.set_ylabel("Weight value")
    ax.grid(True, alpha=0.3)

    # 5. Hard permutation visualization (first block)
    ax = axes[1, 1]
    hard_perm = first_q.hard_permutation()[0].cpu().numpy()  # first block
    ax.scatter(range(len(hard_perm)), hard_perm, s=2, alpha=0.6)
    ax.plot([0, len(hard_perm)], [0, len(hard_perm)], "r--", alpha=0.3, linewidth=0.5)
    ax.set_title("Learned Column Ordering (hard perm)")
    ax.set_xlabel("Position in sorted order")
    ax.set_ylabel("Original column index")
    ax.grid(True, alpha=0.3)

    # 6. Compression summary
    ax = axes[1, 2]
    ax.axis("off")
    qpp_comps = []
    for layer in model.layers:
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(layer["attn"], proj_name)
            qpp_comps.append(proj.compression_ratio())

    summary_text = (
        f"GumbelQPP Training Summary\n"
        f"{'='*40}\n"
        f"Model: {N_LAYERS}L, d={D_MODEL}, h={N_HEAD}\n"
        f"Anchors: K={ANCHORS}, row_block={ROW_BLOCK}\n"
        f"Vocab: {vocab_size} chars\n\n"
        f"Trainable params: {trainable:,}\n"
        f"Dense equivalent: ~{dense_total:,}\n"
        f"QPP param compression: {np.mean(qpp_comps):.1f}x avg\n\n"
        f"Final loss: {losses[-1]:.4f}\n"
        f"Baseline loss: {-math.log(1/vocab_size):.4f}\n"
        f"Loss drop: {losses[0] - losses[-1]:.3f}\n\n"
        f"Perm changes (last 20): {perm_changes[-1] if perm_changes else 'N/A'}\n"
        f"Temperature: {TEMP_INIT:.1f} → {TEMP_MIN:.1f}\n"
        f"Time: {elapsed:.1f}s ({NUM_BATCHES} steps)"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=8, verticalalignment="top")

    plt.tight_layout()
    outpath = Path(__file__).resolve().parent.parent / "outputs" / "gumbel_qpp_quick_test.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {outpath}", flush=True)

    # 7. Verdict
    print("\n" + "="*60)
    print("QUICK VERDICT")
    print("="*60)
    loss_drop = losses[0] - losses[-1]
    if loss_drop > 0.5:
        print(f"✅ Model is LEARNING (loss drop: {loss_drop:.3f})")
    elif loss_drop > 0.1:
        print(f"⚠️  Marginal learning (loss drop: {loss_drop:.3f})")
    else:
        print(f"❌ Not learning — Gumbel-Softmax may need tuning")

    if perm_changes and perm_changes[-1] < perm_changes[0] * 0.3:
        print(f"✅ Permutation is CONVERGING")
    elif perm_changes:
        print(f"⚠️  Permutation still changing (last: {perm_changes[-1]})")
    else:
        print(f"⚠️  Permutation tracking unavailable")

    first_anchors = model.layers[0]["attn"].q_proj.anchor_values.detach().cpu()
    anchor_std = first_anchors.std(dim=0).mean().item()
    if anchor_std > 0.001:
        print(f"✅ Anchors are non-degenerate (mean std: {anchor_std:.4f})")
    else:
        print(f"⚠️  Anchors collapsed (mean std: {anchor_std:.6f})")
    print("="*60)


if __name__ == "__main__":
    train()
