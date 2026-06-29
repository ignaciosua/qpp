#!/usr/bin/env python3
"""Show QPP reconstruction error per layer type at K=32 anchors.
Fixes the transpose bug — paper reports <2% for attention at K=32."""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

print("Loading Qwen2.5-0.5B...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt,
    device_map={"": dev}, local_files_only=True,
).eval()

L = model.model.layers[0]
attn = L.self_attn; mlp = L.mlp; embed = model.model.embed_tokens

# ═══════ Correct QPP implementation ═══════
def interp_basis(cols, anchors):
    pos = np.linspace(0, anchors - 1, cols, dtype=np.float32)
    left = np.floor(pos).astype(int); right = np.clip(left + 1, 0, anchors - 1)
    frac = pos - left
    B = np.zeros((cols, anchors), dtype=np.float32)
    B[np.arange(cols), left] += 1 - frac
    B[np.arange(cols), right] += frac
    return B

def qpp_reconstruct(w, row_block=128, anchors=32):
    rows, cols = w.shape
    basis = interp_basis(cols, anchors)  # (C, K)
    recon = np.empty_like(w)
    for start in range(0, rows, row_block):
        end = min(start + row_block, rows)
        block = w[start:end].astype(np.float64)           # (R_b, C)
        order = np.argsort(block.mean(axis=0))            # (C,)
        sorted_w = block[:, order]                        # (R_b, C) sorted

        # Solve: sorted_w = theta @ basis.T  =>  theta @ basis.T = sorted_w
        # (sorted_w @ basis) (R_b,K) = theta @ (basis.T @ basis) (K,K)
        rhs = sorted_w @ basis                            # (R_b, K)
        gram = basis.T @ basis
        gram.flat[::anchors + 1] += 1e-4                  # ridge
        theta = np.linalg.solve(gram, rhs.T).T            # (R_b, K)

        rec_sorted = theta @ basis.T                      # (R_b, C)
        rec_block = np.empty_like(rec_sorted)
        # rec_block[:, order] = rec_sorted  — doesn't work in numpy
        # Use put_along_axis
        np.put_along_axis(rec_block, order[None, :], rec_sorted, axis=1)
        recon[start:end] = rec_block

    diff = recon - w
    rel_rmse = float(np.sqrt(np.mean(diff**2)) / max(np.sqrt(np.mean(w**2)), 1e-12))
    return recon, rel_rmse

# ═══════ Compute ═══════
print("\nComputing QPP reconstruction error at K=32...", flush=True)
layers = {
    "Q-proj (Attention)":  attn.q_proj.weight.detach().float().cpu().numpy(),
    "K-proj (Attention)":  attn.k_proj.weight.detach().float().cpu().numpy(),
    "V-proj (Attention)":  attn.v_proj.weight.detach().float().cpu().numpy(),
    "O-proj (Attention)":  attn.o_proj.weight.detach().float().cpu().numpy(),
    "gate_proj (MLP)":     mlp.gate_proj.weight.detach().float().cpu().numpy(),
    "up_proj (MLP)":       mlp.up_proj.weight.detach().float().cpu().numpy(),
    "down_proj (MLP)":     mlp.down_proj.weight.detach().float().cpu().numpy(),
}

results = {}
for name, w in layers.items():
    _, r = qpp_reconstruct(w, anchors=32)
    results[name] = r
    print(f"  {name:<22s} QPP relRMSE={r*100:5.2f}%", flush=True)

# Embedding
w_emb = embed.weight.detach().float().cpu().numpy()
_, emb_err = qpp_reconstruct(w_emb, row_block=w_emb.shape[0], anchors=32)
print(f"  {'Embedding':<22s} QPP relRMSE={emb_err*100:5.2f}%", flush=True)

# ═══════ Plot ═══════
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle("QPP Reconstruction Error at K=32 Anchors — Qwen2.5-0.5B\n"
             "Blue = Attention (error OK). Red = MLP (error higher).",
             fontsize=13, fontweight="bold")

for idx, (name, w) in enumerate(layers.items()):
    ax = axes.flat[idx]
    is_attn = "Attention" in name
    color = "#2196F3" if is_attn else "#F44336"
    
    # Plot 5 rows: sorted original
    for i in range(min(5, w.shape[0])):
        ax.plot(np.sort(w[i]), color=color, alpha=0.5+0.1*i, linewidth=1)
    
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3, lw=0.5)
    err = results[name]
    status = "OK" if err < 0.03 else "WARN" if err < 0.10 else "HIGH"
    ax.set_title(f"{name}\nrelRMSE={err*100:.2f}% [{status}]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Sorted column index", fontsize=8)
    ax.set_ylabel("Weight value", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=7)

# Embedding
ax = axes.flat[len(layers)]
for i in range(min(5, w_emb.shape[0])):
    ax.plot(np.sort(w_emb[i]), color="black", alpha=0.5+0.1*i, linewidth=1)
ax.axhline(y=0, color="gray", ls="--", alpha=0.3, lw=0.5)
ax.set_title(f"Embedding\nrelRMSE={emb_err*100:.2f}% [no shared structure]", fontsize=10, fontweight="bold")
ax.set_xlabel("Sorted column index", fontsize=8)
ax.set_ylabel("Weight value", fontsize=8)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=7)

for ax in axes.flat[len(layers)+1:]:
    ax.axis("off")

plt.tight_layout()
out = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/layer_quantile_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}", flush=True)

# ═══════ Summary ═══════
print("\n" + "=" * 65)
print("QPP RECONSTRUCTION ERROR at K=32 (correct implementation)")
print("=" * 65)
print(f"{'Layer':<22s} {'relRMSE':>9s}  {'Verdict'}")
print("-" * 55)
for name, err in results.items():
    if err < 0.02: v = "QPP: lossless"
    elif err < 0.05: v = "QPP: good (low error)"
    elif err < 0.10: v = "QPP: borderline"
    else: v = "QPP: high error"
    print(f"  {name:<22s} {err*100:7.2f}%  {v}")
print(f"  {'Embedding':<22s} {emb_err*100:7.2f}%  QPP: no block structure")

print(f"\nTAKEAWAY:")
attn_errs = [v for k, v in results.items() if "Attn" in k or "Attention" in k]
mlp_errs = [v for k, v in results.items() if "MLP" in k]
print(f"  Attention avg error: {np.mean(attn_errs)*100:.2f}%")
print(f"  MLP avg error:       {np.mean(mlp_errs)*100:.2f}%")
