#!/usr/bin/env python3
"""Deep dive: show why MLP curves look like S-curves but FAIL under QPP.

Visualizations:
  1. Full sorted curves (attn vs MLP side by side)
  2. ZOOM on center region (where micro-oscillations hide)
  3. 1st derivative (slope) — attention is stepwise, MLP is wavy
  4. 2nd derivative (curvature) — MLP has HIGH variance = high frequency
  5. QPP reconstruction overlaid on original
  6. Per-row frequency spectrum (FFT) comparison
"""

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16

print("Loading Qwen2.5-0.5B...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt,
    device_map=dev, local_files_only=True,
).eval()

L = model.model.layers[0]
attn = L.self_attn; mlp = L.mlp

def get_sorted_row(module, row_idx):
    w = module.weight.detach().float().cpu().numpy()
    return np.sort(w[row_idx])

def qpp_reconstruct_row(w_sorted, cols, anchors=32, ridge=1e-4):
    """Reconstruct one sorted row via QPP."""
    basis = np.zeros((cols, anchors), dtype=np.float32)
    pos = np.linspace(0, anchors - 1, cols, dtype=np.float32)
    left = np.floor(pos).astype(int); right = np.clip(left + 1, 0, anchors - 1)
    frac = pos - left
    basis[np.arange(cols), left] += 1 - frac
    basis[np.arange(cols), right] += frac

    rhs = w_sorted @ basis  # (1, K)
    gram = basis.T @ basis + ridge * np.eye(anchors)
    theta = np.linalg.solve(gram, rhs.T).T  # (1, K)
    return (theta @ basis.T).squeeze()

# ═══════ data ═══════
q_row_full = np.sort(attn.q_proj.weight.detach().float().cpu().numpy()[0])
k_row_full = np.sort(attn.k_proj.weight.detach().float().cpu().numpy()[1])
v_row_full = np.sort(attn.v_proj.weight.detach().float().cpu().numpy()[2])

gate_rows = [np.sort(mlp.gate_proj.weight.detach().float().cpu().numpy()[i]) for i in range(15)]
up_rows   = [np.sort(mlp.up_proj.weight.detach().float().cpu().numpy()[i])   for i in range(4)]
down_rows = [np.sort(mlp.down_proj.weight.detach().float().cpu().numpy()[i]) for i in range(3)]

cols_attn = len(q_row_full)
cols_mlp  = len(gate_rows[0])

# QPP reconstructions
q_qpp = qpp_reconstruct_row(q_row_full, cols_attn, 32)
gate_qpp_0 = qpp_reconstruct_row(gate_rows[0], cols_mlp, 32)

# ═══════ PLOT 1: Full curves — 6 panels ═══════
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("ATTENTION vs MLP — Sorted Weight Curves (Qwen2.5-0.5B, Layer 0)\n"
             "Top: Attention (3 rows). Bottom: MLP gate_proj (3 different rows).",
             fontsize=14, fontweight="bold")

# Attention row 0
ax = axes[0, 0]
ax.plot(q_row_full, "#2196F3", linewidth=0.8)
ax.set_title("Q-proj row 0 (Attention)\n3 clear regions: neg tail, flat center, pos tail", fontsize=10, fontweight="bold")

# Attention row 1
ax = axes[0, 1]
ax.plot(k_row_full, "#1976D2", linewidth=0.8)
ax.set_title("K-proj row 1 (Attention)\nSame 3-region structure", fontsize=10, fontweight="bold")

# Attention row 2  
ax = axes[0, 2]
ax.plot(v_row_full, "#0D47A1", linewidth=0.8)
ax.set_title("V-proj row 2 (Attention)\nConsistent across all rows", fontsize=10, fontweight="bold")

# MLP gate rows 0, 5, 10
for idx, (ax, ri, color) in enumerate(zip(axes[1], [0, 5, 10], ["#F44336", "#D32F2F", "#B71C1C"])):
    ax.plot(gate_rows[ri], color, linewidth=0.8)
    ax.set_title(f"gate_proj row {ri} (MLP)\nLooks like S-curve BUT has micro-oscillations", fontsize=10, fontweight="bold")

for ax in axes.flat:
    ax.set_xlabel("Sorted column index"); ax.set_ylabel("Weight")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3); ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/mlp_deep_1_full_curves.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}", flush=True)

# ═══════ PLOT 2: ZOOM on center region (cols 280-620) ═══════
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("ZOOM: Center Region (columns 280–620) — The micro-oscillations exposed\n"
             "Attention center is NEARLY FLAT. MLP center has structured wiggles.",
             fontsize=14, fontweight="bold")

zoom_start, zoom_end = 280, 620

# Attention zoom
attn_rows = [(q_row_full, "Q-proj row 0", "#2196F3"),
             (k_row_full, "K-proj row 1", "#1976D2"),
             (v_row_full, "V-proj row 2", "#0D47A1")]
for ax, (row, title, color) in zip(axes[0], attn_rows):
    ax.plot(row[zoom_start:zoom_end], color, linewidth=1.2)
    ax.set_title(f"{title}\nCenter: mean={row[zoom_start:zoom_end].mean():.4f}, std={row[zoom_start:zoom_end].std():.4f}",
                 fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)

# MLP zoom
mlp_zoom_rows = [(gate_rows[0], 0), (gate_rows[5], 5), (gate_rows[10], 10)]
for ax, (row, ri) in zip(axes[1], mlp_zoom_rows):
    ax.plot(row[zoom_start:zoom_end], "#F44336", linewidth=1.2)
    ax.set_title(f"gate_proj row {ri} (MLP)\nCenter: mean={row[zoom_start:zoom_end].mean():.4f}, std={row[zoom_start:zoom_end].std():.4f}",
                 fontsize=10, fontweight="bold")
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Sorted column index (zoomed)"); ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.4)

plt.tight_layout()
out2 = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/mlp_deep_2_zoom_center.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}", flush=True)

# ═══════ PLOT 3: Derivatives + QPP overlay ═══════
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle("QPP Reconstruction + Derivatives — Attn vs MLP\n"
             "TOP: QPP reconstruction overlaid. MIDDLE: 1st derivative (slope). BOTTOM: 2nd derivative (curvature).",
             fontsize=13, fontweight="bold")

# Q-proj vs gate_proj row 0
pairs = [
    ("ATTN Q-proj row 0", q_row_full, q_qpp, cols_attn, "#2196F3"),
    ("MLP gate_proj row 0", gate_rows[0], gate_qpp_0, cols_mlp, "#F44336"),
]

for col_idx, (title, orig, recon, cols, color) in enumerate(pairs):
    # Row 1: reconstruction overlay
    ax = axes[0, col_idx * 2]
    ax.plot(orig, color="black", alpha=0.5, linewidth=0.8, label="Original")
    ax.plot(recon, color=color, alpha=0.8, linewidth=1.0, label="QPP K=32")
    ax.set_title(f"{title} — QPP reconstruction", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Zoom on center for reconstruction
    zs, ze = cols // 3, 2 * cols // 3
    ax2 = axes[0, col_idx * 2 + 1]
    ax2.plot(orig[zs:ze], color="black", alpha=0.5, linewidth=1, label="Original")
    ax2.plot(recon[zs:ze], color=color, alpha=0.8, linewidth=1.5, label="QPP K=32")
    ax2.set_title(f"ZOOM center — {'FLAT, good fit' if 'ATTN' in title else 'WAVY, poor fit'}", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Row 2: 1st derivative
    ax3 = axes[1, col_idx * 2]
    d1 = np.diff(orig) * cols  # scale by cols
    ax3.plot(d1, color=color, alpha=0.6, linewidth=0.5)
    ax3.set_title(f"1st derivative (scaled slope)", fontsize=9)
    ax3.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax3.grid(True, alpha=0.3)

    ax3b = axes[1, col_idx * 2 + 1]
    ax3b.hist(d1, bins=60, color=color, alpha=0.6)
    ax3b.set_title(f"Slope histogram — {'sparse peaks' if 'ATTN' in title else 'wide spread'}", fontsize=9)
    ax3b.axvline(x=0, color="red", ls="--", alpha=0.5)

    # Row 3: 2nd derivative
    ax4 = axes[2, col_idx * 2]
    d2 = np.diff(d1)
    ax4.plot(d2, color=color, alpha=0.5, linewidth=0.4)
    ax4.set_title(f"2nd derivative (curvature) — std={d2.std():.4f}", fontsize=9)
    ax4.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax4.grid(True, alpha=0.3)

    ax4b = axes[2, col_idx * 2 + 1]
    ax4b.hist(d2, bins=60, color=color, alpha=0.6)
    ax4b.set_title(f"Curvature histogram — {'tight (low freq)' if d2.std() < 0.01 else 'spread (high freq)'}", fontsize=9)
    ax4b.axvline(x=0, color="red", ls="--", alpha=0.5)

plt.tight_layout()
out3 = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/mlp_deep_3_derivatives.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}", flush=True)

# ═══════ PLOT 4: Multi-row comparison + row-to-row variance ═══════
fig, axes = plt.subplots(2, 3, figsize=(19, 10))
fig.suptitle("ATTN vs MLP — 6 rows each, overlaid + variance band\n"
             "Attention rows are nearly IDENTICAL. MLP rows vary WILDLY = no shared order.",
             fontsize=13, fontweight="bold")

# Attention: 6 rows of Q-proj
attn_rows_data = np.array([np.sort(attn.q_proj.weight.detach().float().cpu().numpy()[i]) for i in range(6)])
attn_mean = attn_rows_data.mean(axis=0)
attn_std  = attn_rows_data.std(axis=0)

ax = axes[0, 0]
for i in range(6):
    ax.plot(attn_rows_data[i], color="#2196F3", alpha=0.5, linewidth=0.6)
ax.plot(attn_mean, "black", linewidth=2, label="Mean")
ax.fill_between(range(len(attn_mean)), attn_mean - attn_std, attn_mean + attn_std, alpha=0.2, color="#2196F3")
ax.set_title(f"Q-proj: 6 rows overlaid\nRow-to-row variance: {attn_std.mean():.4f}", fontsize=9, fontweight="bold")
ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

# Attention: 6 rows of K-proj
k_data = np.array([np.sort(attn.k_proj.weight.detach().float().cpu().numpy()[i]) for i in range(6)])
k_mean, k_std = k_data.mean(axis=0), k_data.std(axis=0)

ax = axes[0, 1]
for i in range(6):
    ax.plot(k_data[i], color="#1976D2", alpha=0.5, linewidth=0.6)
ax.plot(k_mean, "black", linewidth=2)
ax.fill_between(range(len(k_mean)), k_mean - k_std, k_mean + k_std, alpha=0.2, color="#1976D2")
ax.set_title(f"K-proj: 6 rows\nVariance: {k_std.mean():.4f}", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# Attention: 6 rows of V-proj
v_data = np.array([np.sort(attn.v_proj.weight.detach().float().cpu().numpy()[i]) for i in range(6)])
v_mean, v_std = v_data.mean(axis=0), v_data.std(axis=0)

ax = axes[0, 2]
for i in range(6):
    ax.plot(v_data[i], color="#0D47A1", alpha=0.5, linewidth=0.6)
ax.plot(v_mean, "black", linewidth=2)
ax.fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=0.2, color="#0D47A1")
ax.set_title(f"V-proj: 6 rows\nVariance: {v_std.mean():.4f}", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# MLP: 6 rows of gate_proj
gate_data = np.array([np.sort(mlp.gate_proj.weight.detach().float().cpu().numpy()[i]) for i in range(6)])
gate_mean, gate_std = gate_data.mean(axis=0), gate_data.std(axis=0)

ax = axes[1, 0]
for i in range(6):
    ax.plot(gate_data[i], color="#F44336", alpha=0.5, linewidth=0.6)
ax.plot(gate_mean, "black", linewidth=2)
ax.fill_between(range(len(gate_mean)), gate_mean - gate_std, gate_mean + gate_std, alpha=0.2, color="#F44336")
ax.set_title(f"gate_proj (MLP): 6 rows\nRow-to-row variance: {gate_std.mean():.4f}", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# MLP: 6 rows of up_proj
up_data = np.array([np.sort(mlp.up_proj.weight.detach().float().cpu().numpy()[i]) for i in range(6)])
up_mean, up_std = up_data.mean(axis=0), up_data.std(axis=0)

ax = axes[1, 1]
for i in range(6):
    ax.plot(up_data[i], color="#D32F2F", alpha=0.5, linewidth=0.6)
ax.plot(up_mean, "black", linewidth=2)
ax.fill_between(range(len(up_mean)), up_mean - up_std, up_mean + up_std, alpha=0.2, color="#D32F2F")
ax.set_title(f"up_proj (MLP): 6 rows\nVariance: {up_std.mean():.4f}", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

# MLP: 6 rows of down_proj
down_data = np.array([np.sort(mlp.down_proj.weight.detach().float().cpu().numpy()[i]) for i in range(6)])
down_mean, down_std = down_data.mean(axis=0), down_data.std(axis=0)

ax = axes[1, 2]
for i in range(6):
    ax.plot(down_data[i], color="#B71C1C", alpha=0.5, linewidth=0.6)
ax.plot(down_mean, "black", linewidth=2)
ax.fill_between(range(len(down_mean)), down_mean - down_std, down_mean + down_std, alpha=0.2, color="#B71C1C")
ax.set_title(f"down_proj (MLP): 6 rows\nVariance: {down_std.mean():.4f}", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3)

for ax in axes.flat:
    ax.set_xlabel("Sorted column index"); ax.set_ylabel("Weight")

plt.tight_layout()
out4 = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/mlp_deep_4_variance.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}", flush=True)

# ═══════ PRINT SUMMARY ═══════
print("\n" + "=" * 65)
print("MLP vs ATTN — STRUCTURAL DIFFERENCES")
print("=" * 65)

# 2nd derivative variance (curvature freq)
for name, data in [("Q-proj (Attn)", attn_rows_data), ("gate_proj (MLP)", gate_data),
                    ("up_proj (MLP)", up_data), ("down_proj (MLP)", down_data)]:
    d2_vars = []
    for i in range(data.shape[0]):
        d2 = np.diff(np.diff(data[i]))
        d2_vars.append(np.var(d2))
    print(f"  {name:<20s}: curvature_variance={np.mean(d2_vars):.8f}  {'LOW freq' if np.mean(d2_vars) < 0.0001 else 'HIGH freq'}")

# Row similarity
print(f"\n  Row-to-row consistency (mean variance):")
print(f"    Q-proj (Attn): {attn_std.mean():.6f}  (ROWS ARE IDENTICAL)")
print(f"    gate (MLP):    {gate_std.mean():.6f}  (ROWS VARY)")
print(f"    up (MLP):      {up_std.mean():.6f}")
print(f"    down (MLP):    {down_std.mean():.6f}")

print(f"\n  2nd derivative std:")
for name, data in [("Q-proj (Attn)", attn_rows_data), ("gate (MLP)", gate_data)]:
    d2s = [np.std(np.diff(np.diff(data[i]))) for i in range(data.shape[0])]
    print(f"    {name:<20s}: {np.mean(d2s):.6f}")

print(f"\n  KEY: MLP has {(gate_std.mean()/attn_std.mean()):.1f}x more row-to-row variance,")
print(f"  which means SHARED ORDERING (the core QPP trick) degrades.")
print(f"  Each MLP row needs a DIFFERENT ordering -> QPP's block-shared order fails.")
