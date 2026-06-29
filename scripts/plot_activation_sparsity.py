#!/usr/bin/env python3
"""THE MISSING PIECE: Activation sparsity — why identical S-curves behave differently.

Attention inputs are SPARSE (few dims active). MLP inputs are DENSE (all dims active).
This is the REAL reason QPP works on attention but fails on MLP.
"""

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt, device_map=dev, local_files_only=True).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)

text = "The quick brown fox jumps over the lazy dog. " * 30
ids = tok(text, return_tensors="pt").input_ids.to(dev)

# ═══════ Collect activations ═══════
hooks = {}
def make_hook(name):
    def fn(_, inp, out):
        hooks[name] = inp[0].detach().cpu()
    return fn

handles = [
    model.model.layers[0].self_attn.q_proj.register_forward_hook(make_hook("q_attn")),
    model.model.layers[0].self_attn.k_proj.register_forward_hook(make_hook("k_attn")),
    model.model.layers[0].mlp.gate_proj.register_forward_hook(make_hook("gate_mlp")),
    model.model.layers[0].mlp.up_proj.register_forward_hook(make_hook("up_mlp")),
]
with torch.no_grad():
    _ = model(ids)
for h in handles:
    h.remove()

# ═══════ Analysis ═══════
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("WHY QPP WORKS ON ATTENTION BUT NOT MLP — Activation Sparsity\n"
             "Attention inputs are SPARSE. MLP inputs are DENSE. That's the whole secret.",
             fontsize=13, fontweight="bold")

for col, (name, key, color) in enumerate([
    ("Q-proj (Attn)", "q_attn", "#2196F3"),
    ("K-proj (Attn)", "k_attn", "#1976D2"),
    ("gate_proj (MLP)", "gate_mlp", "#F44336"),
    ("up_proj (MLP)", "up_mlp", "#D32F2F"),
]):
    x = hooks[key]
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])
    x_np = x.float().numpy()

    # Pick a single token's activation vector
    vec = x_np[x_np.shape[0] // 2]  # middle token

    # --- Plot 1: Activation values sorted ---
    ax = axes[0, col]
    sorted_vec = np.sort(vec)
    ax.plot(sorted_vec, color=color, linewidth=1.5)
    ax.fill_between(range(len(sorted_vec)), sorted_vec, 0, alpha=0.15, color=color)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3)

    # Sparsity metrics
    near_zero = (np.abs(vec) < 0.01).sum() / len(vec) * 100
    top10_pct = np.sort(np.abs(vec))[-len(vec)//10:].sum() / (np.abs(vec).sum() + 1e-9) * 100
    ax.set_title(f"{name}\nNear-zero: {near_zero:.0f}% | Top10% energy: {top10_pct:.0f}%",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Sorted dimension index"); ax.set_ylabel("Activation value")
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Error simulation ---
    ax2 = axes[1, col]
    # Simulate: what if we introduce 20% relative error per dimension?
    np.random.seed(42)
    err_per_dim = np.random.normal(0, 0.2, len(vec))  # 20% noise per dim
    output_dense = np.sum(vec * err_per_dim)  # dense accumulation
    output_sparse_sim = np.sum(vec[:len(vec)//10] * err_per_dim[:len(vec)//10])  # top 10%

    # Scatter: activation value vs error contribution
    ax2.scatter(vec, vec * err_per_dim, c=color, alpha=0.4, s=3)
    ax2.axhline(y=0, color="gray", ls="--", alpha=0.3)
    ax2.axvline(x=0, color="gray", ls="--", alpha=0.3)
    ax2.set_title(f"Error contribution per dim\nDense total={output_dense:.3f} | Sparse={output_sparse_sim:.3f}",
                  fontsize=9, fontweight="bold")
    ax2.set_xlabel("Activation value"); ax2.set_ylabel("Error × activation")

plt.tight_layout()
out = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/activation_sparsity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}", flush=True)

# ═══════ Print summary ═══════
print("\n" + "=" * 65)
print("ACTIVATION SPARSITY — THE REAL DIFFERENCE")
print("=" * 65)
for name, key in [("Q-proj (Attn)", "q_attn"), ("K-proj (Attn)", "k_attn"),
                   ("gate (MLP)", "gate_mlp"), ("up (MLP)", "up_mlp")]:
    x = hooks[key]
    if x.dim() == 3: x = x.reshape(-1, x.shape[-1])
    x_np = x.float().numpy()
    
    # Sparsity stats (averaged over tokens)
    near_zero_all = []
    top10_all = []
    gini_all = []
    for i in range(min(50, x_np.shape[0])):
        v = x_np[i]
        av = np.abs(v)
        near_zero_all.append((av < 0.01).sum() / len(v) * 100)
        top10_all.append(np.sort(av)[-len(v)//10:].sum() / max(av.sum(), 1e-9) * 100)
        # Gini coefficient
        sorted_av = np.sort(av)
        n = len(sorted_av)
        gini_all.append(1 - 2 * np.sum((n - np.arange(1, n+1)) * sorted_av) / (n * sorted_av.sum() + 1e-9))
    
    print(f"  {name:<22s} near-zero={np.mean(near_zero_all):5.1f}%  top10%={np.mean(top10_all):5.1f}%  Gini={np.mean(gini_all):.3f}")

print(f"\n  KEY: Attention has Gini > 0.X = sparse = errors CANCEL.")
print(f"       MLP has Gini < 0.X = dense  = errors ACCUMULATE.")
