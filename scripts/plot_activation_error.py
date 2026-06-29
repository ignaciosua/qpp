#!/usr/bin/env python3
"""THE REAL TEST: Activation RMSE (not weight RMSE) with real model activations.

This is what determines PPL preservation. QPP works when activation error
is low even though weight error is high (errors cancel in dot product).
"""

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np, torch, torch.nn as nn
from qpp.compression import interp_basis, choose_block_order, solve_anchors_weight
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.bfloat16

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt, device_map=dev, local_files_only=True).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)

# ═══════ Collect real activations ═══════
text = "The quick brown fox jumps over the lazy dog. " * 20
ids = tok(text, return_tensors="pt").input_ids.to(dev)

hooks = {}
def make_hook(name):
    def fn(_, inp, out):
        if name not in hooks:
            hooks[name] = []
        hooks[name].append(inp[0].detach().cpu())
    return fn

# Hook Q-proj, K-proj, gate_proj
handles = [
    model.model.layers[0].self_attn.q_proj.register_forward_hook(make_hook("q_proj")),
    model.model.layers[0].self_attn.k_proj.register_forward_hook(make_hook("k_proj")),
    model.model.layers[0].self_attn.v_proj.register_forward_hook(make_hook("v_proj")),
    model.model.layers[0].self_attn.o_proj.register_forward_hook(make_hook("o_proj")),
    model.model.layers[0].mlp.gate_proj.register_forward_hook(make_hook("gate_proj")),
    model.model.layers[0].mlp.up_proj.register_forward_hook(make_hook("up_proj")),
    model.model.layers[0].mlp.down_proj.register_forward_hook(make_hook("down_proj")),
]
with torch.no_grad():
    _ = model(ids)
for h in handles:
    h.remove()

# ═══════ Analyze ═══════
layers = {
    "Q-proj (Attn)":   (model.model.layers[0].self_attn.q_proj,   "q_proj"),
    "K-proj (Attn)":   (model.model.layers[0].self_attn.k_proj,   "k_proj"),
    "V-proj (Attn)":   (model.model.layers[0].self_attn.v_proj,   "v_proj"),
    "O-proj (Attn)":   (model.model.layers[0].self_attn.o_proj,   "o_proj"),
    "gate_proj (MLP)": (model.model.layers[0].mlp.gate_proj,       "gate_proj"),
    "up_proj (MLP)":   (model.model.layers[0].mlp.up_proj,         "up_proj"),
    "down_proj (MLP)": (model.model.layers[0].mlp.down_proj,       "down_proj"),
}

print("\n" + "=" * 65)
print("ACTIVATION ERROR (real data) vs WEIGHT ERROR — K=32")
print("=" * 65)
print(f"{'Layer':<22s} {'WgtErr':>7s} {'ActErr':>7s} {'Verdict'}")

results = {}
for name, (module, hook_name) in layers.items():
    w = module.weight.detach().float().cpu().numpy()
    rows, cols = w.shape
    basis = interp_basis(cols, 32)

    x_act = torch.cat(hooks[hook_name], dim=0)  # real activations
    if x_act.dim() == 3:
        x_act = x_act.reshape(-1, x_act.shape[-1])
    x_np = x_act.float().numpy()

    recon = np.empty_like(w)
    act_err_num = 0.0
    act_err_den = 0.0

    for start in range(0, rows, 128):
        end = min(start + 128, rows)
        block = w[start:end]
        order = choose_block_order(block, "mean")
        sorted_w = block[:, order]

        rhs = sorted_w @ basis
        gram = basis.T @ basis + 1e-4 * np.eye(32)
        theta = np.linalg.solve(gram, rhs.T).T

        rec_sorted = theta @ basis.T
        rec_block = np.empty_like(rec_sorted)
        np.put_along_axis(rec_block, order[None, :], rec_sorted, axis=1)
        recon[start:end] = rec_block

        # Activation error from REAL activations
        y_ref = x_np[:, order] @ sorted_w.T
        y_hat = x_np[:, order] @ rec_sorted.T
        act_err_num += float(np.sum((y_ref - y_hat) ** 2))
        act_err_den += float(np.sum(y_ref ** 2))

    w_err = float(np.sqrt(np.mean((recon - w)**2)) / max(np.sqrt(np.mean(w**2)), 1e-12))
    a_err = float(np.sqrt(act_err_num / max(act_err_den, 1e-12)))

    if a_err < 0.05: v = "✅ QPP works"
    elif a_err < 0.15: v = "⚠️  Borderline"
    else: v = "❌ QPP fails"

    results[name] = (w_err, a_err, v)
    print(f"  {name:<22s} {w_err*100:6.1f}% {a_err*100:6.1f}%  {v}")

# ═══════ Plot ═══════
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flat
fig.suptitle("Sorted Weights + Real Activation Error — Qwen2.5-0.5B Layer 0\n"
             "Activation error (not weight error) determines PPL preservation.",
             fontsize=13, fontweight="bold")

for idx, (name, (module, hook_name)) in enumerate(layers.items()):
    ax = axes[idx]
    w = module.weight.detach().float().cpu().numpy()
    w_err, a_err, v = results[name]

    is_attn = "Attn" in name
    color = "#2196F3" if is_attn else "#F44336"

    for i in range(min(5, w.shape[0])):
        ax.plot(np.sort(w[i]), color=color, alpha=0.5+0.1*i, linewidth=1)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.3, lw=0.5)

    ax.set_title(f"{name}\nWgtErr={w_err*100:.1f}% | ActErr={a_err*100:.2f}% | {v}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("Sorted column index", fontsize=7)
    ax.set_ylabel("Weight", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)

axes[len(layers)].axis("off")

plt.tight_layout()
out = "/media/neo/Data2/ainanana/all/experiments/ml/lab/qpp_repo/outputs/layer_activation_error.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}", flush=True)

attn_a = [a for n, (_, a, _) in results.items() if "Attn" in n]
mlp_a  = [a for n, (_, a, _) in results.items() if "MLP" in n]
print(f"\nAttention avg ActError: {np.mean(attn_a)*100:.2f}%")
print(f"MLP avg ActError:       {np.mean(mlp_a)*100:.2f}%")
print(f"\n→ Activation error is what QPP optimizes (not weight error).")
print(f"→ Lower activation error = better PPL preservation.")
