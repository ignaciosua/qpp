#!/usr/bin/env python3
"""Auto-adaptive QPP: find optimal K per layer.

For each layer, sweep K = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256].
Stop when dPPL < threshold (0.01, 0.05, or 0.10).
Each layer gets the minimal K that preserves quality.

This answers: "Can we make anchors dynamic?"
"""

import torch, gc, math, time, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr
from qpp.benchmark import perplexity, make_text, collect_activations

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16

print("Loading Qwen2.5-0.5B...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt, device_map={"": dev}, local_files_only=True,
).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
text = make_text(8)

ppl_base, _, _ = perplexity(model, tok, text, dev, 2048, 512)
print(f"BF16 PPL: {ppl_base:.4f}", flush=True)

# Select sample layers: few attention + few MLP for speed
names = [
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.10.self_attn.q_proj",
    "model.layers.10.mlp.gate_proj",
    "model.layers.20.self_attn.q_proj",
    "model.layers.20.mlp.gate_proj",
    "model.layers.23.self_attn.q_proj",
    "model.layers.23.mlp.gate_proj",
]

# Collect activations
print("Collecting activations...", flush=True)
activations = collect_activations(model, tok, make_text(4), names, dev, 2048, 512, 256)

K_values = [32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
thresholds = [0.01, 0.05, 0.10]

results = []

print(f"\n{'='*95}")
print(f"AUTO-ADAPTIVE QPP — optimal K per layer (sweep {len(K_values)} values)")
print(f"{'='*95}")
print(f"{'Layer':<42s} {'Type':>5s} {'d_model':>7s} {'OptK':>5s} {'Comp':>6s} {'dPPL@opt':>9s} {'K@0.01':>6s} {'K@0.05':>6s}")
print("-" * 95)

for name in names:
    obj = model
    for p in name.split('.'): obj = getattr(obj, p)
    ow = obj.weight.detach().clone()
    ob = obj.bias.detach().clone() if obj.bias is not None else None
    rows, cols = ow.shape
    calib_x = activations.get(name)
    rb = 64 if 'down_proj' in name else 128
    layer_type = "ATTN" if "attn" in name else "MLP"

    dppl_at_K = {}
    comp_at_K = {}

    for k_val in K_values:
        try:
            comp, stats = build_compressed_linear(obj, rb, k_val, 0, 0, 0, 0, 0, 1e-4, 'mean', calib_x, 'reconstruct')
        except Exception:
            dppl_at_K[k_val] = 999
            comp_at_K[k_val] = 0
            continue

        set_nested_attr(model, name, comp)
        gc.collect(); torch.cuda.empty_cache()

        ppl_k, _, _ = perplexity(model, tok, text, dev, 2048, 512)
        dppl_at_K[k_val] = ppl_k - ppl_base
        comp_at_K[k_val] = stats['runtime_buffer_compression']

        # Rollback
        parent = model
        parts = name.split('.')
        for p in parts[:-1]: parent = getattr(parent, p)
        restore = torch.nn.Linear(cols, rows, bias=ob is not None)
        restore.weight.data.copy_(ow)
        if ob is not None: restore.bias.data.copy_(ob)
        setattr(parent, parts[-1], restore.to(dev))
        gc.collect(); torch.cuda.empty_cache()

    # Find optimal K for each threshold
    opt_info = {}
    for thresh in thresholds:
        best_k = 256
        for k_val in K_values:
            if dppl_at_K.get(k_val, 999) < thresh:
                best_k = k_val
                break
        opt_info[thresh] = best_k

    # Best K with dPPL < 0.01 and best compression
    opt_k = opt_info[0.05]  # default to 0.05 threshold
    opt_dppl = dppl_at_K.get(opt_k, 999)
    opt_comp = comp_at_K.get(opt_k, 0)

    print(f"  {name:<42s} {layer_type:>5s} {rows:>5d}x{cols:<5d} {opt_k:>4d}  {opt_comp:>4.1f}x  {opt_dppl:+8.5f}  {opt_info[0.01]:>4d}  {opt_info[0.05]:>4d}",
          flush=True)

    results.append({
        "name": name,
        "type": layer_type,
        "rows": rows,
        "cols": cols,
        "opt_k_0.01": opt_info[0.01],
        "opt_k_0.05": opt_info[0.05],
        "opt_k_0.10": opt_info[0.10],
        "comp_at_opt": opt_comp,
        "dppl_at_opt": opt_dppl,
        "dppl_curve": dppl_at_K,
        "comp_curve": comp_at_K,
    })

# ── Summary ──
print(f"\n{'='*70}")
print(f"SUMMARY — Optimal K per layer type")
print(f"{'='*70}")

for ltype in ["ATTN", "MLP"]:
    filtered = [r for r in results if r["type"] == ltype]
    if not filtered: continue
    avg_k_001 = np.mean([r["opt_k_0.01"] for r in filtered])
    avg_k_005 = np.mean([r["opt_k_0.05"] for r in filtered])
    avg_comp = np.mean([r["comp_at_opt"] for r in filtered])
    print(f"  {ltype}: avg K@0.01={avg_k_001:.0f}  K@0.05={avg_k_005:.0f}  avgComp={avg_comp:.1f}x")

# Plot dPPL vs K curves
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(16, 12))
colors_attn = plt.cm.Blues(np.linspace(0.4, 1, 7))
colors_mlp = plt.cm.Reds(np.linspace(0.4, 1, 6))

for r in results:
    ax = axes[0] if r["type"] == "ATTN" else axes[1]
    ks = sorted(r["dppl_curve"].keys())
    dppls = [r["dppl_curve"][k] for k in ks]
    color = colors_attn[len(ax.lines) % 7] if r["type"] == "ATTN" else colors_mlp[len(ax.lines) % 6]
    short = r["name"].split(".layers.")[-1].replace(".self_attn.", "/").replace(".mlp.", "/")
    ax.plot(ks, dppls, 'o-', color=color, linewidth=1.5, markersize=4, label=short, alpha=0.8)
    ax.axhline(y=0.01, color="green", ls="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=0.05, color="orange", ls="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=0.0, color="gray", alpha=0.2)
    ax.set_title(["ATTENTION layers", "MLP layers"][0 if r["type"] == "ATTN" else 1], fontsize=12, fontweight="bold")
    ax.set_xlabel("K (anchors)")
    ax.set_ylabel("dPPL")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "outputs" / "auto_anchors.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}", flush=True)

# Save JSON
outdir = Path(__file__).resolve().parent.parent / "outputs" / "auto_anchors"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
print(f"Saved: {outdir / 'results.json'}", flush=True)
