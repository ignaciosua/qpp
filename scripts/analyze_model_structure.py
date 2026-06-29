#!/usr/bin/env python3
"""Quick model structure analyzer."""
import json, sys
from pathlib import Path
from collections import defaultdict
from safetensors import safe_open

sf_dir = Path(sys.argv[1])
index = json.loads((sf_dir / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

# Count layers
layer_indices = set()
for k in weight_map:
    if ".layers." in k:
        layer_indices.add(int(k.split(".layers.")[1].split(".")[0]))
print(f"Layers: {len(layer_indices)} ({min(layer_indices)}–{max(layer_indices)})")

# Categorize
cats = defaultdict(lambda: {"params": 0, "count": 0})
for key, shard in weight_map.items():
    if "embed_tokens" in key:
        cat = "embed_tokens"
    elif "lm_head" in key:
        cat = "lm_head"
    elif ".layers." in key:
        parts = key.split(".")
        li = parts.index("layers")
        rest = ".".join(parts[li + 2 :])
        if "self_attn" in rest:
            cat = "attention"
        elif "mlp" in rest:
            cat = "mlp"
        elif "norm" in key.lower() or "ln" in key.lower():
            cat = "norm_weightless"
        else:
            cat = f"layer_other"
    elif "norm" in key.lower() or "ln" in key.lower():
        cat = "norm_weightless"
    else:
        cat = f"other/{key}"

    sf = safe_open(str(sf_dir / shard), framework="pt")
    shape = sf.get_tensor(key).shape
    p = shape.numel()
    cats[cat]["params"] += p
    cats[cat]["count"] += 1

total = sum(v["params"] for v in cats.values())
print(f"\n{'Category':<18s} {'Count':>5s} {'Params':>14s} {'MB(BF16)':>10s} {'%':>8s}  {'Compress?':<16s} {'MB saved'}")
print("-" * 90)

for cat, info in sorted(cats.items(), key=lambda x: -x[1]["params"]):
    mb = info["params"] * 2 / 1e6
    pct = info["params"] / total * 100

    # Compressibility estimate
    note = ""
    if cat == "embed_tokens":
        note += "INT8 (2x)"
        saved = mb / 2
    elif cat == "lm_head":
        note += "tied→free"
        saved = mb
    elif cat == "attention":
        note += "QPP ~30%"
        saved = mb * 0.30
    elif cat == "mlp":
        note += "QPP ~15%?"
        saved = mb * 0.15
    elif cat == "norm_weightless":
        note += "trivial"
        saved = mb
    else:
        note += "?"
        saved = 0

    print(f"{cat:<18s} {info['count']:>5d} {info['params']:>14,d} {mb:>10.1f} {pct:>7.1f}%  {note:<16s} {saved:>8.1f}")

print("-" * 90)
mb_total = total * 2 / 1e6
print(f"{'TOTAL':<18s}              {total:>14,d} {mb_total:>10.1f} {100:>7.1f}%")
print()

# Show a few MLP shapes
print("=== MLP shapes (first 6) ===")
n = 0
for k, s in weight_map.items():
    if "mlp" in k and n < 6:
        sf = safe_open(str(sf_dir / s), framework="pt")
        shape = list(sf.get_tensor(k).shape)
        print(f"  {k}: {shape}")
        n += 1

# Show a few attention shapes
print("\n=== Attention shapes (first 6) ===")
n = 0
for k, s in weight_map.items():
    if "self_attn" in k and n < 6:
        sf = safe_open(str(sf_dir / s), framework="pt")
        shape = list(sf.get_tensor(k).shape)
        print(f"  {k}: {shape}")
        n += 1

print(f"\n=== lm_head ===")
for k, s in weight_map.items():
    if "lm_head" in k:
        sf = safe_open(str(sf_dir / s), framework="pt")
        shape = list(sf.get_tensor(k).shape)
        print(f"  {k}: {shape}")
