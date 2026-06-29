#!/usr/bin/env python3
"""Validate QPP on MLP — K=128, using build_compressed_linear (the proven function).

Tests: single gate_proj, then greedy all MLP layers.
"""

import gc, math, time, sys, json
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, Int8CompressedLinear, persistent_model_bytes
from qpp.benchmark import perplexity, make_text, collect_activations

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

print("Loading Qwen2.5-0.5B...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt, device_map={"": dev}, local_files_only=True,
).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
text = make_text(8)
calib_text = make_text(4)
ppl_bf16, _, _ = perplexity(model, tok, text, dev, 2048, 512)
print(f"BF16 PPL: {ppl_bf16:.4f}", flush=True)

# Get MLP names
mlp_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and '.mlp.' in n]
print(f"MLP modules: {len(mlp_names)}", flush=True)

# Get all activations
activations = collect_activations(model, tok, calib_text, mlp_names, dev, 2048, 512, 256)

# ── Quick single-layer test first ──
name0 = mlp_names[0]
obj = model
for p in name0.split('.'): obj = getattr(obj, p)
orig_weight = obj.weight.detach().clone()
orig_bias = obj.bias.detach().clone() if obj.bias is not None else None
calib0 = activations.get(name0)

comp0, stats0 = build_compressed_linear(obj, 128, 128, 0, 0, 0, 0, 0, 1e-4, 'mean', calib0, 'reconstruct')
set_nested_attr(model, name0, comp0)
gc.collect(); torch.cuda.empty_cache()
ppl_k128, _, _ = perplexity(model, tok, text, dev, 2048, 512)
print(f"Test: {name0} K=128 -> ppl={ppl_k128:.4f} dPPL={ppl_k128-ppl_bf16:+.4f} comp={stats0['runtime_buffer_compression']:.1f}x", flush=True)

# Restore
obj2 = model
for p in name0.split('.')[:-1]: obj2 = getattr(obj2, p)
restore = nn.Linear(orig_weight.shape[1], orig_weight.shape[0], bias=orig_bias is not None)
restore.weight.data.copy_(orig_weight)
if orig_bias is not None: restore.bias.data.copy_(orig_bias)
setattr(obj2, name0.split('.')[-1], restore.to(dev))
gc.collect(); torch.cuda.empty_cache()

# Verify restoration
ppl_restored, _, _ = perplexity(model, tok, text, dev, 2048, 512)
print(f"Restored: ppl={ppl_restored:.4f} (should be {ppl_bf16:.4f})", flush=True)

# ── Greedy all MLP ──
print(f"\n{'='*70}")
print(f"GREEDY QPP-MLP (K=128, row_block=128/64)")
print(f"{'='*70}")
print(f"{'#':>3s} {'Module':<42s} {'DenseKB':>7s} {'QPPKB':>7s} {'Comp':>5s} {'dPPL':>8s} {'Totd':>8s} {'Acc?'}")
print("-" * 93)

current_ppl = ppl_bf16
total_dense = 0
total_qpp = 0
accepted_count = 0
rejects = []

for idx, name in enumerate(mlp_names, 1):
    obj = model
    for p in name.split('.'): obj = getattr(obj, p)
    ow = obj.weight.detach().clone()
    ob = obj.bias.detach().clone() if obj.bias is not None else None
    calib_x = activations.get(name)
    rb = 64 if 'down_proj' in name else 128

    try:
        comp, stats = build_compressed_linear(obj, rb, 128, 0, 0, 0, 0, 0, 1e-4, 'mean', calib_x, 'reconstruct')
    except Exception as e:
        print(f"{idx:3d} {name:<42s} FAIL: {e}", flush=True)
        rejects.append(name)
        continue

    set_nested_attr(model, name, comp)
    gc.collect(); torch.cuda.empty_cache()

    cand_ppl, _, _ = perplexity(model, tok, text, dev, 2048, 512)
    dppl = cand_ppl - current_ppl
    totd = cand_ppl - ppl_bf16
    dense_kb = stats['dense_bf16_bytes'] // 1024
    qpp_kb = stats['runtime_buffer_bytes'] // 1024
    comp_r = stats['runtime_buffer_compression']

    accept = totd <= 0.5
    tag = "ACCEPT" if accept else "REJECT"

    print(f"{idx:3d} {name:<42s} {dense_kb:7d} {qpp_kb:7d} {comp_r:4.1f}x {dppl:+8.4f} {totd:+8.4f}  {tag}", flush=True)

    if accept:
        current_ppl = cand_ppl
        total_dense += dense_kb * 1024
        total_qpp += qpp_kb * 1024
        accepted_count += 1
    else:
        # Rollback
        parent = model
        parts = name.split('.')
        for p in parts[:-1]: parent = getattr(parent, p)
        restore = nn.Linear(ow.shape[1], ow.shape[0], bias=ob is not None)
        restore.weight.data.copy_(ow)
        if ob is not None: restore.bias.data.copy_(ob)
        setattr(parent, parts[-1], restore.to(dev))
        rejects.append(name)

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"  BF16 PPL:        {ppl_bf16:.4f}")
print(f"  QPP-MLP PPL:     {current_ppl:.4f}  (dPPL={current_ppl-ppl_bf16:+.4f})")
print(f"  Accepted:        {accepted_count}/{len(mlp_names)}")
print(f"  Compression:     {total_dense/total_qpp:.1f}x")
print(f"  MLP saved:       {(total_dense-total_qpp)/1e6:.1f} MB")
print(f"  Total model:     {persistent_model_bytes(model)/1e6:.1f} MB")

report = {
    "bf16_ppl": ppl_bf16,
    "qpp_mlp_ppl": current_ppl,
    "dppl": current_ppl - ppl_bf16,
    "accepted": accepted_count,
    "rejected": len(rejects),
    "total": len(mlp_names),
    "compression": total_dense / max(1, total_qpp),
    "saved_mb": (total_dense - total_qpp) / 1e6,
    "rejected_modules": rejects[:10],
}
outdir = Path(__file__).resolve().parent.parent / "outputs" / "qpp_mlp_validation"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "results.json").write_text(json.dumps(report, indent=2))
print(f"\nReport: {outdir / 'results.json'}")
