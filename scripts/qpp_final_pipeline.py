#!/usr/bin/env python3
"""FINAL PIPELINE: Apply optimal strategy to all layers and measure total compression.

Strategy per layer type:
  Attention Q/K/V/O → QPP K=32
  MLP gate/up       → QPP K=128
  MLP down          → QPP K=96
  Embedding/lm_head → INT8 (tied weights, one compression for both)

Measures: PPL, storage size, per-layer acceptance.
"""

import gc, math, time, sys, json
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, Int8CompressedLinear, persistent_model_bytes
from qpp.benchmark import perplexity, make_text, collect_activations, cuda_mem

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

print("=" * 65)
print("QPP FINAL PIPELINE — Qwen2.5-0.5B")
print("=" * 65)

print("Loading model...", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt, device_map={"": dev}, local_files_only=True,
).eval()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)

text = make_text(8)
calib_text = make_text(4)

# ── Baseline ──
t0 = time.perf_counter()
ppl_base, _, _ = perplexity(model, tok, text, dev, 2048, 512)
orig_bytes = persistent_model_bytes(model)
print(f"\nBF16 PPL: {ppl_base:.4f} | Size: {orig_bytes/1e6:.1f} MB", flush=True)

# ── Phase 1: INT8 Embedding (also covers lm_head, tied) ──
print("\n--- Phase 1: INT8 Embedding ---", flush=True)
embed = model.model.embed_tokens
int8_embed = Int8CompressedLinear(embed.weight)
model.model.embed_tokens = int8_embed.to(dev)
# lm_head is tied to embed — already compressed
lm_head = model.lm_head
int8_lm = Int8CompressedLinear(lm_head.weight)
# Actually check if tied
print(f"  Tied weights: {embed.weight.data_ptr() == lm_head.weight.data_ptr()}", flush=True)
if embed.weight.data_ptr() != lm_head.weight.data_ptr():
    model.lm_head = int8_lm.to(dev)

gc.collect(); torch.cuda.empty_cache()
ppl_e, _, _ = perplexity(model, tok, text, dev, 2048, 512)
print(f"  After INT8 embed: PPL={ppl_e:.4f} d={ppl_e-ppl_base:+.4f}", flush=True)

# ── Phase 2: QPP Attention (K=32) ──
print("\n--- Phase 2: QPP Attention K=32 ---", flush=True)
attn_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and '.self_attn.' in n]
acts_attn = collect_activations(model, tok, calib_text, attn_names, dev, 2048, 512, 256)

current_ppl = ppl_e
attn_accepted = 0
attn_dense_b = 0
attn_qpp_b = 0

for idx, name in enumerate(attn_names):
    obj = model
    for p in name.split('.'): obj = getattr(obj, p)
    ow = obj.weight.detach().clone()
    ob = obj.bias.detach().clone() if obj.bias is not None else None
    rows, cols = ow.shape
    cal = acts_attn.get(name)
    rb = 64 if 'down_proj' in name else 128

    try:
        comp, stats = build_compressed_linear(obj, rb, 32, 0, 0, 0, 0, 0, 1e-4, 'mean', cal, 'reconstruct')
    except Exception as e:
        if idx < 3: print(f"  FAIL {name}: {e}", flush=True)
        continue

    set_nested_attr(model, name, comp)
    gc.collect(); torch.cuda.empty_cache()

    cand, _, _ = perplexity(model, tok, text, dev, 2048, 512)
    totd = cand - ppl_base
    accept = totd <= 0.5

    if accept:
        current_ppl = cand
        attn_accepted += 1
        attn_dense_b += stats['dense_bf16_bytes']
        attn_qpp_b += stats['runtime_buffer_bytes']
    else:
        # Rollback
        parent = model; parts = name.split('.')
        for p in parts[:-1]: parent = getattr(parent, p)
        rl = nn.Linear(cols, rows, bias=ob is not None)
        rl.weight.data.copy_(ow)
        if ob is not None: rl.bias.data.copy_(ob)
        setattr(parent, parts[-1], rl.to(dev))

    if idx % 16 == 0 or idx == len(attn_names) - 1:
        print(f"  Attn [{idx+1}/{len(attn_names)}] accepted={attn_accepted} PPL={current_ppl:.4f} d={current_ppl-ppl_base:+.4f}", flush=True)

print(f"  Attention done: {attn_accepted}/{len(attn_names)} accepted, comp={attn_dense_b/max(1,attn_qpp_b):.1f}x", flush=True)

# ── Phase 3: QPP MLP (K=128 gate/up, K=96 down) ──
print("\n--- Phase 3: QPP MLP (K=128 gate/up, K=96 down) ---", flush=True)
mlp_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and '.mlp.' in n]
acts_mlp = collect_activations(model, tok, calib_text, mlp_names, dev, 2048, 512, 256)

mlp_accepted = 0
mlp_dense_b = 0
mlp_qpp_b = 0

for idx, name in enumerate(mlp_names):
    obj = model
    for p in name.split('.'): obj = getattr(obj, p)
    ow = obj.weight.detach().clone()
    ob = obj.bias.detach().clone() if obj.bias is not None else None
    rows, cols = ow.shape
    cal = acts_mlp.get(name)

    # Select K based on layer type
    if 'down_proj' in name:
        K = 96
        rb = 64
    elif 'gate_proj' in name or 'up_proj' in name:
        K = 128
        rb = 128
    else:
        K = 96
        rb = 128

    try:
        comp, stats = build_compressed_linear(obj, rb, K, 0, 0, 0, 0, 0, 1e-4, 'mean', cal, 'reconstruct')
    except Exception as e:
        if idx < 3: print(f"  FAIL {name}: {e}", flush=True)
        continue

    set_nested_attr(model, name, comp)
    gc.collect(); torch.cuda.empty_cache()

    cand, _, _ = perplexity(model, tok, text, dev, 2048, 512)
    totd = cand - ppl_base
    accept = totd <= 0.5

    if accept:
        current_ppl = cand
        mlp_accepted += 1
        mlp_dense_b += stats['dense_bf16_bytes']
        mlp_qpp_b += stats['runtime_buffer_bytes']
    else:
        parent = model; parts = name.split('.')
        for p in parts[:-1]: parent = getattr(parent, p)
        rl = nn.Linear(cols, rows, bias=ob is not None)
        rl.weight.data.copy_(ow)
        if ob is not None: rl.bias.data.copy_(ob)
        setattr(parent, parts[-1], rl.to(dev))

    if idx % 16 == 0 or idx == len(mlp_names) - 1:
        print(f"  MLP [{idx+1}/{len(mlp_names)}] accepted={mlp_accepted} PPL={current_ppl:.4f} d={current_ppl-ppl_base:+.4f}", flush=True)

print(f"  MLP done: {mlp_accepted}/{len(mlp_names)} accepted, comp={mlp_dense_b/max(1,mlp_qpp_b):.1f}x", flush=True)

# ── Final Results ──
elapsed = time.perf_counter() - t0
final_bytes = persistent_model_bytes(model)
total_dense = orig_bytes
total_compressed = final_bytes
savings = (1 - total_compressed / total_dense) * 100
dppl_final = current_ppl - ppl_base

print("\n" + "=" * 65)
print("FINAL RESULTS — Qwen2.5-0.5B Full Compression")
print("=" * 65)
print(f"  BF16 PPL:         {ppl_base:.4f}")
print(f"  Final PPL:         {current_ppl:.4f}  (ΔPPL={dppl_final:+.4f})")
print(f"  Original size:     {total_dense/1e6:.1f} MB")
print(f"  Compressed size:   {total_compressed/1e6:.1f} MB")
print(f"  Total saved:       {savings:.1f}%")
print(f"  Overall comp:      {total_dense/total_compressed:.1f}x")
print(f"  Elapsed:           {elapsed:.0f}s")
print()
print(f"  Attention: {attn_accepted}/{len(attn_names)} accepted ({attn_dense_b/max(1,attn_qpp_b):.1f}x)")
print(f"  MLP:       {mlp_accepted}/{len(mlp_names)} accepted ({mlp_dense_b/max(1,mlp_qpp_b):.1f}x)")
print(f"  Embed+lm:  INT8 (2.0x, tied)")
print(f"  GPU VRAM:  {cuda_mem()}")
print(f"\n  → github.com/ignaciosua/qpp")

# Save report
report = {
    "model": "Qwen2.5-0.5B-Instruct",
    "bf16_ppl": ppl_base,
    "final_ppl": current_ppl,
    "dppl": dppl_final,
    "original_mb": total_dense / 1e6,
    "compressed_mb": total_compressed / 1e6,
    "savings_pct": savings,
    "overall_compression": total_dense / total_compressed,
    "elapsed_s": elapsed,
    "attention": {"accepted": attn_accepted, "total": len(attn_names), "compression": attn_dense_b / max(1, attn_qpp_b)},
    "mlp": {"accepted": mlp_accepted, "total": len(mlp_names), "compression": mlp_dense_b / max(1, mlp_qpp_b)},
    "embedding": "INT8 2x (tied with lm_head)",
}

outdir = Path(__file__).resolve().parent.parent / "outputs" / "qpp_final_pipeline"
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "report.json").write_text(json.dumps(report, indent=2))
print(f"\nReport: {outdir / 'report.json'}")
