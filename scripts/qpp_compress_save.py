#!/usr/bin/env python3
"""QPP compress + save checkpoint. Reloadable without re-running compression.

Strategy:
  1. Compress ALL layers (attn K=32, MLP K=96-128)
  2. Save full compressed model via torch.save
  3. Save QPP artifact (anchors + orders only)
  4. Fine-tune anchors
  5. Save FT model + artifact
  6. Reload + generate to verify

Can be pointed at any HF model.
"""

import gc, time, sys, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import make_text, collect_activations, perplexity

import argparse
p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="HF model name or path")
p.add_argument("--outdir", required=True, help="Output directory for checkpoint")
p.add_argument("--no-ft", action="store_true", help="Skip fine-tuning")
args = p.parse_args()

dev='cuda'; dt=torch.bfloat16
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"Loading {args.model}...", flush=True)
m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, device_map={"":dev}, local_files_only=True).eval()
tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

text = make_text(8); cal = make_text(4)
def p(): return perplexity(m, tok, text, dev, 2048, 512)[0]

ppl_base = p()
orig_mb = persistent_model_bytes(m) / 1e6
print(f"BF16: PPL={ppl_base:.4f} | {orig_mb:.0f} MB", flush=True)

# ═══════ COMPRESS ═══════
an = [n for n, mo in m.named_modules() if isinstance(mo, nn.Linear) and ".self_attn." in n]
mn = [n for n, mo in m.named_modules() if isinstance(mo, nn.Linear) and ".mlp." in n]
all_names = mn + an
print(f"Compressing {len(all_names)} layers...", flush=True)

acts = collect_activations(m, tok, cal, all_names, dev, 2048, 512, 256)
ok = 0
for i, n in enumerate(all_names):
    obj = m
    for p2 in n.split('.'): obj = getattr(obj, p2)
    rows, cols = obj.weight.shape
    is_mlp = ".mlp." in n; is_down = "down_proj" in n
    if is_mlp: K = 96 if is_down else 128; rb = 64 if is_down else 128
    else: K = 32 if rows <= 2048 else 48; rb = 128
    try:
        co, st = build_compressed_linear(obj, rb, K, 0, 0, 0, 0, 0, 1e-4, "mean", acts.get(n), "reconstruct")
        set_nested_attr(m, n, co); ok += 1
    except Exception as e:
        if i < 3: print(f"FAIL {n}: {e}", flush=True)
    if i % 40 == 0: print(f"  [{i+1}/{len(all_names)}] ok={ok}", flush=True)

gc.collect(); torch.cuda.empty_cache()
ppl_qpp = p()
comp_mb = persistent_model_bytes(m) / 1e6
print(f"QPP only: PPL={ppl_qpp:.2f} | {comp_mb:.0f} MB ({100*(1-comp_mb/orig_mb):.1f}%)", flush=True)

# ═══════ SAVE COMPRESSED MODEL ═══════
print("Saving compressed model...", flush=True)
ckpt_path = outdir / "qpp_compressed.pt"
# Save state dict of all QPP layers + non-QPP params
state = {}
qpp_modules = {}
for name, mod in m.named_modules():
    if mod.__class__.__name__.startswith("QPP"):
        qpp_modules[name] = {
            "anchors": mod.anchors.detach().cpu(),
            "orders_i16": mod.orders_i16.detach().cpu(),
            "basis": mod.basis.detach().cpu() if mod.basis is not None else None,
            "bias_buf": mod.bias_buf.detach().cpu() if hasattr(mod, "bias_buf") and mod.bias_buf is not None else None,
            "row_slices": mod.row_slices,
            "in_features": mod.in_features,
            "out_features": mod.out_features,
        }
    elif hasattr(mod, "state_dict"):
        for k, v in mod.state_dict().items():
            state[f"{name}.{k}"] = v.detach().cpu()

torch.save({"qpp_modules": qpp_modules, "other_params": state, "config": {"model": args.model, "orig_mb": orig_mb, "comp_mb": comp_mb, "ppl_base": ppl_base, "ppl_qpp": ppl_qpp}}, ckpt_path)
print(f"Saved: {ckpt_path} ({ckpt_path.stat().st_size/1e6:.1f} MB)", flush=True)

# ═══════ FT ═══════
if not args.no_ft:
    print("\nFine-tuning anchors...", flush=True)
    trainable = []
    for mod in m.modules():
        if mod.__class__.__name__.startswith("QPP"):
            mod.anchors.requires_grad = True
            trainable.append(mod.anchors)
        else:
            for pm in mod.parameters(): pm.requires_grad = False
    print(f"Trainable: {sum(x.numel() for x in trainable):,}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=1e-3)
    ft_ids = tok(make_text(3), return_tensors="pt").input_ids.to(dev)
    L, bs = 128, 4
    for step in range(150):
        idx = torch.randint(0, max(1, ft_ids.shape[1]-L-1), (bs,))
        x = torch.stack([ft_ids[0, i:i+L] for i in idx])
        out = m(x); logits = out.logits[:, :-1].contiguous(); targets = x[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if step % 30 == 0: print(f"  FT {step:3d} loss={loss.item():.4f}", flush=True)

    gc.collect(); torch.cuda.empty_cache()
    ppl_ft = p()
    final_mb = persistent_model_bytes(m) / 1e6

    # Save FT checkpoint
    ft_path = outdir / "qpp_finetuned.pt"
    state_ft = {}
    qpp_ft = {}
    for name, mod in m.named_modules():
        if mod.__class__.__name__.startswith("QPP"):
            qpp_ft[name] = {
                "anchors": mod.anchors.detach().cpu(),
                "orders_i16": mod.orders_i16.detach().cpu(),
                "basis": mod.basis.detach().cpu() if mod.basis is not None else None,
                "bias_buf": mod.bias_buf.detach().cpu() if hasattr(mod, "bias_buf") and mod.bias_buf is not None else None,
                "row_slices": mod.row_slices,
                "in_features": mod.in_features,
                "out_features": mod.out_features,
            }
        elif hasattr(mod, "state_dict"):
            for k, v in mod.state_dict().items():
                state_ft[f"{name}.{k}"] = v.detach().cpu()
    torch.save({"qpp_modules": qpp_ft, "other_params": state_ft, "config": {"model": args.model, "orig_mb": orig_mb, "final_mb": final_mb, "ppl_base": ppl_base, "ppl_ft": ppl_ft}}, ft_path)
    print(f"Saved FT: {ft_path} ({ft_path.stat().st_size/1e6:.1f} MB)", flush=True)

    savings = (1 - final_mb / orig_mb) * 100
else:
    ppl_ft = ppl_qpp
    final_mb = comp_mb
    savings = (1 - comp_mb / orig_mb) * 100

# ═══════ GENERATION ═══════
print(f"\n{'='*55}")
print("GENERATION")
print(f"{'='*55}")
prompts = ["Explain quantum computing", "The capital of France is", "Artificial intelligence will"]
for prompt in prompts:
    inp = tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.9)
    gen = tok.decode(out[0], skip_special_tokens=True)[len(prompt):].strip()
    print(f"\n[{prompt}]")
    print(f"-> {gen[:200]}")

rpt = {"model": args.model, "bf16_ppl": ppl_base, "qpp_ppl": ppl_qpp, "ft_ppl": ppl_ft,
       "dppl": ppl_ft - ppl_base, "orig_mb": orig_mb, "final_mb": final_mb,
       "savings_pct": savings, "layers_total": len(all_names), "layers_compressed": ok,
       "compressed_ckpt": str(ckpt_path), "ft_ckpt": str(ft_path) if not args.no_ft else None}
(outdir / "report.json").write_text(json.dumps(rpt, indent=2))
print(f"\nSaved: {outdir}/report.json")
print("DONE")
