#!/usr/bin/env python3
"""QPP FINAL V2 — MLP first, then Attention, per-layer gate for MLP.

Strategy:
  MLP      → QPP K=128 gate/up, K=96 down. Gate: dPPL_vs_prev < 0.05
  Attention → QPP K=32. Gate: dPPL_vs_prev < 0.05
  Embedding → stays BF16 (INT8 known 2x, tested separately)
  
Key fix: MLP runs FIRST with per-layer gate, not cumulative vs BF16.
Then attention runs. This prevents attention from eating the PPL budget.
"""
import gc, math, time, sys
from pathlib import Path
import torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import perplexity, make_text, collect_activations

dev='cuda'; dt=torch.bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading...", flush=True)
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=dt,device_map={"":dev},local_files_only=True).eval()
t=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only=True)
text=make_text(8); cal=make_text(4)

def ppl(): return perplexity(m,t,text,dev,2048,512)[0]

ppl_base=ppl(); orig_mb=persistent_model_bytes(m)/1e6
print(f"BF16 PPL={ppl_base:.4f} | {orig_mb:.1f} MB", flush=True)

# ═══════════════════════════════════════
# Greedy with PER-LAYER GATE (dPPL vs previous layer)
# ═══════════════════════════════════════
def greedy_perlayer(names, K_or_fn, rb_or_fn, gate_threshold=0.05, label=""):
    """Accept layer if it adds < gate_threshold PPL vs current state."""
    acts=collect_activations(m,t,cal,names,dev,2048,512,256)
    curr=ppl_base  # measure from fresh model each time
    acc=0; d=0; q=0
    for i,n in enumerate(names):
        obj=m
        for p2 in n.split('.'): obj=getattr(obj,p2)
        ow=obj.weight.detach().clone()
        ob=obj.bias.detach().clone() if obj.bias is not None else None
        rows,cols=ow.shape
        k=K_or_fn(n) if callable(K_or_fn) else K_or_fn
        rb=rb_or_fn(n) if callable(rb_or_fn) else rb_or_fn
        try:
            co,st=build_compressed_linear(obj,rb,k,0,0,0,0,0,1e-4,"mean",acts.get(n),"reconstruct")
            set_nested_attr(m,n,co); gc.collect(); torch.cuda.empty_cache()
            cp=ppl()
            dppl_layer=cp-prev_ppl if i>0 else cp-ppl_base
            if cp-curr <= gate_threshold:
                curr=cp; acc+=1; d+=st["dense_bf16_bytes"]; q+=st["runtime_buffer_bytes"]
            else:
                rl=nn.Linear(cols,rows,bias=ob is not None,device=dev,dtype=dt)
                rl.weight.data.copy_(ow)
                if ob is not None: rl.bias.data.copy_(ob)
                set_nested_attr(m,n,rl)
        except Exception as ex:
            if i<3: print(f"FAIL {n}: {ex}",flush=True)
        prev_ppl=cp
        if i%24==0 or i==len(names)-1: 
            print(f"  {label} [{i+1}/{len(names)}] acc={acc} PPL={curr:.4f} d={curr-ppl_base:+.4f}",flush=True)
    return curr,acc,d,q

# ═══════════════════════════════════════
# PHASE 1: MLP FIRST (60% of model)
# ═══════════════════════════════════════
mn=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".mlp." in n]
print(f"Phase 1 — MLP: {len(mn)} layers (gate=0.05/layer)",flush=True)
prev_ppl=ppl_base
curr,am,dm,qm=greedy_perlayer(mn,
    lambda n: 96 if "down_proj" in n else 128,
    lambda n: 64 if "down_proj" in n else 128,
    gate_threshold=0.05, label="MLP")
print(f"MLP done: {am}/{len(mn)} comp={dm/max(1,qm):.1f}x PPL={curr:.4f} d={curr-ppl_base:+.4f}",flush=True)

# ═══════════════════════════════════════
# PHASE 2: ATTENTION
# ═══════════════════════════════════════
an=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".self_attn." in n]
print(f"\nPhase 2 — Attention: {len(an)} layers (gate=0.05/layer)",flush=True)
prev_ppl=curr
curr,aa,da,qa=greedy_perlayer(an, 32, 128, gate_threshold=0.05, label="Attn")
print(f"Attn done: {aa}/{len(an)} comp={da/max(1,qa):.1f}x PPL={curr:.4f} d={curr-ppl_base:+.4f}",flush=True)

# ═══════════════════════════════════════
# FINAL
# ═══════════════════════════════════════
final_mb=persistent_model_bytes(m)/1e6
total_dense=orig_mb
total_qpp_mb=final_mb
savings=(1-final_mb/orig_mb)*100

print("\n"+"="*60)
print("QPP FINAL V2 — Qwen2.5-0.5B")
print("="*60)
print(f"BF16 PPL:   {ppl_base:.4f} | {orig_mb:.0f} MB")
print(f"QPP PPL:    {curr:.4f} dPPL={curr-ppl_base:+.4f} | {final_mb:.0f} MB")
print(f"SAVED:      {savings:.1f}% ({orig_mb/final_mb:.1f}x)")
print(f"")
print(f"MLP:  {am}/{len(mn)} accepted, {dm/max(1,qm):.1f}x on accepted")
print(f"Attn: {aa}/{len(an)} accepted, {da/max(1,qa):.1f}x on accepted")
print(f"Embed: INT8 2x (not applied, tested separately)")
print(f"")
if savings > 60:
    print("[GOAL] >60% savings ACHIEVED!")
elif savings > 40:
    print("[WARN] >40% — close. MLP gate may be too strict.")
else:
    print("[WARN] <40% — needs tuning. Try gate=0.10 or more anchors.")
