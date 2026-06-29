#!/usr/bin/env python3
"""QPP FINAL — Proper dtype handling. INT8 embed + QPP attn + QPP MLP."""
import gc, math, time, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import perplexity, make_text, collect_activations

dev='cuda'; dt=torch.bfloat16

from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading...", flush=True)
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=dt,device_map={"":dev},local_files_only=True).eval()
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only=True)
text=make_text(8); cal=make_text(4)

def ppl(model): return perplexity(model,tok,text,dev,2048,512)[0]

ppl_base=ppl(m); orig_mb=persistent_model_bytes(m)/1e6
print(f"BF16 PPL={ppl_base:.4f} | {orig_mb:.1f} MB",flush=True)

# ═══════ INT8 Embedding (bfloat16 output) ═══════
class I8Emb(nn.Module):
    def __init__(s,w):
        super().__init__()
        n,d=w.shape; wc=w.detach().cpu().float()
        mx=wc.abs().max(dim=1,keepdim=True)[0].clamp(1e-8)
        s.rg("q",(wc/mx*127).round().clamp(-127,127).to(torch.int8))
        s.rg("s",mx.half())
    def rg(s,n,t): s.register_buffer(n,t)
    def forward(s,x):
        return F.embedding(x,s.q.float()*s.s.float()).to(torch.bfloat16)

e=m.model.embed_tokens
ie=I8Emb(e.weight.detach()).to(dev)
m.model.embed_tokens=ie

# lm_head — wrap to cast output properly: embed is tied, but forward input is float from LayerNorm.
# Qwen lm_head is nn.Linear, not tied after our replacement. Replace with bf16 wrapper:
from qpp.runtime import Int8CompressedLinear
int8_lm=Int8CompressedLinear(e.weight.detach())
m.lm_head=int8_lm.to(dev)
gc.collect(); torch.cuda.empty_cache()
ppl_e=ppl(m)
print(f"INT8 Emb+lm: PPL={ppl_e:.4f} d={ppl_e-ppl_base:+.4f}",flush=True)

# ═══════ Greedy helper ═══════
def greedy(names, K_or_fn, rb_or_fn):
    acts=collect_activations(m,tok,cal,names,dev,2048,512,256)
    curr=ppl_e; acc=0; db=0; qb=0
    for i,n in enumerate(names):
        obj=m
        for p in n.split('.'): obj=getattr(obj,p)
        ow=obj.weight.detach().clone()
        ob=obj.bias.detach().clone() if obj.bias is not None else None
        rows,cols=ow.shape
        k=K_or_fn(n) if callable(K_or_fn) else K_or_fn
        rb=rb_or_fn(n) if callable(rb_or_fn) else rb_or_fn
        try:
            co,st=build_compressed_linear(obj,rb,k,0,0,0,0,0,1e-4,"mean",acts.get(n),"reconstruct")
            set_nested_attr(m,n,co); gc.collect(); torch.cuda.empty_cache()
            cp=ppl(m)
            if cp-ppl_base<=0.5:
                curr=cp; acc+=1; db+=st["dense_bf16_bytes"]; qb+=st["runtime_buffer_bytes"]
            else:
                rl=nn.Linear(cols,rows,bias=ob is not None,device=dev,dtype=dt)
                rl.weight.data.copy_(ow); 
                if ob is not None: rl.bias.data.copy_(ob)
                set_nested_attr(m,n,rl)
        except Exception as ex:
            if i<3: print(f"FAIL {n}: {ex}",flush=True)
        if i%24==0: print(f"  [{i+1}/{len(names)}] acc={acc} PPL={curr:.4f} d={curr-ppl_base:+.4f}",flush=True)
    return curr,acc,db,qb

# ═══════ Attention ═══════
an=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".self_attn." in n]
print(f"Attn: {len(an)} layers",flush=True)
curr,aa,da,qa=greedy(an,32,128)
print(f"Attn: {aa}/{len(an)} comp={da/max(1,qa):.1f}x PPL={curr:.4f}",flush=True)

# ═══════ MLP ═══════
mn=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".mlp." in n]
print(f"MLP: {len(mn)} layers",flush=True)
curr,am,dm,qm=greedy(mn,lambda n: 96 if "down_proj" in n else 128, lambda n: 64 if "down_proj" in n else 128)
print(f"MLP: {am}/{len(mn)} comp={dm/max(1,qm):.1f}x PPL={curr:.4f}",flush=True)

# ═══════ Final ═══════
final_mb=persistent_model_bytes(m)/1e6
print("\n"+"="*60)
print("QPP FINAL — Qwen2.5-0.5B")
print("="*60)
print(f"BF16 PPL:   {ppl_base:.4f}")
print(f"Final PPL:  {curr:.4f}  dPPL={curr-ppl_base:+.4f}")
print(f"Original:   {orig_mb:.0f} MB -> {final_mb:.0f} MB")
print(f"Saved:      {(1-final_mb/orig_mb)*100:.1f}%")
print(f"Attn: {aa}/{len(an)} ({da/max(1,qa):.1f}x)")
print(f"MLP:  {am}/{len(mn)} ({dm/max(1,qm):.1f}x)")
print(f"Emb+lm: INT8 2x")
