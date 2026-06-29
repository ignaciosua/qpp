#!/usr/bin/env python3
"""QPP on Attention + MLP only. Embedding/lm_head stay BF16."""
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

def p(): return perplexity(m,t,text,dev,2048,512)[0]

ppl_base=p(); orig_mb=persistent_model_bytes(m)/1e6
print(f"BF16 PPL={ppl_base:.4f} | {orig_mb:.1f} MB", flush=True)

def greedy(names, K_or_fn, rb_or_fn):
    acts=collect_activations(m,t,cal,names,dev,2048,512,256)
    curr=ppl_base; acc=0; d=0; q=0
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
            cp=p()
            if cp-ppl_base<=0.5:
                curr=cp; acc+=1; d+=st["dense_bf16_bytes"]; q+=st["runtime_buffer_bytes"]
            else:
                rl=nn.Linear(cols,rows,bias=ob is not None,device=dev,dtype=dt)
                rl.weight.data.copy_(ow)
                if ob is not None: rl.bias.data.copy_(ob)
                set_nested_attr(m,n,rl)
        except Exception as ex:
            if i<3: print(f"FAIL {n}: {ex}",flush=True)
        if i%24==0: print(f"  [{i+1}/{len(names)}] acc={acc} PPL={curr:.4f}",flush=True)
    return curr,acc,d,q

an=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".self_attn." in n]
print(f"Attn: {len(an)} layers",flush=True)
curr,aa,da,qa=greedy(an,32,128)
print(f"Attn: {aa}/{len(an)} comp={da/max(1,qa):.1f}x PPL={curr:.4f}",flush=True)

mn=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".mlp." in n]
print(f"MLP: {len(mn)} layers",flush=True)
curr,am,dm,qm=greedy(mn,lambda n: 96 if "down_proj" in n else 128, lambda n: 64 if "down_proj" in n else 128)
print(f"MLP: {am}/{len(mn)} comp={dm/max(1,qm):.1f}x PPL={curr:.4f}",flush=True)

final_mb=persistent_model_bytes(m)/1e6
print("\n"+"="*60)
print("QPP FINAL — Qwen2.5-0.5B")
print("="*60)
print(f"BF16:    {ppl_base:.4f} | {orig_mb:.0f} MB")
print(f"QPP:     {curr:.4f} dPPL={curr-ppl_base:+.4f} | {final_mb:.0f} MB")
print(f"Saved:   {(1-final_mb/orig_mb)*100:.1f}% ({orig_mb/final_mb:.1f}x)")
print(f"Attn: {aa}/{len(an)} ({da/max(1,qa):.1f}x)  MLP: {am}/{len(mn)} ({dm/max(1,qm):.1f}x)")
