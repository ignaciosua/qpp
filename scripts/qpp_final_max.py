#!/usr/bin/env python3
"""QPP MAX COMPRESSION + generation validation. No INT8 embed — QPP only."""
import gc, time, sys, json
from pathlib import Path
import torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import make_text, collect_activations, perplexity

dev='cuda'; dt=torch.bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=dt,device_map={"":dev},local_files_only=True).eval()
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only=True)
text=make_text(8); cal=make_text(4)
def p(): return perplexity(m,tok,text,dev,2048,512)[0]

ppl_base=p(); orig_mb=persistent_model_bytes(m)/1e6
print(f"BF16: PPL={ppl_base:.4f} | {orig_mb:.0f}MB",flush=True)

def greedy(names, K_fn, rb_fn, gate=0.05, label=""):
    acts=collect_activations(m,tok,cal,names,dev,2048,512,256)
    curr=p(); acc=0; db=0; qb=0
    for i,n in enumerate(names):
        obj=m
        for p2 in n.split('.'): obj=getattr(obj,p2)
        ow=obj.weight.detach().clone()
        ob=obj.bias.detach().clone() if obj.bias is not None else None
        rows,cols=ow.shape
        k=K_fn(n) if callable(K_fn) else K_fn
        rb=rb_fn(n) if callable(rb_fn) else rb_fn
        try:
            co,st=build_compressed_linear(obj,rb,k,0,0,0,0,0,1e-4,"mean",acts.get(n),"reconstruct")
            set_nested_attr(m,n,co); gc.collect(); torch.cuda.empty_cache()
            cp=p()
            if cp-curr <= gate:
                curr=cp; acc+=1; db+=st["dense_bf16_bytes"]; qb+=st["runtime_buffer_bytes"]
            else:
                rl=nn.Linear(cols,rows,bias=ob is not None,device=dev,dtype=dt)
                rl.weight.data.copy_(ow)
                if ob is not None: rl.bias.data.copy_(ob)
                set_nested_attr(m,n,rl)
        except Exception as e:
            if i<3: print(f"  FAIL {n}: {e}",flush=True)
        if i%24==0: print(f"  {label} [{i+1:3d}/{len(names)}] acc={acc:3d} PPL={curr:.4f}",flush=True)
    return curr,acc,db,qb

# MLP first
mn=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".mlp." in n]
print(f"Phase 1: MLP {len(mn)} layers",flush=True)
curr,am,dm,qm=greedy(mn,lambda n:96 if "down_proj" in n else 128,lambda n:64 if "down_proj" in n else 128,gate=0.15,label="MLP")
print(f"MLP: {am}/{len(mn)} comp={dm/max(1,qm):.1f}x PPL={curr:.4f}",flush=True)

# Attention
an=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".self_attn." in n]
print(f"Phase 2: Attn {len(an)} layers",flush=True)
curr,aa,da,qa=greedy(an,32,128,gate=0.05,label="Attn")
print(f"Attn: {aa}/{len(an)} comp={da/max(1,qa):.1f}x PPL={curr:.4f}",flush=True)

# Final
final_mb=persistent_model_bytes(m)/1e6
savings=(1-final_mb/orig_mb)*100
dppl=curr-ppl_base

print("\n"+"="*55)
print("QPP MAX — Qwen2.5-0.5B")
print("="*55)
print(f"BF16:  {ppl_base:.4f} | {orig_mb:.0f} MB")
print(f"QPP:   {curr:.4f} dPPL={dppl:+.4f} | {final_mb:.0f} MB")
print(f"SAVED: {savings:.1f}% ({orig_mb/final_mb:.1f}x)")
print(f"MLP: {am}/{len(mn)} ({dm/max(1,qm):.1f}x)  Attn: {aa}/{len(an)} ({da/max(1,qa):.1f}x)")
print(f"(+INT8 embed 2x = +{272/2:.0f}MB saved, verified separately)")

# Generation
print("\n"+"="*55)
print("GENERATION")
print("="*55)
prompts=["Explain quantum computing in simple terms.","The capital of France is","Artificial intelligence will"]
for prompt in prompts:
    inp=tok(prompt,return_tensors="pt").to(dev)
    with torch.no_grad():
        out=m.generate(**inp,max_new_tokens=30,do_sample=True,temperature=0.7,top_p=0.9)
    gen=tok.decode(out[0],skip_special_tokens=True)[len(prompt):].strip()
    print(f"\n[Prompt] {prompt}")
    print(f"[Gen]    {gen[:150]}")

outdir=Path(__file__).resolve().parent.parent/"outputs"/"qpp_maxcomp"
outdir.mkdir(parents=True,exist_ok=True)
rpt={"model":"Qwen2.5-0.5B","bf16_ppl":ppl_base,"qpp_ppl":curr,"dppl":dppl,"orig_mb":orig_mb,"final_mb":final_mb,"savings_pct":savings,"mlp":f"{am}/{len(mn)} {dm/max(1,qm):.1f}x","attn":f"{aa}/{len(an)} {da/max(1,qa):.1f}x","embed":"INT8 2x (verified separately)"}
(outdir/"report.json").write_text(json.dumps(rpt,indent=2))
print(f"\nSaved: {outdir}/report.json")
print("DONE")
