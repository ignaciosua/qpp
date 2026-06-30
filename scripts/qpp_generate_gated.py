#!/usr/bin/env python3
"""QPP with GREEDY GATE + generate. Only keep layers that pass quality check."""
import gc, time, sys
import torch, torch.nn as nn
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import make_text, collect_activations, perplexity

dev='cuda'; dt=torch.bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading...", flush=True)
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",torch_dtype=dt,device_map={"":dev},local_files_only=True).eval()
tok=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only=True)
text=make_text(8); cal=make_text(4)
def ppl(): return perplexity(m,tok,text,dev,2048,512)[0]

ppl_base=ppl(); orig_mb=persistent_model_bytes(m)/1e6
print(f"BF16: PPL={ppl_base:.4f} {orig_mb:.0f}MB", flush=True)

def greedy(names, K_fn, rb_fn, gate=0.05):
    acts=collect_activations(m,tok,cal,names,dev,2048,512,256)
    curr=ppl_base; acc=0; db=0; qb=0
    for i,n in enumerate(names):
        obj=m
        for p in n.split('.'): obj=getattr(obj,p)
        ow=obj.weight.detach().clone(); ob=obj.bias.detach().clone() if obj.bias is not None else None
        rows,cols=ow.shape
        k=K_fn(n) if callable(K_fn) else K_fn; rb=rb_fn(n) if callable(rb_fn) else rb_fn
        try:
            co,st=build_compressed_linear(obj,rb,k,0,0,0,0,0,1e-4,"mean",acts.get(n),"reconstruct")
            set_nested_attr(m,n,co); gc.collect(); torch.cuda.empty_cache()
            cp=ppl()
            if cp-curr <= gate:
                curr=cp; acc+=1; db+=st["dense_bf16_bytes"]; qb+=st["runtime_buffer_bytes"]
            else:
                rl=nn.Linear(cols,rows,bias=ob is not None,device=dev,dtype=dt)
                rl.weight.data.copy_(ow)
                if ob is not None: rl.bias.data.copy_(ob)
                set_nested_attr(m,n,rl)
        except: pass
        if i%30==0: print(f"  [{i+1}/{len(names)}] acc={acc} PPL={curr:.4f}",flush=True)
    return curr,acc,db,qb

an=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".self_attn." in n]
mn=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".mlp." in n]

t0=time.perf_counter()
curr,aa,da,qa=greedy(an, 32, 128)
curr,am,dm,qm=greedy(mn, lambda n: 96 if "down_proj" in n else 128, lambda n: 64 if "down_proj" in n else 128)
elapsed=time.perf_counter()-t0

comp_mb=persistent_model_bytes(m)/1e6
total_save=(1-comp_mb/orig_mb)*100
print(f"\nCompressed: {aa}/{len(an)} attn + {am}/{len(mn)} mlp in {elapsed:.0f}s", flush=True)
print(f"Size: {orig_mb:.0f}->{comp_mb:.0f}MB ({total_save:.1f}% saved)", flush=True)
print(f"PPL: {ppl_base:.4f} -> {curr:.4f} (dPPL={curr-ppl_base:+.4f})", flush=True)

# GENERATE
print("\n" + "=" * 55)
print("GENERATION")
print("=" * 55)
prompts = [
    "Explain quantum computing in simple terms.",
    "The capital of France is",
    "Artificial intelligence will",
]
for prompt in prompts:
    inp=tok(prompt, return_tensors="pt").to(dev)
    with torch.no_grad():
        out=m.generate(**inp, max_new_tokens=30, do_sample=True, temperature=0.7, top_p=0.9)
    text=tok.decode(out[0], skip_special_tokens=True)
    print(f"\n[Prompt] {prompt}")
    print(f"[Gen]    {text[len(prompt):].strip()[:150]}")
print(f"\nComp: {total_save:.1f}% | dPPL={curr-ppl_base:+.3f}")
