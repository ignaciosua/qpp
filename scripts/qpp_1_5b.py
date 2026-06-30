#!/usr/bin/env python3
"""QPP on Qwen2.5-1.5B: compress all layers, FT anchors, generate."""
import gc, time, sys, json
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.runtime import build_compressed_linear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import make_text, collect_activations, perplexity

dev='cuda'; dt=torch.bfloat16
MODEL="/home/neo/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B/snapshots/8a16abf2848eda07cc5253dec660bf1ce007ad7a"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"Loading {MODEL}...",flush=True)
m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=dt,device_map={"":dev},local_files_only=True).eval()
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
text=make_text(8); cal=make_text(4)

def p(): return perplexity(m,tok,text,dev,2048,512)[0]
ppl_base=p(); orig_mb=persistent_model_bytes(m)/1e6
print(f"BF16: PPL={ppl_base:.4f} | {orig_mb:.0f} MB",flush=True)

# ═══════ COMPRESS ALL ═══════
an=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".self_attn." in n]
mn=[n for n,mo in m.named_modules() if isinstance(mo,nn.Linear) and ".mlp." in n]
all_names = mn + an
print(f"Compressing {len(all_names)} layers ({len(an)} attn + {len(mn)} mlp)...",flush=True)

ok=0; db=0; qb=0
acts=collect_activations(m,tok,cal,all_names,dev,2048,512,256)
for i,n in enumerate(all_names):
    obj=m
    for p2 in n.split('.'): obj=getattr(obj,p2)
    rows,cols=obj.weight.shape
    is_mlp=".mlp." in n; is_down="down_proj" in n
    if is_mlp: K=96 if is_down else 128; rb=64 if is_down else 128
    else: K=32 if rows<=2048 else 48; rb=128
    try:
        co,st=build_compressed_linear(obj,rb,K,0,0,0,0,0,1e-4,"mean",acts.get(n),"reconstruct")
        set_nested_attr(m,n,co); ok+=1; db+=st["dense_bf16_bytes"]; qb+=st["runtime_buffer_bytes"]
    except:
        pass
    if i%40==0: print(f"  [{i+1}/{len(all_names)}] ok={ok}",flush=True)

gc.collect(); torch.cuda.empty_cache()
ppl_qpp=p(); comp_mb=persistent_model_bytes(m)/1e6
savings_qpp=(1-comp_mb/orig_mb)*100
print(f"QPP only: PPL={ppl_qpp:.2f} | {comp_mb:.0f} MB ({savings_qpp:.1f}% saved)",flush=True)

# ═══════ FT ═══════
print("\nFine-tuning anchors...",flush=True)
trainable=[]
for mod in m.modules():
    if mod.__class__.__name__.startswith("QPP"):
        mod.anchors.requires_grad=True
        trainable.append(mod.anchors)
    else:
        for pm in mod.parameters(): pm.requires_grad=False
print(f"Trainable: {sum(x.numel() for x in trainable):,} params",flush=True)

opt=torch.optim.AdamW(trainable,lr=1e-3)
ft_ids=tok(make_text(3),return_tensors="pt").input_ids.to(dev)
L,bs=128,4; ft_losses=[]

for step in range(150):
    idx=torch.randint(0,max(1,ft_ids.shape[1]-L-1),(bs,))
    x=torch.stack([ft_ids[0,i:i+L] for i in idx])
    out=m(x); logits=out.logits[:,:-1].contiguous(); targets=x[:,1:].contiguous()
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step()
    ft_losses.append(loss.item())
    if step%30==0: print(f"  FT {step:3d} loss={loss.item():.4f}",flush=True)

gc.collect(); torch.cuda.empty_cache()
ppl_ft=p(); final_mb=persistent_model_bytes(m)/1e6
savings=(1-final_mb/orig_mb)*100

print(f"\n{'='*55}")
print(f"QPP 1.5B — RESULT")
print(f"{'='*55}")
print(f"BF16:    {ppl_base:.4f} | {orig_mb:.0f} MB")
print(f"QPP:     {ppl_qpp:.2f} | {comp_mb:.0f} MB ({savings_qpp:.1f}%)")
print(f"QPP+FT:  {ppl_ft:.4f} | {final_mb:.0f} MB ({savings:.1f}%)")
print(f"dPPL FT: {ppl_ft-ppl_base:+.4f}")

# ═══════ GENERATE ═══════
print(f"\n{'='*55}")
print("GENERATION")
print(f"{'='*55}")
prompts=["Explain quantum computing","The capital of France is","Artificial intelligence will"]
for prompt in prompts:
    inp=tok(prompt,return_tensors="pt").to(dev)
    with torch.no_grad():
        out=m.generate(**inp,max_new_tokens=30,do_sample=True,temperature=0.7,top_p=0.9)
    gen=tok.decode(out[0],skip_special_tokens=True)[len(prompt):].strip()
    print(f"\n[{prompt}]")
    print(f"-> {gen[:250]}")

rpt={"model":MODEL,"bf16_ppl":ppl_base,"qpp_ppl":ppl_qpp,"ft_ppl":ppl_ft,
     "dppl":ppl_ft-ppl_base,"orig_mb":orig_mb,"final_mb":final_mb,
     "savings_pct":savings,"layers_total":len(all_names),"layers_compressed":ok}
outdir=Path(__file__).resolve().parent.parent/"outputs"/"qpp_1_5b"
outdir.mkdir(parents=True,exist_ok=True)
(outdir/"report.json").write_text(json.dumps(rpt,indent=2))
print(f"\nSaved: {outdir}/report.json")
print("DONE")
