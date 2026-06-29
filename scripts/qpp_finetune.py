#!/usr/bin/env python3
"""H1.2: QPP-aware Fine-Tuning — recover PPL after compression.

Pipeline:
  1. Load pretrained model -> measure BF16 baseline PPL
  2. Compress attention layers with QPP + INT8 MLP -> measure PPL loss
  3. QPP-aware FT: freeze orders, train only anchors
  4. Measure recovered PPL
"""

import gc, json, math, time, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.compression import interp_basis, choose_block_order, solve_anchors_weight
from qpp.benchmark import perplexity, make_text

# ═══════ Int8Linear ═══════

class Int8Linear(nn.Module):
    def __init__(self, w, bias=None):
        super().__init__()
        w_cpu = w.detach().cpu().float()
        mx = w_cpu.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = mx / 127.0
        q = (w_cpu / scale).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.half())
        if bias is not None:
            self.register_buffer("bias_buf", bias.detach().cpu())
        else:
            self.bias_buf = None

    def forward(self, x):
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        b = self.bias_buf.to(x.dtype) if self.bias_buf is not None else None
        return F.linear(x, w, b)

# ═══════ QPPTrainableLinear ═══════

class QPPTrainableLinear(nn.Module):
    def __init__(self, weight, row_block=128, anchors=32, bias=None):
        super().__init__()
        rows, cols = weight.shape
        w = weight.detach().float().cpu().numpy()
        blocks = int(math.ceil(rows / row_block))
        basis_np = interp_basis(cols, anchors)
        orders = np.zeros((blocks, cols), dtype=np.int16)
        anchors_init = np.zeros((rows, anchors), dtype=np.float32)
        self.row_slices = []
        for b, start in enumerate(range(0, rows, row_block)):
            end = min(start + row_block, rows)
            self.row_slices.append((start, end))
            block = w[start:end]
            order = choose_block_order(block, "mean")
            orders[b] = order.astype(np.int16)
            anchors_init[start:end] = solve_anchors_weight(block[:, order], basis_np, 1e-4)

        self.anchors = nn.Parameter(torch.from_numpy(anchors_init).float())
        self.register_buffer("orders_i16", torch.from_numpy(orders))
        self.register_buffer("basis", torch.from_numpy(basis_np).float())
        if bias is not None:
            self.register_buffer("bias_buf", bias.detach().clone())
        else:
            self.register_buffer("bias_buf", torch.zeros(rows))

    def forward(self, x):
        original_3d = x.dim() == 3
        if original_3d:
            B, T, D = x.shape
            x = x.reshape(-1, D)
        N = x.shape[0]
        dev, dt = x.device, x.dtype
        out = torch.empty(N, self.anchors.shape[0], device=dev, dtype=dt)
        basis = self.basis.to(dev, dt)
        for bid, (s, e) in enumerate(self.row_slices):
            order = self.orders_i16[bid].long()
            xo = x[:, order]
            z = xo @ basis
            a = self.anchors[s:e].to(dev, dt)
            out[:, s:e] = z @ a.T
        out = out + self.bias_buf.to(dev, dt)
        if original_3d:
            out = out.reshape(B, T, self.anchors.shape[0])
        return out


def set_nested_attr(root, name, value):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], value)


def get_nested_attr(root, name):
    obj = root
    for p in name.split("."):
        obj = getattr(obj, p)
    return obj


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

    print("Loading Qwen2.5-0.5B...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=dt,
        device_map={"": dev}, local_files_only=True,
    ).eval()
    model.config.use_cache = False  # disable KV cache for training
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    text = make_text(8)
    eval_tokens = 2048

    # Phase 1: BF16 baseline
    print("\n--- BF16 Baseline ---", flush=True)
    ppl_bf16, _, _ = perplexity(model, tok, text, dev, eval_tokens, 512)
    print(f"BF16 PPL: {ppl_bf16:.4f}", flush=True)

    # Phase 2: Compress
    print("\n--- Compressing attention + INT8 MLP ---", flush=True)
    attn_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if ".self_attn." in name:
                attn_names.append(name)
            elif ".mlp." in name:
                set_nested_attr(model, name, Int8Linear(mod.weight, mod.bias).to(dev))

    print(f"Compressing {len(attn_names)} attention layers", flush=True)
    for name in attn_names:
        mod = get_nested_attr(model, name)
        qpp = QPPTrainableLinear(mod.weight, anchors=32, bias=mod.bias).to(dev)
        set_nested_attr(model, name, qpp)

    gc.collect(); torch.cuda.empty_cache()
    ppl_qpp, _, _ = perplexity(model, tok, text, dev, eval_tokens, 512)
    delta = ppl_qpp - ppl_bf16
    print(f"QPP PPL: {ppl_qpp:.4f} (dPPL={delta:+.4f})", flush=True)

    # Phase 3: QPP-aware FT
    print("\n--- QPP-aware Fine-Tuning ---", flush=True)
    trainable = []
    for mod in model.modules():
        if isinstance(mod, QPPTrainableLinear):
            mod.anchors.requires_grad = True
            trainable.append(mod.anchors)
        else:
            for p in mod.parameters():
                p.requires_grad = False
    print(f"Trainable: {sum(p.numel() for p in trainable):,} anchor params", flush=True)

    opt = torch.optim.AdamW(trainable, lr=1e-3)
    ft_ids = tok(text[:2000], return_tensors="pt").input_ids.to(dev)
    L, bs = 128, 4
    ft_losses = []
    t0 = time.perf_counter()

    for step in range(100):
        idx = torch.randint(0, max(1, ft_ids.shape[1] - L - 1), (bs,))
        x = torch.stack([ft_ids[0, i:i+L] for i in idx])
        out = model(x)
        logits = out.logits[:, :-1].contiguous()
        targets = x[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        ft_losses.append(loss.item())
        if step % 20 == 0 or step == 99:
            print(f"  FT step {step:3d} loss={loss.item():.4f}", flush=True)

    ft_time = time.perf_counter() - t0
    print(f"FT done in {ft_time:.0f}s", flush=True)

    # Phase 4: Recovered PPL
    ppl_ft, _, _ = perplexity(model, tok, text, dev, eval_tokens, 512)
    delta_ft = ppl_ft - ppl_bf16
    recovered = (delta - delta_ft) / max(delta, 1e-9) * 100

    print("\n" + "=" * 55)
    print("QPP-AWARE FINE-TUNING RESULTS")
    print("=" * 55)
    print(f"  BF16:     {ppl_bf16:.4f}")
    print(f"  QPP:      {ppl_qpp:.4f}  (dPPL={delta:+.4f})")
    print(f"  QPP+FT:   {ppl_ft:.4f}  (dPPL={delta_ft:+.4f})")
    print(f"  Recovered: {recovered:.1f}% of dPPL")
    print(f"  Time: {ft_time:.0f}s ({len(ft_losses)} steps)")
    if recovered > 80:
        print("  [OK] QPP+FT recovers >80% quality")
    elif recovered > 40:
        print("  [WARN] Partial recovery")
    else:
        print("  [FAIL] FT not recovering")


if __name__ == "__main__":
    main()
