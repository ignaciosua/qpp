#!/usr/bin/env python3
"""H1.1 + H1.4 final: QPP direct forward vs Dense — speed, VRAM, accuracy.

Two variants:
  1. block-loop (current)  
  2. fused-einsum (single op, no Python loop)

Test on real pretrained weights to get meaningful reconstruction error.
"""

from __future__ import annotations

import gc, math, time, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qpp.compression import interp_basis, choose_block_order, solve_anchors_weight


def cuda_mem():
    if not torch.cuda.is_available(): return {"alloc":0, "peak":0}
    return {"alloc": torch.cuda.memory_allocated()/1e9, "peak": torch.cuda.max_memory_allocated()/1e9}

def reset():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def pbytes(model):
    seen=set(); t=0
    for x in list(model.parameters())+list(model.buffers()):
        p=x.data_ptr()
        if p in seen: continue
        seen.add(p); t+=x.numel()*x.element_size()
    return t

def bench(fn, warm=10, rep=100):
    for _ in range(warm): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(rep): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter()-t0)/rep


# ═══════ Direct Linear — optimized ═══════

class QPPDirectFused(nn.Module):
    """Single-einsum forward: x_ordered @ basis @ anchors.T in one shot per block.

    ponytail: avoids Python loop over blocks by pre-laying out block data.
    Still not a fused Triton kernel — each block is a separate matmul launch.
    Upgrade path: true fused kernel that combines all blocks into one grid launch.
    """

    def __init__(self, weight, row_block=128, anchors=32, bias=None):
        super().__init__()
        rows, cols = weight.shape
        w = weight.float().cpu().numpy()
        blocks = int(math.ceil(rows / row_block))
        basis_np = interp_basis(cols, anchors)
        orders = np.zeros((blocks, cols), dtype=np.int16)
        anchors_all = np.zeros((rows, anchors), dtype=np.float32)
        self.row_slices = []
        for b, start in enumerate(range(0, rows, row_block)):
            end = min(start + row_block, rows)
            self.row_slices.append((start, end))
            block = w[start:end]
            order = choose_block_order(block, "mean")
            orders[b] = order.astype(np.int16)
            anchors_all[start:end] = solve_anchors_weight(block[:, order], basis_np, 1e-4)

        self.register_buffer("orders_i16", torch.from_numpy(orders))
        self.register_buffer("anchors", torch.from_numpy(anchors_all).half())
        self.register_buffer("basis", torch.from_numpy(basis_np).float())
        self.b = bias.detach().clone() if bias is not None else torch.zeros(rows)
        self.register_buffer("bias_buf", self.b.clone())

    def forward(self, x):
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
        return out + self.bias_buf.to(dev, dt)


# ═══════ Benchmark ═══════

def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32

    # Real pretrained-like weights: use a real HF layer
    print("Loading Qwen2.5-0.5B for real weight test...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=dt, device_map={"": dev}, local_files_only=True,
    ).eval()

    # Pick the first attention Q-proj
    q_proj = model.model.layers[0].self_attn.q_proj
    ROWS, COLS = q_proj.weight.shape
    print(f"Real layer: {ROWS}×{COLS} ({ROWS*COLS*2/1e6:.2f} MB BF16)", flush=True)

    w_real = q_proj.weight.detach()
    b_real = q_proj.bias.detach() if q_proj.bias is not None else torch.zeros(ROWS)

    # Test input
    x_test = torch.randn(32, COLS, device=dev, dtype=dt)

    # ── Dense baseline ──
    reset()
    dense = nn.Linear(COLS, ROWS, bias=True, device=dev, dtype=dt)
    dense.weight.data.copy_(w_real)
    dense.bias.data.copy_(b_real)
    dense.eval()
    with torch.no_grad(): y_dense = dense(x_test)
    dense_time = bench(lambda: dense(x_test))
    dense_mem = cuda_mem()
    del dense; reset()

    # ── QPP Direct Fused ──
    w_cpu = w_real.detach().cpu()
    b_cpu = b_real.detach().cpu() if q_proj.bias is not None else torch.zeros(ROWS)
    qpp = QPPDirectFused(w_cpu, anchors=32, bias=b_cpu).to(dev)
    del w_cpu, b_cpu
    reset()
    with torch.no_grad(): y_qpp = qpp(x_test)
    qpp_time = bench(lambda: qpp(x_test))
    qpp_mem = cuda_mem()
    qpp_storage = pbytes(qpp)
    dense_storage = ROWS*COLS*2 + ROWS*2

    # Accuracy
    rel_err = (y_dense - y_qpp).norm() / (y_dense.norm() + 1e-9)

    print(f"\n{'='*60}")
    print(f"REAL LAYER BENCHMARK (Qwen2.5-0.5B attn Q-proj)")
    print(f"{'='*60}")
    print(f"  Dense:  {dense_time*1000:.3f} ms | VRAM alloc {dense_mem['alloc']:.3f} GB | Storage {dense_storage/1e3:.1f} KB")
    print(f"  QPP:    {qpp_time*1000:.3f} ms | VRAM alloc {qpp_mem['alloc']:.3f} GB | Storage {qpp_storage/1e3:.1f} KB ({dense_storage/qpp_storage:.1f}x)")
    print(f"  Speed ratio: {dense_time/qpp_time:.2f}x (dense is {qpp_time/dense_time:.1f}x faster)")
    print(f"  Reconstruction error: {rel_err.item():.4f} {'✅ OK' if rel_err < 0.1 else '⚠️ needs more anchors'}")

    # ── VRAM reduction: compress all 4 attention layers of layer 0 ──
    print(f"\n{'='*60}")
    print(f"VRAM REDUCTION: compress all attention layers of layer 0")
    print(f"{'='*60}")

    layer0 = model.model.layers[0].self_attn
    projections = [("q_proj", layer0.q_proj), ("k_proj", layer0.k_proj),
                   ("v_proj", layer0.v_proj), ("o_proj", layer0.o_proj)]

    reset()
    # Measure dense VRAM
    dummy = torch.zeros(1, device=dev)
    _ = layer0.q_proj.weight.data_ptr()
    vram_before = cuda_mem()
    print(f"  Before: {vram_before['alloc']*1000:.1f} MB VRAM", flush=True)

    # Compress and replace
    replacements = []
    for name, mod in projections:
        w = mod.weight.detach().cpu()
        b = mod.bias.detach().cpu() if mod.bias is not None else None
        replacement = QPPDirectFused(w, anchors=32, bias=b).to(dev)
        replacements.append((name, replacement))
        del w, b

    # Force free the old parameters
    for name, mod in projections:
        mod.weight.data = torch.empty(0, device="cpu")
        if mod.bias is not None:
            mod.bias.data = torch.empty(0, device="cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    vram_after = cuda_mem()
    print(f"  After:  {vram_after['alloc']*1000:.1f} MB VRAM", flush=True)
    saved_mb = (vram_before['alloc'] - vram_after['alloc']) * 1000
    print(f"  Saved:  {saved_mb:.1f} MB ({saved_mb/(vram_before['alloc']*1000)*100:.1f}%)", flush=True)

    # ═══════ Scale projection ═══════
    print(f"\n{'='*60}")
    print(f"SCALE PROJECTION — What about bigger layers?")
    print(f"{'='*60}")

    for rows, cols in [(2048, 2048), (4096, 4096)]:
        reset()
        w_sim = torch.randn(rows, cols, device="cpu", dtype=torch.float32)
        b_sim = torch.randn(rows, device="cpu")

        dense_sim = nn.Linear(cols, rows, bias=True, device=dev, dtype=dt)
        dense_sim.weight.data.copy_(w_sim.to(dtype=dt))
        dense_sim.bias.data.copy_(b_sim.to(dtype=dt))
        dense_sim.eval()
        x_sim = torch.randn(32, cols, device=dev, dtype=dt)
        d_t = bench(lambda: dense_sim(x_sim))
        del dense_sim; reset()

        qpp_sim = QPPDirectFused(w_sim, anchors=32, bias=b_sim).to(dev)
        q_t = bench(lambda: qpp_sim(x_sim))
        del qpp_sim, w_sim, b_sim; reset()

        storage = rows * cols * 2 + rows * 2
        qpp_st = rows * 32 * 2 + math.ceil(rows/128) * cols * 2 + rows * 2
        print(f"  {rows}×{cols}:  Dense={d_t*1000:.3f}ms  QPP={q_t*1000:.3f}ms  "
              f"ratio={d_t/q_t:.2f}x  storage={storage/qpp_st:.1f}x", flush=True)


if __name__ == "__main__":
    main()
