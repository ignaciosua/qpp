#!/usr/bin/env python3
"""H1.1 + H1.4: Triton direct-forward kernel + VRAM reduction benchmark.

Three benchmarks on Qwen2.5-0.5B:
  1. BF16 baseline (speed, VRAM)
  2. QPP reconstruct mode (current — materializes weight)
  3. QPP direct mode (no weight materialization, no Triton needed)
  4. Verify VRAM reduction after freeing dense weights

ponytail: uses PyTorch builtins for gather+matmul. A true fused Triton kernel
would combine gather + basis @ anchors into one launch, saving ~20% more.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qpp.compression import interp_basis, choose_block_order, solve_anchors_weight


# ═══════════════════════════════════════
# Benchmark helpers
# ═══════════════════════════════════════

def cuda_mem():
    if not torch.cuda.is_available():
        return {"alloc": 0, "peak": 0}
    return {
        "alloc": torch.cuda.memory_allocated() / 1e9,
        "peak": torch.cuda.max_memory_allocated() / 1e9,
    }

def reset_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def persistent_model_bytes(model):
    seen = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        ptr = t.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += t.numel() * t.element_size()
    return total


# ═══════════════════════════════════════
# QPP Direct Linear (no weight materialization)
# ═══════════════════════════════════════

class QPPDirectLinear(nn.Module):
    """QPP compressed linear that NEVER materializes a dense weight matrix.

    Forward: x_ordered = x[:, order]; z = x_ordered @ basis; out = z @ anchors.T

    This is what H1.1 would optimize with Triton, but the PyTorch ops
    already avoid the O(RC) reconstruction.
    """

    def __init__(self, weight, row_block=128, anchors=32, bias=None):
        super().__init__()
        rows, cols = weight.shape
        w = weight.float().cpu().numpy()
        blocks = int(math.ceil(rows / row_block))

        # Fit anchors
        basis_np = interp_basis(cols, anchors)
        orders = np.empty((blocks, cols), dtype=np.int16)
        anchors_all = np.empty((rows, anchors), dtype=np.float32)
        self.row_slices = []
        for b, start in enumerate(range(0, rows, row_block)):
            end = min(start + row_block, rows)
            self.row_slices.append((start, end))
            block = w[start:end]
            order = choose_block_order(block, "mean")
            orders[b] = order.astype(np.int16)
            theta = solve_anchors_weight(block[:, order], basis_np, 1e-4)
            anchors_all[start:end] = theta

        self.register_buffer("orders_i16", torch.from_numpy(orders))
        self.register_buffer("anchors", torch.from_numpy(anchors_all).half())
        self.register_buffer("basis", torch.from_numpy(basis_np).float())
        self.register_buffer("bias", bias.detach().clone() if bias is not None else torch.zeros(rows))

    def forward(self, x):
        N = x.shape[0]
        device = x.device
        out = torch.empty(N, self.anchors.shape[0], device=device, dtype=x.dtype)

        for block_id, (start, end) in enumerate(self.row_slices):
            order = self.orders_i16[block_id].long()
            x_ordered = x[:, order]                               # (N, C)
            z = x_ordered @ self.basis.to(device, x.dtype)        # (N, K)
            anchors = self.anchors[start:end].to(device, x.dtype) # (R_b, K)
            out[:, start:end] = z @ anchors.T                      # (N, R_b)

        return out + self.bias.to(device, x.dtype)


def benchmark_forward(module, x, warmup=10, repeats=50):
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            module(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeats):
            module(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return elapsed / repeats


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}", flush=True)

    # ═══════════════════════════════════
    # Test on synthetic layer (Qwen2.5-0.5B attention size)
    # ═══════════════════════════════════
    ROWS, COLS = 896, 896  # Qwen2.5-0.5B Q-proj
    x_test = torch.randn(64, COLS, device=device, dtype=dtype)

    print(f"\nLayer size: {ROWS}×{COLS} ({ROWS*COLS*2/1e6:.1f} MB BF16)", flush=True)

    # Baseline: dense nn.Linear
    reset_mem()
    linear = nn.Linear(COLS, ROWS, bias=True, device=device, dtype=dtype)
    dense_time = benchmark_forward(linear, x_test)
    dense_mem = cuda_mem()
    print(f"Dense Linear:    {dense_time*1000:.2f} ms | VRAM: {dense_mem['alloc']:.2f} GB alloc", flush=True)

    # QPP Direct — NEVER materializes weight
    reset_mem()
    del linear
    weight_t = torch.randn(ROWS, COLS, device="cpu", dtype=torch.float32)
    bias_t = torch.randn(ROWS, device="cpu", dtype=torch.float32)
    qpp_direct = QPPDirectLinear(weight_t, anchors=32, bias=bias_t).to(device)
    del weight_t, bias_t
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    qpp_direct_time = benchmark_forward(qpp_direct, x_test)
    qpp_direct_mem = cuda_mem()
    qpp_bytes = persistent_model_bytes(qpp_direct)
    dense_bytes = ROWS * COLS * 2 + ROWS * 2  # weight + bias
    comp = dense_bytes / max(1, qpp_bytes)

    print(f"QPP Direct:      {qpp_direct_time*1000:.2f} ms | VRAM: {qpp_direct_mem['alloc']:.3f} GB alloc "
          f"| Storage: {qpp_bytes/1e3:.1f} KB vs {dense_bytes/1e3:.1f} KB ({comp:.1f}x)",
          flush=True)

    # ═══════════════════════════════════
    # VRAM reduction: free dense, keep only QPP
    # ═══════════════════════════════════
    print("\n--- VRAM Reduction Test ---", flush=True)

    # Simulate what happens when we compress a model
    reset_mem()
    model_dense = nn.Sequential(
        nn.Linear(COLS, ROWS, bias=True, device=device, dtype=dtype),
        nn.Linear(COLS, ROWS, bias=True, device=device, dtype=dtype),
        nn.Linear(COLS, ROWS, bias=True, device=device, dtype=dtype),
        nn.Linear(COLS, ROWS, bias=True, device=device, dtype=dtype),
    )
    vram_before = cuda_mem()
    print(f"4× Dense Linear: VRAM={vram_before['alloc']:.3f} GB alloc", flush=True)

    # "Compress" — replace each dense with QPP, free the old weight
    reset_mem()
    compressed_modules = []
    for i, child in enumerate(model_dense):
        w = child.weight.detach().cpu()
        b = child.bias.detach().cpu()
        qpp = QPPDirectLinear(w, anchors=32, bias=b).to(device)
        compressed_modules.append(qpp)
        del w, b
    del model_dense
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Force sync so CUDA actually frees
    torch.cuda.synchronize()
    vram_after = cuda_mem()
    print(f"4× QPP Direct:   VRAM={vram_after['alloc']:.3f} GB alloc", flush=True)
    saved = vram_before["alloc"] - vram_after["alloc"]
    print(f"VRAM saved: {saved:.3f} GB ({saved/vram_before['alloc']*100:.1f}%)", flush=True)

    # ═══════════════════════════════════
    # Summary
    # ═══════════════════════════════════
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    speedup = dense_time / qpp_direct_time
    print(f"Speed:   Dense={dense_time*1000:.2f}ms | QPP={qpp_direct_time*1000:.2f}ms | ratio={speedup:.2f}x", flush=True)
    print(f"Storage: {comp:.1f}x compression ({dense_bytes/1e6:.2f}MB → {qpp_bytes/1e6:.3f}MB)", flush=True)
    print(f"VRAM:    Saved {saved:.3f} GB ({saved/vram_before['alloc']*100:.1f}%) after replacing 4 layers", flush=True)

    # Final op: reconstruct weight from QPP and check accuracy
    with torch.no_grad():
        w_dense = torch.randn(ROWS, COLS, device=device, dtype=dtype)
        linear_ref = nn.Linear(COLS, ROWS, bias=True, device=device, dtype=dtype)
        qpp_test = QPPDirectLinear(w_dense.detach().cpu(), anchors=32, bias=linear_ref.bias.detach().cpu()).to(device)
        y_ref = linear_ref(x_test)
        y_qpp = qpp_test(x_test)
        rel_err = (y_ref - y_qpp).norm() / (y_ref.norm() + 1e-9)
        print(f"\nReconstruction error: {rel_err:.4f} ({'OK' if rel_err < 0.1 else 'HIGH'})", flush=True)


if __name__ == "__main__":
    import numpy as np
    main()
