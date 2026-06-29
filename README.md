# QPP: Quantile Piecewise Perceptron

**Parametric compression for LLM attention layers — reduce the *number of parameters*, not just their bit precision.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![tests](https://github.com/ignaciosua/qpp/actions/workflows/tests.yml/badge.svg)](https://github.com/ignaciosua/qpp/actions)

QPP exploits a previously untapped structure in LLM weight matrices: when you sort the weights of a perceptron (row of a `Linear` layer) from smallest to largest, they form a **smooth, monotonic quantile curve** with three distinct regions (negative tail, center near zero, positive tail). This curve is regular enough to be approximated by a **linear interpolation basis with K anchors** (K ≪ columns). Instead of storing R×C floats, QPP stores R×K anchors + block-shared ordering.

Unlike traditional quantization (GGUF, AWQ, GPTQ) which reduces bit precision (max ~4×), QPP is **orthogonal and complementary** — it reduces the parameter count itself, achieving **21× parametric compression** on attention layers alone.

## Key Results (Qwen3-4B)

| Variant | PPL | ΔPPL | Size | Savings | tok/s |
|---------|-----|------|------|---------|-------|
| BF16 baseline | 3.919 | — | 8,045 MB | 0% | 30.0 |
| **QPP attention-only** | **3.923** | **+0.005** | **7,381 MB** | **8.25%** | **26.5** |
| QPP+CB+INT8 hybrid | 4.421 | +0.502 | 4,738 MB | 41.1% | 26.0 |
| GGUF Q4_K_M | 3.933 | +0.015 | 2,497 MB | 69.0% | 15.4 |

**QPP vs GGUF on the same 13 attention modules (351 MB BF16):**
- GGUF Q4_K_M: 109 MB (3.2×)
- QPP+CB_2b: 17 MB (21×) — **7× more effective**

**QPP + GGUF are orthogonal**: QPP(21×) × GGUF(3.2×) = 67× theoretical on attention.

## Quickstart

```bash
pip install -e .
# or with HuggingFace support:
pip install -e ".[hf]"
```

### Compress a single weight matrix (no model loading needed)

```python
import torch
from qpp import compress_weight_shared_order

weight = torch.randn(1024, 2560, dtype=torch.float32)  # a Linear layer
reconstructed, stats = compress_weight_shared_order(
    weight, row_block=128, anchors=32, outlier_topk=0,
    ridge=1e-4, order_mode="mean", calib_x=None,
)
print(f"Compression: {stats['compression_vs_bf16']:.1f}x")
print(f"Weight RMSE: {stats['weight_rel_rmse']:.4f}")
```

### Full pipeline: compress a HuggingFace model

```bash
qpp-compress --model Qwen/Qwen2.5-0.5B-Instruct --outdir outputs/my_run --save-artifact
```

Or from Python:

```python
from qpp.hybrid import HybridPipeline

pipe = HybridPipeline(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    anchors=32,
    total_delta_gate=0.5,
)
summary = pipe.run()
print(f"Saved {summary['savings_pct']:.1f}% with ΔPPL={summary['delta_ppl']:+.4f}")
```

### Load a saved QPP artifact

```python
import torch
from qpp.runtime import QPPCompressedLinear, set_nested_attr

artifact = torch.load("outputs/my_run/qpp_compressed_artifact.pt", weights_only=False)
for name, spec in artifact["modules"].items():
    # Create QPPCompressedLinear from artifact spec
    compressed = QPPCompressedLinear(
        anchors=spec["tensors"]["anchors"],
        orders_i16=spec["tensors"]["orders_i16"],
        row_slices=[tuple(s) for s in spec["row_slices"]],
        original_shape=(spec["out_features"], spec["in_features"]),
        bias=spec["tensors"].get("bias"),
    )
    set_nested_attr(model, name, compressed)
```

## How It Works

```
Sorted weights of a perceptron:     QPP approximation with K=6 anchors:
                                   
  +▄                                +▄
  │ ▀▄                              │ ▀▄
  │   ▀▄          ← anchors →       │ ●──●──●──●──●──●
  │     ▀▄                          │   ╱  ╱  ╱  ╱  ╱
  │       ▀▄                        │  ╱  ╱  ╱  ╱  ╱
  │         ▀▀▀▀▀▀▀▀▀▀            │ ╱  ╱  ╱  ╱  ╱
──┼────────────────────→          ──┼────────────────────→
  sorted column index               sorted column index
```

1. **Block partition**: Group rows into blocks (B=128)
2. **Shared ordering**: Compute one column permutation per block (mean-based)
3. **Anchor fitting**: Solve least-squares for K anchor values per row against interpolation basis
4. **Reconstruct**: Ŵ_r = basis · Θ_r[π⁻¹]
5. **Greedy gating**: Compress → measure ΔPPL → accept if ≤ threshold, else rollback

## Installation

```bash
# Core (only numpy + torch)
pip install -e .

# With HuggingFace support
pip install -e ".[hf]"

# Dev (tests, plotting, pandas)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## Project Structure

```
qpp/
├── src/qpp/
│   ├── __init__.py        # Public API
│   ├── compression.py     # Core QPP algorithm (numpy, no HF dependency)
│   ├── runtime.py         # QPPCompressedLinear, Int8CompressedLinear nn.Modules
│   ├── codebook.py        # Lloyd-Max 1D quantizer for anchor compression
│   ├── benchmark.py       # PPL eval, generation benchmarks, corpus
│   └── hybrid.py          # Full pipeline: QPP attn + INT8 MLP/embed
├── tests/
│   ├── test_compression.py  # Unit tests for core algorithm (CPU-only)
│   └── test_runtime.py      # Unit tests for compressed modules (CPU-only)
├── paper/
│   └── qpp_paper.tex       # LaTeX paper
├── docs/                    # Full documentation
└── scripts/                 # Analysis scripts
```

## Current Status & Known Limitations

**Production-ready:**
- QPP compression/decompression algorithm ✓
- QPPCompressedLinear drop-in nn.Module ✓
- Greedy PPL-gated hybrid pipeline ✓
- INT8 per-channel for MLP/embeddings (lossless, 2×) ✓
- Artifact save/reload ✓

**Needs work (great community contributions!):**
- **Fused Triton/CUDA kernel**: `forward()` reconstructs dense weights temporarily — real speedup needs a fused kernel
- **VRAM reduction end-to-end**: persistent bytes drop but CUDA allocated doesn't (due to reconstruction)
- **QPP for MLP**: MLP weights are zero-centered Gaussian (no 3-region quantile curve) — needs different approach
- **More models**: currently tested on Qwen2.5, Qwen3, Phi-2, Gemma
- **Quantization-aware training (QAT)**: train with QPP in the loop
- **Anchor quantization**: currently FP16 anchors — explore lower precision

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

## Citation

```bibtex
@misc{suarez2026qpp,
  title={QPP: Parametric Compression via Quantile Curves for Large Language Models},
  author={Suárez Hernández, Ignacio Fernando},
  year={2026},
  month={6},
  howpublished={\url{https://github.com/ignaciosua/qpp}},
}
```

## License

MIT — see [LICENSE](LICENSE).
