"""QPP: Quantile Piecewise Perceptron — parametric compression for LLM attention layers.

QPP reduces the *number of parameters* (not their bit precision) by exploiting
the smooth, monotonic quantile curve that emerges when perceptron weights are
sorted. This is orthogonal to traditional quantization (GGUF, AWQ, GPTQ).

Core modules:
- compression: shared-order QPP fitting, interp_basis, compress/decompress
- runtime: QPPCompressedLinear, Int8CompressedLinear drop-in nn.Modules
- codebook: Lloyd-Max codebook quantization for QPP anchors
- hybrid: end-to-end hybrid pipeline (QPP attn + INT8 MLP/embed)
- benchmark: PPL evaluation, generation benchmarks, model loading utilities
"""

from qpp.compression import (
    choose_block_order,
    compress_weight_shared_order,
    interp_basis,
    row_block_for,
    solve_anchors_activation,
    solve_anchors_weight,
    target_linears,
)
from qpp.runtime import (
    Int8CompressedLinear,
    QPPCompressedLinear,
    build_compressed_linear,
    set_nested_attr,
)
from qpp.codebook import lloyd_max_1d

__version__ = "0.1.0"
__all__ = [
    # compression
    "interp_basis",
    "choose_block_order",
    "solve_anchors_weight",
    "solve_anchors_activation",
    "compress_weight_shared_order",
    "row_block_for",
    "target_linears",
    # runtime
    "QPPCompressedLinear",
    "Int8CompressedLinear",
    "build_compressed_linear",
    "set_nested_attr",
    # codebook
    "lloyd_max_1d",
]
