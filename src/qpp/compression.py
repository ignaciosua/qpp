"""Core QPP compression: interp_basis, shared-order fitting, solve_anchors.

This module contains the pure-algorithm core of QPP with no model-dependant code.
Works with raw numpy arrays — no HuggingFace required.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch


def interp_basis(cols: int, anchors: int) -> np.ndarray:
    """Linear interpolation basis from K anchors to `cols` positions.

    Returns (cols, anchors) float32 array where each row sums to 1.
    """
    if anchors < 2:
        raise ValueError("anchors must be >= 2")
    pos = np.linspace(0.0, anchors - 1, cols, dtype=np.float32)
    left = np.floor(pos).astype(np.int64)
    right = np.clip(left + 1, 0, anchors - 1)
    frac = pos - left
    basis = np.zeros((cols, anchors), dtype=np.float32)
    rows = np.arange(cols)
    basis[rows, left] += 1.0 - frac
    basis[rows, right] += frac
    return basis


def choose_block_order(
    w_block: np.ndarray, mode: Literal["mean", "median", "abs_mean"]
) -> np.ndarray:
    """Compute a shared column ordering for a block of rows.

    Args:
        w_block: (block_rows, cols) float32 weights
        mode: aggregation strategy — "mean" (default), "median", "abs_mean"

    Returns:
        order: (cols,) int64 array — permutation that sorts columns
    """
    if mode == "mean":
        score = w_block.mean(axis=0)
    elif mode == "median":
        score = np.median(w_block, axis=0)
    elif mode == "abs_mean":
        score = np.mean(np.abs(w_block), axis=0)
    else:
        raise ValueError(f"unknown order mode: {mode}")
    return np.argsort(score).astype(np.int64)


def solve_anchors_weight(
    w_sorted: np.ndarray, basis: np.ndarray, ridge: float
) -> np.ndarray:
    """Solve for QPP anchors minimizing weight reconstruction error.

    Args:
        w_sorted: (rows, cols) sorted weights
        basis: (cols, anchors) interpolation basis
        ridge: L2 regularization

    Returns:
        theta: (rows, anchors) float32 anchor values
    """
    gram = basis.T @ basis
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = w_sorted @ basis
    return np.linalg.solve(gram, rhs.T).T.astype(np.float32)


def solve_anchors_activation(
    x_ordered: np.ndarray, y: np.ndarray, basis: np.ndarray, ridge: float
) -> np.ndarray:
    """Solve for QPP anchors minimizing activation reconstruction error.

    Args:
        x_ordered: (calib_samples, cols) column-ordered activations
        y: (calib_samples, rows) target outputs
        basis: (cols, anchors) interpolation basis
        ridge: L2 regularization

    Returns:
        theta: (rows, anchors) float32 anchor values
    """
    z = x_ordered @ basis  # (samples, anchors)
    gram = z.T @ z
    gram.flat[:: gram.shape[0] + 1] += ridge
    rhs = z.T @ y
    return np.linalg.solve(gram, rhs).T.astype(np.float32)


def compress_weight_shared_order(
    weight: torch.Tensor,
    row_block: int,
    anchors: int,
    outlier_topk: int,
    ridge: float,
    order_mode: str,
    calib_x: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float | int | None]]:
    """Compress a single weight matrix with shared-order QPP.

    This is the core compression routine. It:
    1. Partitions rows into blocks of size ``row_block``
    2. Computes a shared column ordering per block
    3. Fits K anchor values per row via least-squares
    4. Optionally stores top-k outlier residuals

    Args:
        weight: (out_features, in_features) weight matrix
        row_block: number of rows per order-sharing block
        anchors: number of QPP anchors (K)
        outlier_topk: top-k sparse outliers to preserve per row
        ridge: L2 regularization for anchor fitting
        order_mode: "mean", "median", or "abs_mean"
        calib_x: optional calibration activations for activation-aware fitting

    Returns:
        (reconstructed_weight, stats_dict)
    """
    dtype = weight.dtype
    device = weight.device
    w = weight.detach().float().cpu().numpy()
    rows, cols = w.shape
    basis = interp_basis(cols, anchors)
    recon = np.empty_like(w)
    blocks = int(math.ceil(rows / row_block))
    act_err_num = 0.0
    act_err_den = 0.0

    x_np = None
    if calib_x is not None:
        x_np = calib_x.detach().float().cpu().numpy()

    for start in range(0, rows, row_block):
        end = min(start + row_block, rows)
        block = w[start:end]
        order = choose_block_order(block, order_mode)
        block_sorted = block[:, order]

        if x_np is not None:
            y = x_np @ block.T
            theta = solve_anchors_activation(x_np[:, order], y, basis, ridge)
        else:
            theta = solve_anchors_weight(block_sorted, basis, ridge)

        rec_sorted = theta @ basis.T
        rec_block = np.empty_like(rec_sorted)
        np.put_along_axis(rec_block, order[None, :], rec_sorted, axis=1)

        if outlier_topk > 0:
            residual = block - rec_block
            k = min(outlier_topk, cols)
            idx = np.argpartition(np.abs(residual), -k, axis=1)[:, -k:]
            row_idx = np.arange(end - start)[:, None]
            rec_block[row_idx, idx] += residual[row_idx, idx]

        recon[start:end] = rec_block

        if x_np is not None:
            y_ref = x_np @ block.T
            y_hat = x_np @ rec_block.T
            act_err_num += float(np.sum((y_ref - y_hat) ** 2))
            act_err_den += float(np.sum(y_ref**2))

    diff = recon - w
    dense_bf16_bytes = rows * cols * 2
    qpp_param_bytes = rows * anchors * 2
    shared_order_bytes = blocks * cols * (2 if cols <= 65535 else 4)
    outlier_bytes = rows * outlier_topk * ((2 if cols <= 65535 else 4) + 2)
    total_qpp_bytes = qpp_param_bytes + shared_order_bytes + outlier_bytes

    stats: dict[str, float | int | None] = {
        "rows": rows,
        "cols": cols,
        "blocks": blocks,
        "dense_bf16_bytes": dense_bf16_bytes,
        "qpp_param_bytes": qpp_param_bytes,
        "shared_order_bytes": shared_order_bytes,
        "outlier_bytes": outlier_bytes,
        "total_qpp_bytes": total_qpp_bytes,
        "compression_vs_bf16": dense_bf16_bytes / max(1, total_qpp_bytes),
        "params_only_compression_vs_bf16": dense_bf16_bytes / max(1, qpp_param_bytes),
        "weight_rel_rmse": float(
            np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(w**2)) + 1e-12)
        ),
        "weight_mae": float(np.mean(np.abs(diff))),
        "activation_rel_rmse": (
            None
            if x_np is None or act_err_den <= 0
            else float(math.sqrt(act_err_num / act_err_den))
        ),
    }
    return torch.from_numpy(recon).to(device=device, dtype=dtype), stats


def row_block_for(
    name: str, rows: int, default_block: int, down_block: int
) -> int:
    """Heuristic: use smaller row_block for down-projection layers."""
    if ".down_proj" in name:
        return down_block
    return default_block


def target_linears(
    model: torch.nn.Module, target: str, max_modules: int
) -> list[tuple[str, torch.nn.Linear]]:
    """Find nn.Linear modules by component type.

    Args:
        model: a torch module tree
        target: "attention", "mlp", or "all"
        max_modules: 0 = no limit

    Returns:
        list of (name, nn.Linear) tuples
    """
    out: list[tuple[str, torch.nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) or "lm_head" in name:
            continue
        if target == "attention" and ".self_attn." not in name:
            continue
        if target == "mlp" and ".mlp." not in name:
            continue
        if target == "all" and not (".self_attn." in name or ".mlp." in name):
            continue
        out.append((name, module))
    return out[:max_modules] if max_modules > 0 else out
