"""Runtime compressed modules: QPPCompressedLinear, Int8CompressedLinear.

Drop-in replacements for nn.Linear that store compressed weights instead
of dense parameters. Forward reconstructs weights on-the-fly.

No optimized kernel yet — real speedup requires a fused Triton/CUDA kernel.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qpp.compression import choose_block_order, interp_basis, solve_anchors_activation, solve_anchors_weight


def set_nested_attr(root: nn.Module, name: str, value: nn.Module) -> None:
    """Set a module attribute by dotted path, e.g. 'model.layers.5.mlp.gate_proj'."""
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


class QPPCompressedLinear(nn.Module):
    """nn.Linear replacement storing QPP-compressed weights.

    Stores anchors, shared column orders, optional outliers, residual, and basis.
    Supports two forward modes:
    - "reconstruct": build dense weight then F.linear (simple, robust)
    - "direct": F.linear via ordered columns + basis (no dense weight materialization)

    Codebook-quantized anchors supported via anchor_cb + anchor_codes buffers.
    """

    def __init__(
        self,
        anchors: torch.Tensor,
        orders_i16: torch.Tensor,
        row_slices: list[tuple[int, int]],
        original_shape: tuple[int, int],
        bias: torch.Tensor | None = None,
        outlier_idx_i16: torch.Tensor | None = None,
        outlier_val: torch.Tensor | None = None,
        residual_a: torch.Tensor | None = None,
        residual_b: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        orders_i32: torch.Tensor | None = None,
        direct_vectorize_max_tokens: int = 16,
        forward_mode: Literal["reconstruct", "direct"] = "reconstruct",
        anchor_cb: torch.Tensor | None = None,
        anchor_codes: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        rows, cols = original_shape
        self.out_features = rows
        self.in_features = cols
        self.row_slices = row_slices
        self.forward_mode = forward_mode
        self.direct_vectorize_max_tokens = direct_vectorize_max_tokens
        self.uniform_row_block = (
            len(row_slices) > 0
            and all(
                (end - start) == (row_slices[0][1] - row_slices[0][0])
                for start, end in row_slices
            )
        )

        self.register_buffer("anchors", anchors.contiguous())
        self.register_buffer("orders_i16", orders_i16.contiguous())
        self.orders_i32 = orders_i32.contiguous() if orders_i32 is not None else None
        self.basis = basis.contiguous() if basis is not None else None
        self.bias = bias.contiguous() if bias is not None else None

        if outlier_idx_i16 is not None and outlier_val is not None:
            self.register_buffer("outlier_idx_i16", outlier_idx_i16.contiguous())
            self.register_buffer("outlier_val", outlier_val.contiguous())
        else:
            self.outlier_idx_i16 = None
            self.outlier_val = None

        if residual_a is not None and residual_b is not None:
            self.register_buffer("residual_a", residual_a.contiguous())
            self.register_buffer("residual_b", residual_b.contiguous())
        else:
            self.residual_a = None
            self.residual_b = None

        if anchor_cb is not None and anchor_codes is not None:
            self.register_buffer("anchor_cb", anchor_cb.contiguous())
            self.register_buffer("anchor_codes", anchor_codes.contiguous())
        else:
            self.anchor_cb = None
            self.anchor_codes = None

    def _get_anchors(self) -> torch.Tensor:
        """Decode anchors: either raw or from codebook."""
        if self.anchor_cb is not None and self.anchor_codes is not None:
            return self.anchor_cb.gather(1, self.anchor_codes.long())
        return self.anchors

    def reconstruct_weight(self) -> torch.Tensor:
        rows, cols = self.out_features, self.in_features
        weight = torch.empty(rows, cols, device=self.anchors.device, dtype=self.anchors.dtype)
        anchors_decoded = self._get_anchors()
        for block_id, (start, end) in enumerate(self.row_slices):
            order = self.orders_i16[block_id].to(torch.long)
            rec_sorted = F.interpolate(
                anchors_decoded[start:end].float().unsqueeze(1),
                size=cols,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
            rec_block = torch.empty(end - start, cols, device=weight.device, dtype=torch.float32)
            rec_block[:, order] = rec_sorted
            if self.outlier_idx_i16 is not None:
                idx = self.outlier_idx_i16[start:end].to(torch.long)
                vals = self.outlier_val[start:end].float()
                if idx.numel() > 0:
                    rr = torch.arange(end - start, device=weight.device)[:, None]
                    rec_block[rr, idx] += vals
            weight[start:end] = rec_block.to(weight.dtype)
        if self.residual_a is not None and self.residual_b is not None:
            weight = weight + (self.residual_a.float() @ self.residual_b.float()).to(weight.dtype)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.forward_mode == "direct" and self.outlier_idx_i16 is None:
            return self._forward_direct(x)
        weight = self.reconstruct_weight().to(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def _forward_direct(self, x: torch.Tensor) -> torch.Tensor:
        if self.basis is None:
            raise RuntimeError("direct QPP forward requires a precomputed basis buffer")
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        if (
            self.orders_i32 is not None
            and self.uniform_row_block
            and 0 < x_flat.shape[0] <= self.direct_vectorize_max_tokens
        ):
            return self._forward_direct_vectorized_small(x_flat, original_shape)
        out = torch.empty(x_flat.shape[0], self.out_features, device=x.device, dtype=x.dtype)
        basis = self.basis.to(dtype=x.dtype)
        for block_id, (start, end) in enumerate(self.row_slices):
            order = (
                self.orders_i32[block_id]
                if self.orders_i32 is not None
                else self.orders_i16[block_id].to(torch.int32)
            )
            x_ordered = x_flat.index_select(1, order)
            z = x_ordered @ basis
            out[:, start:end] = z @ self.anchors[start:end].to(dtype=x.dtype).T
        if self.residual_a is not None and self.residual_b is not None:
            residual_mid = x_flat @ self.residual_b.to(dtype=x.dtype).T
            out = out + residual_mid @ self.residual_a.to(dtype=x.dtype).T
        if self.bias is not None:
            out = out + self.bias.to(dtype=x.dtype)
        return out.reshape(*original_shape, self.out_features)

    def _forward_direct_vectorized_small(self, x_flat: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        block_rows = self.row_slices[0][1] - self.row_slices[0][0]
        blocks = len(self.row_slices)
        basis = self.basis.to(dtype=x_flat.dtype)
        x_ordered = x_flat[:, self.orders_i32]
        z = torch.einsum("nbc,ck->nbk", x_ordered, basis)
        anchors_3d = self.anchors.reshape(blocks, block_rows, -1).to(dtype=x_flat.dtype)
        out = torch.einsum("nbk,brk->nbr", z, anchors_3d).reshape(x_flat.shape[0], self.out_features)
        if self.residual_a is not None and self.residual_b is not None:
            residual_mid = x_flat @ self.residual_b.to(dtype=x_flat.dtype).T
            out = out + residual_mid @ self.residual_a.to(dtype=x_flat.dtype).T
        if self.bias is not None:
            out = out + self.bias.to(dtype=x_flat.dtype)
        return out.reshape(*original_shape, self.out_features)

    def runtime_buffer_bytes(self) -> int:
        total = 0
        if self.anchor_cb is not None and self.anchor_codes is not None:
            total += self.anchor_cb.numel() * self.anchor_cb.element_size()
            total += self.anchor_codes.numel() * self.anchor_codes.element_size()
        else:
            total += self.anchors.numel() * self.anchors.element_size()
        total += self.orders_i16.numel() * self.orders_i16.element_size()
        if self.orders_i32 is not None:
            total += self.orders_i32.numel() * self.orders_i32.element_size()
        if self.basis is not None:
            total += self.basis.numel() * self.basis.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        if self.outlier_idx_i16 is not None:
            total += self.outlier_idx_i16.numel() * self.outlier_idx_i16.element_size()
            total += self.outlier_val.numel() * self.outlier_val.element_size()
        if self.residual_a is not None and self.residual_b is not None:
            total += self.residual_a.numel() * self.residual_a.element_size()
            total += self.residual_b.numel() * self.residual_b.element_size()
        return total


class Int8CompressedLinear(nn.Module):
    """INT8 per-channel quantized nn.Linear.

    Stores int8 qweight + fp16 scale + optional fp16 bias.
    Dequantizes on-the-fly in forward, matching input dtype.
    """

    def __init__(self, w: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.in_features = w.shape[1]
        self.out_features = w.shape[0]
        device_orig = w.device
        # Move to CPU first to avoid GPU OOM during quantization
        w_cpu = w.detach().cpu().float()
        max_abs = w_cpu.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
        scale = max_abs / 127.0
        q = (w_cpu / scale).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q.contiguous().to(device_orig))
        self.register_buffer("scale", scale.half().contiguous().to(device_orig))
        b_cpu = bias.detach().cpu() if bias is not None else None
        if b_cpu is not None:
            self.register_buffer("bias", b_cpu.half().to(device_orig))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_deq = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_deq, b)


def build_compressed_linear(
    module: nn.Linear,
    row_block: int,
    anchors_k: int,
    outlier_topk: int,
    residual_rank: int,
    train_residual_steps: int,
    train_residual_lr: float,
    train_residual_batch: int,
    ridge: float,
    order_mode: str,
    calib_x: torch.Tensor | None,
    forward_mode: str,
) -> tuple[QPPCompressedLinear, dict[str, float | int | None]]:
    """Build a QPPCompressedLinear from an existing nn.Linear module.

    This is the main entry point for compressing a single layer. It partitions
    rows into blocks, fits anchors, optionally trains a low-rank residual,
    and returns the compressed module + stats.

    Returns:
        (compressed_module, stats_dict)
    """
    w_t = module.weight.detach()
    device = w_t.device
    dtype = w_t.dtype
    w = w_t.float().cpu().numpy()
    rows, cols = w.shape
    basis_np = interp_basis(cols, anchors_k)
    blocks = int(math.ceil(rows / row_block))
    orders = np.empty((blocks, cols), dtype=np.int16)
    anchors_np = np.empty((rows, anchors_k), dtype=np.float32)
    recon = np.empty_like(w)
    row_slices: list[tuple[int, int]] = []
    act_err_num = 0.0
    act_err_den = 0.0
    out_idx = np.zeros((rows, outlier_topk), dtype=np.int16) if outlier_topk > 0 else None
    out_val = np.zeros((rows, outlier_topk), dtype=np.float32) if outlier_topk > 0 else None
    x_np = None if calib_x is None or calib_x.numel() == 0 else calib_x.float().cpu().numpy()

    for block_id, start in enumerate(range(0, rows, row_block)):
        end = min(start + row_block, rows)
        row_slices.append((start, end))
        block = w[start:end]
        order = choose_block_order(block, order_mode)
        if cols > np.iinfo(np.int16).max:
            raise ValueError("int16 orders only support <= 32767 columns")
        orders[block_id] = order.astype(np.int16)
        sorted_block = block[:, order]
        if x_np is not None:
            y = x_np @ block.T
            theta = solve_anchors_activation(x_np[:, order], y, basis_np, ridge)
        else:
            theta = solve_anchors_weight(sorted_block, basis_np, ridge)
        rec_sorted = theta @ basis_np.T
        rec_block = np.empty_like(rec_sorted)
        np.put_along_axis(rec_block, order[None, :], rec_sorted, axis=1)

        if outlier_topk > 0:
            residual = block - rec_block
            idx = np.argpartition(np.abs(residual), -outlier_topk, axis=1)[:, -outlier_topk:]
            rr = np.arange(end - start)[:, None]
            vals = residual[rr, idx]
            rec_block[rr, idx] += vals
            out_idx[start:end] = idx.astype(np.int16)
            out_val[start:end] = vals.astype(np.float32)

        anchors_np[start:end] = theta
        recon[start:end] = rec_block
        if x_np is not None:
            y_ref = x_np @ block.T
            y_hat = x_np @ rec_block.T
            act_err_num += float(np.sum((y_ref - y_hat) ** 2))
            act_err_den += float(np.sum(y_ref**2))

    # Low-rank residual (optional)
    residual_a_t = None
    residual_b_t = None
    residual_train_loss_initial = None
    residual_train_loss_final = None
    if residual_rank > 0:
        rank = min(residual_rank, rows, cols)
        residual = torch.from_numpy(w - recon).to(device=device, dtype=torch.float32)
        q = min(rank + 8, min(rows, cols))
        u, s, v = torch.pca_lowrank(residual, q=q, center=False, niter=4)
        residual_a_f = (u[:, :rank] * s[:rank]).contiguous()
        residual_b_f = v[:, :rank].T.contiguous()
        if train_residual_steps > 0 and x_np is not None:
            x_train = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
            target_delta = x_train @ residual.T
            a_param = nn.Parameter(residual_a_f.clone())
            b_param = nn.Parameter(residual_b_f.clone())
            opt = torch.optim.AdamW([a_param, b_param], lr=train_residual_lr, weight_decay=0.0)
            batch = min(max(1, train_residual_batch), x_train.shape[0])
            with torch.no_grad():
                pred0 = (x_train @ b_param.T) @ a_param.T
                residual_train_loss_initial = float(F.mse_loss(pred0, target_delta).detach().cpu())
            for step in range(train_residual_steps):
                st = (step * batch) % x_train.shape[0]
                ed = min(st + batch, x_train.shape[0])
                xb, yb = x_train[st:ed], target_delta[st:ed]
                pred = (xb @ b_param.T) @ a_param.T
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred1 = (x_train @ b_param.T) @ a_param.T
                residual_train_loss_final = float(F.mse_loss(pred1, target_delta).detach().cpu())
            residual_a_f = a_param.detach().contiguous()
            residual_b_f = b_param.detach().contiguous()
            del x_train, target_delta, a_param, b_param, opt
        residual_a_t = residual_a_f.to(dtype=dtype)
        residual_b_t = residual_b_f.to(dtype=dtype)
        recon += (residual_a_t.float() @ residual_b_t.float()).detach().cpu().numpy()
        del residual, u, s, v, residual_a_f, residual_b_f

    diff = recon - w
    if x_np is not None:
        act_diff = x_np @ diff.T
        act_ref = x_np @ w.T
        act_err_num = float(np.sum(act_diff**2))
        act_err_den = float(np.sum(act_ref**2))

    dense_bf16_bytes = rows * cols * 2
    qpp_param_bytes = rows * anchors_k * 2
    shared_order_bytes = blocks * cols * 2
    outlier_bytes = rows * outlier_topk * 4
    residual_bytes = (
        0
        if residual_rank <= 0
        else (rows * min(residual_rank, rows, cols) + min(residual_rank, rows, cols) * cols) * 2
    )
    theoretical_qpp_bytes = qpp_param_bytes + shared_order_bytes + outlier_bytes + residual_bytes
    bias = None if module.bias is None else module.bias.detach().to(device=device, dtype=dtype)

    compressed = QPPCompressedLinear(
        anchors=torch.from_numpy(anchors_np).to(device=device, dtype=dtype),
        orders_i16=torch.from_numpy(orders).to(device=device),
        row_slices=row_slices,
        original_shape=(rows, cols),
        bias=bias,
        outlier_idx_i16=None if out_idx is None else torch.from_numpy(out_idx).to(device=device),
        outlier_val=None if out_val is None else torch.from_numpy(out_val).to(device=device, dtype=dtype),
        residual_a=residual_a_t,
        residual_b=residual_b_t,
        basis=None if forward_mode != "direct" else torch.from_numpy(basis_np).to(device=device, dtype=dtype),
        orders_i32=None if forward_mode != "direct" else torch.from_numpy(orders.astype(np.int32)).to(device=device),
        forward_mode=forward_mode,
    )
    runtime_bytes = compressed.runtime_buffer_bytes()
    stats: dict[str, float | int | None] = {
        "rows": rows,
        "cols": cols,
        "blocks": blocks,
        "dense_bf16_bytes": dense_bf16_bytes,
        "theoretical_qpp_bytes": theoretical_qpp_bytes,
        "residual_bytes": residual_bytes,
        "runtime_buffer_bytes": runtime_bytes,
        "theoretical_compression": dense_bf16_bytes / max(1, theoretical_qpp_bytes),
        "runtime_buffer_compression": dense_bf16_bytes / max(1, runtime_bytes),
        "weight_rel_rmse": float(np.sqrt(np.mean(diff**2)) / (np.sqrt(np.mean(w**2)) + 1e-12)),
        "activation_rel_rmse": None if x_np is None or act_err_den <= 0 else float(math.sqrt(act_err_num / act_err_den)),
        "residual_train_loss_initial": residual_train_loss_initial,
        "residual_train_loss_final": residual_train_loss_final,
    }
    return compressed, stats


def persistent_model_bytes(model: nn.Module) -> int:
    """Total bytes of unique parameter+buffer tensors in the model."""
    seen: set[int] = set()
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        ptr = tensor.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += tensor.numel() * tensor.element_size()
    return total
