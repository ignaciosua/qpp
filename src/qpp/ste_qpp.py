"""STE QPP: Straight-Through Estimator with periodic hard reordering.

Solves the Gumbel-Softmax plateau on real data by alternating:
  1. Train anchors with fixed hard permutation (N steps)
  2. Materialize dense weight → recompute ordering via argsort
  3. Update hard permutation
  4. Repeat

This co-evolution converges reliably: better anchors → better ordering →
better anchors. No Gumbel noise, no soft permutation, no temperature annealing.

ponytail: the simplest thing that could possibly work. Upgrade path:
learned ordering via Hungarian algorithm on the materialized weights,
Sinkhorn-regularized soft permutation as a regularizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpp.compression import interp_basis


class STEQPPLinear(nn.Module):
    """QPP layer where ordering is periodically recomputed from materialized weights.

    Stores: anchors (R, K), hard_order (B, C) as int16 buffer.
    Every `reorder_every` steps, materializes dense weights, recomputes
    the ordering via block-wise mean argsort, and updates the hard_order buffer.

    Between reordering steps, trains anchors normally via gradient descent.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        anchors: int = 32,
        row_block: int = 16,
        bias: bool = True,
        reorder_every: int = 200,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.anchors_k = anchors
        self.row_block = row_block
        self.reorder_every = reorder_every

        rows, cols = out_features, in_features
        blocks = int(math.ceil(rows / row_block))

        # Learnable anchors: (rows, anchors) — the ONLY trained parameters
        self.anchor_values = nn.Parameter(torch.randn(rows, anchors) * 0.02)

        # Hard permutation: (blocks, cols) int16 — periodically recomputed
        # Identity ordering initially: position k = column k
        identity = torch.arange(cols, dtype=torch.int16).unsqueeze(0).expand(blocks, -1).contiguous()
        self.register_buffer("orders_i16", identity)

        # Interpolation basis: (cols, anchors)
        basis_np = interp_basis(cols, anchors)
        self.register_buffer("basis", torch.from_numpy(basis_np).float())

        self.bias_param = nn.Parameter(torch.zeros(rows)) if bias else None

        self.row_slices = [
            (b * row_block, min((b + 1) * row_block, rows)) for b in range(blocks)
        ]

        # Ordering initialized as identity above. recompute_ordering() called at step boundaries.

        # Step counter (not a parameter, not a buffer — plain int)
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    def recompute_ordering(self):
        """Materialize dense weight from current anchors, recompute block-wise ordering."""
        with torch.no_grad():
            weight = self._materialize_dense()
            w_np = weight.float().cpu().numpy()
            for block_id, (start, end) in enumerate(self.row_slices):
                block = w_np[start:end]
                order = block.mean(axis=0).argsort().astype(int)
                self.orders_i16[block_id] = torch.from_numpy(order.astype(
                    int if w_np.shape[1] <= 32767 else object
                )).to(self.orders_i16.device)

    def _materialize_dense(self) -> torch.Tensor:
        """Reconstruct dense weight matrix from anchors + current ordering."""
        rows, cols = self.out_features, self.in_features
        weight = torch.empty(rows, cols, device=self.anchor_values.device, dtype=torch.float32)
        for block_id, (start, end) in enumerate(self.row_slices):
            order = self.orders_i16[block_id].long()
            inv_order = torch.empty_like(order)
            inv_order[order.contiguous()] = torch.arange(cols, device=order.device)
            anchors = self.anchor_values[start:end].float()  # (R_b, K)
            rec_sorted = self.basis.to(anchors.device) @ anchors.T  # (C, R_b)
            weight[start:end] = rec_sorted.T[:, inv_order]
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with current hard permutation. No Gumbel noise."""
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features).to(dtype=torch.float32)
        out = torch.empty(x_flat.shape[0], self.out_features, device=x.device, dtype=torch.float32)

        for block_id, (start, end) in enumerate(self.row_slices):
            order = self.orders_i16[block_id].long()
            x_ordered = x_flat[:, order]  # (N, C) — hard permutation
            z = x_ordered @ self.basis.to(device=x.device)  # (N, K)
            block_anchors = self.anchor_values[start:end].float()  # (R_b, K)
            out[:, start:end] = z @ block_anchors.T

        if self.bias_param is not None:
            out = out + self.bias_param.float()

        return out.reshape(*original_shape, self.out_features).to(dtype=x.dtype)

    def step_and_maybe_reorder(self) -> bool:
        """Increment step counter. Return True if reordering was triggered."""
        self._step += 1
        if self._step % self.reorder_every == 0:
            self.recompute_ordering()
            return True
        return False

    def get_dense_weight(self):
        with torch.no_grad():
            w = self._materialize_dense()
            b = self.bias_param if self.bias_param is not None else None
            return w, b

    def compression_ratio(self) -> float:
        rows, cols = self.out_features, self.in_features
        dense_bytes = rows * cols * 2
        qpp_bytes = rows * self.anchors_k * 2  # anchors FP16
        qpp_bytes += self.orders_i16.numel() * 2  # orders int16
        if self.bias_param is not None:
            qpp_bytes += self.bias_param.numel() * 2
        return dense_bytes / max(1, qpp_bytes)


class STEAttention(nn.Module):
    """Multi-head attention with STE QPP projections."""

    def __init__(self, d_model: int, n_head: int, anchors: int = 32, row_block: int = 16, reorder_every: int = 200):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        kw = {"anchors": anchors, "row_block": row_block, "bias": True, "reorder_every": reorder_every}
        self.q_proj = STEQPPLinear(d_model, d_model, **kw)
        self.k_proj = STEQPPLinear(d_model, d_model, **kw)
        self.v_proj = STEQPPLinear(d_model, d_model, **kw)
        self.out_proj = STEQPPLinear(d_model, d_model, **kw)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.d_k)
        attn = F.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

    def step_all(self) -> list[bool]:
        return [p.step_and_maybe_reorder() for p in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]]
