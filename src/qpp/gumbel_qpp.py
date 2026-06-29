"""Gumbel-Softmax QPP: learn the column ordering during training.

This module replaces nn.Linear with a compressed representation where
the column ordering is NOT imposed (argsort of weights) but LEARNED
via Gumbel-Softmax as a differentiable proxy for permutation.

Research question: does the quantile curve structure emerge naturally
during training, or is it an artifact of pretrained weight statistics?

Classes:
  GumbelQPPLinear        — full-rank order_logits (O(C²), d_model ≤ 256)
  LowRankGumbelQPPLinear — factorized U@V.T (O(C×R), scales to d=768+)

ponytail: LowRankGumbelQPPLinear with R=16 drops memory 25× vs full-rank
for d_model=768. Upgrade path: Sinkhorn normalization, shared U/V across
Q/K/V/O, gradient checkpointing.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpp.compression import interp_basis


class GumbelQPPLinear(nn.Module):
    """QPP with learned column ordering via Gumbel-Softmax.

    ponytail: O(C²) per block for order_logits. Use LowRankGumbelQPPLinear
    for d_model > 256.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        anchors: int = 8,
        row_block: int = 16,
        bias: bool = True,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.anchors_k = anchors
        self.row_block = row_block
        self.temp_min = temp_min

        rows, cols = out_features, in_features
        blocks = int(math.ceil(rows / row_block))

        self.order_logits = nn.Parameter(torch.randn(blocks, cols, cols) * 0.01)
        self.anchor_values = nn.Parameter(torch.randn(rows, anchors) * 0.02)
        basis_np = interp_basis(cols, anchors)
        self.register_buffer("basis", torch.from_numpy(basis_np).float())

        self.bias_param = nn.Parameter(torch.zeros(rows)) if bias else None
        self.register_buffer("temperature", torch.tensor(temp_init))
        self.row_slices = [
            (b * row_block, min((b + 1) * row_block, rows)) for b in range(blocks)
        ]

    def _get_logits(self, block_id: int) -> torch.Tensor:
        """Override in LowRankGumbelQPPLinear for factorized variant."""
        return self.order_logits[block_id]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        out = torch.empty(x_flat.shape[0], self.out_features, device=x.device, dtype=x.dtype)

        for block_id, (start, end) in enumerate(self.row_slices):
            logits = self._get_logits(block_id)
            soft_perm = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
            x_ordered = x_flat @ soft_perm
            z = x_ordered @ self.basis.to(dtype=x_flat.dtype)
            block_anchors = self.anchor_values[start:end].to(dtype=x_flat.dtype)
            out[:, start:end] = z @ block_anchors.T

        if self.bias_param is not None:
            out = out + self.bias_param.to(dtype=x.dtype)
        return out.reshape(*original_shape, self.out_features)

    def hard_permutation(self) -> torch.Tensor:
        """Discretize to hard permutation via greedy assignment per block.

        Returns (blocks, C) where position k maps to original column index.
        """
        with torch.no_grad():
            B, C = len(self.row_slices), self.in_features
            device = self.order_logits.device
            hard = torch.empty(B, C, dtype=torch.long, device=device)
            for b in range(B):
                logits = self._get_logits(b)
                best_pos = torch.argmax(logits, dim=-1)
                conf = logits[torch.arange(C, device=device), best_pos]
                _, col_order = torch.sort(conf, descending=True)
                assigned = torch.full((C,), -1, dtype=torch.long, device=device)
                for col in col_order:
                    pos = best_pos[col].item()
                    if assigned[pos] == -1:
                        assigned[pos] = col.item()
                unassigned = (assigned == -1).nonzero(as_tuple=True)[0]
                cols_left = [c for c in range(C) if c not in assigned.tolist()]
                for pos, col in zip(unassigned.tolist(), cols_left):
                    assigned[pos] = col
                hard[b] = assigned
            return hard

    def get_dense_weight(self) -> torch.Tensor:
        """Materialize equivalent dense weight matrix (for analysis only)."""
        with torch.no_grad():
            hard = self.hard_permutation()
            rows, cols = self.out_features, self.in_features
            weight = torch.empty(rows, cols, device=self.anchor_values.device)
            for block_id, (start, end) in enumerate(self.row_slices):
                order = hard[block_id]
                inv_order = torch.empty_like(order)
                inv_order[order] = torch.arange(cols, device=order.device)
                anchors = self.anchor_values[start:end]
                rec_sorted = self.basis.to(anchors.device) @ anchors.T
                weight[start:end] = rec_sorted.T[:, inv_order]
            if self.bias_param is not None:
                return weight, self.bias_param
            return weight, None

    def compression_ratio(self) -> float:
        rows, cols = self.out_features, self.in_features
        dense_bytes = rows * cols * 2
        qpp_bytes = rows * self.anchors_k * 2
        qpp_bytes += self.order_logits.numel() * 4
        if self.bias_param is not None:
            qpp_bytes += self.bias_param.numel() * 2
        return dense_bytes / max(1, qpp_bytes)


class LowRankGumbelQPPLinear(GumbelQPPLinear):
    """Low-rank GumbelQPP: order_logits ≈ U @ V.T → O(C×R) vs O(C²).

    For d_model=768 and R=16: 24K floats/block vs 590K → 25× less memory.
    Enables scaling to real models like GPT-2 124M.

    ponytail: R=16 is conservative. Upgrade path: Sinkhorn normalization,
    shared U/V across Q/K/V/O of the same layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        anchors: int = 8,
        row_block: int = 16,
        bias: bool = True,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
        perm_rank: int = 16,
    ):
        # Skip GumbelQPPLinear.__init__ — build manually
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.anchors_k = anchors
        self.row_block = row_block
        self.temp_min = temp_min
        self.perm_rank = perm_rank

        rows, cols = out_features, in_features
        blocks = int(math.ceil(rows / row_block))

        # Low-rank: U, V of shape (blocks, cols, rank) → logits = U @ V.T
        self.order_U = nn.Parameter(torch.randn(blocks, cols, perm_rank) * 0.01)
        self.order_V = nn.Parameter(torch.randn(blocks, cols, perm_rank) * 0.01)
        self.order_logits = None  # kept as None for compatibility with parent methods

        self.anchor_values = nn.Parameter(torch.randn(rows, anchors) * 0.02)
        basis_np = interp_basis(cols, anchors)
        self.register_buffer("basis", torch.from_numpy(basis_np).float())

        self.bias_param = nn.Parameter(torch.zeros(rows)) if bias else None
        self.register_buffer("temperature", torch.tensor(temp_init))
        self.row_slices = [
            (b * row_block, min((b + 1) * row_block, rows)) for b in range(blocks)
        ]

    def _get_logits(self, block_id: int) -> torch.Tensor:
        return self.order_U[block_id] @ self.order_V[block_id].T

    def hard_permutation(self) -> torch.Tensor:
        """Same greedy assignment but using U @ V.T on-the-fly."""
        with torch.no_grad():
            B, C = len(self.row_slices), self.in_features
            device = self.order_U.device
            hard = torch.empty(B, C, dtype=torch.long, device=device)
            for b in range(B):
                logits = self._get_logits(b)
                best_pos = torch.argmax(logits, dim=-1)
                conf = logits[torch.arange(C, device=device), best_pos]
                _, col_order = torch.sort(conf, descending=True)
                assigned = torch.full((C,), -1, dtype=torch.long, device=device)
                for col in col_order:
                    pos = best_pos[col].item()
                    if assigned[pos] == -1:
                        assigned[pos] = col.item()
                unassigned = (assigned == -1).nonzero(as_tuple=True)[0]
                cols_left = [c for c in range(C) if c not in assigned.tolist()]
                for pos, col in zip(unassigned.tolist(), cols_left):
                    assigned[pos] = col
                hard[b] = assigned
            return hard

    def compression_ratio(self) -> float:
        rows, cols = self.out_features, self.in_features
        dense_bytes = rows * cols * 2
        qpp_bytes = rows * self.anchors_k * 2
        qpp_bytes += (self.order_U.numel() + self.order_V.numel()) * 4
        if self.bias_param is not None:
            qpp_bytes += self.bias_param.numel() * 2
        return dense_bytes / max(1, qpp_bytes)
