"""Gumbel-Softmax QPP: learn the column ordering during training.

This module replaces nn.Linear with a compressed representation where
the column ordering is NOT imposed (argsort of weights) but LEARNED
via Gumbel-Softmax as a differentiable proxy for permutation.

Research question: does the quantile curve structure emerge naturally
during training, or is it an artifact of pretrained weight statistics?

ponytail: prototype — O(C²) per block for Gumbel-Softmax, only viable for
small C (d_model ≤ 256). Upgrade path: low-rank factorized permutation,
Sinkhorn normalization, or Hungarian straight-through estimator.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qpp.compression import interp_basis


class GumbelQPPLinear(nn.Module):
    """Linear layer where weights are stored as QPP anchors + learned column ordering.

    Forward pass:
      1. Gumbel-Softmax: soft_perm = gumbel_softmax(order_logits)
      2. Reorder activations: x_ordered = x @ soft_perm.T
      3. Interpolate: z = x_ordered @ basis   (basis from interp_basis)
      4. Project: out = z @ anchors.T

    At the end of training, soft_perm is discretized to a hard permutation
    for inference (no Gumbel noise).
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

        # Per-block learnable ordering logits: (blocks, cols, cols)
        # order_logits[b, i, j] = logit for column i being at position j in block b
        self.order_logits = nn.Parameter(torch.randn(blocks, cols, cols) * 0.01)

        # Learnable anchors: (rows, anchors)
        self.anchor_values = nn.Parameter(torch.randn(rows, anchors) * 0.02)

        # Fixed interpolation basis: (cols, anchors)
        basis_np = interp_basis(cols, anchors)
        self.register_buffer("basis", torch.from_numpy(basis_np).float())

        if bias:
            self.bias_param = nn.Parameter(torch.zeros(rows))
        else:
            self.bias_param = None

        # Temperature (can be annealed externally)
        self.register_buffer("temperature", torch.tensor(temp_init))

        self.row_slices = [
            (b * row_block, min((b + 1) * row_block, rows)) for b in range(blocks)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft permutation.

        Args:
            x: (..., in_features)

        Returns:
            (..., out_features)
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        out = torch.empty(
            x_flat.shape[0], self.out_features,
            device=x.device, dtype=x.dtype,
        )

        for block_id, (start, end) in enumerate(self.row_slices):
            # Gumbel-Softmax permutation for this block
            logits = self.order_logits[block_id]  # (C, C)
            soft_perm = F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)
            # soft_perm[k, j] ≈ P(col k goes to position j)
            # x_ordered[n, k] = Σ_j x[n, j] · soft_perm[j, k] = (x @ soft_perm)[n, k]
            x_ordered = x_flat @ soft_perm  # (N, C)

            # Interpolate: (N, C) @ (C, K) = (N, K)
            z = x_ordered @ self.basis.to(dtype=x_flat.dtype)

            # Project: (N, K) @ (K, block_rows) = (N, block_rows)
            block_anchors = self.anchor_values[start:end].to(dtype=x_flat.dtype)  # (R_b, K)
            out[:, start:end] = z @ block_anchors.T

        if self.bias_param is not None:
            out = out + self.bias_param.to(dtype=x.dtype)

        return out.reshape(*original_shape, self.out_features)

    def hard_permutation(self) -> torch.Tensor:
        """Discretize to hard permutation for inference (no Gumbel noise).

        Returns (blocks, C) where position k maps to original column index.
        Uses greedy column-to-position assignment with tiebreaking.
        ponytail: O(C²) greedy per block. Upgrade path: Hungarian / Sinkhorn.
        """
        with torch.no_grad():
            B, C = self.order_logits.shape[0], self.in_features
            hard = torch.empty(B, C, dtype=torch.long, device=self.order_logits.device)
            for b in range(B):
                logits = self.order_logits[b]  # (C, C)
                best_pos = torch.argmax(logits, dim=-1)  # (C,)
                conf = logits[torch.arange(C, device=logits.device), best_pos]
                _, col_order = torch.sort(conf, descending=True)
                assigned = torch.full((C,), -1, dtype=torch.long, device=logits.device)
                for col in col_order:
                    pos = best_pos[col].item()
                    if assigned[pos] == -1:
                        assigned[pos] = col.item()
                # Fill remaining: unassigned columns → unassigned positions
                unassigned = (assigned == -1).nonzero(as_tuple=True)[0]
                cols_left = [c for c in range(C) if c not in assigned.tolist()]
                for pos, col in zip(unassigned.tolist(), cols_left):
                    assigned[pos] = col
                hard[b] = assigned
            return hard

    def get_dense_weight(self) -> torch.Tensor:
        """Materialize the equivalent dense weight matrix (for analysis)."""
        with torch.no_grad():
            hard = self.hard_permutation()  # (blocks, C)
            rows, cols = self.out_features, self.in_features
            weight = torch.empty(rows, cols, device=self.anchor_values.device)
            for block_id, (start, end) in enumerate(self.row_slices):
                order = hard[block_id]  # (C,)
                inv_order = torch.empty_like(order)
                inv_order[order] = torch.arange(cols, device=order.device)
                anchors = self.anchor_values[start:end]  # (R_b, K)
                rec_sorted = self.basis.to(anchors.device) @ anchors.T  # (C, R_b)
                weight[start:end] = rec_sorted.T[:, inv_order]
            if self.bias_param is not None:
                return weight, self.bias_param
            return weight, None

    def compression_ratio(self) -> float:
        """Parametric compression vs dense BF16."""
        rows, cols = self.out_features, self.in_features
        dense_bytes = rows * cols * 2
        qpp_bytes = rows * self.anchors_k * 2  # anchors
        qpp_bytes += self.order_logits.numel() * 4  # logits (fp32 during training)
        qpp_bytes += (self.bias_param.numel() * 2 if self.bias_param is not None else 0)
        return dense_bytes / max(1, qpp_bytes)
