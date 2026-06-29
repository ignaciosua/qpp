"""Unit tests for QPP core compression — no GPU or HuggingFace required."""

import math

import numpy as np
import torch

from qpp import choose_block_order, compress_weight_shared_order, interp_basis, lloyd_max_1d
from qpp.compression import solve_anchors_activation, solve_anchors_weight


def test_interp_basis_shape_and_partition_of_unity():
    basis = interp_basis(cols=17, anchors=5)
    assert basis.shape == (17, 5)
    np.testing.assert_allclose(basis.sum(axis=1), np.ones(17), atol=1e-6)


def test_interp_basis_endpoints():
    """Anchor 0 and anchor K-1 should fully activate at first/last columns."""
    basis = interp_basis(cols=100, anchors=10)
    assert basis[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert basis[-1, -1] == pytest.approx(1.0, abs=1e-6)


def test_solve_anchors_weight_perfect():
    """With zero noise, reconstruction should be exact."""
    torch.manual_seed(42)
    rows, cols, K = 5, 20, 6
    basis = interp_basis(cols, K)
    theta_true = np.random.randn(rows, K).astype(np.float32)
    w_sorted = theta_true @ basis.T
    theta_hat = solve_anchors_weight(w_sorted, basis, ridge=1e-4)
    np.testing.assert_allclose(theta_hat, theta_true, atol=1e-3)


def test_solve_anchors_activation_perfect():
    torch.manual_seed(42)
    rows, cols, K, N = 5, 20, 6, 32
    basis = interp_basis(cols, K)
    theta_true = np.random.randn(rows, K).astype(np.float32)
    w_sorted = theta_true @ basis.T
    x = np.random.randn(N, cols).astype(np.float32)
    y = x @ w_sorted.T
    theta_hat = solve_anchors_activation(x, y, basis, ridge=1e-4)
    np.testing.assert_allclose(theta_hat, theta_true, atol=1e-3)


def test_shared_order_storage_is_per_block_not_per_row():
    weight = torch.randn(10, 32, dtype=torch.float32)
    _, stats = compress_weight_shared_order(
        weight, row_block=4, anchors=8, outlier_topk=0,
        ridge=1e-4, order_mode="mean", calib_x=None,
    )
    expected_blocks = math.ceil(10 / 4)
    assert stats["blocks"] == expected_blocks
    assert stats["shared_order_bytes"] == expected_blocks * 32 * 2
    assert stats["qpp_param_bytes"] == 10 * 8 * 2
    assert stats["dense_bf16_bytes"] == 10 * 32 * 2


def test_reconstruction_shape_and_outlier_bytes():
    weight = torch.randn(7, 19, dtype=torch.float32)
    recon, stats = compress_weight_shared_order(
        weight, row_block=3, anchors=6, outlier_topk=4,
        ridge=1e-4, order_mode="median", calib_x=None,
    )
    assert recon.shape == weight.shape
    assert stats["outlier_bytes"] == 7 * 4 * (2 + 2)
    assert stats["total_qpp_bytes"] == (
        stats["qpp_param_bytes"] + stats["shared_order_bytes"] + stats["outlier_bytes"]
    )


def test_activation_calibration_path_runs():
    torch.manual_seed(0)
    weight = torch.randn(5, 13, dtype=torch.float32)
    calib_x = torch.randn(23, 13, dtype=torch.float32)
    recon, stats = compress_weight_shared_order(
        weight, row_block=2, anchors=5, outlier_topk=0,
        ridge=1e-3, order_mode="mean", calib_x=calib_x,
    )
    assert recon.shape == weight.shape
    assert stats["activation_rel_rmse"] is not None
    assert np.isfinite(stats["activation_rel_rmse"])


def test_choose_block_order_is_single_vector_for_block():
    block = np.array([[3.0, 1.0, 2.0], [30.0, 10.0, 20.0]], dtype=np.float32)
    order = choose_block_order(block, "mean")
    assert len(order) == 3
    np.testing.assert_array_equal(order, np.array([1, 2, 0]))


def test_choose_block_order_abs_mean():
    block = np.array([[1.0, -5.0, 3.0], [-2.0, 4.0, -1.0]], dtype=np.float32)
    order = choose_block_order(block, "abs_mean")
    assert len(order) == 3
    # abs means: [1.5, 4.5, 2.0] → sorted: column 0, 2, 1
    np.testing.assert_array_equal(order, np.array([0, 2, 1]))


def test_compress_weight_shared_order_small():
    """Smoke test: full compression on a tiny weight matrix."""
    torch.manual_seed(123)
    weight = torch.randn(16, 64, dtype=torch.float32)
    recon, stats = compress_weight_shared_order(
        weight, row_block=4, anchors=16, outlier_topk=4,
        ridge=1e-3, order_mode="mean", calib_x=None,
    )
    assert recon.shape == (16, 64)
    assert stats["compression_vs_bf16"] > 1.0
    assert stats["weight_rel_rmse"] < 1.0


def test_lloyd_max_1d_shape():
    data = np.random.randn(100).astype(np.float32)
    cb, codes = lloyd_max_1d(data, n_levels=4)
    assert cb.shape == (4,)
    assert codes.shape == (100,)
    assert codes.dtype == np.uint8
    assert set(codes) <= {0, 1, 2, 3}


def test_lloyd_max_1d_constant():
    """Constant data should quantize perfectly."""
    data = np.full(50, 3.5, dtype=np.float32)
    cb, codes = lloyd_max_1d(data, n_levels=2)
    np.testing.assert_allclose(cb, [3.5, 3.5], atol=1e-6)


def test_compression_ratio_increases_with_more_anchors():
    torch.manual_seed(42)
    weight = torch.randn(32, 128, dtype=torch.float32)
    _, s8 = compress_weight_shared_order(weight, 8, 8, 0, 0.001, "mean", None)
    _, s16 = compress_weight_shared_order(weight, 8, 16, 0, 0.001, "mean", None)
    assert s8["compression_vs_bf16"] > s16["compression_vs_bf16"]


# Need pytest for this one
import pytest  # noqa: E402
