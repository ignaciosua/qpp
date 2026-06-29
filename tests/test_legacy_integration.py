import math

import numpy as np
import torch

from qpp.compression import (
    choose_block_order,
    compress_weight_shared_order,
    interp_basis,
)


def test_interp_basis_shape_and_partition_of_unity():
    basis = interp_basis(cols=17, anchors=5)
    assert basis.shape == (17, 5)
    np.testing.assert_allclose(basis.sum(axis=1), np.ones(17), atol=1e-6)


def test_shared_order_storage_is_per_block_not_per_row():
    weight = torch.randn(10, 32, dtype=torch.float32)
    _, stats = compress_weight_shared_order(
        weight,
        row_block=4,
        anchors=8,
        outlier_topk=0,
        ridge=1e-4,
        order_mode="mean",
        calib_x=None,
    )
    expected_blocks = math.ceil(10 / 4)
    assert stats["blocks"] == expected_blocks
    assert stats["shared_order_bytes"] == expected_blocks * 32 * 2
    assert stats["qpp_param_bytes"] == 10 * 8 * 2
    assert stats["dense_bf16_bytes"] == 10 * 32 * 2


def test_reconstruction_shape_and_outlier_bytes():
    weight = torch.randn(7, 19, dtype=torch.float32)
    recon, stats = compress_weight_shared_order(
        weight,
        row_block=3,
        anchors=6,
        outlier_topk=4,
        ridge=1e-4,
        order_mode="median",
        calib_x=None,
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
        weight,
        row_block=2,
        anchors=5,
        outlier_topk=0,
        ridge=1e-3,
        order_mode="mean",
        calib_x=calib_x,
    )
    assert recon.shape == weight.shape
    assert stats["activation_rel_rmse"] is not None
    assert np.isfinite(stats["activation_rel_rmse"])


def test_choose_block_order_is_single_vector_for_block():
    block = np.array([[3.0, 1.0, 2.0], [30.0, 10.0, 20.0]], dtype=np.float32)
    order = choose_block_order(block, "mean")
    assert order.shape == (3,)
    np.testing.assert_array_equal(order, np.array([1, 2, 0]))
