"""Lloyd-Max 1D quantizer for QPP anchor codebook compression.

The codebook step is optional and orthogonal to QPP: it further compresses
the K anchor values per row via a per-row vector quantizer.
"""

from __future__ import annotations

import numpy as np


def lloyd_max_1d(
    data: np.ndarray, n_levels: int, max_iter: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max 1D scalar quantizer.

    Args:
        data: (N,) float array of values to quantize
        n_levels: number of quantization levels (e.g., 4 for 2-bit)
        max_iter: maximum Lloyd iterations

    Returns:
        (codebook, codes) where codebook is (n_levels,) float32 and
        codes is (N,) uint8 with indices into the codebook
    """
    sd = np.sort(data)
    idx = np.linspace(0, len(sd) - 1, n_levels + 1).astype(int)
    centroids = np.array([sd[idx[i] : idx[i + 1]].mean() for i in range(n_levels)], dtype=np.float32)

    for _ in range(max_iter):
        dists = np.abs(data[:, None] - centroids[None, :])
        assigns = np.argmin(dists, axis=1)
        new_c = np.array(
            [
                data[assigns == i].mean() if np.any(assigns == i) else centroids[i]
                for i in range(n_levels)
            ],
            dtype=np.float32,
        )
        if np.allclose(centroids, new_c, atol=1e-6):
            break
        centroids = new_c

    codes = np.argmin(np.abs(data[:, None] - centroids[None, :]), axis=1).astype(np.uint8)
    return centroids, codes
