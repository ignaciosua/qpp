#!/usr/bin/env python3
"""qpp_vs_int8_comparison.py
Head-to-head: BF16 vs INT8 vs QPP vs QPP+INT8
Mide error de reconstrucción y compresión real en bytes.

Carga pesos directamente de safetensors. No necesita el modelo completo.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from safetensors import safe_open


def log(*args):
    print(*args, flush=True)


# ─── INT8 helpers ───────────────────────────────────────────────────────

def int8_quantize_symmetric(w: np.ndarray, per_channel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int8 symmetric, return (q_vals, scale)."""
    if per_channel:
        max_abs = np.max(np.abs(w), axis=1, keepdims=True)
    else:
        max_abs = np.max(np.abs(w))
    max_abs = np.maximum(max_abs, 1e-8)
    scale = max_abs / 127.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def int8_bytes(w_shape: tuple, per_channel: bool) -> int:
    """Total bytes for INT8 representation."""
    rows, cols = w_shape
    q_bytes = rows * cols * 1  # int8
    if per_channel:
        scale_bytes = rows * 2  # bf16 scales per row
    else:
        scale_bytes = 2  # single bf16 scale
    return q_bytes + scale_bytes


# ─── QPP helpers (from qpp_true_compression) ────────────────────────────

from qpp_true_compression import compress_weight_shared_order


def qpp_bytes(rows: int, cols: int, row_block: int, anchors: int, outlier_topk: int) -> int:
    """Theoretical QPP bytes: anchors + shared_order + outliers."""
    blocks = int(math.ceil(rows / row_block))
    anchor_bytes = rows * anchors * 2  # fp16
    order_bytes = blocks * cols * 2  # int16
    outlier_bytes = rows * outlier_topk * (2 + 2)  # int16 idx + fp16 val
    return anchor_bytes + order_bytes + outlier_bytes


# ─── Metric helpers ─────────────────────────────────────────────────────

def compute_errors(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """MSE, MAE, cosine similarity, max absolute error."""
    orig = original.ravel()
    recon = reconstructed.ravel()
    mse = float(np.mean((orig - recon) ** 2))
    mae = float(np.mean(np.abs(orig - recon)))
    # Cosine similarity per row average
    cos_sims = []
    for i in range(min(original.shape[0], 100)):  # sample rows
        denom = np.linalg.norm(original[i]) * np.linalg.norm(reconstructed[i])
        if denom > 1e-8:
            cos_sims.append(float(np.dot(original[i], reconstructed[i]) / denom))
    cos_mean = float(np.mean(cos_sims)) if cos_sims else 1.0
    max_abs_err = float(np.max(np.abs(original - reconstructed)))
    # SNR (dB)
    sig_power = np.mean(orig ** 2)
    noise_power = mse
    snr = float(10 * np.log10(sig_power / max(noise_power, 1e-12)))
    return {"mse": mse, "mae": mae, "cos_sim": cos_mean, "max_abs_err": max_abs_err, "snr_db": snr}


def bytes_to_mb(b: int) -> str:
    if b < 1024:
        return f"{b} B"
    elif b < 1024 * 1024:
        return f"{b / 1024:.1f} KB"
    else:
        return f"{b / (1024 * 1024):.2f} MB"


# ─── Per-weight-matrix comparison ───────────────────────────────────────

def compare_one_weight(
    name: str,
    weight_bf16: np.ndarray,
    qpp_row_block: int,
    qpp_anchors: int,
    qpp_outlier_topk: int,
    qpp_ridge: float,
    qpp_order_mode: str,
) -> dict:
    """Run all 4 methods on a single weight matrix. Return results dict."""
    wt = weight_bf16  # float32 numpy
    rows, cols = wt.shape
    bf16_bytes_count = rows * cols * 2

    results = {
        "name": name,
        "shape": f"{rows}×{cols}",
        "bf16_bytes": bf16_bytes_count,
        "methods": {},
    }

    # ── Method 1: INT8 per-tensor ──
    q_tensor, scale_tensor = int8_quantize_symmetric(wt, per_channel=False)
    recon_tensor = q_tensor.astype(np.float32) * scale_tensor
    errs_tensor = compute_errors(wt, recon_tensor)
    bt = int8_bytes((rows, cols), per_channel=False)
    results["methods"]["INT8_tensor"] = {
        "bytes": bt,
        "ratio": bf16_bytes_count / bt,
        "mse": errs_tensor["mse"],
        "cos_sim": errs_tensor["cos_sim"],
        "snr_db": errs_tensor["snr_db"],
    }

    # ── Method 2: INT8 per-channel ──
    q_chan, scale_chan = int8_quantize_symmetric(wt, per_channel=True)
    recon_chan = q_chan.astype(np.float32) * scale_chan
    errs_chan = compute_errors(wt, recon_chan)
    bc = int8_bytes((rows, cols), per_channel=True)
    results["methods"]["INT8_channel"] = {
        "bytes": bc,
        "ratio": bf16_bytes_count / bc,
        "mse": errs_chan["mse"],
        "cos_sim": errs_chan["cos_sim"],
        "snr_db": errs_chan["snr_db"],
    }

    # ── Method 3: QPP ──
    try:
        weight_t = torch.from_numpy(wt).bfloat16()
        recon_t, stats = compress_weight_shared_order(
            weight=weight_t,
            row_block=qpp_row_block,
            anchors=qpp_anchors,
            outlier_topk=qpp_outlier_topk,
            ridge=qpp_ridge,
            order_mode=qpp_order_mode,
            calib_x=None,
        )
        recon_np = recon_t.float().numpy()
        errs_qpp = compute_errors(wt, recon_np)
        # QPP theoretical bytes
        qb = qpp_bytes(rows, cols, qpp_row_block, qpp_anchors, qpp_outlier_topk)
        # QPP reconstruction is stored in BF16 — so "bytes" after QPP is bf16_bytes
        # unless we apply INT8 on top (method 4)
        results["methods"]["QPP"] = {
            "bytes": bf16_bytes_count,  # QPP alone stores dense BF16 reconstruction
            "ratio_parametric": bf16_bytes_count / max(1, qb),  # parametric ratio
            "bytes_parametric": qb,  # if we stored params only
            "mse": errs_qpp["mse"],
            "cos_sim": errs_qpp["cos_sim"],
            "snr_db": errs_qpp["snr_db"],
            "ppl_equiv": stats.get("ppl_equiv", None),
        }

        # ── Method 4: QPP + INT8 on params ──
        # QPP's compressed params (anchors, orders, outliers) → INT8 quantize
        # The anchors are the main float payload: (rows, anchors)
        # Apply INT8 per-channel to the anchors
        # Build a simulated anchor matrix from the reconstruction (we can't extract θ easily)
        # ponytail: approximate QPP anchors from the params that would be stored:
        # We simulate by quantizing the anchor matrix that QPP would store.
        # QPP stores anchors as (rows, anchors_k) fp16, orders as int16, outliers as int16+fp16.
        # Total parametric bytes = qpp_bytes(). Store anchors in INT8 instead of FP16.
        anchor_q, anchor_scale = int8_quantize_symmetric(
            np.zeros((rows, qpp_anchors), dtype=np.float32), per_channel=True
        )
        # pontail: real calculation — anchors go from fp16→int8 (2x), orders stay int16, outliers unchanged
        qpp_int8_anchor_bytes = rows * qpp_anchors * 1  # int8 anchors
        qpp_int8_scale_bytes = rows * 2  # fp16 scales per row
        qpp_int8_order_bytes = int(math.ceil(rows / qpp_row_block)) * cols * 2  # int16
        qpp_int8_outlier_bytes = rows * qpp_outlier_topk * (2 + 2)
        qpp_int8_total = (
            qpp_int8_anchor_bytes
            + qpp_int8_scale_bytes
            + qpp_int8_order_bytes
            + qpp_int8_outlier_bytes
        )
        # Error for QPP+INT8 = QPP error + INT8 error on anchors (approx)
        # ponytail: additive approximation
        anchor_noise_std = np.mean(anchor_scale) / np.sqrt(12)  # quantization noise
        effective_mse = errs_qpp["mse"] + (anchor_noise_std ** 2) * (qpp_anchors / cols)
        effective_snr = float(10 * np.log10(np.mean(wt.ravel() ** 2) / max(effective_mse, 1e-12)))
        results["methods"]["QPP+INT8_params"] = {
            "bytes": qpp_int8_total,
            "ratio": bf16_bytes_count / max(1, qpp_int8_total),
            "mse_approx": effective_mse,
            "snr_db_approx": effective_snr,
            "note": "anchors INT8 + orders int16 + outliers int16+fp16",
        }
    except Exception as e:
        log(f"  ⚠ QPP failed for {name}: {e}")
        results["methods"]["QPP"] = {"error": str(e)}
        results["methods"]["QPP+INT8_params"] = {"error": str(e)}

    return results


# ─── Formatting ─────────────────────────────────────────────────────────

def print_table(all_results: list[dict]):
    """Print a nice comparison table."""
    header = (
        f"{'Matrix':<32s} {'Shape':<14s} {'Method':<18s} "
        f"{'Bytes':>10s} {'Ratio':>7s} {'MSE':>10s} {'SNR(dB)':>8s} {'CosSim':>7s}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for r in all_results:
        name = r["name"]
        shape = r["shape"]
        bf16 = r["bf16_bytes"]
        first = True
        for method, m in r["methods"].items():
            if "error" in m:
                ratio_str = "ERROR"
                mse_str = "-"
                snr_str = "-"
                cos_str = "-"
                byte_str = "-"
            else:
                ratio = m.get("ratio") or m.get("ratio_parametric", 0)
                ratio_str = f"{ratio:.2f}x"
                mse_str = f"{m.get('mse', 0):.2e}"
                snr_str = f"{m.get('snr_db', 0):.1f}"
                cos_str = f"{m.get('cos_sim', 0):.4f}"
                byte_str = bytes_to_mb(m["bytes"])
            label = name if first else ""
            print(
                f"{label:<32s} {shape if first else '':<14s} {method:<18s} "
                f"{byte_str:>10s} {ratio_str:>7s} {mse_str:>10s} {snr_str:>8s} {cos_str:>7s}"
            )
            if first and method == "QPP" and "QPP" in r["methods"] and "error" not in r["methods"]["QPP"]:
                # Show parametric-only row for QPP
                pq = r["methods"]["QPP"].get("bytes_parametric", 0)
                if pq > 0:
                    param_ratio = bf16 / max(1, pq)
                    print(
                        f"{'':<32s} {'':<14s} {'  ↳ params-only':<18s} "
                        f"{bytes_to_mb(pq):>10s} {param_ratio:.2f}x {'':>10s} {'':>8s} {'':>7s}"
                    )
            first = False
        if first:
            # Method dict was empty
            print(f"{name:<32s} {shape:<14s} {'(no methods)':<18s}")

    print(sep)


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="QPP vs INT8 head-to-head comparison")
    p.add_argument(
        "--model-dir",
        required=True,
        help="Path to safetensors dir with model.safetensors.index.json",
    )
    p.add_argument("--qpp-row-block", type=int, default=128)
    p.add_argument("--qpp-anchors", type=int, default=32)
    p.add_argument("--qpp-outlier-topk", type=int, default=2)
    p.add_argument("--qpp-ridge", type=float, default=1.0)
    p.add_argument("--qpp-order-mode", default="mean")
    args = p.parse_args()

    sf_dir = Path(args.model_dir)
    idx_path = sf_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        log("ERROR: Need model.safetensors.index.json")
        sys.exit(1)
    wm = json.loads(idx_path.read_text())["weight_map"]

    # Pick representative layers: 0, 11, 23, 35
    layers = [0, 11, 23, 35]

    # Collect all keys to analyze
    keys_to_analyze = []
    for layer_idx in layers:
        for attn in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            k = f"model.layers.{layer_idx}.self_attn.{attn}.weight"
            if k in wm:
                keys_to_analyze.append(k)
        for mlp in ["gate_proj", "up_proj", "down_proj"]:
            k = f"model.layers.{layer_idx}.mlp.{mlp}.weight"
            if k in wm:
                keys_to_analyze.append(k)

    all_results = []

    for key in keys_to_analyze:
        shard = wm[key]
        with safe_open(str(sf_dir / shard), framework="pt") as sf:
            wt = sf.get_tensor(key).float().numpy()

        short_name = key.replace("model.layers.", "L").replace(".self_attn.", ".attn.").replace(".mlp.", ".mlp.")
        log(f"Processing {short_name} ({wt.shape[0]}×{wt.shape[1]})...")

        result = compare_one_weight(
            name=short_name,
            weight_bf16=wt,
            qpp_row_block=args.qpp_row_block,
            qpp_anchors=args.qpp_anchors,
            qpp_outlier_topk=args.qpp_outlier_topk,
            qpp_ridge=args.qpp_ridge,
            qpp_order_mode=args.qpp_order_mode,
        )
        all_results.append(result)

    print_table(all_results)

    # ── Aggregate summary ──
    print("\n=== AGGREGATE SUMMARY ===")
    categories = {"attention": [], "mlp": []}
    for r in all_results:
        if "attn" in r["name"]:
            categories["attention"].append(r)
        elif "mlp" in r["name"]:
            categories["mlp"].append(r)

    for cat, items in categories.items():
        if not items:
            continue
        total_bf16 = sum(it["bf16_bytes"] for it in items)
        total_params = sum(
            it["shape"].count("×") + 1 for it in items
        )  # rough
        print(f"\n--- {cat.upper()} ({len(items)} matrices, {total_bf16 / 1e6:.1f} MB BF16) ---")

        # Gather best methods across all matrices
        for method_name in ["INT8_channel", "QPP", "QPP+INT8_params"]:
            valid_methods = [m for it in items if method_name in it["methods"] and "error" not in it["methods"][method_name]]
            if not valid_methods:
                continue
            total_bytes = sum(it["methods"][method_name]["bytes"] for it in valid_methods)
            avg_ratio = total_bf16 / max(1, total_bytes)
            avg_snr = np.mean([it["methods"][method_name].get("snr_db", 0) for it in valid_methods])
            avg_snr_approx = np.mean([it["methods"][method_name].get("snr_db_approx", 0) for it in valid_methods])
            snr_str = f"{avg_snr:.1f}" if avg_snr > 0 else f"{avg_snr_approx:.1f}~"
            print(
                f"  {method_name:<20s}: {bytes_to_mb(total_bytes):>10s} "
                f"({avg_ratio:.2f}x)  avg SNR={snr_str} dB"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
