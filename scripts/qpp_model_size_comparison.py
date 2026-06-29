#!/usr/bin/env python3
"""Compare full QPP runs across model sizes."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-qpp-model-compare")

import matplotlib.pyplot as plt
import pandas as pd


RUNS = [
    ("Qwen2.5-0.5B-Instruct", "outputs/qpp_full_all_rank256_s200_gate05"),
    ("Qwen2-1.5B", "outputs/qpp_qwen2_1p5b_full_all_rank256_s200_gate05"),
    ("Qwen3-4B", "outputs/qpp_qwen3_4b_full_all_rank256_s200_gate05"),
    ("Phi-2", "outputs/qpp_phi2_full_all_rank256_s200_gate05"),
]


def load_run(model: str, outdir: str) -> dict:
    data = json.loads((Path(outdir) / "qpp_runtime_results.json").read_text(encoding="utf-8"))
    run = data["run"]
    base = float(run["persistent_model_mb_after_load"])
    qpp = float(run["persistent_model_mb_after_qpp"])
    saved = base - qpp
    return {
        "model": model,
        "outdir": outdir,
        "selected_modules": run["selected_modules"],
        "accepted_modules": run["accepted_modules"],
        "accepted_pct": 100.0 * run["accepted_modules"] / max(1, run["selected_modules"]),
        "bf16_ppl": run["baseline_ppl"],
        "qpp_ppl": run["runtime_qpp_ppl"],
        "delta_ppl": run["delta_ppl"],
        "bf16_model_mb": base,
        "qpp_model_mb": qpp,
        "saved_mb": saved,
        "saved_pct_total": 100.0 * saved / base,
        "model_remaining_pct": 100.0 * qpp / base,
        "accepted_dense_bf16_mb": run["accepted_dense_bf16_mb"],
        "accepted_qpp_runtime_mb": run["accepted_runtime_buffer_mb"],
        "accepted_subset_compression": run["runtime_buffer_compression_vs_bf16"],
        "slowdown_vs_bf16": run["slowdown_vs_bf16"],
        "generation_seconds": run["generation_seconds"],
    }


def write_png(df: pd.DataFrame, outdir: Path) -> None:
    show = df[[
        "model",
        "accepted_modules",
        "delta_ppl",
        "bf16_model_mb",
        "qpp_model_mb",
        "saved_mb",
        "saved_pct_total",
        "model_remaining_pct",
        "accepted_subset_compression",
        "slowdown_vs_bf16",
    ]].copy()
    for col in show.columns:
        if col not in ("model", "accepted_modules"):
            show[col] = show[col].map(lambda x: f"{x:.3f}")
    fig, ax = plt.subplots(figsize=(17, 3.8))
    ax.axis("off")
    ax.set_title("Full QPP run: model size comparison", fontsize=15, weight="bold", pad=12)
    table = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f8fa")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(outdir / "qpp_model_size_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = Path("outputs/qpp_model_size_comparison")
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([load_run(*run) for run in RUNS])
    df.to_csv(outdir / "qpp_model_size_comparison.csv", index=False)
    write_png(df, outdir)
    report = [
        "# QPP Model Size Comparison",
        "",
        df.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Notes:",
        "- All runs use anchors=32, residual_rank=256, 200 residual training steps, eval_tokens=8192 and total_delta_gate=0.5.",
        "- Qwen2-1.5B has the best total percentage saving so far under this policy; Qwen3-4B saves the most absolute MB and compresses accepted modules strongly.",
        "- Generation text remains poor/repetitive and speed is slower than BF16 due to dense reconstruction.",
    ]
    (outdir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {outdir.resolve()}")


if __name__ == "__main__":
    main()
