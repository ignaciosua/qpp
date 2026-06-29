#!/usr/bin/env python3
"""Summarize Qwen3 QPP speed/quality variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path


RUNS = [
    ("BF16", Path("outputs/qpp_qwen3_4b_bf16_generation_baseline/bf16_generation_benchmark.json"), "bf16"),
    ("QPP full direct+vec", Path("outputs/qpp_qwen3_4b_direct_cached_vectorized_bench/qpp_artifact_benchmark.json"), "qpp"),
    ("QPP attention-only", Path("outputs/qpp_qwen3_4b_direct_attention_only_vectorized_bench/qpp_artifact_benchmark.json"), "qpp"),
    ("QPP attn+lateMLP", Path("outputs/qpp_qwen3_4b_direct_attention_late_mlp_vectorized_bench/qpp_artifact_benchmark.json"), "qpp"),
    ("QPP attn+MLP26/31", Path("outputs/qpp_qwen3_4b_direct_attention_mlp26_31_vectorized_bench/qpp_artifact_benchmark.json"), "qpp"),
]


def quality_note(text: str) -> str:
    low = text.lower()
    if "0000000000" in low:
        return "bad repetition"
    if low.count("quantization is the process") >= 3 or low.count("the quantization") >= 4:
        return "repetitive"
    return "coherent smoke"


def rows() -> list[dict[str, object]]:
    base = json.loads(RUNS[0][1].read_text())
    base_mb = float(base["persistent_model_mb"])
    base_tok_s = float(base["new_tokens_per_second"])
    base_gen_seconds = float(base["seconds"])
    out = [
        {
            "mode": "BF16",
            "loaded_modules": 0,
            "ppl": 3.9187336886416997,
            "delta_ppl": 0.0,
            "model_mb": base_mb,
            "saved_mb": 0.0,
            "saved_pct": 0.0,
            "eval_seconds": 4.20810413302388,
            "gen_seconds": base_gen_seconds,
            "tok_s": base_tok_s,
            "speed_vs_bf16": 1.0,
            "quality": "baseline",
        }
    ]
    for label, path, _kind in RUNS[1:]:
        data = json.loads(path.read_text())
        gen = data["generation"]
        qpp_mb = float(data["qpp_persistent_mb"])
        tok_s = float(gen["new_tokens_per_second"])
        out.append(
            {
                "mode": label,
                "loaded_modules": int(data.get("loaded_modules", data.get("accepted_modules", 0))),
                "ppl": float(data["ppl"]),
                "delta_ppl": float(data["ppl"]) - 3.9187336886416997,
                "model_mb": qpp_mb,
                "saved_mb": base_mb - qpp_mb,
                "saved_pct": 100.0 * (base_mb - qpp_mb) / base_mb,
                "eval_seconds": float(data["eval_seconds"]),
                "gen_seconds": float(gen["seconds"]),
                "tok_s": tok_s,
                "speed_vs_bf16": tok_s / base_tok_s,
                "quality": quality_note(gen["generated_text"]),
            }
        )
    return out


def fmt(x: object, d: int = 3) -> str:
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def write_report(data: list[dict[str, object]], outdir: Path) -> None:
    lines = [
        "# Qwen3 QPP Speed/Quality Summary",
        "",
        "| Modo | Modulos | PPL | Delta PPL | Ahorro | Modelo MB | Eval s | tok/s | Speed vs BF16 | Calidad smoke |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in data:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["mode"]),
                    str(r["loaded_modules"]),
                    fmt(r["ppl"], 4),
                    fmt(r["delta_ppl"], 4),
                    f"{fmt(r['saved_pct'], 2)}%",
                    fmt(r["model_mb"], 1),
                    fmt(r["eval_seconds"], 3),
                    fmt(r["tok_s"], 2),
                    f"{fmt(r['speed_vs_bf16'], 2)}x",
                    str(r["quality"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Decision actual: `QPP attention-only` es el primer perfil competitivo contra BF16 porque ahorra VRAM, supera tok/s BF16 en este smoke y mantiene generacion coherente. Los perfiles con MLP aumentan ahorro, pero introducen repeticion.",
        ]
    )
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(data: list[dict[str, object]], outdir: Path) -> None:
    import matplotlib.pyplot as plt

    table_rows = [
        [
            r["mode"],
            r["loaded_modules"],
            fmt(r["ppl"], 3),
            f"{fmt(r['saved_pct'], 1)}%",
            fmt(r["tok_s"], 2),
            f"{fmt(r['speed_vs_bf16'], 2)}x",
            r["quality"],
        ]
        for r in data
    ]
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["Modo", "Modulos", "PPL", "Ahorro", "tok/s", "Speed", "Calidad"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#111827")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")
    fig.tight_layout()
    fig.savefig(outdir / "qpp_speed_quality_summary.png", dpi=180)
    plt.close(fig)


def main() -> None:
    outdir = Path("outputs/qpp_speed_quality_summary")
    outdir.mkdir(parents=True, exist_ok=True)
    data = rows()
    with (outdir / "qpp_speed_quality_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    write_report(data, outdir)
    write_png(data, outdir)
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    main()
