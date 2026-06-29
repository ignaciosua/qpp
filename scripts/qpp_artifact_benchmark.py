#!/usr/bin/env python3
"""Load a saved QPP artifact and benchmark it without requantizing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpp.runtime import QPPCompressedLinear, set_nested_attr, persistent_model_bytes
from qpp.benchmark import make_text, perplexity, cuda_mem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-tokens", type=int, default=8192)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--text-repeat", type=int, default=16)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--prompt", default="Explain quantization in one short paragraph.")
    p.add_argument("--cache-indices", action="store_true")
    p.add_argument("--direct-vectorize-max-tokens", type=int, default=16)
    p.add_argument("--include-substrings", nargs="*", default=None)
    p.add_argument("--skip-substrings", nargs="*", default=None)
    p.add_argument("--max-artifact-modules", type=int, default=0)
    return p.parse_args()


def load_artifact(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def module_from_artifact(
    spec: dict,
    device: str,
    dtype: torch.dtype,
    cache_indices: bool,
    direct_vectorize_max_tokens: int,
) -> QPPCompressedLinear:
    tensors = spec["tensors"]
    orders_i16 = tensors["orders_i16"].to(device=device)
    orders_i32 = None
    if cache_indices:
        if "orders_i32" in tensors:
            orders_i32 = tensors["orders_i32"].to(device=device)
        else:
            orders_i32 = orders_i16.to(torch.int32)
    kwargs = {
        "anchors": tensors["anchors"].to(device=device, dtype=dtype),
        "orders_i16": orders_i16,
        "orders_i32": orders_i32,
        "row_slices": [tuple(x) for x in spec["row_slices"]],
        "original_shape": (int(spec["out_features"]), int(spec["in_features"])),
        "bias": None if "bias" not in tensors else tensors["bias"].to(device=device, dtype=dtype),
        "basis": None if "basis" not in tensors else tensors["basis"].to(device=device, dtype=dtype),
        "outlier_idx_i16": None if "outlier_idx_i16" not in tensors else tensors["outlier_idx_i16"].to(device=device),
        "outlier_val": None if "outlier_val" not in tensors else tensors["outlier_val"].to(device=device, dtype=dtype),
        "residual_a": None if "residual_a" not in tensors else tensors["residual_a"].to(device=device, dtype=dtype),
        "residual_b": None if "residual_b" not in tensors else tensors["residual_b"].to(device=device, dtype=dtype),
        "forward_mode": spec.get("forward_mode", "direct"),
        "direct_vectorize_max_tokens": direct_vectorize_max_tokens,
    }
    return QPPCompressedLinear(**kwargs)


def generation_benchmark(model, tokenizer, args: argparse.Namespace) -> dict:
    encoded = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    input_tokens = int(encoded["input_ids"].shape[-1])
    output_tokens = int(out.shape[-1])
    new_tokens = max(0, output_tokens - input_tokens)
    return {
        "seconds": seconds,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "new_tokens": new_tokens,
        "new_tokens_per_second": new_tokens / max(seconds, 1e-9),
        "generated_text": generated,
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    artifact = load_artifact(Path(args.artifact))
    model_path = args.model or artifact["summary"]["config"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()
    base_mb = persistent_model_bytes(model) / 1e6
    selected_modules = []
    for name, spec in artifact["modules"].items():
        if args.include_substrings and not any(s in name for s in args.include_substrings):
            continue
        if args.skip_substrings and any(s in name for s in args.skip_substrings):
            continue
        selected_modules.append((name, spec))
    if args.max_artifact_modules > 0:
        selected_modules = selected_modules[: args.max_artifact_modules]
    for name, spec in selected_modules:
        set_nested_attr(
            model,
            name,
            module_from_artifact(
                spec,
                args.device,
                dtype,
                args.cache_indices,
                args.direct_vectorize_max_tokens,
            ),
        )
    qpp_mb = persistent_model_bytes(model) / 1e6
    text = make_text(args.text_repeat)
    qpp_ppl, qpp_sec, tokens = perplexity(model, tokenizer, text, args.device, args.eval_tokens, args.ctx)
    gen = generation_benchmark(model, tokenizer, args) if args.generate else None
    result = {
        "artifact": str(Path(args.artifact)),
        "model": model_path,
        "cache_indices": args.cache_indices,
        "direct_vectorize_max_tokens": args.direct_vectorize_max_tokens,
        "artifact_modules": len(artifact["modules"]),
        "loaded_modules": len(selected_modules),
        "include_substrings": args.include_substrings,
        "skip_substrings": args.skip_substrings,
        "base_persistent_mb": base_mb,
        "qpp_persistent_mb": qpp_mb,
        "persistent_delta_mb": qpp_mb - base_mb,
        "ppl": qpp_ppl,
        "eval_seconds": qpp_sec,
        "tokens": tokens,
        "memory": cuda_mem(),
        "generation": gen,
    }
    (outdir / "qpp_artifact_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    gen_lines = []
    if gen is not None:
        gen_lines = [
            f"- generation_seconds: {gen['seconds']:.6f}",
            f"- generation_tok_s: {gen['new_tokens_per_second']:.6f}",
            "",
            "```text",
            gen["generated_text"],
            "```",
        ]
    (outdir / "REPORT.md").write_text(
        "# QPP Artifact Benchmark\n\n"
        f"- cache_indices: {args.cache_indices}\n"
        f"- direct_vectorize_max_tokens: {args.direct_vectorize_max_tokens}\n"
        f"- artifact_modules: {len(artifact['modules'])}\n"
        f"- loaded_modules: {len(selected_modules)}\n"
        f"- qpp_persistent_mb: {qpp_mb:.3f}\n"
        f"- persistent_delta_mb: {qpp_mb - base_mb:.3f}\n"
        f"- ppl: {qpp_ppl:.6f}\n"
        f"- eval_seconds: {qpp_sec:.6f}\n"
        + ("\n".join(gen_lines) + "\n" if gen_lines else ""),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
