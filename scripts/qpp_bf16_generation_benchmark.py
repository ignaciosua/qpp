#!/usr/bin/env python3
"""Small BF16 generation benchmark for local causal LMs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qpp.runtime import persistent_model_bytes
from qpp.benchmark import cuda_mem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--prompt", default="Explain quantization in one short paragraph.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()
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
    result = {
        "model": args.model,
        "dtype": str(dtype),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "new_tokens": new_tokens,
        "seconds": seconds,
        "new_tokens_per_second": new_tokens / max(seconds, 1e-9),
        "persistent_model_mb": persistent_model_bytes(model) / 1e6,
        "memory": cuda_mem(),
        "generated_text": generated,
    }
    (outdir / "bf16_generation_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (outdir / "REPORT.md").write_text(
        "# BF16 Generation Benchmark\n\n"
        f"- seconds: {seconds:.6f}\n"
        f"- new_tokens: {new_tokens}\n"
        f"- new_tokens_per_second: {result['new_tokens_per_second']:.6f}\n"
        f"- persistent_model_mb: {result['persistent_model_mb']:.3f}\n\n"
        "```text\n"
        f"{generated}\n"
        "```\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
