"""Benchmark utilities: PPL evaluation, generation benchmarking, model loading.

ponytail: These require HuggingFace transformers. Import only if needed.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn


@torch.no_grad()
def perplexity(
    model: nn.Module,
    tokenizer,
    text: str,
    device: str,
    max_tokens: int,
    ctx: int,
) -> tuple[float, float, int]:
    """Compute perplexity of a model on a text using sliding window.

    Returns:
        (perplexity, elapsed_seconds, num_tokens_evaluated)
    """
    model.eval()
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
    if ids.shape[1] < 2:
        return float("inf"), 0.0, int(ids.shape[1])
    ctx = min(ctx, ids.shape[1] - 1)
    stride = max(1, ctx // 2)
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    start = time.perf_counter()
    for begin in range(0, ids.shape[1], stride):
        end = min(begin + ctx, ids.shape[1])
        trg_len = end - prev_end
        batch = ids[:, begin:end]
        labels = batch.clone()
        labels[:, :-trg_len] = -100
        out = model(batch, labels=labels)
        valid = int((labels != -100).sum().item())
        nll_sum += float(out.loss) * valid
        n_tokens += valid
        prev_end = end
        if end == ids.shape[1]:
            break
    loss = nll_sum / max(1, n_tokens)
    return math.exp(loss), time.perf_counter() - start, n_tokens


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    tokenizer,
    text: str,
    target_names: list[str],
    device: str,
    max_tokens: int,
    ctx: int,
    num_calib_rows: int,
) -> dict[str, torch.Tensor]:
    """Collect activation vectors for calibration of QPP anchors.

    Returns:
        dict mapping module name to (samples, in_features) activation tensor
    """
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens].to(device)
    ctx = min(ctx, ids.shape[1] - 1)
    hooks = []
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}

    def hook_fn(name: str):
        def fn(mod, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            outputs[name].append(x.cpu())
        return fn

    for name in target_names:
        try:
            mod = model
            for part in name.split("."):
                mod = getattr(mod, part)
            hooks.append(mod.register_forward_hook(hook_fn(name)))
        except (AttributeError, KeyError):
            continue

    model.eval()
    stride = max(1, ctx // 2)
    for begin in range(0, ids.shape[1], stride):
        end = min(begin + ctx, ids.shape[1])
        batch = ids[:, begin:end]
        _ = model(batch)
        if sum(len(v) for v in outputs.values()) >= num_calib_rows:
            break

    for h in hooks:
        h.remove()

    result: dict[str, torch.Tensor] = {}
    for name, chunks in outputs.items():
        if chunks:
            result[name] = torch.cat(chunks, dim=0)[:num_calib_rows]
    return result


def cuda_mem() -> dict[str, float]:
    """Return current CUDA memory stats in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
    }


def generation_benchmark(
    model, tokenizer, prompt: str, device: str, max_new_tokens: int = 64
) -> dict:
    """Run a generation benchmark and return timing/quality stats.

    Returns dict with keys: seconds, new_tokens, new_tokens_per_second, generated_text, memory
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    input_tokens = int(encoded["input_ids"].shape[-1])
    output_tokens = int(out.shape[-1])
    new_tokens = max(0, output_tokens - input_tokens)
    return {
        "seconds": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "new_tokens": new_tokens,
        "new_tokens_per_second": new_tokens / max(elapsed, 1e-9),
        "generated_text": generated,
        "memory": cuda_mem(),
    }


def default_corpus() -> str:
    """Default calibration/eval corpus for QPP experiments."""
    return """
Quantization compresses model weights while trying to preserve the output
distribution of the original network. The critical metric for language models
is perplexity, because small changes in internal projections can compound over
many layers. A useful compression method must report real storage, including
metadata and residuals, not only idealized parameter counts.

QPP shared-order compression stores one column order per block of output rows
and then approximates each row with a small set of quantile anchors. This avoids
the fatal per-row permutation cost from the first QPP experiment. Activation
calibration can fit anchors against observed layer inputs so that the compressed
projection preserves outputs rather than only minimizing weight error.
"""


def make_text(repeat: int, extra_path: str | None = None) -> str:
    """Build a repeated calibration text from the default corpus + optional extra."""
    text = default_corpus()
    if extra_path:
        extra = Path(extra_path)
        if extra.exists():
            text += "\n\n" + extra.read_text(encoding="utf-8")
    return (text.strip() + "\n\n") * repeat
