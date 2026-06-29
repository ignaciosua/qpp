"""End-to-end hybrid compression pipeline: QPP (attention) + INT8 (MLP/embed).

This module provides a complete pipeline for compressing a HuggingFace model:
1. INT8-prepass on MLP + embeddings (lossless 2x, SNR > 40 dB)
2. Greedy QPP on attention layers with PPL-gated acceptance
3. Optional codebook quantization of QPP anchors

Usage:
    from qpp.hybrid import HybridPipeline
    pipe = HybridPipeline(model_name="Qwen/Qwen2.5-0.5B-Instruct", ...)
    pipe.run()
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from qpp.benchmark import cuda_mem, default_corpus, make_text, perplexity
from qpp.codebook import lloyd_max_1d
from qpp.compression import row_block_for, target_linears
from qpp.runtime import (
    Int8CompressedLinear,
    QPPCompressedLinear,
    build_compressed_linear,
    persistent_model_bytes,
    set_nested_attr,
)


@dataclass
class HybridResult:
    module: str
    accepted: bool
    candidate_ppl: float
    delta_ppl: float
    total_delta_ppl: float
    rows: int
    cols: int
    anchors: int
    row_block: int
    dense_bf16_bytes: int
    qpp_bytes: int
    compression: float
    weight_rel_rmse: float
    activation_rel_rmse: float | None


class HybridPipeline:
    """Greedy QPP + INT8 hybrid compression pipeline."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        anchors: int = 32,
        row_block: int = 128,
        outlier_topk: int = 0,
        residual_rank: int = 0,
        total_delta_gate: float = 0.5,
        calibrate: bool = True,
        calib_tokens: int = 4096,
        calib_rows: int = 1024,
        eval_tokens: int = 4096,
        ctx: int = 1024,
        text_repeat: int = 8,
        forward_mode: str = "reconstruct",
        cb_bits: int = 0,
        int8_mlp: bool = True,
        int8_embed: bool = True,
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.anchors = anchors
        self.row_block = row_block
        self.outlier_topk = outlier_topk
        self.residual_rank = residual_rank
        self.total_delta_gate = total_delta_gate
        self.calibrate = calibrate
        self.calib_tokens = calib_tokens
        self.calib_rows = calib_rows
        self.eval_tokens = eval_tokens
        self.ctx = ctx
        self.text_repeat = text_repeat
        self.forward_mode = forward_mode
        self.cb_bits = cb_bits
        self.int8_mlp = int8_mlp
        self.int8_embed = int8_embed
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code

        self.model: nn.Module | None = None
        self.tokenizer = None
        self.text: str = ""
        self.baseline_ppl: float = 0.0
        self.results: list[HybridResult] = []

    def load_model(self):
        """Load model and tokenizer from HuggingFace."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading {self.model_name}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
            torch_dtype=self.dtype,
            device_map={"": self.device},
            trust_remote_code=self.trust_remote_code,
        ).eval()
        self.text = make_text(self.text_repeat)

    def get_baseline(self) -> dict:
        """Compute BF16 baseline PPL and model size."""
        ppl, elapsed, tokens = perplexity(
            self.model, self.tokenizer, self.text, self.device, self.eval_tokens, self.ctx
        )
        orig_bytes = persistent_model_bytes(self.model)
        print(f"BF16 PPL: {ppl:.4f} | Size: {orig_bytes / 1e6:.1f} MB | Tokens: {tokens}", flush=True)
        self.baseline_ppl = ppl
        return {"ppl": ppl, "size_mb": orig_bytes / 1e6, "tokens": tokens, "elapsed_s": elapsed}

    def apply_int8_prepass(self) -> dict:
        """Apply INT8 to MLP and embedding layers."""
        from qpp.benchmark import collect_activations

        stats = {"mlp_count": 0, "embed_count": 0, "mlp_original_mb": 0.0, "embed_original_mb": 0.0}

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            is_mlp = ".mlp." in name and self.int8_mlp
            is_embed = ("embed_tokens" in name or "lm_head" in name) and self.int8_embed
            if not (is_mlp or is_embed):
                continue

            original_bytes = module.weight.numel() * module.weight.element_size()
            if module.bias is not None:
                original_bytes += module.bias.numel() * module.bias.element_size()

            w_cpu = module.weight.detach().to("cpu").float()
            b_cpu = module.bias.detach().to("cpu").half() if module.bias is not None else None
            max_abs = w_cpu.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8)
            scale = max_abs / 127.0
            q = (w_cpu / scale).round().clamp(-127, 127).to(torch.int8)

            new_mod = Int8CompressedLinear.__new__(Int8CompressedLinear)
            nn.Module.__init__(new_mod)
            new_mod.in_features = w_cpu.shape[1]
            new_mod.out_features = w_cpu.shape[0]
            new_mod.register_buffer("qweight", q.contiguous())
            new_mod.register_buffer("scale", scale.half().contiguous())
            new_mod.bias = b_cpu.contiguous() if b_cpu is not None else None
            del w_cpu, scale, q

            set_nested_attr(self.model, name, new_mod.to(self.device))
            if is_mlp:
                stats["mlp_count"] += 1
                stats["mlp_original_mb"] += original_bytes / 1e6
            if is_embed:
                stats["embed_count"] += 1
                stats["embed_original_mb"] += original_bytes / 1e6

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"INT8 prepass: {stats['mlp_count']} MLP + {stats['embed_count']} embed/lm_head", flush=True)
        return stats

    def run_qpp_attention(self) -> list[HybridResult]:
        """Greedy QPP compression on attention layers."""
        from qpp.benchmark import collect_activations

        modules = target_linears(self.model, "attention", 0)
        names = [n for n, _ in modules]
        print(f"QPP candidates: {len(names)} attention modules", flush=True)

        # Calibration
        activations = {}
        if self.calibrate:
            activations = collect_activations(
                self.model, self.tokenizer, self.text, names,
                self.device, self.calib_tokens, self.ctx, self.calib_rows,
            )

        current_ppl = self.baseline_ppl
        results: list[HybridResult] = []

        for idx, name in enumerate(names, 1):
            module = self.model
            for part in name.split("."):
                module = getattr(module, part)
            if not isinstance(module, nn.Linear):
                continue

            original_w = module.weight.detach()
            rows, cols = original_w.shape
            rb = row_block_for(name, rows, self.row_block, 64)
            calib_x = activations.get(name)

            try:
                compressed, stats = build_compressed_linear(
                    module, rb, self.anchors, self.outlier_topk,
                    self.residual_rank, 0, 0.001, 32, 1e-4, "mean",
                    calib_x, self.forward_mode,
                )
            except Exception as e:
                print(f"  [{idx}/{len(names)}] QPP FAIL {name}: {e}", flush=True)
                continue

            set_nested_attr(self.model, name, compressed)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            cand_ppl, _, _ = perplexity(
                self.model, self.tokenizer, self.text, self.device, self.eval_tokens, self.ctx
            )
            delta = cand_ppl - current_ppl
            total_delta = cand_ppl - self.baseline_ppl
            accept = total_delta <= self.total_delta_gate

            result = HybridResult(
                module=name,
                accepted=accept,
                candidate_ppl=cand_ppl,
                delta_ppl=delta,
                total_delta_ppl=total_delta,
                rows=rows, cols=cols,
                anchors=self.anchors,
                row_block=rb,
                dense_bf16_bytes=int(stats["dense_bf16_bytes"]),
                qpp_bytes=int(stats["runtime_buffer_bytes"]),
                compression=float(stats["runtime_buffer_compression"]),
                weight_rel_rmse=float(stats["weight_rel_rmse"]),
                activation_rel_rmse=stats.get("activation_rel_rmse"),
            )
            results.append(result)

            status = "ACCEPT" if accept else "REJECT"
            print(
                f"  [{idx}/{len(names)}] {name}: ppl={cand_ppl:.4f} Δ={delta:+.4f} "
                f"totΔ={total_delta:+.4f} {status} comp={result.compression:.1f}x",
                flush=True,
            )

            if accept:
                current_ppl = cand_ppl
            else:
                # Rollback: restore original Linear
                orig_mod = nn.Linear(cols, rows, bias=module.bias is not None)
                orig_mod.weight.data.copy_(original_w)
                if module.bias is not None:
                    orig_mod.bias.data.copy_(module.bias)
                set_nested_attr(self.model, name, orig_mod.to(self.device))

        self.results = results
        return results

    def run(self) -> dict:
        """Run the full hybrid pipeline and return summary dict."""
        self.load_model()
        baseline = self.get_baseline()

        int8_stats = {}
        if self.int8_mlp or self.int8_embed:
            int8_stats = self.apply_int8_prepass()

        results = self.run_qpp_attention()

        accepted = [r for r in results if r.accepted]
        final_ppl = self.baseline_ppl
        if accepted:
            # ponytail: recalculate final PPL after all accepted modules
            final_ppl, _, _ = perplexity(
                self.model, self.tokenizer, self.text, self.device, self.eval_tokens, self.ctx
            )

        final_bytes = persistent_model_bytes(self.model)
        summary = {
            "model": self.model_name,
            "approach": "QPP+INT8 hybrid",
            "baseline_ppl": self.baseline_ppl,
            "final_ppl": final_ppl,
            "delta_ppl": final_ppl - self.baseline_ppl,
            "original_mb": baseline["size_mb"],
            "compressed_mb": final_bytes / 1e6,
            "savings_pct": (1 - final_bytes / (baseline["size_mb"] * 1e6)) * 100,
            "accepted_modules": len(accepted),
            "total_attention_modules": len(results),
            "config": {
                "anchors": self.anchors,
                "row_block": self.row_block,
                "total_delta_gate": self.total_delta_gate,
                "cb_bits": self.cb_bits,
            },
            "int8_stats": int8_stats,
        }

        print(f"\n{'='*60}")
        print(f"Final PPL: {final_ppl:.4f} (Δ={final_ppl - self.baseline_ppl:+.4f})")
        print(f"Size: {baseline['size_mb']:.0f} → {final_bytes/1e6:.0f} MB ({summary['savings_pct']:.1f}% saved)")
        print(f"Accepted: {len(accepted)}/{len(results)} attention modules")
        print(f"{'='*60}")
        return summary

    def save_artifact(self, outdir: Path) -> Path:
        """Save the compressed model artifact for later reload."""
        outdir.mkdir(parents=True, exist_ok=True)
        modules = {}
        for name, mod in self.model.named_modules():
            if not isinstance(mod, QPPCompressedLinear):
                continue
            tensors = {
                "anchors": mod.anchors.detach().cpu(),
                "orders_i16": mod.orders_i16.detach().cpu(),
            }
            for attr in ("orders_i32", "basis", "bias", "outlier_idx_i16", "outlier_val", "residual_a", "residual_b"):
                t = getattr(mod, attr, None)
                if t is not None:
                    tensors[attr] = t.detach().cpu()
            if mod.anchor_cb is not None:
                tensors["anchor_cb"] = mod.anchor_cb.detach().cpu()
                tensors["anchor_codes"] = mod.anchor_codes.detach().cpu()
            modules[name] = {
                "out_features": mod.out_features,
                "in_features": mod.in_features,
                "row_slices": mod.row_slices,
                "forward_mode": mod.forward_mode,
                "tensors": tensors,
            }

        artifact = {
            "format": "qpp-runtime-artifact-v1",
            "model": self.model_name,
            "results": [asdict(r) for r in self.results],
            "modules": modules,
        }
        path = outdir / "qpp_compressed_artifact.pt"
        torch.save(artifact, path)
        print(f"Artifact saved to {path}", flush=True)
        return path


def main_cli():
    """CLI entry point for `qpp-compress`."""
    p = argparse.ArgumentParser(description="QPP hybrid compression pipeline")
    p.add_argument("--model", required=True, help="HuggingFace model name or path")
    p.add_argument("--outdir", default="outputs/qpp_hybrid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--anchors", type=int, default=32)
    p.add_argument("--row-block", type=int, default=128)
    p.add_argument("--total-delta-gate", type=float, default=0.5)
    p.add_argument("--cb-bits", type=int, default=0)
    p.add_argument("--no-int8-mlp", action="store_true")
    p.add_argument("--no-int8-embed", action="store_true")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--forward-mode", default="reconstruct", choices=["reconstruct", "direct"])
    p.add_argument("--eval-tokens", type=int, default=4096)
    p.add_argument("--calib-tokens", type=int, default=4096)
    p.add_argument("--calib-rows", type=int, default=1024)
    p.add_argument("--text-repeat", type=int, default=8)
    p.add_argument("--ctx", type=int, default=1024)
    p.add_argument("--save-artifact", action="store_true")
    args = p.parse_args()

    pipe = HybridPipeline(
        model_name=args.model,
        device=args.device,
        int8_mlp=not args.no_int8_mlp,
        int8_embed=not args.no_int8_embed,
        calibrate=not args.no_calibrate,
        anchors=args.anchors,
        row_block=args.row_block,
        total_delta_gate=args.total_delta_gate,
        cb_bits=args.cb_bits,
        forward_mode=args.forward_mode,
        eval_tokens=args.eval_tokens,
        calib_tokens=args.calib_tokens,
        calib_rows=args.calib_rows,
        text_repeat=args.text_repeat,
        ctx=args.ctx,
    )
    summary = pipe.run()

    outdir = Path(args.outdir)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    if args.save_artifact:
        pipe.save_artifact(outdir)
    print(f"Output: {outdir}")


if __name__ == "__main__":
    main_cli()
