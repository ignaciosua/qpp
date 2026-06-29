# Contributing to QPP

Thanks for your interest! QPP is research code looking for community help. Here's how to contribute.

## Getting Started

```bash
git clone git@github.com:ignaciosua/qpp.git
cd qpp
pip install -e ".[dev]"
pytest tests/ -v
```

## What to Work On

### 🔰 Good First Issues
- Add support for a new model architecture (Llama, Mistral, DeepSeek)
- Benchmark QPP on a model we haven't tested yet
- Improve type annotations
- Add more unit tests

### 🔧 Core Development
- **Fused Triton/CUDA kernel** for `QPPCompressedLinear.forward()` — this is the biggest impact item
- VRAM reduction: free dense weights after compression instead of keeping them
- QPP for MLP layers (Gaussian distribution needs different approach)
- Anchor quantization (currently FP16, explore FP8/INT8)

### 📚 Research Questions
- Can the ordering be learned instead of imposed?
- QPP + quantization-aware training (QAT)
- Combining QPP with sparse attention / MoE
- Theoretical compression bounds for quantile curve approximation

## Development Workflow

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes, add tests
4. Run tests: `pytest tests/ -v`
5. Run lint: `ruff check src/ tests/`
6. Commit with a clear message
7. Push and open a PR

## Code Style

- Python 3.10+ with type hints
- numpy-style docstrings
- Line length: 120 chars
- Use `ruff format` and `ruff check` (no separate formatter needed)

## Running Tests

```bash
# CPU-only unit tests (fast, no GPU/transformers needed)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=qpp --cov-report=term
```

## Project Philosophy

QPP is **parametric compression** — reducing the *number of parameters*, not their bit precision. It's orthogonal to traditional quantization. The codebase keeps this distinction clear:

- `src/qpp/compression.py` — pure algorithm, numpy-only, no ML framework dependency
- `src/qpp/runtime.py` — torch nn.Modules for inference
- `src/qpp/hybrid.py` — HuggingFace integration (optional dependency)

New features should respect these dependency boundaries.

## Questions?

Open a [GitHub Discussion](https://github.com/ignaciosua/qpp/discussions) for research questions, or an Issue for bugs/features.
