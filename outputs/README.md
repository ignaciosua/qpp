# Outputs directory

This directory is gitignored except for reference result files.
Large experiment outputs (model artifacts, safetensors, JSON logs) live here locally.

## Reproducing results

Run the hybrid pipeline:
```bash
qpp-compress --model Qwen/Qwen2.5-0.5B-Instruct --outdir outputs/my_run --save-artifact
```

Reference results from the Qwen3-4B paper experiments are in `reference_results.json`.
