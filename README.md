# Generation Collapse under Network Pruning

This repository studies when and why pruning hurts autoregressive generation in LLMs, even when non-generative metrics look stable.

It includes three parts:
- `inter-layer`: drop full attention/MLP layers or full transformer blocks.
- `intra-layer`: sparse pruning inside layers (WANDA, SparseGPT, magnitude, etc.).
- `gen-collapse`: analyze hidden/logit/probability divergence between dense and pruned models.

## Project Layout

- `inter-layer/`: layer/block dropping pipeline based on `llmtuner`.
- `intra-layer/`: classic sparsification pipeline with perplexity and zero-shot eval hooks.
- `gen-collapse/`: scripts for cosine/KL-based generation-collapse analysis.
- `REDAME.md`: this file. (Name kept for compatibility with existing repo structure.)

## Environment

Recommended:
- Linux + NVIDIA GPU
- CUDA 11.8+ (or compatible with your installed PyTorch)
- Python 3.10 (3.9 also works in many setups)

Core dependencies:
- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `peft`
- `trl`
- `tqdm`

Install example:

```bash
cd inter-layer
pip install -e .
```

For `intra-layer` and `gen-collapse`, install missing libs as needed in the same Python env:

```bash
pip install torch transformers accelerate datasets tqdm
```

## Quick Start

### 1) Inter-layer pruning (layer/block dropping)

Run from `inter-layer/`.

Example (layer drop):

```bash
bash scripts/dropping/layer_drop.sh
```

Example (block drop):

```bash
bash scripts/dropping/block_drop.sh
```

The default scripts run two phases:
- phase A: compute similarities and write reserved layers.
- phase B (`post_dropping`): export final dropped model config/weights for loading.

### 2) Intra-layer pruning (WANDA / SparseGPT / magnitude)

Run from `intra-layer/`.

```bash
bash scripts/prune.sh
```

Or single run:

```bash
python main.py \
  --model /path/to/model \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --save /path/to/log_dir \
  --save_model /path/to/pruned_model
```

### 3) Generation-collapse analysis

Run from `gen-collapse/` after a dropped model is produced:

```bash
python cosine_layerwise.py --model_root_path ... --dropped_root_path ...
python cosine_final.py --model_root_path ... --dropped_root_path ...
python cosine_subspace.py --model_root_path ... --dropped_root_path ...
```

## Key Arguments

Inter-layer pruning:
- `--prune_method`: `layer_drop` or `block_drop`
- `--layer_drop_method`: `discrete` or `post_dropping`
- `--block_drop_method`: `discrete` / `consecutive` / `post_dropping`
- `--target_layer`: `attn` / `mlp` / `all`
- `--drop_n`: number of layers (or blocks) to drop
- `--n_calibration_samples`: calibration samples used to compute similarity
- `--similarity_cache_file`: cache path to avoid recomputing similarities

Intra-layer pruning:
- `--prune_method`: `wanda` / `sparsegpt` / `magnitude` / `neuron_partition`
- `--sparsity_ratio`: target sparsity
- `--sparsity_type`: `unstructured` / `2:4` / `4:8`
- `--nsamples`: number of calibration samples

## Outputs

Inter-layer outputs usually include:
- `reserved_layers.json`: kept layer indices after dropping decision.
- `layer_mapping.json` (block drop): old-to-new index mapping.
- `config.json`: contains `drop_attn_list` / `drop_mlp_list`.
- model weights/tokenizer files (if `only_update_config=False`).

Intra-layer outputs usually include:
- `log_<method>.txt`: sparsity and perplexity summary.
- saved pruned model directory when `--save_model` is set.

## Reproducibility Notes

- Set `--seed` / `--prune_seed` and keep decoding args fixed (`temperature`, `top_k`, `top_p`) for fair dense-vs-pruned comparison.
- Use the same prompts and max length during generation-collapse analysis.
- Keep `n_calibration_samples` and dataset split fixed across pruning variants.

## Known Issues

- The filename is `REDAME.md` (typo). Rename to `README.md` if preferred.
- Some scripts assume local paths like `your_model_root_path`; update before running.
- `intra-layer/main.py` imports modules (`lib.eval`, `lib.ablate`) that must exist in your local copy/environment.
- `gen-collapse/cosine_subspace.py` references `args.mode` but does not define it; add this argument or adjust the log filename before running.

## Citation

If this repo helps your research, please cite your corresponding paper/project for this codebase.
