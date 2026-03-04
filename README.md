# Demystifying When Pruning Works via Representation Hierarchies

This repository is the codebase for the paper **“Demystifying When Pruning Works via Representation Hierarchies” (ICML 2026 submission)**.

The core question is why pruning can preserve non-generative performance while degrading autoregressive generation.
We analyze pruning through three sequential representation spaces:
- **Embedding space**: hidden states
- **Logit space**: pre-softmax outputs
- **Probability space**: post-softmax distributions

## Repository Structure

- `inter-layer/`: layer/block dropping pipeline.
- `intra-layer/`: intra-layer sparsification (WANDA / SparseGPT / magnitude / neuron partition).
- `representation-analysis/`: paper-aligned analysis scripts for representation hierarchy.

## Environment

Recommended:
- Linux + NVIDIA GPU
- CUDA-compatible PyTorch
- Python 3.9+

Core dependencies:
- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `tqdm`

Example:

```bash
cd inter-layer
pip install -e .

# for analysis scripts
pip install torch transformers accelerate datasets tqdm
```

## Analysis Scripts (Paper-Aligned)

All three scripts support:
- `--analysis_mode dropped` (dense vs dropped behavior in one model via drop masks)
- `--analysis_mode pruned` (dense model vs separately loaded pruned model)

### 1) Layerwise transition analysis

File: `representation-analysis/transition_layerwise_compare.py`

Purpose:
- Compare **attn/mlp sublayer transitions** at the same layer and same context.
- Log transition metrics in embedding/logit/probability spaces.

### 2) Generation-time divergence analysis

File: `representation-analysis/compare_generation_metrics.py`

Purpose:
- Compare dense vs target trajectories across decoding steps.
- Report cosine/KL and second-order estimates tied to the paper’s Section 6 formulas.

### 3) Task subspace analysis (MCQ)

File: `representation-analysis/compare_mcq_subspace_metrics.py`

Purpose:
- Compare global vocabulary-space behavior vs answer-option subspace behavior.
- Mirrors the non-generative subspace robustness discussion in the paper.

## Quick Start

Run from `representation-analysis/`.

### Dropped mode

```bash
python transition_layerwise_compare.py \
  --analysis_mode dropped \
  --model_root_path /path/to/model_root \
  --model_postfix Qwen/Qwen2.5-7B-Instruct \
  --dropped_root_path /path/to/dropped_results \
  --target_layer attn \
  --drop_n 8

python compare_generation_metrics.py \
  --analysis_mode dropped \
  --model_root_path /path/to/model_root \
  --model_postfix Qwen/Qwen2.5-7B-Instruct \
  --dropped_root_path /path/to/dropped_results \
  --target_layer attn \
  --drop_n 8

python compare_mcq_subspace_metrics.py \
  --analysis_mode dropped \
  --model_root_path /path/to/model_root \
  --model_postfix Qwen/Qwen2.5-7B-Instruct \
  --dropped_root_path /path/to/dropped_results \
  --target_layer attn \
  --drop_n 8
```

### Pruned mode

```bash
python transition_layerwise_compare.py \
  --analysis_mode pruned \
  --model_name /path/to/dense_model \
  --pruned_model_name /path/to/pruned_model

python compare_generation_metrics.py \
  --analysis_mode pruned \
  --model_name /path/to/dense_model \
  --pruned_model_name /path/to/pruned_model

python compare_mcq_subspace_metrics.py \
  --analysis_mode pruned \
  --model_name /path/to/dense_model \
  --pruned_model_name /path/to/pruned_model
```

## Notes on Metric Definitions

- Probability-space metrics use `softmax(logits / T)` where `T` is analysis temperature.
- If generation is greedy (`temperature=0`), analysis falls back to `T=1.0` for stable probability-space comparison.
- In generation analysis, KL and cosine estimates are logged in the second-order form with `1 / (2T^2)` scaling.

## Outputs

Analysis logs are written under `representation-analysis/cosine_logs/` by default, with subfolders per script/mode/temperature.

## Citation

If this repository helps your research, please cite the corresponding paper.
