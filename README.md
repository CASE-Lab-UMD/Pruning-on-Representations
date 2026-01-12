# Demystifying When Pruning Works via Representation Hierarchies

This repository investigates **when and why network pruning fails for generative tasks in large language models**, despite strong performance on non-generative benchmarks.  
We provide a systematic study of **inter-layer pruning**, **intra-layer pruning**, and a dedicated **analysis of generation collapse**.


---

## Inter-layer Pruning 

This folder contains experiments that **remove or disable entire layers or blocks** in pretrained LLMs.

Typical settings include:
- Attention layer dropping
- MLP layer dropping
- Block-level pruning

We evaluate these models on:
- Non-generative tasks (e.g., multiple-choice, classification)
- Generative tasks (e.g., chain-of-thought reasoning, open-ended generation)

This part highlights the **mismatch between structural redundancy and generative robustness**.

---

## Intra-layer Pruning 

This folder focuses on **within-layer sparsification**, including:
- Unstructured weight pruning (e.g., WANDA, SparseGPT)
- Structured intra-layer pruning (attention heads / neurons)

Despite preserving most layers, these methods can still cause:
- Severe distributional shift in logits
- Instability during autoregressive decoding

We use identical prompts and decoding settings to enable fair comparison with inter-layer pruning.

---

## Generation Collapse Analysis (`./gen-collapse`)

This folder provides a **fine-grained analysis of why pruning breaks generation**, including:
- Step-wise comparison of hidden states, logits, and probabilities
- Cosine similarity and KL divergence across decoding steps
- Case studies of probability collapse under pruning

Key observation:
> **Small perturbations before the LM head are often amplified by the softmax, leading to catastrophic divergence during generation.**

This analysis explains why pruning appears harmless on single-step evaluation but fails in multi-step decoding.

---

## Key Takeaways

- Structural redundancy in LLMs does **not** imply robustness in generation.
- Generation is particularly sensitive to perturbations introduced by pruning.
- Evaluation limited to non-generative tasks can be misleading.

---

## Models and Benchmarks

- Models: Mistral, LLaMA, Qwen, Mixtral
- Benchmarks:
  - lm-eval-harness (BBH, GSM, etc.)
  - Open-ended generation and long-context prompts
