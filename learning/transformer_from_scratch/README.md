# Transformer From Scratch

## Purpose

Implement a minimal transformer architecture from scratch to develop a precise mechanical understanding of the attention mechanism, positional encoding, multi-head attention, layer normalization, and the encoder-decoder structure. Implementation will use NumPy and/or PyTorch at the tensor operation level, without relying on high-level transformer libraries.

## Why This Matters for the Platform

Evaluating hallucination, faithfulness, and groundedness in LLM outputs requires understanding how transformers attend to context, how they generate tokens, and where in the generation process failures are most likely to originate. This implementation is a prerequisite for reasoning about failure modes at a mechanistic level.

## Intended Scope

- Scaled dot-product attention implemented manually.
- Multi-head attention with explicit head splitting and concatenation.
- Positional encoding (sinusoidal, following the original Vaswani et al. formulation).
- A minimal encoder block or encoder-decoder architecture sufficient to understand the full mechanism.
- Trained on a small sequence-to-sequence task to verify correctness (e.g., copying, reversing a sequence).
- Documented relationship between each implementation step and the original paper.

## What This Is Not

This is not a production language model. It will not be trained at scale, fine-tuned, or integrated into any evaluation pipeline. Its purpose is strictly educational.

## Reference

Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
https://arxiv.org/abs/1706.03762

## Status

Not yet started.
