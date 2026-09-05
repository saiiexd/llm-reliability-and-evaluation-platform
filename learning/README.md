# Learning

This directory contains temporary educational implementations used to develop a working understanding of the underlying concepts and mechanisms that the platform will rely on. These are not platform components.

## Purpose

Understanding a system well enough to evaluate it requires understanding how it works internally. Before integrating a concept into production platform code, it is implemented here from first principles to confirm genuine understanding rather than assumed understanding.

## Important Constraints

- Code in `learning/` is not automatically production code. It exists to develop understanding, not to be shipped.
- Implementations here are allowed to be incomplete, experimental, and imperfect.
- When a concept is understood well enough to inform the platform, the relevant insight is documented in `research/notes/` and the platform implementation is written independently in `src/`.
- Do not copy learning code directly into `src/`. The platform implementation should be a fresh, informed implementation, not a copy of the educational version.

## Contents

```
learning/
  neural_network_from_scratch/    Build a feedforward network from scratch using only NumPy.
  transformer_from_scratch/       Build a minimal transformer from scratch using only NumPy/PyTorch.
```

## Status

Neither learning project has been started yet. They will be implemented incrementally as the research phase identifies specific conceptual gaps that require hands-on implementation to resolve.
