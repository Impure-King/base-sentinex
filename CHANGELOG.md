# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

## [1.0.0.dev0+reference] - 2026-10-10

Reference implementation on the `reference/v2` branch. Not a release: it is not
published to PyPI and the branch is not meant to be merged. The next Sentinex
release will be rebuilt on `main`. This code was first written under the name
Voltax and renamed to Sentinex.

### Added

- `sentinex.ops`: one op table (primitives with a `jax.numpy` implementation and
  per-input VJP rules) that runs eagerly, under `jax.jit`, and in graph mode.
- Graph mode, `sx.function`: traces a Python function once per input signature into
  Sentinex's own SSA graph, caches it, and retraces on new shapes, dtypes, pytree
  structure or Python-scalar values. Includes:
  - symbolic reverse-mode autodiff over the graph (`sx.grad`, `sx.value_and_grad`),
  - passes: identity forwarding, constant folding, CSE, DCE,
  - an op-by-op executor with a global per-op kernel cache, parallel kernel
    compilation, buffer freeing after last use, and a per-op profiler,
  - `jit_compile=True` to lower the same graph to one XLA program,
  - composition with `jax.jit` / `jax.grad` (the graph is replayed through JAX).
- `sentinex.nn`: Dense, Conv2D, MaxPool2D, AvgPool2D, Flatten, Dropout, LayerNorm,
  Embedding, PositionalEmbedding, MultiHeadAttention, Residual, Sequential, MLP,
  TransformerBlock, initializers.
- `sentinex.losses`: cross-entropy, binary cross-entropy, MSE, accuracy.
- `sentinex.optim`: Optax-style gradient transformations (SGD, momentum, Nesterov,
  Adam, AdamW, RMSProp, global-norm clipping, weight decay, chain) and schedules.
- `sentinex.train`: `TrainState`, `make_train_step`, `compile_step` for the run
  modes; `sentinex.data.batches`.
- Tests (pytest): every op's forward pass and gradient against JAX, tracing and
  retracing rules, passes, layers, optimizers and losses against Optax, cross-mode
  training agreement, benchmark-baseline parity, packaging.
- Benchmarks against raw JAX + Optax (`benchmarks/`), with recorded CPU results.
- Design walkthrough (`docs/WALKTHROUGH.md`) and examples.
- Packaging: `pyproject.toml` (hatchling), MIT license, `py.typed`, ruff config,
  GitHub Actions CI on CPU.

### Changed

- Rewritten from scratch with a functional design: layers are configuration and
  parameters are plain pytrees, instead of the 0.0.x object-oriented `Module` API.
- Requires Python 3.12+ and JAX 0.11+.

### Removed

- The 0.0.x code (Equinox-based modules, wrapped Optax optimizers, the model zoo
  and training loop) is not on this branch. It remains in `main`'s history.

## [0.0.1] and [0.0.0]

Earlier object-oriented versions, published to PyPI. See the `main` branch.
