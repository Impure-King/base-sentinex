# Benchmark results

Apple M5 (10 cores, 24 GB), JAX 0.11.2 on the CPU backend, Python 3.12, macOS 27.
Run on 2026-10-10 at commit `d132355`. Raw outputs:

* [results/2026-10-10-cpu.md](results/2026-10-10-cpu.md) (training steps, `bench.py`)
* [results/2026-10-10-cpu-overhead.md](results/2026-10-10-cpu-overhead.md) (`overhead.py`)
* [results/2026-10-10-cpu-first-call.md](results/2026-10-10-cpu-first-call.md) (`first_call.py`)

The `.json` files next to them have every timed round and the environment.

## How it was measured

* Same initial weights and the same 4 synthetic batches (cycled) in every mode. Adam,
  lr 1e-3, float32.
* The raw JAX + Optax baselines (`workloads.py`) are written independently with
  jnp/lax/jax.nn/optax. `tests/test_benchmark_parity.py` checks they compute the same
  losses as the Voltax models.
* After the first call and 10 warmup steps, 15 timed rounds per mode, interleaved
  across modes in rotating order. Each round is ~0.3 s of raw JAX work; all modes run
  the same steps per round. A round is timed from the first dispatch to
  `block_until_ready` on the last result.
* The machine was otherwise idle: 1-minute load 3.7 at the start (the 5- and 15-minute
  averages were still high from earlier jobs). A first attempt that overlapped another
  CPU-heavy job was discarded.
* The loss after the first step is identical in every mode for every model (same
  weights, same batch). Final losses agree closely after hundreds to thousands of Adam
  steps (the transformer ends at 0.025635 in every mode). The CNN drifts most: 0.00341 raw,
  0.00333 Voltax jit, 0.00325 graph mode after 356 steps, because fused and unfused
  convolutions round differently and memorizing random labels amplifies it.

## Training step time

Median step time in ms, with the change versus raw JAX + Optax.

| workload | raw JAX + Optax | Voltax `jax.jit` | Voltax graph → XLA | Voltax graph mode | Voltax eager |
|---|---:|---:|---:|---:|---:|
| MLP 784-512-512-10, batch 512 | 4.26 | 4.23 (−0.8%) | 4.17 (−2.0%) | 5.60 (+31%) | 26.2 (+515%) |
| same MLP, batch 32 | 0.713 | 0.723 (+1.4%) | 0.727 (+2.0%) | 2.17 (+204%) | 6.31 (+785%) |
| CNN, 28×28×1, batch 128 | 20.0 | 20.6 (+3.0%) | 20.7 (+3.4%) | 26.8 (+34%) | 32.2 (+61%) |
| transformer, 2 layers, d=128, seq 64, batch 32 | 25.8 | 24.0 (−7.2%) | 26.6 (+3.2%) | 32.3 (+25%) | 51.0 (+97%) |

Throughput from the medians:

| workload | raw JAX + Optax | Voltax `jax.jit` | Voltax graph mode |
|---|---:|---:|---:|
| MLP, batch 512 | 120,258 ex/s | 121,194 ex/s | 91,505 ex/s |
| MLP, batch 32 | 44,867 ex/s | 44,238 ex/s | 14,782 ex/s |
| CNN, batch 128 | 6,393 ex/s | 6,207 ex/s | 4,775 ex/s |
| transformer, batch 32 | 1,239 seq/s (79k tokens/s) | 1,335 seq/s (85k tokens/s) | 991 seq/s (63k tokens/s) |

Kernels per step (XLA's fused kernels for the jit modes; one-op kernels for graph mode):

| workload | raw JAX + Optax | Voltax `jax.jit` | Voltax graph mode |
|---|---:|---:|---:|
| MLP | 50 | 47 | 143 |
| CNN | 85 | 83 | 185 |
| transformer | 300 | 297 | 895 |

## What the numbers say

**Voltax under `jax.jit` costs nothing at run time.** It is within −7% to +3% of hand-written
JAX + Optax. That's expected: under `jax.jit` Voltax ops are just `jax.numpy` calls, and both
versions become one XLA program. The programs are not identical, because the model code differs in
equivalent ways. For example, Optax's cross-entropy gathers the label logit while Voltax multiplies by
a one-hot mask, and Optax's Adam guards its step counter against overflow. On the transformer,
swapping the forward pass, loss or optimizer one at a time between the two versions moves the step
time by −2.7% to +1.2%; all three together give −7.4%. So the gap comes from XLA's whole-program
fusion decisions, not from any single component.

**A Voltax graph lowered to XLA is as good as `jax.jit`** (`graph → XLA`, within 4% of raw).
The graph IR doesn't lose anything that XLA needs.

**Graph mode costs 25–34% on these CPU steps, and 3x on the small-batch MLP.** It runs 2–3x more
kernels than XLA (no fusion), and each one pays a dispatch and a full round trip through memory.
With batch 512, the matmuls dominate and run the same way in every mode, so the extra kernels add
about 1.3 ms to a 4.3 ms step. With batch 32, the step is so small that the 142 dispatches *are* the
step: about 1.45 ms extra, roughly 10 µs per kernel.

**Graph mode beats eager by 1.2x to 4.7x.** Tracing once removes the model's Python code and JAX's
op-by-op autodiff from every step. This is the reason `tf.function` exists.

**The CNN is dominated by a few XLA:CPU kernels.** Graph mode's profiler (`Executable.profile`)
attributes 64% of the CNN step to the convolution backward kernels and the max-pool backward
(select-and-scatter). They cost the same in every mode, which is why eager is only 61% slower there.

## Per-op overhead (`overhead.py`)

A chain of `x = x * 1.0001 + 0.0001`, time per call in µs:

| array | ops | eager | `jax.jit` | graph mode | graph µs/op |
|---:|---:|---:|---:|---:|---:|
| 1,000 floats | 64 | 330 | 3.1 | 217 | 3.4 |
| 1,000 floats | 512 | 2,605 | 24 | 1,701 | 3.3 |
| 1,000,000 floats | 64 | 2,862 | 205 | 2,807 | 43.9 |
| 1,000,000 floats | 512 | 28,233 | 5,389 | 31,875 | 62.3 |

* Small arrays: graph mode costs ~3.4 µs per op, all dispatch. `jax.jit` runs 64 ops in 3.1 µs
  total because they are one fused kernel.
* Large arrays: each graph-mode op reads and writes 4 MB, ~44 µs. Fused, the 64-op chain takes
  205 µs instead of 2,807 µs (13.7x). At 512 ops the fused loop becomes compute-bound and the gap
  shrinks to 5.9x.

## First call (`first_call.py`)

| step | Voltax trace + passes | kernel compile, 1 thread | kernel compile, 8 threads | `jax.jit` trace + compile | graph-mode cache hit |
|---|---:|---:|---:|---:|---:|
| MLP (143 kernels, 94 unique) | 0.06 s | 0.68 s | 0.17 s | 0.08 s | 8 µs (22 arg leaves) |
| CNN (185, 152 unique) | 0.07 s | 1.54 s | 0.38 s | 0.33 s | 12 µs (28 leaves) |
| transformer (895, 255 unique) | 0.12 s | 2.19 s | 0.59 s | 0.37 s | 42 µs (94 leaves) |

Graph mode's first call is *slower* than `jax.jit`'s: compiling a few hundred tiny XLA programs
costs more than compiling one big one, because each compile has fixed overhead. Compiling on a
thread pool cuts it by about 4x. TensorFlow avoids this cost entirely by shipping precompiled
kernels. Once compiled, kernels are shared across graphs and retraces. A cache hit (flatten the
arguments, build the signature key, look it up) costs 8–42 µs in Python, growing with the number
of argument leaves.

## Not measured

* GPU. The code is device-agnostic, but this machine has no CUDA. On a GPU, fusion and launch
  overhead usually matter more, so expect graph mode to fall further behind `jax.jit`; that is a
  prediction, not a measurement.
* Real datasets. Throughput doesn't depend on the data values, so the batches are synthetic.

## Re-running

```bash
uv run python benchmarks/bench.py --wait-for-idle 4
```

```bash
uv run python benchmarks/overhead.py
```

```bash
uv run python benchmarks/first_call.py
```

Close other heavy programs first. On a fanless laptop, sustained load also throttles the CPU, so
compare ratios within a run rather than absolute times across runs.
