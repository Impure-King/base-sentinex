# Sentinex

A small functional deep-learning library on JAX, with its own graph mode.

You write a model once with `sentinex.ops`, `sentinex.nn`, `sentinex.losses` and
`sentinex.optim`. Then you choose how the training step runs:

| mode | what happens |
|---|---|
| eager | call the step directly; every op is dispatched from Python |
| `jax.jit` | JAX traces the step to a jaxpr; XLA compiles and fuses it into one program |
| `sx.function` | Sentinex traces the step into **its own graph**, caches it per input signature, and runs it **op by op** (like `tf.function` without `jit_compile`) |
| `sx.function(jit_compile=True)` | the same Sentinex graph, lowered to XLA as one program |

Graph mode is the reason this repo exists. It has its own tracer, SSA graph IR,
symbolic reverse-mode autodiff, optimization passes (constant folding, CSE, DCE)
and an op-by-op executor, so you can compare it with the jaxpr → XLA path on the
same model. [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) explains how it works.

## Quick look

```python
import jax
import sentinex as sx
from sentinex import losses, nn, optim, train

model = nn.Sequential([nn.Dense(512), nn.relu, nn.Dense(10)])
params = model.init(jax.random.PRNGKey(0), (None, 784))   # None = any batch size
opt = optim.adam(1e-3)
x = jax.random.normal(jax.random.PRNGKey(1), (128, 784))
y = jax.random.randint(jax.random.PRNGKey(2), (128,), 0, 10)

def loss_fn(params, batch, key):
    x, y = batch
    return losses.cross_entropy(model.apply(params, x), y)

step = train.make_train_step(loss_fn, opt)   # pure: (state, batch) -> (state, metrics)
step = sx.function(step)                      # or jax.jit(step), or leave it eager

state = train.TrainState.create(params, opt)
state, metrics = step(state, (x, y))          # first call traces; later calls reuse the graph

print(step.graph(state, (x, y)).pretty())     # the Sentinex graph that runs
```

## Setup

Needs [uv](https://docs.astral.sh/uv/). JAX runs on CPU here; the code is
device-agnostic and runs on GPU with a CUDA build of jaxlib.

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run python examples/quickstart.py
```

```bash
uv run python examples/inspect_graph.py
```

## Benchmarks

```bash
uv run python benchmarks/bench.py --wait-for-idle 3
```

```bash
uv run python benchmarks/overhead.py
```

`bench.py` trains an MLP, a CNN and a small transformer with raw JAX + Optax and
with Sentinex in every mode, from the same weights on the same batches. Results go
to `benchmarks/results/`. On an Apple M5 (CPU):

* Sentinex under `jax.jit` runs within −7% to +3% of hand-written JAX + Optax
  (about 120k examples/s for a 670k-parameter MLP at batch 512).
* Graph mode is 25–34% slower than XLA on these steps, and 3x slower on a
  small-batch step where per-op dispatch dominates. It is 1.2–4.7x faster than eager.

Full tables and how to read them: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Layout

```
src/sentinex/
  primitive.py      Primitive: impl + VJP rules + the eager/graph dispatch
  ops.py            the op set models are written with
  autodiff.py       sx.grad / sx.value_and_grad (JAX AD, or graph AD while tracing)
  graph/            graph mode
    tracing.py      Symbol and GraphBuilder: recording ops
    ir.py           Aval, Literal, Node, Graph (+ pretty printer)
    kernels.py      shape inference and the per-op kernel cache
    autodiff.py     symbolic reverse mode over the graph
    passes.py       identity forwarding, constant folding, CSE, DCE
    executor.py     op-by-op executor, and the interpreter used for XLA lowering
    function.py     sx.function: signature cache, retracing
  nn/               layers and initializers
  losses.py  optim.py  train.py  data.py
tests/              pytest suite (ops and gradients vs JAX, optimizers vs Optax, ...)
benchmarks/         workloads, raw JAX baselines, bench.py, overhead.py, results/
examples/           quickstart.py, inspect_graph.py
docs/WALKTHROUGH.md design walkthrough
```

## Limitations

Graph mode has no control-flow primitives (`cond`, `while`, `scan`) and only
static shapes. The executor runs one op at a time from Python. Layers have no
mutable state (no BatchNorm). See the walkthrough for the trade-offs.
