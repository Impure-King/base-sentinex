"""Look inside both compilation paths for the same training step.

    uv run python examples/inspect_graph.py

Prints, for one small Dense + softmax-cross-entropy SGD step:
  1. the Voltax graph as traced, and the op counts after each optimization pass
  2. the optimized Voltax graph (what the executor runs, one kernel per op)
  3. the jaxpr JAX traces for jax.jit of the same function
  4. what XLA turns that into: the fused kernels of the optimized HLO
  5. retracing: which calls reuse the cached graph and which trace again
  6. a per-op profile of the graph executor
"""

import re

import jax
import jax.numpy as jnp

import voltax as vx
from voltax import losses, ops


def loss_fn(params, x, y):
    logits = ops.relu(x @ params["w"] + params["b"])
    return losses.cross_entropy(logits, y)


def sgd_step(params, x, y, lr):
    loss, grads = vx.value_and_grad(loss_fn)(params, x, y)
    return jax.tree.map(lambda p, g: p - lr * g, params, grads), loss


def banner(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    key = jax.random.PRNGKey(0)
    params = {"w": jax.random.normal(key, (16, 4)) * 0.1, "b": jnp.zeros(4)}
    x, y = jax.random.normal(key, (8, 16)), jnp.arange(8) % 4

    step = vx.function(sgd_step)
    cf = step.get_concrete_function(params, x, y, 0.1)

    banner("1. Voltax graph as traced (forward + symbolic backward), and after each pass")
    print(cf.raw_graph.summary())
    print("ops after each pass:", cf.pass_stats)

    banner("2. Optimized Voltax graph: the executor runs these one by one")
    print(cf.graph.pretty())

    banner("3. The jaxpr JAX traces for jax.jit(sgd_step)")
    jaxpr = jax.make_jaxpr(sgd_step, static_argnums=3)(params, x, y, 0.1)
    print(jaxpr)

    banner("4. What XLA runs: the optimized HLO entry computation (fused kernels)")
    hlo = jax.jit(sgd_step, static_argnums=3).lower(params, x, y, 0.1).compile().as_text()
    entry = hlo[hlo.index("ENTRY") :]
    entry = entry[: entry.index("\n}") + 2]
    for line in entry.splitlines():
        print(re.sub(r", metadata=\{.*?\}", "", line)[:150])
    n_fusions = len(re.findall(r"\bfusion\(", entry))
    print(f"\n{len(cf.graph.ops)} Voltax kernels per step vs {n_fusions} XLA fusions (plus a few unfused dots).")

    banner("5. Caching and retracing")
    step(params, x, y, 0.1)
    print("same shapes, same lr      -> trace_count", step.trace_count)
    step(params, x + 1, y, 0.1)
    print("new array values          -> trace_count", step.trace_count, "(values are not in the signature)")
    step(params, x, y, 0.05)
    print("new Python float lr       -> trace_count", step.trace_count, "(Python scalars are static)")
    step(params, x[:4], y[:4], 0.1)
    print("new batch size            -> trace_count", step.trace_count, "(shapes are in the signature)")
    step(params, x, y, jnp.float32(0.05))
    step(params, x, y, jnp.float32(0.01))
    print("lr as a 0-d array, twice  -> trace_count", step.trace_count, "(one new trace, then reused)")

    banner("6. Per-op profile of the graph executor (blocking after every op)")
    flat = jax.tree.leaves((params, x, y))
    for op, count, seconds in cf.executable.profile(flat):
        print(f"  {op:16s} x{count:<3d} {seconds * 1e6:9.1f} us")


if __name__ == "__main__":
    main()
