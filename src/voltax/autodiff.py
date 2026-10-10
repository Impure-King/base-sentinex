"""`vx.grad` / `vx.value_and_grad`: one API, two autodiff engines.

Outside graph tracing (eager, or inside `jax.jit`) these are `jax.value_and_grad`.
While a `vx.function` is being traced they run Voltax's symbolic reverse mode
(`voltax.graph.autodiff.backward`), which appends gradient nodes to the graph.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import numpy as np

from voltax.graph.autodiff import backward
from voltax.graph.tracing import Symbol, current_builder


def value_and_grad(fn: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False) -> Callable:
    def wrapped(*args, **kwargs):
        builder = current_builder()
        if builder is None:
            return jax.value_and_grad(fn, argnums, has_aux)(*args, **kwargs)
        return _graph_value_and_grad(builder, fn, argnums, has_aux, args, kwargs)

    return wrapped


def grad(fn: Callable, argnums: int | Sequence[int] = 0, has_aux: bool = False) -> Callable:
    vg = value_and_grad(fn, argnums, has_aux)

    def wrapped(*args, **kwargs):
        value, g = vg(*args, **kwargs)
        return (g, value[1]) if has_aux else g

    return wrapped


def _graph_value_and_grad(builder, fn, argnums, has_aux, args, kwargs):
    from voltax import ops

    nums = (argnums,) if isinstance(argnums, int) else tuple(argnums)
    args = list(args)
    wrt, trees = [], []
    for i in nums:
        leaves, tree = jax.tree.flatten(args[i])
        # A fresh identity node per leaf: each leaf is an independent variable,
        # matching JAX even if two leaves are the same Symbol (tied weights).
        fresh = [ops.identity(builder.as_symbol(leaf)) for leaf in leaves]
        for s in fresh:
            if not np.issubdtype(s.dtype, np.inexact):
                raise TypeError(f"grad requires float inputs, got {s.aval}")
        args[i] = jax.tree.unflatten(tree, fresh)
        wrt.append(fresh)
        trees.append(tree)

    result = fn(*args, **kwargs)
    out, aux = result if has_aux else (result, None)
    if ops.shape_of(out) != () or not np.issubdtype(ops.dtype_of(out), np.floating):
        raise TypeError(f"grad needs a scalar float output, got shape {ops.shape_of(out)}")

    flat = [s for group in wrt for s in group]
    cts = backward(builder, out, flat) if isinstance(out, Symbol) else [None] * len(flat)
    grads, k = [], 0
    for group, tree in zip(wrt, trees):
        g = [ct if ct is not None else ops.zeros(s.shape, s.dtype) for ct, s in zip(cts[k : k + len(group)], group)]
        grads.append(jax.tree.unflatten(tree, g))
        k += len(group)
    grads = grads[0] if isinstance(argnums, int) else tuple(grads)
    return ((out, aux) if has_aux else out), grads
