"""Kernels: one primitive call, specialised and compiled on its own.

Graph mode needs two things per node:
  * shape inference while tracing (`abstract_eval`), and
  * an executable to run at execution time (`compiled_kernel`).

Both come from the primitive's `impl`. Shape inference runs `jax.eval_shape` on it,
so the shape rules can never disagree with the kernel. Execution compiles the impl,
alone, into a one-op XLA executable. That is the "kernel" in the TensorFlow sense:
it is reused for every node with the same (op, params, literals, input avals), across
all graphs and retraces, and nothing is fused across nodes.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import numpy as np

from sentinex.graph.ir import Aval, Literal, Node

# (prim, spec, params) -> jitted kernel; (that key, input avals) -> aval / executable.
_JITTED: dict[tuple, Any] = {}
_AVALS: dict[tuple, Aval] = {}
_COMPILED: dict[tuple, Any] = {}
_LOCK = threading.Lock()


class KernelStats:
    """Counters for the global kernel cache (useful in tests and the walkthrough)."""

    compiles = 0
    compile_seconds = 0.0
    hits = 0

    @classmethod
    def reset(cls):
        cls.compiles, cls.compile_seconds, cls.hits = 0, 0.0, 0


def clear_cache():
    _JITTED.clear()
    _AVALS.clear()
    _COMPILED.clear()
    KernelStats.reset()


def freeze(value: Any, what: str = "param") -> Any:
    """Make a param hashable so it can be part of cache keys."""
    if isinstance(value, dict):
        return tuple(sorted((k, freeze(v, what)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v, what) for v in value)
    if isinstance(value, np.dtype) or (isinstance(value, type) and issubclass(value, np.generic)):
        return np.dtype(value)
    try:
        hash(value)
    except TypeError:
        raise TypeError(f"{what} {value!r} is not hashable; primitive params must be static") from None
    return value


def literal_spec(inputs: tuple) -> tuple:
    """Which positions are literals (and their values); None marks an array input."""
    return tuple(i.key if isinstance(i, Literal) else None for i in inputs)


def _kernel_fn(prim, spec: tuple, params: tuple):
    """The impl with literals and params closed over; takes only the array inputs."""
    kwargs = dict(params)
    literals = [None if s is None else s[1] for s in spec]

    def kernel(*arrays):
        it = iter(arrays)
        args = [next(it) if s is None else lit for s, lit in zip(spec, literals)]
        return prim.impl(*args, **kwargs)

    kernel.__name__ = f"sentinex_{prim.name}"
    return kernel


def _jitted(prim, spec: tuple, params: tuple):
    key = (prim, spec, params)
    fn = _JITTED.get(key)
    if fn is None:
        fn = _JITTED[key] = jax.jit(_kernel_fn(prim, spec, params))
    return fn


def abstract_eval(prim, inputs: tuple, params: tuple) -> Aval:
    """Output aval of `prim` on these inputs, without running anything."""
    spec = literal_spec(inputs)
    in_avals = tuple(i.aval for i in inputs if isinstance(i, Node))
    key = (prim, spec, params, in_avals)
    aval = _AVALS.get(key)
    if aval is None:
        out = jax.eval_shape(_jitted(prim, spec, params), *(a.to_struct() for a in in_avals))
        if not isinstance(out, jax.ShapeDtypeStruct):
            raise TypeError(f"primitive {prim.name} must return a single array, got {out}")
        aval = _AVALS[key] = Aval(tuple(out.shape), np.dtype(out.dtype), bool(out.weak_type))
    return aval


def _compile_key(node: Node) -> tuple:
    in_avals = tuple(i.aval for i in node.inputs if isinstance(i, Node))
    return (node.op, literal_spec(node.inputs), node.params, in_avals)


def _compile(key: tuple):
    prim, spec, params, in_avals = key
    t0 = time.perf_counter()
    exe = _jitted(prim, spec, params).lower(*(a.to_struct() for a in in_avals)).compile()
    with _LOCK:
        KernelStats.compiles += 1
        KernelStats.compile_seconds += time.perf_counter() - t0
        _COMPILED[key] = exe
    return exe


def compiled_kernels(nodes: list[Node], max_workers: int = 8) -> list:
    """XLA executables for these nodes, from the global kernel cache.

    Missing kernels are compiled on a thread pool: XLA compilation releases the
    GIL, so this is several times faster than compiling one by one."""
    keys = [_compile_key(n) for n in nodes]
    missing = list(dict.fromkeys(k for k in keys if k not in _COMPILED))
    KernelStats.hits += len(keys) - len(missing)
    if len(missing) > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers) as pool:
            list(pool.map(_compile, missing))
    else:
        for k in missing:
            _compile(k)
    return [_COMPILED[k] for k in keys]
