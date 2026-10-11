"""`sx.function`: trace once per input signature, cache the graph, run it op by op.

    step = sx.function(train_step)          # like tf.function (no jit_compile)
    step = sx.function(train_step, jit_compile=True)   # graph -> one XLA program

On each call:
  1. Flatten the arguments into leaves and build a signature: for each array leaf
     its (shape, dtype, weak_type); for each other leaf its Python value.
  2. Look the signature up in this function's cache.
  3. On a miss, trace: run the Python function on Symbols, optimize the graph and
     prepare an executable. This is the only time the Python body runs.
  4. Run the cached executable on the array leaves and rebuild the output tree.

Python scalars (and other non-array leaves) are static, as in tf.function: a new
value means a new graph. That is a common cause of retracing; see `trace_count`.
"""

from __future__ import annotations

import dataclasses
import functools
import time
import warnings
from typing import Any, Callable

import jax
import numpy as np

from sentinex.graph import passes
from sentinex.graph.executor import Executable, interpret
from sentinex.graph.ir import Aval, Graph
from sentinex.graph.tracing import GraphBuilder, current_builder

RETRACE_WARNING_THRESHOLD = 5


@dataclasses.dataclass
class ConcreteFunction:
    """One traced specialization: the graph for a single input signature."""

    signature: tuple
    raw_graph: Graph  # as traced
    graph: Graph  # after optimization passes
    out_tree: Any
    run: Callable[[list], list] | None  # None until compiled
    pass_stats: dict
    trace_seconds: float
    optimize_seconds: float
    compile_seconds: float
    executable: Executable | None = None  # graph mode: the op-by-op program
    xla_executable: Any = None  # jit_compile=True: the single XLA program


def _leaf_signature(leaf) -> tuple:
    if isinstance(leaf, jax.Array):  # includes JAX tracers
        return ("array", tuple(leaf.shape), leaf.dtype, bool(leaf.weak_type))
    if isinstance(leaf, (np.ndarray, np.generic)):
        return ("array", tuple(leaf.shape), jax.dtypes.canonicalize_dtype(leaf.dtype), False)
    try:
        hash(leaf)
    except TypeError:
        raise TypeError(
            f"sx.function argument leaf of type {type(leaf).__name__} is neither an array "
            "nor hashable, so it cannot be part of the trace signature"
        ) from None
    return ("static", type(leaf), leaf)


class Function:
    """A Python function compiled to Sentinex graphs, one per input signature."""

    def __init__(self, fn: Callable, *, jit_compile: bool = False, optimize: bool = True):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self.jit_compile = jit_compile
        self.optimize = optimize
        self._cache: dict[tuple, ConcreteFunction] = {}
        self.trace_count = 0
        self._warned = False

    def __call__(self, *args, **kwargs):
        if current_builder() is not None:
            # Called while tracing another sx.function: just inline the Python body.
            return self._fn(*args, **kwargs)
        key, treedef, leaves, sig = self._signature(args, kwargs)
        concrete = self._cache.get(key)
        if concrete is None:
            concrete = self._cache[key] = self._trace_and_compile(key, treedef, leaves, sig)
        arrays = [leaf for leaf, s in zip(leaves, sig) if s[0] == "array"]
        if any(isinstance(a, jax.core.Tracer) for a in arrays):
            # Inside jax.jit / jax.grad: replay the graph through JAX instead.
            outs = interpret(concrete.graph, arrays)
        else:
            outs = concrete.run(arrays)
        return jax.tree.unflatten(concrete.out_tree, outs)

    @staticmethod
    def _signature(args, kwargs):
        leaves, treedef = jax.tree.flatten((args, kwargs))
        sig = [_leaf_signature(leaf) for leaf in leaves]
        return (treedef, tuple(sig)), treedef, leaves, sig

    # -- tracing ----------------------------------------------------------------
    def _build(self, key, treedef, leaves, sig) -> ConcreteFunction:
        """Trace the Python function on Symbols and optimize the graph (no compile)."""
        name = getattr(self._fn, "__name__", "fn")
        t0 = time.perf_counter()
        builder = GraphBuilder(name)
        with builder:
            traced = [
                builder.add_input(Aval(s[1], np.dtype(s[2]), s[3])) if s[0] == "array" else leaf
                for leaf, s in zip(leaves, sig)
            ]
            args, kwargs = jax.tree.unflatten(treedef, traced)
            out = self._fn(*args, **kwargs)
            out_leaves, out_tree = jax.tree.flatten(out)
            builder.set_outputs(out_leaves)
        raw = builder.graph
        t1 = time.perf_counter()
        graph, stats = passes.optimize(raw) if self.optimize else (passes.renumber(raw), {})
        t2 = time.perf_counter()
        return ConcreteFunction(
            signature=key,
            raw_graph=raw,
            graph=graph,
            out_tree=out_tree,
            run=None,
            pass_stats=stats,
            trace_seconds=t1 - t0,
            optimize_seconds=t2 - t1,
            compile_seconds=0.0,
        )

    def _trace_and_compile(self, key, treedef, leaves, sig) -> ConcreteFunction:
        concrete = self._build(key, treedef, leaves, sig)
        graph = concrete.graph
        t0 = time.perf_counter()
        if self.jit_compile:
            structs = [n.aval.to_struct() for n in graph.inputs]
            compiled = jax.jit(functools.partial(interpret, graph)).lower(structs).compile()
            concrete.xla_executable = compiled
            concrete.run = lambda arrays: compiled(arrays)
        else:
            concrete.executable = concrete.run = Executable(graph)
        concrete.compile_seconds = time.perf_counter() - t0

        self.trace_count += 1
        if self.trace_count > RETRACE_WARNING_THRESHOLD and not self._warned:
            self._warned = True
            warnings.warn(
                f"sx.function '{graph.name}' has been traced {self.trace_count} times. Each new "
                "input shape/dtype or Python-scalar value makes a new graph. Pass changing "
                "numbers as arrays, and keep batch shapes fixed.",
                stacklevel=3,
            )
        return concrete

    # -- inspection -------------------------------------------------------------
    def trace(self, *args, **kwargs) -> ConcreteFunction:
        """Trace and optimize for these arguments without compiling or caching."""
        return self._build(*self._signature(args, kwargs))

    def get_concrete_function(self, *args, **kwargs) -> ConcreteFunction:
        """The cached specialization for these arguments (traces and compiles if needed)."""
        key, treedef, leaves, sig = self._signature(args, kwargs)
        if key not in self._cache:
            self._cache[key] = self._trace_and_compile(key, treedef, leaves, sig)
        return self._cache[key]

    def graph(self, *args, **kwargs) -> Graph:
        return self.get_concrete_function(*args, **kwargs).graph

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self):
        self._cache.clear()


def function(fn: Callable | None = None, *, jit_compile: bool = False, optimize: bool = True):
    """Decorator/wrapper: compile `fn` to Sentinex graphs (see module docstring)."""
    if fn is None:
        return lambda f: Function(f, jit_compile=jit_compile, optimize=optimize)
    return Function(fn, jit_compile=jit_compile, optimize=optimize)
