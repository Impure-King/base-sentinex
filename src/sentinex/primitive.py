"""Primitives: the one op table every Sentinex mode goes through.

A primitive has
  * `impl`: the computation, written with jax.numpy / lax. Eager mode and `jax.jit`
    call it directly. The graph executor compiles it into a one-op XLA kernel.
  * VJP rules (optional, one per input), written with Sentinex ops. Only graph-mode
    autodiff uses them; under eager/jit, JAX differentiates `impl` itself.

Calling a primitive dispatches on its arguments: if any argument is a graph
`Symbol`, the call is recorded into that Symbol's graph; otherwise `impl` runs now.
That single `if` is what lets one model definition run in every mode.
"""

from __future__ import annotations

from typing import Callable

import jax

from sentinex.graph.tracing import Symbol


class Primitive:
    def __init__(self, name: str, impl: Callable, *, nondiff: bool = False):
        self.name = name
        self.impl = impl
        self.nondiff = nondiff  # output carries no gradient (comparisons, argmax, rng)
        self.vjp_rules: tuple | None = None  # one rule (or None) per positional input
        self._derived: dict[int, Primitive] = {}

    def __call__(self, *args, **params):
        for a in args:
            if type(a) is Symbol:
                return a.builder.record(self, args, params)
        return self.impl(*args, **params)

    def defvjp(self, *rules: Callable | None) -> "Primitive":
        """Register VJP rules, one per positional input.

        A rule is called as `rule(g, out, *inputs, **params)` and returns the
        cotangent for its input. `None` means "not differentiable in this input".
        """
        self.vjp_rules = rules
        return self

    def vjp_rule(self, argnum: int) -> Callable | None:
        if self.nondiff:
            return None
        if self.vjp_rules is not None:
            return self.vjp_rules[argnum]
        return self.derived_vjp_rule(argnum)

    def derived_vjp_rule(self, argnum: int) -> Callable:
        """Fallback: a separate backward primitive whose kernel is `jax.vjp` of impl.

        This mirrors TensorFlow, where Conv2D's gradient is its own pair of kernels
        (Conv2DBackpropInput / Conv2DBackpropFilter). The graph still gets a node per
        backward op; only the kernel inside that node is derived by JAX.
        """
        backward = self._derived.get(argnum)
        if backward is None:
            backward = self._derived[argnum] = _derived_backward(self, argnum)
        return lambda g, out, *args, **params: backward(*args, g, **params)

    def __repr__(self) -> str:
        return f"Primitive({self.name})"


def _derived_backward(prim: Primitive, argnum: int) -> Primitive:
    def impl(*args, **params):
        *primals, g = args

        def f(x):
            xs = list(primals)
            xs[argnum] = x
            return prim.impl(*xs, **params)

        _, pullback = jax.vjp(f, primals[argnum])
        return pullback(g)[0]

    return Primitive(f"{prim.name}_grad{argnum}", impl)


def primitive(name: str, *, nondiff: bool = False):
    """Decorator: turn an impl function into a Primitive."""

    def wrap(impl: Callable) -> Primitive:
        return Primitive(name, impl, nondiff=nondiff)

    return wrap
