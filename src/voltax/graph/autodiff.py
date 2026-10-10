"""Graph-mode reverse-mode autodiff: a graph-to-graph transformation.

`backward(builder, out, wrt)` walks the recorded graph from `out` back to `wrt`
and appends gradient nodes to the same graph, using each primitive's VJP rules.
This is what TensorFlow's `tf.gradients` does in graph mode. JAX instead
differentiates while tracing (JVP + transpose on the jaxpr); the result is the
same kind of thing, a bigger graph that computes both value and gradient.

Because the rules are written with Voltax ops, the gradient nodes are ordinary
nodes: they get optimized, compiled and executed like everything else, and can
themselves be differentiated again.
"""

from __future__ import annotations

import numpy as np

from voltax.graph.ir import Node
from voltax.graph.tracing import GraphBuilder, Symbol


def _is_float(dtype) -> bool:
    return np.issubdtype(dtype, np.inexact)


def backward(builder: GraphBuilder, out: Symbol, wrt: list[Symbol]) -> list[Symbol | None]:
    """Cotangents of scalar `out` with respect to each Symbol in `wrt` (None = zero)."""
    from voltax import ops  # late import: ops -> primitive -> tracing

    nodes = builder.graph.nodes
    wrt_ids = {w.node.id for w in wrt}
    lo, hi = min(wrt_ids), out.node.id

    # 1. Forward sweep: which nodes depend on a wrt node? Only those can carry
    #    gradient. Node ids are a topological order, so one pass is enough.
    active = set(wrt_ids)
    for node in nodes[lo : hi + 1]:
        if not node.op.nondiff and any(isinstance(i, Node) and i.id in active for i in node.inputs):
            active.add(node.id)
    if hi not in active:
        return [None] * len(wrt)

    # 2. Backward sweep in reverse topological order, accumulating cotangents.
    #    The seed is a const node so every rule sees a Symbol `g`.
    cts: dict[int, Symbol] = {hi: builder.symbol(builder.add_const(np.ones((), out.dtype)))}
    for node in reversed(nodes[lo : hi + 1]):
        if node.id in wrt_ids or node.id not in cts:
            continue
        g = cts.pop(node.id)
        out_sym = builder.symbol(node)
        args = [builder.symbol(i) if isinstance(i, Node) else i.value for i in node.inputs]
        params = dict(node.params)
        for k, inp in enumerate(node.inputs):
            if not isinstance(inp, Node) or inp.id not in active or not _is_float(inp.aval.dtype):
                continue
            rule = node.op.vjp_rule(k)
            if rule is None:
                continue
            ct = rule(g, out_sym, *args, **params)
            if ct is None:
                continue
            cts[inp.id] = ct if inp.id not in cts else ops.add(cts[inp.id], ct)
    return [cts.get(w.node.id) for w in wrt]
