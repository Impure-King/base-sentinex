"""Graph optimization passes. Each pass takes a graph and returns a new one.

These are the classic "graph-level" optimizations a graph runtime does before it
executes anything (TensorFlow's Grappler does these and many more):

  forward_identities  drop identity nodes (graph-mode autodiff inserts them)
  fold_constants      evaluate nodes whose inputs are all constants, once, now
  cse                 merge nodes that compute the same thing from the same inputs
  dce                 drop nodes no output depends on

None of them fuse ops. Fusion (turning several ops into one loop over memory) is
what XLA adds on top, and it is the main reason `jax.jit` beats graph mode.
"""

from __future__ import annotations

import collections
from typing import Callable

from voltax.graph.ir import CONST, INPUT, Graph, Literal, Node, aval_of


def _rewrite(graph: Graph, visit: Callable[[Node, tuple], Node | None]) -> Graph:
    """Rebuild `graph` node by node. `visit(node, new_inputs)` returns the node to
    use in its place: a new node, an existing (already rebuilt) node, or None to
    keep the node with its inputs remapped."""
    new = Graph(graph.name)
    remap: dict[int, Node] = {}
    for node in graph.nodes:
        inputs = tuple(remap[i.id] if isinstance(i, Node) else i for i in node.inputs)
        repl = visit(node, inputs)
        if repl is None:
            repl = Node(node.id, node.op, inputs, node.params, node.aval, node.value)
        if repl.id == node.id:  # a rebuilt node, not a forward to an earlier one
            new.nodes.append(repl)
        remap[node.id] = repl
    new.inputs = [remap[n.id] for n in graph.inputs]
    new.outputs = [remap[n.id] for n in graph.outputs]
    return new


def forward_identities(graph: Graph) -> Graph:
    def visit(node, inputs):
        if node.kind == "op" and node.op.name == "identity" and isinstance(inputs[0], Node):
            return inputs[0]
        return None

    return _rewrite(graph, visit)


def fold_constants(graph: Graph, max_size: int = 4096) -> Graph:
    """Precompute small all-constant subgraphs. The size cap stops folding from
    materialising big arrays (e.g. a broadcast seed) that are cheap to compute."""

    def visit(node, inputs):
        if node.kind != "op" or node.aval.size > max_size:
            return None
        if not all(isinstance(i, Literal) or i.op is CONST for i in inputs):
            return None
        args = [i.value for i in inputs]
        value = node.op.impl(*args, **dict(node.params))
        return Node(node.id, CONST, (), (), aval_of(value), value)

    return _rewrite(graph, visit)


def cse(graph: Graph) -> Graph:
    """Common-subexpression elimination by hash-consing (op, inputs, params).

    Safe because every Voltax primitive is a pure function of its inputs; even RNG
    ops take their key as an input. (TF must exclude its stateful ops here.)"""
    seen: dict[tuple, Node] = {}

    def visit(node, inputs):
        if node.kind == "input":
            return None
        if node.kind == "const":
            key = ("const", id(node.value))
        else:
            ins = tuple(i.key if isinstance(i, Literal) else i.id for i in inputs)
            key = (node.op, ins, node.params)
        hit = seen.get(key)
        if hit is not None:
            return hit
        rebuilt = Node(node.id, node.op, inputs, node.params, node.aval, node.value)
        seen[key] = rebuilt
        return rebuilt

    return _rewrite(graph, visit)


def dce(graph: Graph) -> Graph:
    """Dead-code elimination: keep what the outputs need (and all inputs)."""
    live = {n.id for n in graph.outputs}
    for node in reversed(graph.nodes):
        if node.id in live:
            live.update(i.id for i in node.inputs if isinstance(i, Node))
    new = Graph(graph.name)
    new.nodes = [n for n in graph.nodes if n.id in live or n.op is INPUT]
    new.inputs, new.outputs = list(graph.inputs), list(graph.outputs)
    return new


def renumber(graph: Graph) -> Graph:
    """Give nodes dense ids 0..n-1 (the executor uses ids as register slots)."""
    ids = {n.id: k for k, n in enumerate(graph.nodes)}
    remap: dict[int, Node] = {}
    new = Graph(graph.name)
    for node in graph.nodes:
        inputs = tuple(remap[i.id] if isinstance(i, Node) else i for i in node.inputs)
        remap[node.id] = Node(ids[node.id], node.op, inputs, node.params, node.aval, node.value)
        new.nodes.append(remap[node.id])
    new.inputs = [remap[n.id] for n in graph.inputs]
    new.outputs = [remap[n.id] for n in graph.outputs]
    return new


PIPELINE = (
    ("forward_identities", forward_identities),
    ("fold_constants", fold_constants),
    ("cse", cse),
    ("dce", dce),
)


def optimize(graph: Graph) -> tuple[Graph, dict]:
    """Run the standard pipeline. Returns the graph and op counts after each pass."""
    stats = collections.OrderedDict(traced=len(graph.ops))
    for name, run in PIPELINE:
        graph = run(graph)
        stats[name] = len(graph.ops)
    return renumber(graph), dict(stats)
