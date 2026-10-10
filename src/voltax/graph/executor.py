"""Graph execution: run a graph one kernel at a time.

`Executable` turns an optimized graph into a flat program:

    for kernel, input_slots, out_slot, dead_slots in program:
        regs[out_slot] = kernel(*regs[input_slots])
        regs[dead_slots] = None          # free buffers after their last use

Each `kernel` is a one-op XLA executable from the global kernel cache. Every op
writes its full result to memory and the next op reads it back; nothing is fused.
That is the defining difference from `jax.jit`, where XLA sees the whole program,
fuses elementwise chains into single loops and plans memory globally.

JAX dispatch is asynchronous, so the Python loop usually runs ahead of the device
and per-op overhead overlaps with compute until ops get very small.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Sequence

import jax

from voltax.graph import kernels
from voltax.graph.ir import CONST, INPUT, Graph, Node


class Executable:
    def __init__(self, graph: Graph):
        self.graph = graph
        n = len(graph.nodes)
        assert all(node.id == k for k, node in enumerate(graph.nodes)), "renumber the graph first"

        self._template: list[Any] = [None] * n
        for node in graph.nodes:
            if node.op is CONST:
                self._template[node.id] = node.value
        self._input_slots = [node.id for node in graph.inputs]
        self._output_slots = [node.id for node in graph.outputs]

        # Liveness: the last instruction that reads each slot.
        ops = [node for node in graph.nodes if node.kind == "op"]
        last_use: dict[int, int] = {}
        for pc, node in enumerate(ops):
            for i in node.inputs:
                if isinstance(i, Node):
                    last_use[i.id] = pc
        keep = set(self._output_slots)

        t0 = time.perf_counter()
        compiled = kernels.compiled_kernels(ops)
        self.compile_seconds = time.perf_counter() - t0
        self.program = []
        for pc, (node, kernel) in enumerate(zip(ops, compiled)):
            ins = tuple(i.id for i in node.inputs if isinstance(i, Node))
            dead = tuple(sorted({s for s in ins if last_use[s] == pc and s not in keep}))
            self.program.append((kernel, ins, node.id, dead))
        self.nodes = ops

    @property
    def num_kernels(self) -> int:
        return len(self.program)

    def __call__(self, args: Sequence[Any]) -> list[Any]:
        regs = self._template.copy()
        for slot, a in zip(self._input_slots, args):
            regs[slot] = a
        for kernel, ins, out, dead in self.program:
            regs[out] = kernel(*[regs[i] for i in ins])
            for s in dead:
                regs[s] = None
        return [regs[s] for s in self._output_slots]

    def profile(self, args: Sequence[Any], repeats: int = 5) -> list[tuple[str, int, float]]:
        """Time each op (blocking after every kernel). Returns (op, count, seconds)
        per op type, slowest first, using the best of `repeats` runs per node."""
        best = [float("inf")] * len(self.program)
        for _ in range(repeats):
            regs = self._template.copy()
            for slot, a in zip(self._input_slots, args):
                regs[slot] = a
            jax.block_until_ready([r for r in regs if r is not None])
            for k, (kernel, ins, out, _) in enumerate(self.program):
                t0 = time.perf_counter()
                regs[out] = jax.block_until_ready(kernel(*[regs[i] for i in ins]))
                best[k] = min(best[k], time.perf_counter() - t0)
        totals: dict[str, list] = collections.defaultdict(lambda: [0, 0.0])
        for node, t in zip(self.nodes, best):
            totals[node.op.name][0] += 1
            totals[node.op.name][1] += t
        return sorted(((k, c, t) for k, (c, t) in totals.items()), key=lambda r: -r[2])


def interpret(graph: Graph, args: Sequence[Any]) -> list[Any]:
    """Evaluate a graph by calling each primitive's impl directly.

    With concrete arrays this is plain eager execution. With JAX tracers (inside
    `jax.jit`) it emits one jaxpr for the whole graph, which XLA then compiles and
    fuses: that is how `vx.function(..., jit_compile=True)` lowers a Voltax graph.
    """
    env: dict[int, Any] = {}
    for node, a in zip(graph.inputs, args):
        env[node.id] = a
    for node in graph.nodes:
        if node.op is INPUT:
            continue
        if node.op is CONST:
            env[node.id] = node.value
            continue
        vals = [env[i.id] if isinstance(i, Node) else i.value for i in node.inputs]
        env[node.id] = node.op.impl(*vals, **dict(node.params))
    return [env[n.id] for n in graph.outputs]

