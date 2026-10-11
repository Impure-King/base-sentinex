"""Sentinex's graph IR.

A `Graph` is a flat list of `Node`s in topological order (SSA style): each node
defines exactly one array value, and its inputs are earlier nodes or `Literal`s.

    %3 = matmul(%0, %1)        f32[256,512]
    %4 = add(%3, %2)           f32[256,512]
    %5 = relu(%4)              f32[256,512]

There are three kinds of node:
  * input  - a placeholder for one array argument of the traced function,
  * const  - an array captured while tracing (closed-over or created eagerly),
  * op     - a call to a `Primitive` with static `params`.

Compare with a jaxpr: a jaxpr is the same idea (typed SSA over primitives), but
JAX hands the whole jaxpr to XLA, which fuses and schedules it as one program.
Sentinex keeps the graph and runs it node by node (see executor.py).
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any

import jax
import numpy as np


@dataclasses.dataclass(frozen=True)
class Aval:
    """Abstract value: everything the tracer knows about an array."""

    shape: tuple[int, ...]
    dtype: np.dtype
    weak_type: bool = False  # JAX's "Python scalar" typing, kept so promotion matches

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    def to_struct(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(self.shape, self.dtype, weak_type=self.weak_type)

    def __str__(self) -> str:
        return f"{_short_dtype(self.dtype)}[{','.join(map(str, self.shape))}]"


def aval_of(x: Any) -> Aval:
    """Aval of a concrete array (numpy or jax)."""
    dtype = jax.dtypes.canonicalize_dtype(x.dtype)
    return Aval(tuple(x.shape), dtype, bool(getattr(x, "weak_type", False)))


@dataclasses.dataclass(frozen=True)
class Literal:
    """A Python scalar baked into a node, like a jaxpr literal.

    Literals stay Python scalars (not arrays) so `x * 2.0` keeps x's dtype,
    exactly as JAX's weak typing does.
    """

    value: bool | int | float | complex

    @property
    def key(self) -> tuple:
        # 1, 1.0 and True compare equal in Python, so the type is part of the key.
        return (type(self.value), self.value)

    def __str__(self) -> str:
        return repr(self.value)


class _Special:
    """Marker op for input and const nodes. They have no kernel and no gradient."""

    nondiff = True

    def __init__(self, name: str):
        self.name = name

    def vjp_rule(self, argnum):
        return None

    def __repr__(self) -> str:
        return self.name


INPUT = _Special("input")
CONST = _Special("const")


class Node:
    """One SSA value in a graph."""

    __slots__ = ("id", "op", "inputs", "params", "aval", "value")

    def __init__(self, id: int, op, inputs: tuple, params: tuple, aval: Aval, value=None):
        self.id = id
        self.op = op  # a Primitive, or INPUT / CONST
        self.inputs = inputs  # tuple of Node | Literal
        self.params = params  # sorted tuple of (name, frozen value)
        self.aval = aval
        self.value = value  # the array, for const nodes only

    @property
    def kind(self) -> str:
        if self.op is INPUT:
            return "input"
        if self.op is CONST:
            return "const"
        return "op"

    def __repr__(self) -> str:
        return f"%{self.id}"


class Graph:
    """A traced function: placeholder inputs, a node list and output nodes."""

    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: list[Node] = []
        self.inputs: list[Node] = []
        self.outputs: list[Node] = []

    @property
    def ops(self) -> list[Node]:
        return [n for n in self.nodes if n.kind == "op"]

    def op_counts(self) -> collections.Counter:
        return collections.Counter(n.op.name for n in self.ops)

    def summary(self) -> str:
        consts = sum(n.kind == "const" for n in self.nodes)
        return f"{len(self.inputs)} inputs, {consts} consts, {len(self.ops)} ops"

    def pretty(self, max_ops: int | None = None) -> str:
        lines = [f"graph {self.name} ({self.summary()}) {{"]
        lines.append("  inputs: " + " ".join(f"%{n.id}:{n.aval}" for n in self.inputs))
        shown = 0
        for n in self.nodes:
            if n.kind == "input":
                continue
            if max_ops is not None and shown >= max_ops:
                lines.append(f"  ... {len(self.nodes) - len(self.inputs) - shown} more")
                break
            shown += 1
            if n.kind == "const":
                desc = "const"
                if n.aval.size == 1:
                    desc += f" {np.asarray(n.value).reshape(())!r}"
            else:
                args = ", ".join(str(i) if isinstance(i, Literal) else f"%{i.id}" for i in n.inputs)
                params = " ".join(f"{k}={_fmt_param(v)}" for k, v in n.params)
                desc = f"{n.op.name}({args})" + (f" {params}" if params else "")
            lines.append(f"  %{n.id} = {desc:<52} {n.aval}")
        lines.append("  return (" + ", ".join(f"%{n.id}" for n in self.outputs) + ")")
        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()


def _short_dtype(dtype) -> str:
    dtype = np.dtype(dtype)
    names = {"float": "f", "int": "i", "uint": "u", "bool": "bool", "complex": "c", "bfloat": "bf"}
    for prefix, short in names.items():
        if dtype.name.startswith(prefix):
            return short + dtype.name[len(prefix):]
    return dtype.name


def _fmt_param(v) -> str:
    if isinstance(v, (np.dtype, type)):
        return _short_dtype(v)
    return str(v)
