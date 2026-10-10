"""Tracing: run a Python function on `Symbol`s and record what it does.

A `Symbol` stands in for an array whose value is not known yet. It has a shape and
dtype, and every op applied to it is appended to the active `GraphBuilder` instead
of being computed. When the function returns, the builder holds the whole graph.

This is the same idea as JAX's tracers or TensorFlow's symbolic tensors. The
difference from JAX: here the recording is done by Voltax code you can read in one
file, and the result is Voltax's own `Graph`, not a jaxpr.
"""

from __future__ import annotations

import threading
from typing import Any

import jax
import numpy as np

from voltax.graph import kernels
from voltax.graph.ir import CONST, INPUT, Aval, Graph, Literal, Node, aval_of


class TracingError(Exception):
    """Raised when traced code needs a concrete value that does not exist yet."""


_STATE = threading.local()


def _stack() -> list:
    if not hasattr(_STATE, "stack"):
        _STATE.stack = []
    return _STATE.stack


def current_builder() -> "GraphBuilder | None":
    stack = _stack()
    return stack[-1] if stack else None


_SCALARS = (bool, int, float, complex)


class GraphBuilder:
    """Accumulates nodes while a function is traced. Use as a context manager."""

    def __init__(self, name: str = "graph"):
        self.graph = Graph(name)
        self._consts: dict[int, tuple[Any, Node]] = {}  # id(array) -> (array, node)
        self.closed = False

    def __enter__(self):
        _stack().append(self)
        return self

    def __exit__(self, *exc):
        _stack().pop()
        self.closed = True

    # -- creating nodes ---------------------------------------------------
    def _add(self, op, inputs, params, aval, value=None) -> Node:
        node = Node(len(self.graph.nodes), op, inputs, params, aval, value)
        self.graph.nodes.append(node)
        return node

    def add_input(self, aval: Aval) -> "Symbol":
        node = self._add(INPUT, (), (("index", len(self.graph.inputs)),), aval)
        self.graph.inputs.append(node)
        return Symbol(node, self)

    def add_const(self, value: Any) -> Node:
        """Capture a concrete array. The same object is captured only once."""
        hit = self._consts.get(id(value))
        if hit is not None:
            return hit[1]
        if isinstance(value, jax.core.Tracer):
            raise TracingError(
                "a JAX tracer reached a Voltax graph. Don't call a vx.function's Python "
                "body inside jax.jit/grad; call the vx.function itself instead."
            )
        array = jax.numpy.asarray(value)
        node = self._add(CONST, (), (), aval_of(array), array)
        self._consts[id(value)] = (value, node)  # keep `value` alive so its id stays unique
        return node

    def as_symbol(self, x: Any) -> "Symbol":
        if isinstance(x, Symbol):
            self._check_owner(x)
            return x
        if isinstance(x, _SCALARS):
            x = np.asarray(x)
        return Symbol(self.add_const(x), self)

    def symbol(self, node: Node) -> "Symbol":
        return Symbol(node, self)

    def record(self, prim, args: tuple, params: dict) -> "Symbol":
        """Append `prim(*args, **params)` to the graph and return its Symbol."""
        if self.closed or current_builder() is not self:
            raise TracingError(
                f"a Symbol from a finished trace was used by '{prim.name}'. Symbols only "
                "exist while their function is being traced; return arrays instead of "
                "storing Symbols in globals or closures."
            )
        inputs = tuple(self._input(prim, a) for a in args)
        frozen = tuple(sorted((k, kernels.freeze(v, f"{prim.name} param {k}")) for k, v in params.items()))
        aval = kernels.abstract_eval(prim, inputs, frozen)
        return Symbol(self._add(prim, inputs, frozen, aval), self)

    def _input(self, prim, a: Any):
        if type(a) is Symbol:
            self._check_owner(a)
            return a.node
        if isinstance(a, _SCALARS):
            return Literal(a)
        if isinstance(a, (np.ndarray, np.generic, jax.Array)):
            return self.add_const(a)
        raise TypeError(f"{prim.name}: unsupported argument of type {type(a).__name__}")

    def _check_owner(self, s: "Symbol"):
        if s.builder is not self:
            raise TracingError("a Symbol from a different trace was used; Symbols cannot cross traces")

    def set_outputs(self, leaves: list) -> None:
        """Mark the function's output leaves (concrete leaves become consts)."""
        self.graph.outputs = [self.as_symbol(leaf).node for leaf in leaves]


def _ops():
    from voltax import ops  # late import: ops imports this module

    return ops


def _concrete_error(sym: "Symbol", what: str) -> TracingError:
    return TracingError(
        f"cannot convert traced Symbol {sym.aval} to {what}. Graph mode traces the "
        "function once per input signature, so array values are unknown during tracing. "
        "Use vx.ops.where instead of Python if/else on arrays, or pass the value as a "
        "Python scalar so it becomes part of the signature."
    )


class Symbol:
    """A symbolic array inside a trace. Ops on it are recorded, not computed."""

    __slots__ = ("node", "builder")
    __array_ufunc__ = None  # make numpy return NotImplemented so our reflected ops run

    def __init__(self, node: Node, builder: GraphBuilder):
        self.node = node
        self.builder = builder

    # -- metadata ---------------------------------------------------------
    @property
    def aval(self) -> Aval:
        return self.node.aval

    @property
    def shape(self) -> tuple[int, ...]:
        return self.node.aval.shape

    @property
    def dtype(self) -> np.dtype:
        return self.node.aval.dtype

    @property
    def weak_type(self) -> bool:
        return self.node.aval.weak_type

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return self.node.aval.size

    def __repr__(self) -> str:
        return f"Symbol({self.aval}, %{self.node.id})"

    def __len__(self) -> int:
        if not self.shape:
            raise TypeError("len() of a 0-d Symbol")
        return self.shape[0]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    # -- values that cannot exist at trace time --------------------------------
    def __bool__(self):
        raise _concrete_error(self, "a Python bool")

    def __int__(self):
        raise _concrete_error(self, "a Python int")

    def __float__(self):
        raise _concrete_error(self, "a Python float")

    def __index__(self):
        raise _concrete_error(self, "an index")

    def __array__(self, *args, **kwargs):
        raise _concrete_error(self, "a numpy array")

    # No __jax_array__: JAX would then treat Symbols as arrays and call it from
    # `jax_array * symbol` instead of deferring to Symbol.__rmul__. Passing a Symbol
    # to jax.numpy therefore fails with JAX's "not a valid JAX type" TypeError.

    __hash__ = None  # arrays are unhashable; == is elementwise

    # -- operators ----------------------------------------------------------
    def __add__(self, o): return _ops().add(self, o)
    def __radd__(self, o): return _ops().add(o, self)
    def __sub__(self, o): return _ops().sub(self, o)
    def __rsub__(self, o): return _ops().sub(o, self)
    def __mul__(self, o): return _ops().mul(self, o)
    def __rmul__(self, o): return _ops().mul(o, self)
    def __truediv__(self, o): return _ops().div(self, o)
    def __rtruediv__(self, o): return _ops().div(o, self)
    def __pow__(self, o): return _ops().pow(self, o)
    def __rpow__(self, o): return _ops().pow(o, self)
    def __matmul__(self, o): return _ops().matmul(self, o)
    def __rmatmul__(self, o): return _ops().matmul(o, self)
    def __neg__(self): return _ops().neg(self)
    def __pos__(self): return self
    def __abs__(self): return _ops().abs(self)
    def __lt__(self, o): return _ops().less(self, o)
    def __le__(self, o): return _ops().less_equal(self, o)
    def __gt__(self, o): return _ops().greater(self, o)
    def __ge__(self, o): return _ops().greater_equal(self, o)
    def __eq__(self, o): return _ops().equal(self, o)
    def __ne__(self, o): return _ops().not_equal(self, o)

    def __getitem__(self, index): return _ops().getitem(self, index)

    # -- array methods ------------------------------------------------------
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _ops().reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return _ops().transpose(self, axes or None)

    @property
    def T(self):
        return _ops().transpose(self)

    def astype(self, dtype):
        return _ops().cast(self, dtype)

    def sum(self, axis=None, keepdims=False):
        return _ops().sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return _ops().mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return _ops().max(self, axis, keepdims)
