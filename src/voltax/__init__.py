"""Voltax: a small functional deep-learning library on JAX.

Write a model once with `voltax.ops`, `voltax.nn`, `voltax.losses` and
`voltax.optim`, then run it in any of these modes:

  eager             call it directly (op-by-op, Python in the loop)
  jax.jit           JAX traces to a jaxpr, XLA compiles and fuses the whole step
  vx.function       Voltax traces to its own graph, caches it per signature and
                    runs it op by op (like tf.function without jit_compile)
  vx.function(..., jit_compile=True)
                    Voltax graph lowered to XLA as one program
"""

from voltax import data, graph, losses, nn, ops, optim, train
from voltax.autodiff import grad, value_and_grad
from voltax.graph import Symbol, TracingError, function

__version__ = "0.2.0"

__all__ = [
    "Symbol",
    "TracingError",
    "data",
    "function",
    "grad",
    "graph",
    "losses",
    "nn",
    "ops",
    "optim",
    "train",
    "value_and_grad",
]
