"""Sentinex: a small functional deep-learning library on JAX.

Write a model once with `sentinex.ops`, `sentinex.nn`, `sentinex.losses` and
`sentinex.optim`, then run it in any of these modes:

  eager             call it directly (op-by-op, Python in the loop)
  jax.jit           JAX traces to a jaxpr, XLA compiles and fuses the whole step
  sx.function       Sentinex traces to its own graph, caches it per signature and
                    runs it op by op (like tf.function without jit_compile)
  sx.function(..., jit_compile=True)
                    Sentinex graph lowered to XLA as one program
"""

from sentinex import data, graph, losses, nn, ops, optim, train
from sentinex.autodiff import grad, value_and_grad
from sentinex.graph import Symbol, TracingError, function

__version__ = "1.0.0.dev0+reference"

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
