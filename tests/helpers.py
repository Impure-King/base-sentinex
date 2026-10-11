"""Shared test helpers."""

import jax
import jax.numpy as jnp
import numpy as np

import sentinex as sx
from sentinex import ops


def tree_allclose(a, b, rtol=1e-5, atol=1e-6):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), (len(la), len(lb))
    for x, y in zip(la, lb):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=rtol, atol=atol)


def weighted_sum(fn):
    """Scalar test loss: sum(fn(*args) * w) for a fixed random w of the output shape."""

    def loss(*args):
        out = fn(*args)
        w = np.random.default_rng(0).standard_normal(ops.shape_of(out)).astype(np.float32)
        return ops.sum(out * w)

    return loss


def graph_value_and_grad(fn, *args, argnums=0):
    """value_and_grad computed by Sentinex's graph-mode autodiff."""
    return sx.function(lambda *a: sx.value_and_grad(fn, argnums)(*a))(*args)


def jax_value_and_grad(fn, *args, argnums=0):
    """Reference: JAX autodiff of the same function (ops call jnp impls eagerly)."""
    return jax.value_and_grad(fn, argnums)(*args)


def rand(shape, seed=0, positive=False, dtype=jnp.float32):
    x = np.random.default_rng(seed).standard_normal(shape).astype(dtype)
    return jnp.asarray(np.abs(x) + 0.5 if positive else x)
