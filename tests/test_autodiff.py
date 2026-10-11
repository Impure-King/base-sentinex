"""sx.value_and_grad / sx.grad in graph mode against JAX."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sentinex as sx
from sentinex import losses, nn, ops

from helpers import rand, tree_allclose


def _mlp_loss(params, x, y):
    h = ops.relu(x @ params["w1"] + params["b1"])
    return losses.cross_entropy(h @ params["w2"] + params["b2"], y)


def _params():
    return {"w1": rand((8, 16)), "b1": rand((16,), 1), "w2": rand((16, 3), 2), "b2": rand((3,), 3)}


def test_mlp_grads_match_jax():
    x, y = rand((5, 8), 4), jnp.array([0, 1, 2, 1, 0])
    ref = jax.value_and_grad(_mlp_loss)(_params(), x, y)
    out = sx.function(sx.value_and_grad(_mlp_loss))(_params(), x, y)
    tree_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_argnums_and_has_aux():
    def f(a, b, c):
        return ops.sum(a * b) + ops.sum(c), {"aux": a * 2}

    args = (rand((3,)), rand((3,), 1), rand((2,), 2))
    ref = jax.value_and_grad(f, argnums=(0, 2), has_aux=True)(*args)
    out = sx.function(sx.value_and_grad(f, argnums=(0, 2), has_aux=True))(*args)
    tree_allclose(out, ref)
    g = sx.function(sx.grad(f, argnums=1, has_aux=True))(*args)
    tree_allclose(g, jax.grad(f, argnums=1, has_aux=True)(*args))


def test_tied_leaves_get_separate_gradients():
    def outer(w):
        return sx.grad(lambda p: ops.sum(p["a"] * 2.0 + p["b"] * 3.0))({"a": w, "b": w})

    g = sx.function(outer)(jnp.ones(2))
    np.testing.assert_allclose(g["a"], [2.0, 2.0])
    np.testing.assert_allclose(g["b"], [3.0, 3.0])


def test_unused_input_gets_zero_gradient():
    g = sx.function(sx.grad(lambda x, y: ops.sum(x * x), argnums=1))(jnp.ones(2), jnp.ones(3))
    np.testing.assert_array_equal(g, np.zeros(3))


def test_second_order():
    f = lambda x: ops.sum(ops.sin(x) * x**3)
    x = rand((4,))
    hvp_graph = sx.function(sx.grad(lambda x: ops.sum(sx.grad(f)(x))))(x)
    hvp_jax = jax.grad(lambda x: jnp.sum(jax.grad(f)(x)))(x)
    np.testing.assert_allclose(hvp_graph, hvp_jax, rtol=1e-4, atol=1e-5)


def test_grad_through_layers_matches_jax():
    model = nn.Sequential([nn.Conv2D(4), nn.relu, nn.MaxPool2D(), nn.Flatten(), nn.Dense(3)])
    params = model.init(jax.random.PRNGKey(0), (None, 6, 6, 2))
    x, y = rand((2, 6, 6, 2)), jnp.array([0, 2])
    loss = lambda p: losses.cross_entropy(model.apply(p, x), y)
    tree_allclose(sx.function(sx.grad(loss))(params), jax.grad(loss)(params), rtol=1e-4, atol=1e-6)


def test_errors():
    with pytest.raises(TypeError, match="scalar"):
        sx.function(sx.grad(lambda x: x * 2))(jnp.ones(3))
    with pytest.raises(TypeError, match="float"):
        sx.function(sx.grad(lambda x: ops.sum(ops.cast(x, jnp.float32))))(jnp.ones(3, jnp.int32))


def test_outside_graph_mode_it_is_jax():
    x = rand((3,))
    tree_allclose(sx.grad(lambda x: ops.sum(ops.exp(x)))(x), jnp.exp(x))
    tree_allclose(jax.jit(sx.grad(lambda x: ops.sum(ops.exp(x))))(x), jnp.exp(x))
