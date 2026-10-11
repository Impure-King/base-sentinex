"""Every op: graph-mode forward matches eager, graph-mode gradients match JAX."""

import jax.numpy as jnp
import numpy as np
import pytest

import sentinex as sx
from sentinex import ops

from helpers import graph_value_and_grad, jax_value_and_grad, rand, tree_allclose, weighted_sum

# (name, fn, args, argnums to differentiate)
CASES = [
    ("add_broadcast", lambda x, y: x + y, [rand((3, 4)), rand((4,), 1)], (0, 1)),
    ("sub_broadcast", lambda x, y: x - y, [rand((3, 1)), rand((3, 4), 1)], (0, 1)),
    ("mul_broadcast", lambda x, y: x * y, [rand((2, 3, 4)), rand((3, 1), 1)], (0, 1)),
    ("div", lambda x, y: x / y, [rand((3, 4)), rand((3, 4), 1, positive=True)], (0, 1)),
    ("pow_literal", lambda x: x**3, [rand((5,))], (0,)),
    ("pow_arrays", lambda x, y: x**y, [rand((4,), positive=True), rand((4,), 1)], (0, 1)),
    ("maximum_ties", lambda x, y: ops.maximum(x, y), [jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 0.0, 4.0])], (0, 1)),
    ("minimum", lambda x, y: ops.minimum(x, y), [rand((3, 4)), rand((4,), 1)], (0, 1)),
    ("where", lambda x, y: ops.where(x > 0, x, y), [rand((3, 4)), rand((3, 4), 1)], (0, 1)),
    ("neg", lambda x: -x, [rand((4,))], (0,)),
    ("exp", ops.exp, [rand((4,))], (0,)),
    ("log", ops.log, [rand((4,), positive=True)], (0,)),
    ("log1p", ops.log1p, [rand((4,), positive=True)], (0,)),
    ("sqrt", ops.sqrt, [rand((4,), positive=True)], (0,)),
    ("rsqrt", ops.rsqrt, [rand((4,), positive=True)], (0,)),
    ("tanh", ops.tanh, [rand((4,))], (0,)),
    ("sigmoid", ops.sigmoid, [rand((4,))], (0,)),
    ("sin", ops.sin, [rand((4,))], (0,)),
    ("cos", ops.cos, [rand((4,))], (0,)),
    ("abs", ops.abs, [rand((4,))], (0,)),
    ("relu", ops.relu, [rand((10,))], (0,)),
    ("cast_roundtrip", lambda x: ops.cast(ops.cast(x, jnp.float16), jnp.float32), [rand((4,))], (0,)),
    ("sum_axis", lambda x: ops.sum(x, 1), [rand((3, 4, 5))], (0,)),
    ("sum_keepdims", lambda x: ops.sum(x, (0, 2), keepdims=True), [rand((3, 4, 5))], (0,)),
    ("max_axis", lambda x: ops.max(x, -1), [rand((3, 4))], (0,)),
    ("max_ties", lambda x: ops.max(x, 0), [jnp.array([[1.0, 2.0], [1.0, 0.0]])], (0,)),
    ("mean", lambda x: ops.mean(x, 0), [rand((3, 4))], (0,)),
    ("var", lambda x: ops.var(x, -1, keepdims=True), [rand((3, 4))], (0,)),
    ("reshape", lambda x: ops.reshape(x, (4, -1)), [rand((2, 3, 4))], (0,)),
    ("transpose", lambda x: ops.transpose(x, (2, 0, 1)), [rand((2, 3, 4))], (0,)),
    ("broadcast_to", lambda x: ops.broadcast_to(x, (2, 3, 4)), [rand((3, 1))], (0,)),
    ("getitem", lambda x: x[1:, None, ::2], [rand((3, 5))], (0,)),
    ("getitem_int", lambda x: x[..., 0], [rand((3, 5))], (0,)),
    ("concatenate", lambda x, y: ops.concatenate([x, y], 1), [rand((2, 3)), rand((2, 2), 1)], (0, 1)),
    ("take", lambda t: ops.take(t, jnp.array([[0, 2], [2, 1]]), 0), [rand((3, 4))], (0,)),
    ("matmul_2d", lambda a, b: a @ b, [rand((3, 4)), rand((4, 5), 1)], (0, 1)),
    ("matmul_batched", lambda a, b: a @ b, [rand((2, 3, 4)), rand((2, 4, 5), 1)], (0, 1)),
    ("matmul_broadcast", lambda a, b: a @ b, [rand((2, 3, 4)), rand((4, 5), 1)], (0, 1)),
    ("matmul_vector", lambda a, b: a @ b, [rand((3, 4)), rand((4,), 1)], (0, 1)),
    ("einsum", lambda a, b: ops.einsum("bij,bjk->bik", a, b), [rand((2, 3, 4)), rand((2, 4, 5), 1)], (0, 1)),
    ("conv2d_same", lambda x, w: ops.conv2d(x, w), [rand((2, 6, 6, 3)), rand((3, 3, 3, 4), 1)], (0, 1)),
    ("conv2d_valid_strided", lambda x, w: ops.conv2d(x, w, (2, 2), "VALID"), [rand((2, 7, 7, 2)), rand((3, 3, 2, 3), 1)], (0, 1)),
    ("max_pool", lambda x: ops.max_pool2d(x), [rand((2, 4, 4, 3))], (0,)),
    ("avg_pool", lambda x: ops.avg_pool2d(x, (3, 3), (1, 1), "SAME"), [rand((2, 4, 4, 3))], (0,)),
    ("softmax", lambda x: ops.softmax(x, -1), [rand((3, 5))], (0,)),
    ("log_softmax", lambda x: ops.log_softmax(x, 0), [rand((3, 5))], (0,)),
    ("logsumexp", lambda x: ops.logsumexp(x, -1), [rand((3, 5))], (0,)),
    ("gelu", ops.gelu, [rand((6,))], (0,)),
    ("clip", lambda x: ops.clip(x, -0.5, 0.5), [rand((6,))], (0,)),
    ("stop_gradient", lambda x: ops.stop_gradient(x) * x, [rand((4,))], (0,)),
]


@pytest.mark.parametrize("name,fn,args,argnums", CASES, ids=[c[0] for c in CASES])
def test_forward_matches_eager(name, fn, args, argnums):
    eager = fn(*args)
    graph = sx.function(fn)(*args)
    tree_allclose(eager, graph, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name,fn,args,argnums", CASES, ids=[c[0] for c in CASES])
def test_graph_gradients_match_jax(name, fn, args, argnums):
    loss = weighted_sum(fn)
    v_graph, g_graph = graph_value_and_grad(loss, *args, argnums=argnums)
    v_jax, g_jax = jax_value_and_grad(loss, *args, argnums=argnums)
    np.testing.assert_allclose(v_graph, v_jax, rtol=1e-5, atol=1e-5)
    tree_allclose(g_graph, g_jax, rtol=1e-4, atol=1e-5)


def test_nondiff_ops_have_zero_gradient():
    x = rand((5,))
    loss = lambda x: ops.sum(ops.cast(ops.argmax(x[None], -1), jnp.float32)) + ops.sum(ops.sign(x))
    _, g = graph_value_and_grad(loss, x)
    np.testing.assert_array_equal(g, np.zeros(5, np.float32))


def test_one_hot_and_random_ops_run_in_graph():
    import jax

    key = jax.random.PRNGKey(3)
    f = lambda k, y: (ops.one_hot(y, 4), ops.bernoulli(ops.random_split(k, 2)[1], 0.5, (8,)))
    eager = f(key, jnp.array([0, 3, 1]))
    graph = sx.function(f)(key, jnp.array([0, 3, 1]))
    tree_allclose(eager, graph)


def test_ops_on_jax_arrays_are_plain_jax():
    x = rand((3,))
    assert isinstance(ops.exp(x), type(jnp.exp(x)))
    np.testing.assert_allclose(ops.exp(x), jnp.exp(x))
