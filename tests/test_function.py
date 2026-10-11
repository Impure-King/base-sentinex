"""sx.function: tracing, signature caching, retracing, errors, composition."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sentinex as sx
from sentinex import ops
from sentinex.graph import TracingError, function

from helpers import rand, tree_allclose


def test_traces_once_per_signature():
    calls = []

    @sx.function
    def f(x, y):
        calls.append(1)  # Python side effects only happen while tracing
        return ops.tanh(x) * y

    for seed in range(4):
        f(rand((3,), seed), rand((3,), seed + 1))
    assert f.trace_count == 1 and len(calls) == 1 and f.cache_size == 1


@pytest.mark.parametrize(
    "first,second",
    [
        ((jnp.ones(3),), (jnp.ones(4),)),  # new shape
        ((jnp.ones(3),), (jnp.ones(3, jnp.int32),)),  # new dtype
        ((jnp.ones(3),), ((jnp.ones(3), jnp.ones(3)),)),  # new pytree structure
    ],
)
def test_retraces_on_new_signature(first, second):
    f = sx.function(lambda x: jax.tree.map(lambda a: a + 1, x))
    f(*first)
    f(*first)
    f(*second)
    assert f.trace_count == 2


def test_python_scalars_are_static():
    f = sx.function(lambda x, n: x * n)
    np.testing.assert_allclose(f(jnp.ones(2), 2.0), [2.0, 2.0])
    np.testing.assert_allclose(f(jnp.ones(2), 3.0), [3.0, 3.0])
    f(jnp.ones(2), 3.0)
    assert f.trace_count == 2
    # The same number as an array is a traced input: no retrace for new values.
    g = sx.function(lambda x, n: x * n)
    g(jnp.ones(2), jnp.float32(2.0))
    g(jnp.ones(2), jnp.float32(3.0))
    assert g.trace_count == 1


def test_static_python_control_flow_is_allowed():
    @sx.function
    def f(x, mode):
        return ops.relu(x) if mode == "relu" else ops.tanh(x)

    x = rand((5,))
    np.testing.assert_allclose(f(x, "relu"), ops.relu(x))
    np.testing.assert_allclose(f(x, "tanh"), ops.tanh(x), rtol=1e-6)


def test_retrace_warning():
    f = sx.function(lambda x: x + 1)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for n in range(1, 8):
            f(jnp.ones(n))
    assert any("traced" in str(m.message) for m in w)


def test_bool_of_symbol_raises():
    @sx.function
    def f(x):
        if ops.sum(x) > 0:  # data-dependent Python control flow
            return x
        return -x

    with pytest.raises(TracingError, match="where"):
        f(jnp.ones(3))


def test_jax_numpy_on_symbol_raises():
    f = sx.function(lambda x: jnp.sin(x))  # graph mode needs sx.ops, not jax.numpy
    with pytest.raises(TypeError, match="Symbol"):
        f(jnp.ones(3))


def test_mixed_jax_array_and_symbol_arithmetic():
    # jax.Array operators return NotImplemented for Symbols, so Python falls back
    # to Symbol's reflected operators and the op is recorded.
    w = jnp.array([1.0, 2.0])
    x = jnp.array([2.0, 4.0])
    f = sx.function(lambda x: (w * x, w + x, w - x, w / x, w @ x))
    tree_allclose(f(x), (w * x, w + x, w - x, w / x, w @ x))


def test_leaked_symbol_raises():
    leaked = []

    @sx.function
    def f(x):
        leaked.append(x)
        return x * 2

    f(jnp.ones(2))
    g = sx.function(lambda y: y + leaked[0])
    with pytest.raises(TracingError):
        g(jnp.ones(2))


def test_closures_are_captured_by_value():
    w = jnp.array([1.0, 2.0])
    f = sx.function(lambda x: x * w)
    np.testing.assert_allclose(f(jnp.ones(2)), [1.0, 2.0])
    w = jnp.array([5.0, 5.0])  # rebinding does not affect the cached graph
    np.testing.assert_allclose(f(jnp.ones(2)), [1.0, 2.0])


def test_nested_functions_inline():
    inner = sx.function(lambda x: ops.exp(x))
    outer = sx.function(lambda x: inner(x) + 1)
    np.testing.assert_allclose(outer(jnp.zeros(2)), [2.0, 2.0])
    assert inner.trace_count == 0 and outer.trace_count == 1


def test_output_trees_constants_and_passthrough():
    @sx.function
    def f(x, d):
        return {"same": x, "const": np.float32(3.0), "pair": (d["a"] * 2, None)}

    out = f(jnp.ones(2), {"a": jnp.ones(3)})
    np.testing.assert_allclose(out["same"], [1.0, 1.0])
    np.testing.assert_allclose(out["const"], 3.0)
    np.testing.assert_allclose(out["pair"][0], [2.0, 2.0, 2.0])
    assert out["pair"][1] is None


def test_kwargs_and_numpy_inputs():
    f = sx.function(lambda x, *, scale: x * scale)
    np.testing.assert_allclose(f(np.ones(3, np.float32), scale=jnp.float32(2.0)), [2.0] * 3)


def test_jit_compile_mode_matches():
    fn = lambda x, w: ops.softmax(x @ w, -1)
    x, w = rand((4, 3)), rand((3, 5), 1)
    tree_allclose(function(fn)(x, w), function(fn, jit_compile=True)(x, w))


def test_composes_with_jax_transforms():
    # Under jax.jit / jax.grad the cached graph is replayed through JAX.
    f = sx.function(lambda x: ops.sum(ops.sin(x) * x))
    x = rand((5,))
    np.testing.assert_allclose(jax.jit(f)(x), jnp.sum(jnp.sin(x) * x), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(f)(x), jnp.sin(x) + x * jnp.cos(x), rtol=1e-5)


def test_graph_inspection():
    f = sx.function(lambda x: ops.relu(x @ x.T))
    g = f.graph(rand((3, 4)))
    text = g.pretty()
    assert "matmul" in text and "relu" in text and "transpose" in text
    assert g.op_counts()["matmul"] == 1
    cf = f.get_concrete_function(rand((3, 4)))
    assert cf.pass_stats["traced"] >= len(g.ops)
