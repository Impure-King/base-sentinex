"""Graph passes and the executor."""

import jax.numpy as jnp
import numpy as np
from helpers import rand, tree_allclose

import sentinex as sx
from sentinex import ops
from sentinex.graph import KernelStats, function, passes


def test_dce_removes_unused_work():
    def f(x):
        unused = ops.exp(x) * 3  # noqa: F841
        return x + 1

    g = function(f).graph(rand((3,)))
    assert g.op_counts() == {"add": 1}


def test_cse_merges_duplicates():
    g = function(lambda x: ops.exp(x) + ops.exp(x)).graph(rand((3,)))
    assert g.op_counts()["exp"] == 1


def test_ops_on_plain_arrays_run_at_trace_time():
    # No Symbol argument -> the op runs eagerly while tracing; the result is a const.
    small = np.arange(4, dtype=np.float32)
    g = function(lambda x: x + ops.exp(small)).graph(rand((4,)))
    assert g.op_counts() == {"add": 1}


def test_constant_folding_and_size_cap():
    # d mean(x)/dx = 1/n everywhere: it depends only on the constant gradient seed.
    grad_mean = function(sx.grad(lambda x: ops.mean(x)))
    small = grad_mean.graph(rand((4,)))
    assert small.ops == [] and small.outputs[0].kind == "const"
    np.testing.assert_allclose(grad_mean(rand((4,))), np.full(4, 0.25))
    big = grad_mean.graph(rand((100, 100)))  # 10k elements: over the fold cap
    assert big.op_counts() == {"broadcast_to": 1}


def test_identity_nodes_are_forwarded():
    def f(x):
        return sx.value_and_grad(lambda y: ops.sum(y * y))(x)

    traced = function(f, optimize=False).graph(rand((3,)))
    optimized = function(f).graph(rand((3,)))
    assert traced.op_counts()["identity"] == 1
    assert "identity" not in optimized.op_counts()


def test_optimization_preserves_results():
    def f(x, w):
        loss, g = sx.value_and_grad(lambda w: ops.sum(ops.tanh(x @ w) ** 2))(w)
        return loss, g

    args = (rand((4, 3)), rand((3, 2), 1))
    tree_allclose(function(f)(*args), function(f, optimize=False)(*args))


def test_pipeline_reports_counts():
    raw = function(lambda x: ops.exp(x) + ops.exp(x), optimize=False).graph(rand((3,)))
    g, stats = passes.optimize(raw)
    assert list(stats) == ["traced", "forward_identities", "fold_constants", "cse", "dce"]
    assert stats["traced"] == 3 and stats["dce"] == 2 and len(g.ops) == 2


def test_executor_frees_each_intermediate_once():
    f = function(lambda x: ops.tanh(ops.exp(x) * 2) + x)
    cf = f.get_concrete_function(rand((5,)))
    program = cf.executable.program
    freed = [s for (_, _, _, dead) in program for s in dead]
    outputs = {n.id for n in cf.graph.outputs}
    assert len(freed) == len(set(freed))
    assert not outputs & set(freed)
    intermediates = {out for (_, _, out, _) in program} - outputs
    assert intermediates <= set(freed)


def test_kernel_cache_is_shared_across_graphs():
    x = rand((7, 3))
    function(lambda a: ops.sigmoid(a) * 3.5)(x)
    before = KernelStats.compiles
    function(lambda b: ops.sigmoid(b) * 3.5)(x)  # same ops, params, literals, shapes
    assert KernelStats.compiles == before


def test_profile_reports_ops():
    f = function(lambda x: ops.exp(x) @ x)
    cf = f.get_concrete_function(rand((8, 8)))
    rows = cf.executable.profile([rand((8, 8))], repeats=2)
    assert {r[0] for r in rows} == {"exp", "matmul"}
    assert all(t >= 0 for _, _, t in rows)


def test_literals_keep_weak_typing():
    f = function(lambda x: x * 2.0 + 1)
    out = f(jnp.ones(3, jnp.float16))
    assert out.dtype == jnp.float16
