"""Optimizers, losses and schedules against Optax."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import sentinex as sx
from sentinex import losses, ops, optim

from helpers import rand, tree_allclose

PAIRS = [
    ("sgd", optim.sgd(0.1), optax.sgd(0.1)),
    ("momentum", optim.sgd(0.1, momentum=0.9), optax.sgd(0.1, momentum=0.9)),
    ("nesterov", optim.sgd(0.1, momentum=0.9, nesterov=True), optax.sgd(0.1, momentum=0.9, nesterov=True)),
    ("adam", optim.adam(1e-2), optax.adam(1e-2)),
    ("adamw", optim.adamw(1e-2, weight_decay=0.1), optax.adamw(1e-2, weight_decay=0.1)),
    ("rmsprop", optim.rmsprop(1e-2), optax.rmsprop(1e-2)),
    ("clip+adam", optim.chain(optim.clip_by_global_norm(0.5), optim.adam(1e-2)),
     optax.chain(optax.clip_by_global_norm(0.5), optax.adam(1e-2))),
    ("adam_schedule", optim.adam(optim.cosine_decay_schedule(1e-2, 10)), optax.adam(optax.cosine_decay_schedule(1e-2, 10))),
]


def _loss(params):
    return jnp.sum((params["w"] - 1.0) ** 2) + jnp.sum(jnp.sin(params["b"]) * 3)


def _run(opt, steps=6):
    params = {"w": rand((3, 4)), "b": rand((4,), 1)}
    state = opt.init(params)
    for _ in range(steps):
        grads = jax.grad(_loss)(params)
        updates, state = opt.update(grads, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
    return params


@pytest.mark.parametrize("name,ours,ref", PAIRS, ids=[p[0] for p in PAIRS])
def test_matches_optax(name, ours, ref):
    tree_allclose(_run(ours), _run(ref), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name,ours,ref", PAIRS, ids=[p[0] for p in PAIRS])
def test_update_in_graph_mode_matches_eager(name, ours, ref):
    params = {"w": rand((3, 4)), "b": rand((4,), 1)}
    grads = jax.grad(_loss)(params)
    state = ours.init(params)
    eager = ours.update(grads, state, params)
    graph = sx.function(ours.update)(grads, state, params)
    tree_allclose(graph, eager, rtol=1e-6, atol=1e-7)


def test_schedules_match_optax():
    pairs = [
        (optim.linear_schedule(1.0, 0.1, 10), optax.linear_schedule(1.0, 0.1, 10)),
        (optim.cosine_decay_schedule(0.5, 20, alpha=0.1), optax.cosine_decay_schedule(0.5, 20, alpha=0.1)),
        (optim.warmup_cosine_decay_schedule(1.0, 5, 20, 0.05), optax.warmup_cosine_decay_schedule(0.0, 1.0, 5, 20, 0.05)),
    ]
    for ours, ref in pairs:
        for step in [0, 1, 4, 5, 6, 12, 20, 25]:
            np.testing.assert_allclose(ours(jnp.int32(step)), ref(step), rtol=1e-6, atol=1e-7)


def test_global_norm():
    tree = {"a": rand((3,)), "b": [rand((2, 2), 1)]}
    np.testing.assert_allclose(optim.global_norm(tree), optax.tree.norm(tree), rtol=1e-6)


def test_losses_match_optax():
    logits, labels = rand((4, 3, 5)), jnp.array([[0, 1, 4], [2, 2, 3], [1, 0, 0], [4, 4, 1]])
    ref = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    np.testing.assert_allclose(losses.cross_entropy(logits, labels, "none"), ref, rtol=1e-6)
    np.testing.assert_allclose(losses.cross_entropy(logits, labels), ref.mean(), rtol=1e-6)
    targets = jax.random.uniform(jax.random.PRNGKey(0), (6,))
    np.testing.assert_allclose(
        losses.binary_cross_entropy(rand((6,)) * 5, targets, "none"),
        optax.sigmoid_binary_cross_entropy(rand((6,)) * 5, targets), rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(losses.mse(rand((5,)), rand((5,), 1)), jnp.mean((rand((5,)) - rand((5,), 1)) ** 2), rtol=1e-6)
    np.testing.assert_allclose(losses.accuracy(jnp.eye(3)[jnp.array([0, 2, 1, 1])], jnp.array([0, 2, 0, 1])), 0.75)


def test_losses_in_graph_mode():
    logits, labels = rand((4, 5)), jnp.array([0, 1, 4, 2])
    f = lambda z, y: (losses.cross_entropy(z, y), losses.accuracy(z, y), losses.binary_cross_entropy(z, ops.sigmoid(z)))
    tree_allclose(sx.function(f)(logits, labels), f(logits, labels), rtol=1e-6)
