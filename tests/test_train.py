"""Training: all run modes compute the same training run."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from voltax import data, losses, nn, optim, train

from helpers import tree_allclose


def _task(kind):
    key = jax.random.PRNGKey(0)
    if kind == "mlp":
        model, in_shape = nn.MLP([32, 32, 4]), (None, 10)
        x = jax.random.normal(key, (16, 10))
        y = jnp.arange(16) % 4
    elif kind == "cnn":
        model = nn.Sequential([nn.Conv2D(4), nn.relu, nn.MaxPool2D(), nn.Flatten(), nn.Dense(3)])
        in_shape, x, y = (None, 8, 8, 1), jax.random.normal(key, (6, 8, 8, 1)), jnp.arange(6) % 3
    else:
        model = nn.Sequential([
            nn.Embedding(20, 16), nn.PositionalEmbedding(8),
            nn.TransformerBlock(16, 2, 32, dropout=0.1), nn.LayerNorm(), nn.Dense(20),
        ])
        in_shape = (None, 8)
        x = jax.random.randint(key, (4, 8), 0, 20)
        y = jnp.roll(x, -1, axis=1)
    params = model.init(jax.random.PRNGKey(1), in_shape)

    def loss_fn(p, batch, key):
        xb, yb = batch
        logits = model.apply(p, xb, key=key, train=key is not None)
        return losses.cross_entropy(logits, yb), {"acc": losses.accuracy(logits, yb)}

    return params, (x, y), loss_fn


@pytest.mark.parametrize("kind", ["mlp", "cnn", "transformer"])
def test_modes_agree(kind):
    params, batch, loss_fn = _task(kind)
    opt = optim.sgd(0.05, momentum=0.9)
    key = jax.random.PRNGKey(7) if kind == "transformer" else None  # dropout on
    results = {}
    for mode in train.MODES:
        state = train.TrainState.create(params, opt, key=key)
        step = train.compile_step(train.make_train_step(loss_fn, opt, has_aux=True), mode)
        history = []
        for _ in range(4):
            state, metrics = step(state, batch)
            history.append(float(metrics["loss"]))
        results[mode] = (state, history)
    ref_state, ref_hist = results["eager"]
    for mode, (state, hist) in results.items():
        np.testing.assert_allclose(hist, ref_hist, rtol=1e-5, err_msg=mode)
        tree_allclose(state.params, ref_state.params, rtol=1e-4, atol=1e-5)
        assert int(state.step) == 4
    assert ref_hist[-1] < ref_hist[0]


def test_graph_mode_learns_and_is_reproducible():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 2)) * 4
    labels = rng.integers(0, 3, 600)
    xs = (centers[labels] + rng.normal(size=(600, 2))).astype(np.float32)
    model = nn.MLP([32, 3])
    opt = optim.adam(1e-2)

    def loss_fn(p, batch, key):
        return losses.cross_entropy(model.apply(p, batch[0]), batch[1])

    def run():
        state = train.TrainState.create(model.init(jax.random.PRNGKey(0), (None, 2)), opt)
        step = train.compile_step(train.make_train_step(loss_fn, opt), "graph")
        hist = []
        for batch in data.batches((xs, labels), 64, seed=1, epochs=5):
            state, m = step(state, batch)
            hist.append(float(m["loss"]))
        return state, hist, step.trace_count

    s1, hist, traces = run()
    s2, _, _ = run()
    assert hist[-1] < hist[0] / 2 and traces == 1
    tree_allclose(s1.params, s2.params, rtol=0, atol=0)


def test_batches():
    xs, ys = np.arange(10), np.arange(10) * 2
    out = list(data.batches((xs, ys), 3, shuffle=False))
    assert [b[0].tolist() for b in out] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert len(list(data.batches((xs, ys), 3, drop_remainder=False))) == 4
    a = [b[0].tolist() for b in data.batches((xs, ys), 5, seed=3)]
    b = [b[0].tolist() for b in data.batches((xs, ys), 5, seed=3)]
    assert a == b and sorted(sum(a, [])) == list(range(10))
    with pytest.raises(ValueError):
        next(data.batches((xs, ys[:5]), 2))


def test_unknown_mode():
    with pytest.raises(ValueError):
        train.compile_step(lambda s, b: s, "turbo")
