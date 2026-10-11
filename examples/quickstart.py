"""Train one model in every run mode and compare.

    uv run python examples/quickstart.py

A small MLP classifies points from three Gaussian blobs. The model, loss and
optimizer are defined once; only the way the train step is run changes.
"""

import time

import jax
import numpy as np

from sentinex import data, losses, nn, optim, train


def make_dataset(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(3, 2)) * 3
    labels = rng.integers(0, 3, n)
    points = centers[labels] + rng.normal(size=(n, 2))
    return points.astype(np.float32), labels.astype(np.int32)


model = nn.Sequential([nn.Dense(64), nn.relu, nn.Dropout(0.1), nn.Dense(64), nn.relu, nn.Dense(3)])
optimizer = optim.adamw(3e-3, weight_decay=1e-4)


def loss_fn(params, batch, key):
    x, y = batch
    logits = model.apply(params, x, key=key, train=True)
    return losses.cross_entropy(logits, y), {"accuracy": losses.accuracy(logits, y)}


def main():
    xs, ys = make_dataset()
    params = model.init(jax.random.PRNGKey(0), (None, 2))
    print(f"model: {nn.param_count(params):,} parameters")
    for mode in train.MODES:
        state = train.TrainState.create(params, optimizer, key=jax.random.PRNGKey(1))
        step = train.compile_step(train.make_train_step(loss_fn, optimizer, has_aux=True), mode)
        t0 = time.perf_counter()
        for batch in data.batches((xs, ys), 100, seed=0, epochs=3):
            state, metrics = step(state, batch)
        jax.block_until_ready(state)
        extra = f", traced {step.trace_count}x" if hasattr(step, "trace_count") else ""
        print(
            f"{mode:10s} loss {float(metrics['loss']):.4f}  accuracy {float(metrics['accuracy']):.3f}  "
            f"{int(state.step)} steps in {time.perf_counter() - t0:.2f}s (incl. compile{extra})"
        )


if __name__ == "__main__":
    main()
