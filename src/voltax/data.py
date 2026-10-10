"""Minimal batching for in-memory datasets (numpy in, device arrays out)."""

from __future__ import annotations

from typing import Iterator, Sequence

import jax
import numpy as np


def batches(
    arrays: Sequence[np.ndarray],
    batch_size: int,
    *,
    shuffle: bool = True,
    seed: int = 0,
    drop_remainder: bool = True,
    epochs: int | None = 1,
) -> Iterator[tuple]:
    """Yield tuples of batches. Fixed batch shapes (drop_remainder=True) avoid
    retracing in jax.jit and vx.function."""
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays need the same leading dimension")
    rng = np.random.default_rng(seed)
    epoch = 0
    while epochs is None or epoch < epochs:
        order = rng.permutation(n) if shuffle else np.arange(n)
        stop = n - n % batch_size if drop_remainder else n
        for start in range(0, stop, batch_size):
            idx = order[start : start + batch_size]
            yield tuple(jax.device_put(a[idx]) for a in arrays)
        epoch += 1
