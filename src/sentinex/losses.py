"""Losses and metrics, written with sentinex.ops so they work in every mode."""

from __future__ import annotations

import jax.numpy as jnp

from sentinex import ops


def _reduce(x, reduction: str):
    if reduction == "mean":
        return ops.mean(x)
    if reduction == "sum":
        return ops.sum(x)
    if reduction == "none":
        return x
    raise ValueError(f"unknown reduction {reduction!r}")


def cross_entropy(logits, labels, reduction: str = "mean"):
    """Softmax cross-entropy with integer labels. logits: (..., classes)."""
    logp = ops.log_softmax(logits, -1)
    onehot = ops.one_hot(labels, ops.shape_of(logits)[-1], ops.dtype_of(logits))
    return _reduce(-ops.sum(onehot * logp, -1), reduction)


def log_sigmoid(x):
    # -softplus(-x), computed stably: min(x, 0) - log1p(exp(-|x|)).
    return ops.minimum(x, 0.0) - ops.log1p(ops.exp(-ops.abs(x)))


def binary_cross_entropy(logits, targets, reduction: str = "mean"):
    """Sigmoid cross-entropy on logits; targets in [0, 1]."""
    loss = -(targets * log_sigmoid(logits) + (1.0 - targets) * log_sigmoid(-logits))
    return _reduce(loss, reduction)


def mse(pred, target, reduction: str = "mean"):
    return _reduce(ops.square(pred - target), reduction)


def accuracy(logits, labels):
    hits = ops.equal(ops.argmax(logits, -1), labels)
    return ops.mean(ops.cast(hits, jnp.float32))
