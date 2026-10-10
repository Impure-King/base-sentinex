"""Training utilities: an explicit train state, a pure train step, and run modes.

    state = TrainState.create(params, optimizer, key=jax.random.PRNGKey(0))
    step = make_train_step(loss_fn, optimizer)       # pure: (state, batch) -> (state, metrics)
    step = compile_step(step, "graph")                # or "eager", "jit", "graph_xla"
    for batch in data:
        state, metrics = step(state, batch)

The step is pure: everything that changes (params, optimizer state, step count,
RNG key) goes in and comes out. That is what lets jax.jit and vx.function cache
it, and what makes a run reproducible from its seed.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from voltax import ops
from voltax.autodiff import value_and_grad
from voltax.graph.function import function
from voltax.optim import GradientTransformation, apply_updates

MODES = ("eager", "jit", "graph", "graph_xla")


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    step: Any  # int32 scalar array
    key: Any = None  # uint32[2] PRNG key (jax.random.PRNGKey), or None if unused

    @classmethod
    def create(cls, params, optimizer: GradientTransformation, key=None) -> "TrainState":
        return cls(params, optimizer.init(params), jnp.zeros([], jnp.int32), key)


def make_train_step(loss_fn: Callable, optimizer: GradientTransformation, has_aux: bool = False):
    """Build `train_step(state, batch) -> (state, metrics)`.

    `loss_fn(params, batch, key)` returns the scalar loss, or `(loss, metrics_dict)`
    when `has_aux=True`. `key` is a fresh PRNG key per step (None if the state has
    no key).
    """

    def train_step(state: TrainState, batch):
        key, next_key = None, None
        if state.key is not None:
            keys = ops.random_split(state.key, 2)
            next_key, key = keys[0], keys[1]
        (loss, aux), grads = value_and_grad(_with_aux(loss_fn, has_aux), has_aux=True)(state.params, batch, key)
        updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
        params = apply_updates(state.params, updates)
        return TrainState(params, opt_state, state.step + 1, next_key), {"loss": loss, **aux}

    return train_step


def _with_aux(loss_fn, has_aux):
    if has_aux:
        return loss_fn
    return lambda params, batch, key: (loss_fn(params, batch, key), {})


def compile_step(step: Callable, mode: str) -> Callable:
    """Wrap a pure step function for one of the run modes in `MODES`."""
    if mode == "eager":
        return step
    if mode == "jit":
        return jax.jit(step)
    if mode == "graph":
        return function(step)
    if mode == "graph_xla":
        return function(step, jit_compile=True)
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
