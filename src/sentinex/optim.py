"""Optimizers as composable gradient transformations (the Optax design).

A `GradientTransformation` is a pair of pure functions:

    state = opt.init(params)
    updates, state = opt.update(grads, state, params)
    params = apply_updates(params, updates)

Transformations compose with `chain`; `adam` is `chain(scale_by_adam(...),
scale_by_learning_rate(lr))`. The math follows Optax step for step, so results
match `optax` to float rounding (tests/test_optim.py checks this). Because the
update functions only use sentinex.ops and array operators, optimizers trace into
graph mode like everything else.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from sentinex import ops

Schedule = Callable[[Any], Any]


class GradientTransformation(NamedTuple):
    init: Callable[[Any], Any]
    update: Callable[..., tuple[Any, Any]]


class EmptyState(NamedTuple):
    pass


def _tmap(fn, *trees):
    return jax.tree.map(fn, *trees)


def _zeros_like_tree(params):
    return _tmap(lambda p: jnp.zeros(p.shape, p.dtype), params)


def _count():
    return jnp.zeros([], jnp.int32)


def apply_updates(params, updates):
    return _tmap(lambda p, u: p + u, params, updates)


def global_norm(tree):
    return ops.sqrt(sum(ops.sum(ops.square(x)) for x in jax.tree.leaves(tree)))


def chain(*transforms: GradientTransformation) -> GradientTransformation:
    def init(params):
        return tuple(t.init(params) for t in transforms)

    def update(updates, state, params=None):
        new_state = []
        for t, s in zip(transforms, state):
            updates, s = t.update(updates, s, params)
            new_state.append(s)
        return updates, tuple(new_state)

    return GradientTransformation(init, update)


def identity() -> GradientTransformation:
    return GradientTransformation(lambda params: EmptyState(), lambda u, s, p=None: (u, s))


def scale(factor: float) -> GradientTransformation:
    def update(updates, state, params=None):
        return _tmap(lambda g: factor * g, updates), state

    return GradientTransformation(lambda params: EmptyState(), update)


class ScaleByScheduleState(NamedTuple):
    count: Any


def scale_by_schedule(schedule: Schedule) -> GradientTransformation:
    def init(params):
        return ScaleByScheduleState(_count())

    def update(updates, state, params=None):
        step_size = schedule(state.count)
        return _tmap(lambda g: step_size * g, updates), ScaleByScheduleState(state.count + 1)

    return GradientTransformation(init, update)


def scale_by_learning_rate(lr: float | Schedule) -> GradientTransformation:
    if callable(lr):
        return scale_by_schedule(lambda count: -1.0 * lr(count))
    return scale(-lr)


class TraceState(NamedTuple):
    trace: Any


def trace(decay: float, nesterov: bool = False) -> GradientTransformation:
    """Momentum: t <- g + decay * t."""

    def update(updates, state, params=None):
        f = lambda g, t: g + decay * t  # noqa: E731
        new_trace = _tmap(f, updates, state.trace)
        updates = _tmap(f, updates, new_trace) if nesterov else new_trace
        return updates, TraceState(new_trace)

    return GradientTransformation(lambda params: TraceState(_zeros_like_tree(params)), update)


class ScaleByAdamState(NamedTuple):
    count: Any
    mu: Any
    nu: Any


def scale_by_adam(b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0) -> GradientTransformation:
    def init(params):
        return ScaleByAdamState(_count(), _zeros_like_tree(params), _zeros_like_tree(params))

    def update(updates, state, params=None):
        mu = _tmap(lambda g, t: (1 - b1) * g + b1 * t, updates, state.mu)
        nu = _tmap(lambda g, t: (1 - b2) * (g * g) + b2 * t, updates, state.nu)
        count = state.count + 1
        bc1, bc2 = 1 - b1**count, 1 - b2**count
        mu_hat = _tmap(lambda t: t / bc1, mu)
        nu_hat = _tmap(lambda t: t / bc2, nu)
        updates = _tmap(lambda m, v: m / (ops.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
        return updates, ScaleByAdamState(count, mu, nu)

    return GradientTransformation(init, update)


class ScaleByRmsState(NamedTuple):
    nu: Any


def scale_by_rms(decay=0.9, eps=1e-8) -> GradientTransformation:
    def init(params):
        return ScaleByRmsState(_zeros_like_tree(params))

    def update(updates, state, params=None):
        nu = _tmap(lambda g, t: (1 - decay) * (g * g) + decay * t, updates, state.nu)
        updates = _tmap(lambda g, n: g * ops.rsqrt(n + eps), updates, nu)
        return updates, ScaleByRmsState(nu)

    return GradientTransformation(init, update)


def add_decayed_weights(weight_decay: float) -> GradientTransformation:
    def update(updates, state, params=None):
        return _tmap(lambda g, p: g + weight_decay * p, updates, params), state

    return GradientTransformation(lambda params: EmptyState(), update)


def clip_by_global_norm(max_norm: float) -> GradientTransformation:
    def update(updates, state, params=None):
        norm = global_norm(updates)
        small = norm < max_norm
        return _tmap(lambda t: ops.where(small, t, (t / norm) * max_norm), updates), state

    return GradientTransformation(lambda params: EmptyState(), update)


# ------------------------------------------------------------------- optimizers


def sgd(lr: float | Schedule, momentum: float | None = None, nesterov: bool = False):
    parts = [trace(momentum, nesterov)] if momentum is not None else []
    return chain(*parts, scale_by_learning_rate(lr))


def adam(lr: float | Schedule, b1=0.9, b2=0.999, eps=1e-8):
    return chain(scale_by_adam(b1, b2, eps), scale_by_learning_rate(lr))


def adamw(lr: float | Schedule, b1=0.9, b2=0.999, eps=1e-8, weight_decay=1e-4):
    return chain(scale_by_adam(b1, b2, eps), add_decayed_weights(weight_decay), scale_by_learning_rate(lr))


def rmsprop(lr: float | Schedule, decay=0.9, eps=1e-8):
    return chain(scale_by_rms(decay, eps), scale_by_learning_rate(lr))


# -------------------------------------------------------------------- schedules
# A schedule maps the step count (an int32 array or Symbol) to a learning rate.


def constant_schedule(value: float) -> Schedule:
    return lambda count: value


def linear_schedule(init_value: float, end_value: float, transition_steps: int) -> Schedule:
    def schedule(count):
        frac = ops.clip(ops.cast(count, jnp.float32) / transition_steps, 0.0, 1.0)
        return init_value + (end_value - init_value) * frac

    return schedule


def cosine_decay_schedule(init_value: float, decay_steps: int, alpha: float = 0.0) -> Schedule:
    def schedule(count):
        frac = ops.minimum(ops.cast(count, jnp.float32), float(decay_steps)) / decay_steps
        cosine = 0.5 * (1.0 + ops.cos(jnp.pi * frac))
        return init_value * ((1 - alpha) * cosine + alpha)

    return schedule


def warmup_cosine_decay_schedule(peak_value: float, warmup_steps: int, decay_steps: int, end_value=0.0) -> Schedule:
    warm = linear_schedule(0.0, peak_value, warmup_steps)
    alpha = end_value / peak_value if peak_value else 0.0
    cos = cosine_decay_schedule(peak_value, decay_steps - warmup_steps, alpha)

    def schedule(count):
        c = ops.cast(count, jnp.float32)
        return ops.where(c < warmup_steps, warm(c), cos(ops.maximum(c - warmup_steps, 0.0)))

    return schedule
