"""Parameter initializers: `init(key, shape, dtype) -> array`.

Initialization runs eagerly with jax.random, outside any trace.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def _fans(shape) -> tuple[int, int]:
    # Dense (in, out); conv (kh, kw, in, out): the receptive field multiplies both.
    receptive = math.prod(shape[:-2]) if len(shape) > 2 else 1
    fan_in = shape[-2] * receptive if len(shape) > 1 else shape[0]
    return fan_in, shape[-1] * receptive


def zeros(key, shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)


def ones(key, shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype)


def normal(stddev: float = 1e-2):
    def init(key, shape, dtype=jnp.float32):
        return stddev * jax.random.normal(key, shape, dtype)

    return init


def glorot_uniform(key, shape, dtype=jnp.float32):
    fan_in, fan_out = _fans(shape)
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype, -limit, limit)


def he_normal(key, shape, dtype=jnp.float32):
    fan_in, _ = _fans(shape)
    return math.sqrt(2.0 / fan_in) * jax.random.normal(key, shape, dtype)


def lecun_normal(key, shape, dtype=jnp.float32):
    fan_in, _ = _fans(shape)
    return math.sqrt(1.0 / fan_in) * jax.random.normal(key, shape, dtype)
