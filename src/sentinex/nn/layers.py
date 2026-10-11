"""Layers: configuration objects plus pure functions of explicit parameters.

A layer holds only hyperparameters. Its parameters are a separate pytree:

    model = nn.Sequential([nn.Dense(512), nn.relu, nn.Dense(10)])
    params = model.init(key, (None, 784))      # None = any batch size
    logits = model.apply(params, x)

`init(key, input_shape)` creates parameters, `output_shape(input_shape)` does
shape inference (so containers can chain layers), and `apply(params, x, key=None,
train=False)` is the forward pass. `apply` only uses `sentinex.ops` and array
operators, so the same layer runs eagerly, under `jax.jit`, and in graph mode.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sentinex import ops
from sentinex.nn import initializers as init_

Shape = tuple  # dims are ints, or None for "any size"


class Layer:
    def init(self, key, input_shape: Shape) -> Any:
        return {}

    def output_shape(self, input_shape: Shape) -> Shape:
        return tuple(input_shape)

    def apply(self, params, x, *, key=None, train: bool = False):
        raise NotImplementedError

    def __call__(self, params, x, *, key=None, train: bool = False):
        return self.apply(params, x, key=key, train=train)

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({fields})"


class Lambda(Layer):
    """A parameter-free layer from a function (activations, reshapes, ...)."""

    def __init__(self, fn: Callable, output_shape: Callable[[Shape], Shape] | None = None):
        self.fn = fn
        self._shape_fn = output_shape

    def output_shape(self, input_shape):
        return self._shape_fn(input_shape) if self._shape_fn else tuple(input_shape)

    def apply(self, params, x, *, key=None, train=False):
        return self.fn(x)

    def __repr__(self) -> str:
        return f"Lambda({getattr(self.fn, '__name__', self.fn)})"


def _as_layer(layer) -> Layer:
    if isinstance(layer, Layer):
        return layer
    if callable(layer):
        return Lambda(layer)
    raise TypeError(f"not a layer: {layer!r}")


class Sequential(Layer):
    """Run layers in order. Params are a tuple with one entry per layer."""

    def __init__(self, layers: Sequence[Layer | Callable]):
        self.layers = [_as_layer(layer) for layer in layers]

    def init(self, key, input_shape):
        params, shape = [], tuple(input_shape)
        for layer, k in zip(self.layers, jax.random.split(key, len(self.layers))):
            params.append(layer.init(k, shape))
            shape = layer.output_shape(shape)
        return tuple(params)

    def output_shape(self, input_shape):
        shape = tuple(input_shape)
        for layer in self.layers:
            shape = layer.output_shape(shape)
        return shape

    def apply(self, params, x, *, key=None, train=False):
        keys = [None] * len(self.layers)
        if key is not None:
            split = ops.random_split(key, len(self.layers))
            keys = [split[i] for i in range(len(self.layers))]
        for layer, p, k in zip(self.layers, params, keys):
            x = layer.apply(p, x, key=k, train=train)
        return x


class Residual(Layer):
    """x + layer(x)."""

    def __init__(self, layer: Layer):
        self.layer = _as_layer(layer)

    def init(self, key, input_shape):
        return self.layer.init(key, input_shape)

    def apply(self, params, x, *, key=None, train=False):
        return x + self.layer.apply(params, x, key=key, train=train)


class Dense(Layer):
    def __init__(self, units: int, use_bias: bool = True, w_init=init_.glorot_uniform, b_init=init_.zeros):
        self.units, self.use_bias = units, use_bias
        self.w_init, self.b_init = w_init, b_init

    def init(self, key, input_shape):
        kw, kb = jax.random.split(key)
        params = {"w": self.w_init(kw, (input_shape[-1], self.units))}
        if self.use_bias:
            params["b"] = self.b_init(kb, (self.units,))
        return params

    def output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def apply(self, params, x, *, key=None, train=False):
        y = x @ params["w"]
        return y + params["b"] if self.use_bias else y


class Conv2D(Layer):
    """2-D convolution on NHWC inputs with HWIO kernels."""

    def __init__(self, filters: int, kernel_size=(3, 3), strides=(1, 1), padding: str = "SAME"):
        self.filters = filters
        self.kernel_size = tuple(kernel_size)
        self.strides = tuple(strides)
        self.padding = padding

    def init(self, key, input_shape):
        kw, kb = jax.random.split(key)
        shape = (*self.kernel_size, input_shape[-1], self.filters)
        return {"w": init_.glorot_uniform(kw, shape), "b": init_.zeros(kb, (self.filters,))}

    def output_shape(self, input_shape):
        n, h, w, _ = input_shape
        spatial = _window_out((h, w), self.kernel_size, self.strides, self.padding)
        return (n, *spatial, self.filters)

    def apply(self, params, x, *, key=None, train=False):
        return ops.conv2d(x, params["w"], self.strides, self.padding) + params["b"]


def _window_out(spatial, window, strides, padding):
    if padding == "SAME":
        return tuple(-(-d // s) for d, s in zip(spatial, strides))
    return tuple(-(-(d - k + 1) // s) for d, k, s in zip(spatial, window, strides))


class _Pool(Layer):
    def __init__(self, window=(2, 2), strides=None, padding: str = "VALID"):
        self.window = tuple(window)
        self.strides = tuple(strides or window)
        self.padding = padding

    def output_shape(self, input_shape):
        n, h, w, c = input_shape
        return (n, *_window_out((h, w), self.window, self.strides, self.padding), c)


class MaxPool2D(_Pool):
    def apply(self, params, x, *, key=None, train=False):
        return ops.max_pool2d(x, self.window, self.strides, self.padding)


class AvgPool2D(_Pool):
    def apply(self, params, x, *, key=None, train=False):
        return ops.avg_pool2d(x, self.window, self.strides, self.padding)


class Flatten(Layer):
    """Flatten everything but the batch axis."""

    def output_shape(self, input_shape):
        return (input_shape[0], math.prod(input_shape[1:]))

    def apply(self, params, x, *, key=None, train=False):
        return ops.reshape(x, (ops.shape_of(x)[0], -1))


class Dropout(Layer):
    """Inverted dropout. Needs `key` when `train=True`."""

    def __init__(self, rate: float):
        self.rate = rate

    def apply(self, params, x, *, key=None, train=False):
        if not train or self.rate == 0.0:
            return x
        if key is None:
            raise ValueError("Dropout(train=True) needs a PRNG key")
        keep = ops.bernoulli(key, 1.0 - self.rate, ops.shape_of(x))
        return ops.where(keep, x / (1.0 - self.rate), 0.0)


class LayerNorm(Layer):
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def init(self, key, input_shape):
        d = input_shape[-1]
        return {"scale": jnp.ones((d,)), "bias": jnp.zeros((d,))}

    def apply(self, params, x, *, key=None, train=False):
        mu = ops.mean(x, -1, keepdims=True)
        var = ops.var(x, -1, keepdims=True)
        return (x - mu) * ops.rsqrt(var + self.eps) * params["scale"] + params["bias"]


class Embedding(Layer):
    """Integer ids (..., ) -> vectors (..., dim)."""

    def __init__(self, vocab_size: int, dim: int, stddev: float = 0.02):
        self.vocab_size, self.dim, self.stddev = vocab_size, dim, stddev

    def init(self, key, input_shape):
        return {"table": init_.normal(self.stddev)(key, (self.vocab_size, self.dim))}

    def output_shape(self, input_shape):
        return (*input_shape, self.dim)

    def apply(self, params, x, *, key=None, train=False):
        return ops.take(params["table"], x, axis=0)


class PositionalEmbedding(Layer):
    """Adds a learned position vector to each step of a (batch, time, dim) input."""

    def __init__(self, max_len: int, stddev: float = 0.02):
        self.max_len, self.stddev = max_len, stddev

    def init(self, key, input_shape):
        return {"pos": init_.normal(self.stddev)(key, (self.max_len, input_shape[-1]))}

    def apply(self, params, x, *, key=None, train=False):
        t = ops.shape_of(x)[-2]
        return x + params["pos"][:t]


class MultiHeadAttention(Layer):
    """Self-attention over (batch, time, dim) with fused QKV projection."""

    def __init__(self, num_heads: int, causal: bool = False):
        self.num_heads, self.causal = num_heads, causal

    def init(self, key, input_shape):
        d = input_shape[-1]
        if d % self.num_heads:
            raise ValueError(f"dim {d} not divisible by {self.num_heads} heads")
        k1, k2 = jax.random.split(key)
        return {
            "w_qkv": init_.glorot_uniform(k1, (d, 3 * d)),
            "b_qkv": jnp.zeros((3 * d,)),
            "w_out": init_.glorot_uniform(k2, (d, d)),
            "b_out": jnp.zeros((d,)),
        }

    def apply(self, params, x, *, key=None, train=False):
        b, t, d = ops.shape_of(x)
        h = self.num_heads
        qkv = x @ params["w_qkv"] + params["b_qkv"]  # (B, T, 3D)
        qkv = ops.transpose(ops.reshape(qkv, (b, t, 3, h, d // h)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, Dh)
        scores = (q @ ops.swapaxes(k, -1, -2)) * (1.0 / math.sqrt(d // h))
        if self.causal:
            mask = np.tril(np.ones((t, t), dtype=bool))
            scores = ops.where(mask, scores, float(np.finfo(np.float32).min))
        out = ops.softmax(scores, -1) @ v  # (B, H, T, Dh)
        out = ops.reshape(ops.transpose(out, (0, 2, 1, 3)), (b, t, d))
        return out @ params["w_out"] + params["b_out"]


# Activations usable directly inside Sequential.
relu = ops.relu
gelu = ops.gelu
tanh = ops.tanh
sigmoid = ops.sigmoid


def MLP(sizes: Sequence[int], activation: Callable = ops.relu) -> Sequential:
    """Dense layers with `activation` between them (none after the last)."""
    layers: list = []
    for i, n in enumerate(sizes):
        layers.append(Dense(n))
        if i < len(sizes) - 1:
            layers.append(activation)
    return Sequential(layers)


def TransformerBlock(d_model: int, num_heads: int, mlp_dim: int, causal: bool = True, dropout: float = 0.0):
    """Pre-LayerNorm transformer block, built only from the layers above."""
    attn = [LayerNorm(), MultiHeadAttention(num_heads, causal)]
    mlp = [LayerNorm(), Dense(mlp_dim), gelu, Dense(d_model)]
    if dropout:
        attn.append(Dropout(dropout))
        mlp.append(Dropout(dropout))
    return Sequential([Residual(Sequential(attn)), Residual(Sequential(mlp))])


def param_count(params) -> int:
    return sum(int(np.prod(p.shape)) for p in jax.tree.leaves(params))
