"""Benchmark workloads: each model in Sentinex and in raw JAX + Optax.

The raw JAX versions are written the way you would write them without Sentinex
(jnp / lax / jax.nn / optax, `jax.jit` on the whole step). They consume the same
parameter pytree as the Sentinex model so both start from identical weights, and
tests/test_benchmark_parity.py checks that both compute the same function.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax

from sentinex import losses, nn, optim, train

LEARNING_RATE = 1e-3


@dataclasses.dataclass
class Workload:
    name: str
    description: str
    batch_size: int
    model: nn.Layer
    input_shape: tuple
    make_batch: Callable[[int], tuple]  # seed -> (x, y) on device
    raw_forward: Callable[[Any, Any], Any]  # (params, x) -> logits, raw JAX

    def init_params(self):
        return self.model.init(jax.random.PRNGKey(0), self.input_shape)

    # -- Sentinex ---------------------------------------------------------------
    def sentinex_step(self, mode: str):
        opt = optim.adam(LEARNING_RATE)

        def loss_fn(params, batch, key):
            x, y = batch
            return losses.cross_entropy(self.model.apply(params, x), y)

        step = train.compile_step(train.make_train_step(loss_fn, opt), mode)
        state = train.TrainState.create(self.init_params(), opt)
        return state, step, lambda out: out[1]["loss"]

    # -- raw JAX + Optax ----------------------------------------------------------
    def raw_step(self):
        opt = optax.adam(LEARNING_RATE)
        forward = self.raw_forward

        def loss_fn(params, x, y):
            return optax.softmax_cross_entropy_with_integer_labels(forward(params, x), y).mean()

        @jax.jit
        def step(state, batch):
            params, opt_state = state
            loss, grads = jax.value_and_grad(loss_fn)(params, *batch)
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        params = self.init_params()
        return (params, opt.init(params)), step, lambda out: out[1]


def _classification_batch(shape, classes):
    def make(seed):
        k1, k2 = jax.random.split(jax.random.PRNGKey(100 + seed))
        return jax.random.normal(k1, shape), jax.random.randint(k2, (shape[0],), 0, classes)

    return make


def _token_batch(batch, seq, vocab):
    def make(seed):
        x = jax.random.randint(jax.random.PRNGKey(100 + seed), (batch, seq + 1), 0, vocab)
        return x[:, :-1], x[:, 1:]

    return make


# ------------------------------------------------------------------ raw JAX models


def raw_mlp(params, x):
    dense = [p for p in params if p]  # activation layers have empty params
    for i, p in enumerate(dense):
        x = x @ p["w"] + p["b"]
        if i < len(dense) - 1:
            x = jax.nn.relu(x)
    return x


def _raw_conv(x, p):
    y = lax.conv_general_dilated(x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def _raw_max_pool(x):
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")


def raw_cnn(params, x):
    conv1, _, _, conv2, _, _, _, fc1, _, fc2 = params
    x = _raw_max_pool(jax.nn.relu(_raw_conv(x, conv1)))
    x = _raw_max_pool(jax.nn.relu(_raw_conv(x, conv2)))
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ fc1["w"] + fc1["b"])
    return x @ fc2["w"] + fc2["b"]


def _raw_layer_norm(x, p, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mean) * lax.rsqrt(var + eps) * p["scale"] + p["bias"]


def _raw_attention(x, p, heads):
    b, t, d = x.shape
    qkv = (x @ p["w_qkv"] + p["b_qkv"]).reshape(b, t, 3, heads, d // heads)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))  # (B, H, T, Dh)
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(d // heads)
    mask = jnp.tril(jnp.ones((t, t), bool))
    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    out = jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(scores, -1), v)
    return out.transpose(0, 2, 1, 3).reshape(b, t, d) @ p["w_out"] + p["b_out"]


def make_raw_transformer(heads):
    def forward(params, ids):
        emb, pos, *blocks, ln_f, head = params
        x = emb["table"][ids] + pos["pos"][: ids.shape[1]]
        for attn_block, mlp_block in blocks:
            ln1, mha = attn_block
            x = x + _raw_attention(_raw_layer_norm(x, ln1), mha, heads)
            ln2, fc1, _, fc2 = mlp_block
            h = jax.nn.gelu(_raw_layer_norm(x, ln2) @ fc1["w"] + fc1["b"])
            x = x + h @ fc2["w"] + fc2["b"]
        return _raw_layer_norm(x, ln_f) @ head["w"] + head["b"]

    return forward


# ---------------------------------------------------------------------- workloads


def mlp(batch_size=512):
    return Workload(
        name="mlp" if batch_size == 512 else f"mlp_b{batch_size}",
        description=f"MLP 784-512-512-10, batch {batch_size}",
        batch_size=batch_size,
        model=nn.MLP([512, 512, 10]),
        input_shape=(None, 784),
        make_batch=_classification_batch((batch_size, 784), 10),
        raw_forward=raw_mlp,
    )


def cnn(batch_size=128):
    model = nn.Sequential([
        nn.Conv2D(32), nn.relu, nn.MaxPool2D(),
        nn.Conv2D(64), nn.relu, nn.MaxPool2D(),
        nn.Flatten(), nn.Dense(128), nn.relu, nn.Dense(10),
    ])
    return Workload(
        name="cnn",
        description=f"CNN conv32-pool-conv64-pool-fc128-fc10 on 28x28x1, batch {batch_size}",
        batch_size=batch_size,
        model=model,
        input_shape=(None, 28, 28, 1),
        make_batch=_classification_batch((batch_size, 28, 28, 1), 10),
        raw_forward=raw_cnn,
    )


def transformer(batch_size=32, seq=64, vocab=256, d_model=128, heads=4, layers=2):
    model = nn.Sequential([
        nn.Embedding(vocab, d_model),
        nn.PositionalEmbedding(seq),
        *[nn.TransformerBlock(d_model, heads, 4 * d_model) for _ in range(layers)],
        nn.LayerNorm(),
        nn.Dense(vocab),
    ])
    return Workload(
        name="transformer",
        description=f"decoder-only transformer, {layers} layers, d={d_model}, {heads} heads, "
        f"seq {seq}, vocab {vocab}, batch {batch_size}",
        batch_size=batch_size,
        model=model,
        input_shape=(None, seq),
        make_batch=_token_batch(batch_size, seq, vocab),
        raw_forward=make_raw_transformer(heads),
    )


WORKLOADS = {"mlp": mlp, "mlp_b32": lambda: mlp(32), "cnn": cnn, "transformer": transformer}


def param_count(workload: Workload) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree.leaves(workload.init_params())))
