"""Layers: shapes, behaviour, and graph mode == eager."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import rand, tree_allclose

import sentinex as sx
from sentinex import nn

KEY = jax.random.PRNGKey(0)

LAYERS = [
    ("dense", nn.Dense(7), (None, 5), rand((3, 5))),
    ("dense_3d", nn.Dense(4, use_bias=False), (None, 2, 5), rand((3, 2, 5))),
    ("conv_same", nn.Conv2D(4), (None, 6, 6, 3), rand((2, 6, 6, 3))),
    ("conv_valid_strided", nn.Conv2D(4, (3, 3), (2, 2), "VALID"), (None, 9, 9, 2), rand((2, 9, 9, 2))),
    ("max_pool", nn.MaxPool2D(), (None, 6, 6, 3), rand((2, 6, 6, 3))),
    ("avg_pool_same", nn.AvgPool2D((3, 3), (2, 2), "SAME"), (None, 5, 5, 3), rand((2, 5, 5, 3))),
    ("flatten", nn.Flatten(), (None, 2, 3, 4), rand((2, 2, 3, 4))),
    ("layernorm", nn.LayerNorm(), (None, 6), rand((4, 6))),
    ("embedding", nn.Embedding(10, 4), (None, 5), jnp.array([[1, 2, 3, 4, 9], [0, 0, 1, 1, 2]])),
    ("pos_embedding", nn.PositionalEmbedding(8), (None, 5, 4), rand((2, 5, 4))),
    ("attention", nn.MultiHeadAttention(2), (None, 5, 8), rand((2, 5, 8))),
    ("causal_attention", nn.MultiHeadAttention(4, causal=True), (None, 5, 8), rand((2, 5, 8))),
    ("transformer_block", nn.TransformerBlock(8, 2, 16), (None, 5, 8), rand((2, 5, 8))),
    ("mlp", nn.MLP([8, 8, 3], nn.gelu), (None, 4), rand((2, 4))),
]


@pytest.mark.parametrize("name,layer,in_shape,x", LAYERS, ids=[c[0] for c in LAYERS])
def test_output_shape_and_graph_mode(name, layer, in_shape, x):
    params = layer.init(KEY, in_shape)
    y = layer.apply(params, x)
    expected = layer.output_shape((x.shape[0], *in_shape[1:]))
    assert y.shape == tuple(expected)
    tree_allclose(sx.function(layer.apply)(params, x), y, rtol=1e-5, atol=1e-5)


def test_causal_attention_ignores_the_future():
    layer = nn.MultiHeadAttention(2, causal=True)
    params = layer.init(KEY, (None, 6, 8))
    x = rand((1, 6, 8))
    x2 = x.at[:, 4:].set(rand((1, 2, 8), 9))
    y, y2 = layer.apply(params, x), layer.apply(params, x2)
    np.testing.assert_allclose(y[:, :4], y2[:, :4], rtol=1e-5, atol=1e-6)
    assert not np.allclose(y[:, 4:], y2[:, 4:])


def test_layernorm_normalizes():
    layer = nn.LayerNorm()
    y = layer.apply(layer.init(KEY, (None, 64)), rand((8, 64)) * 3 + 2)
    np.testing.assert_allclose(jnp.mean(y, -1), 0.0, atol=1e-5)
    np.testing.assert_allclose(jnp.var(y, -1), 1.0, atol=1e-3)


def test_dropout():
    layer = nn.Dropout(0.25)
    x = jnp.ones((200, 50))
    np.testing.assert_array_equal(layer.apply({}, x), x)  # train=False: identity
    y = layer.apply({}, x, key=KEY, train=True)
    kept = np.asarray(y != 0)
    assert abs(kept.mean() - 0.75) < 0.02
    np.testing.assert_allclose(np.asarray(y)[kept], 1 / 0.75, rtol=1e-6)
    tree_allclose(sx.function(lambda k: layer.apply({}, x, key=k, train=True))(KEY), y)
    with pytest.raises(ValueError):
        layer.apply({}, x, train=True)


def test_sequential_params_and_count():
    model = nn.Sequential([nn.Dense(4), nn.relu, nn.Dense(2)])
    params = model.init(KEY, (None, 3))
    assert len(params) == 3 and params[1] == {}
    assert nn.param_count(params) == 3 * 4 + 4 + 4 * 2 + 2
    assert model.output_shape((None, 3)) == (None, 2)


def test_init_is_deterministic():
    model = nn.MLP([16, 4])
    tree_allclose(model.init(KEY, (None, 3)), model.init(KEY, (None, 3)), rtol=0, atol=0)
