"""Sentinex ops: the array operations models are written with.

Primitives (defined with `@primitive`) are the units graph mode records, compiles and
differentiates. Composites (plain functions like `softmax`) are built from primitives,
so in graph mode they show up as several nodes, the way they would in TensorFlow.

Every op works on jax arrays (eager or under `jax.jit`) and on graph `Symbol`s.
VJP rules are written with these same ops, so gradient graphs are ordinary graphs.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from sentinex.primitive import primitive

# --------------------------------------------------------------------------- helpers


def shape_of(x) -> tuple[int, ...]:
    return tuple(x.shape) if hasattr(x, "shape") else ()


def ndim_of(x) -> int:
    return len(shape_of(x))


def dtype_of(x) -> np.dtype:
    return np.dtype(x.dtype) if hasattr(x, "dtype") else np.dtype(jnp.result_type(x))


def _axes(axis, nd: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(nd))
    if isinstance(axis, int):
        axis = (axis,)
    return tuple(sorted(a % nd for a in axis))


def _unbroadcast(g, target_shape: tuple[int, ...]):
    """Sum `g` down to `target_shape`: the adjoint of numpy-style broadcasting."""
    gshape = shape_of(g)
    if gshape == tuple(target_shape):
        return g
    lead = len(gshape) - len(target_shape)
    axes = tuple(range(lead)) + tuple(
        i + lead for i, d in enumerate(target_shape) if d == 1 and gshape[i + lead] != 1
    )
    if axes:
        g = sum(g, axes)
    return reshape(g, target_shape)


# --------------------------------------------------------------- elementwise binary


@primitive("add")
def add(x, y):
    return jnp.add(x, y)


@primitive("sub")
def sub(x, y):
    return jnp.subtract(x, y)


@primitive("mul")
def mul(x, y):
    return jnp.multiply(x, y)


@primitive("div")
def div(x, y):
    return jnp.true_divide(x, y)


@primitive("pow")
def pow(x, y):
    return jnp.power(x, y)


@primitive("maximum")
def maximum(x, y):
    return jnp.maximum(x, y)


@primitive("minimum")
def minimum(x, y):
    return jnp.minimum(x, y)


add.defvjp(
    lambda g, out, x, y: _unbroadcast(g, shape_of(x)),
    lambda g, out, x, y: _unbroadcast(g, shape_of(y)),
)
sub.defvjp(
    lambda g, out, x, y: _unbroadcast(g, shape_of(x)),
    lambda g, out, x, y: _unbroadcast(neg(g), shape_of(y)),
)
mul.defvjp(
    lambda g, out, x, y: _unbroadcast(g * y, shape_of(x)),
    lambda g, out, x, y: _unbroadcast(g * x, shape_of(y)),
)
div.defvjp(
    lambda g, out, x, y: _unbroadcast(g / y, shape_of(x)),
    lambda g, out, x, y: _unbroadcast(neg(g) * out / y, shape_of(y)),
)
pow.defvjp(
    lambda g, out, x, y: _unbroadcast(g * y * pow(x, y - 1), shape_of(x)),
    lambda g, out, x, y: _unbroadcast(g * log(x) * out, shape_of(y)),
)


def _balanced_eq(x, z, y):
    # JAX's tie rule for max/min: split the gradient evenly when x == y.
    return where(equal(x, z), where(equal(y, z), 0.5, 1.0), 0.0)


maximum.defvjp(
    lambda g, out, x, y: _unbroadcast(g * _balanced_eq(x, out, y), shape_of(x)),
    lambda g, out, x, y: _unbroadcast(g * _balanced_eq(y, out, x), shape_of(y)),
)
minimum.defvjp(
    lambda g, out, x, y: _unbroadcast(g * _balanced_eq(x, out, y), shape_of(x)),
    lambda g, out, x, y: _unbroadcast(g * _balanced_eq(y, out, x), shape_of(y)),
)

# Comparisons produce booleans: no gradient flows through them.


@primitive("equal", nondiff=True)
def equal(x, y):
    return jnp.equal(x, y)


@primitive("not_equal", nondiff=True)
def not_equal(x, y):
    return jnp.not_equal(x, y)


@primitive("less", nondiff=True)
def less(x, y):
    return jnp.less(x, y)


@primitive("less_equal", nondiff=True)
def less_equal(x, y):
    return jnp.less_equal(x, y)


@primitive("greater", nondiff=True)
def greater(x, y):
    return jnp.greater(x, y)


@primitive("greater_equal", nondiff=True)
def greater_equal(x, y):
    return jnp.greater_equal(x, y)


@primitive("where")
def where(cond, x, y):
    return jnp.where(cond, x, y)


where.defvjp(
    None,
    lambda g, out, c, x, y: _unbroadcast(where(c, g, 0.0), shape_of(x)),
    lambda g, out, c, x, y: _unbroadcast(where(c, 0.0, g), shape_of(y)),
)

# ---------------------------------------------------------------- elementwise unary


@primitive("neg")
def neg(x):
    return jnp.negative(x)


@primitive("exp")
def exp(x):
    return jnp.exp(x)


@primitive("log")
def log(x):
    return jnp.log(x)


@primitive("log1p")
def log1p(x):
    return jnp.log1p(x)


@primitive("sqrt")
def sqrt(x):
    return jnp.sqrt(x)


@primitive("rsqrt")
def rsqrt(x):
    return lax.rsqrt(x)


@primitive("tanh")
def tanh(x):
    return jnp.tanh(x)


@primitive("sigmoid")
def sigmoid(x):
    return jax.nn.sigmoid(x)


@primitive("sin")
def sin(x):
    return jnp.sin(x)


@primitive("cos")
def cos(x):
    return jnp.cos(x)


@primitive("abs")
def abs(x):
    return jnp.abs(x)


@primitive("sign", nondiff=True)
def sign(x):
    return jnp.sign(x)


@primitive("relu")
def relu(x):
    return jax.nn.relu(x)


neg.defvjp(lambda g, out, x: neg(g))
exp.defvjp(lambda g, out, x: g * out)
log.defvjp(lambda g, out, x: g / x)
log1p.defvjp(lambda g, out, x: g / (x + 1.0))
sqrt.defvjp(lambda g, out, x: g * (0.5 / out))
rsqrt.defvjp(lambda g, out, x: g * (-0.5 * out / x))
tanh.defvjp(lambda g, out, x: g * (1.0 - out * out))
sigmoid.defvjp(lambda g, out, x: g * (out * (1.0 - out)))
sin.defvjp(lambda g, out, x: g * cos(x))
cos.defvjp(lambda g, out, x: neg(g * sin(x)))
abs.defvjp(lambda g, out, x: g * sign(x))
relu.defvjp(lambda g, out, x: where(greater(x, 0.0), g, 0.0))


@primitive("cast")
def _cast(x, *, dtype):
    return lax.convert_element_type(x, dtype)


_cast.defvjp(lambda g, out, x, dtype: _cast(g, dtype=dtype_of(x)))


def cast(x, dt):
    return _cast(x, dtype=np.dtype(dt))


@primitive("stop_gradient", nondiff=True)
def stop_gradient(x):
    return lax.stop_gradient(x)


@primitive("identity")
def identity(x):
    return x


identity.defvjp(lambda g, out, x: g)

# ---------------------------------------------------------------------- reductions


@primitive("sum")
def _sum(x, *, axes, keepdims):
    return jnp.sum(x, axis=axes, keepdims=keepdims)


@primitive("max")
def _max(x, *, axes, keepdims):
    return jnp.max(x, axis=axes, keepdims=keepdims)


@primitive("argmax", nondiff=True)
def _argmax(x, *, axis):
    return jnp.argmax(x, axis=axis)


def _keep_shape(x, axes):
    return tuple(1 if i in axes else d for i, d in enumerate(shape_of(x)))


def _sum_vjp(g, out, x, axes, keepdims):
    if not keepdims:
        g = reshape(g, _keep_shape(x, axes))
    return broadcast_to(g, shape_of(x))


def _max_vjp(g, out, x, axes, keepdims):
    # Like JAX: split the gradient evenly over all positions equal to the max.
    if not keepdims:
        g = reshape(g, _keep_shape(x, axes))
        out = reshape(out, _keep_shape(x, axes))
    mask = cast(equal(x, out), dtype_of(g))
    return g * mask / _sum(mask, axes=axes, keepdims=True)


_sum.defvjp(_sum_vjp)
_max.defvjp(_max_vjp)


def sum(x, axis=None, keepdims=False):
    return _sum(x, axes=_axes(axis, ndim_of(x)), keepdims=keepdims)


def max(x, axis=None, keepdims=False):
    return _max(x, axes=_axes(axis, ndim_of(x)), keepdims=keepdims)


def argmax(x, axis=-1):
    return _argmax(x, axis=axis % ndim_of(x))


def mean(x, axis=None, keepdims=False):
    axes = _axes(axis, ndim_of(x))
    n = int(np.prod([shape_of(x)[a] for a in axes], dtype=np.int64))
    return sum(x, axes, keepdims) / n


def var(x, axis=None, keepdims=False):
    centered = x - mean(x, axis, keepdims=True)
    return mean(centered * centered, axis, keepdims)


# ------------------------------------------------------------------ shape ops


@primitive("reshape")
def _reshape(x, *, shape):
    return jnp.reshape(x, shape)


@primitive("transpose")
def _transpose(x, *, axes):
    return jnp.transpose(x, axes)


@primitive("broadcast_to")
def _broadcast_to(x, *, shape):
    return jnp.broadcast_to(x, shape)


_reshape.defvjp(lambda g, out, x, shape: reshape(g, shape_of(x)))
_transpose.defvjp(lambda g, out, x, axes: transpose(g, tuple(int(a) for a in np.argsort(axes))))
_broadcast_to.defvjp(lambda g, out, x, shape: _unbroadcast(g, shape_of(x)))


def reshape(x, shape):
    shape = tuple(int(d) for d in shape)
    if -1 in shape:
        known = int(np.prod([d for d in shape if d != -1], dtype=np.int64))
        size = int(np.prod(shape_of(x), dtype=np.int64))
        shape = tuple(size // known if d == -1 else d for d in shape)
    if shape == shape_of(x):
        return x
    return _reshape(x, shape=shape)


def transpose(x, axes=None):
    axes = tuple(reversed(range(ndim_of(x)))) if axes is None else tuple(a % ndim_of(x) for a in axes)
    if axes == tuple(range(ndim_of(x))):
        return x
    return _transpose(x, axes=axes)


def broadcast_to(x, shape):
    shape = tuple(int(d) for d in shape)
    if shape_of(x) == shape:
        return x
    return _broadcast_to(x, shape=shape)


def swapaxes(x, a, b):
    axes = list(range(ndim_of(x)))
    axes[a], axes[b] = axes[b], axes[a]
    return transpose(x, axes)


def expand_dims(x, axis):
    s = list(shape_of(x))
    s.insert(axis % (len(s) + 1), 1)
    return reshape(x, s)


def squeeze(x, axis):
    s = list(shape_of(x))
    s.pop(axis % len(s))
    return reshape(x, s)


def _freeze_index(index):
    if not isinstance(index, tuple):
        index = (index,)
    out = []
    for i in index:
        if isinstance(i, slice):
            out.append(("slice", i.start, i.stop, i.step))
        elif i is None:
            out.append(("none",))
        elif i is Ellipsis:
            out.append(("ellipsis",))
        elif isinstance(i, (int, np.integer)) and not isinstance(i, bool):
            out.append(("int", int(i)))
        else:
            raise TypeError(
                f"graph mode supports static indexing only (ints, slices, None, ...); "
                f"got {type(i).__name__}. Use sx.ops.take for array indices."
            )
    return tuple(out)


def _thaw_index(frozen):
    parts = []
    for kind, *rest in frozen:
        if kind == "slice":
            parts.append(slice(*rest))
        elif kind == "none":
            parts.append(None)
        elif kind == "ellipsis":
            parts.append(Ellipsis)
        else:
            parts.append(rest[0])
    return tuple(parts)


@primitive("getitem")
def _getitem(x, *, index):
    return x[_thaw_index(index)]


def getitem(x, index):
    return _getitem(x, index=_freeze_index(index))


@primitive("concatenate")
def _concatenate(*xs, axis):
    return jnp.concatenate(xs, axis=axis)


def concatenate(xs, axis=0):
    return _concatenate(*xs, axis=axis % ndim_of(xs[0]))


@primitive("take")
def _take(x, indices, *, axis):
    return jnp.take(x, indices, axis=axis)


_take.defvjp(_take.derived_vjp_rule(0), None)


def take(x, indices, axis=0):
    return _take(x, indices, axis=axis % ndim_of(x))


@primitive("one_hot", nondiff=True)
def _one_hot(x, *, num_classes, dtype):
    return jax.nn.one_hot(x, num_classes, dtype=dtype)


def one_hot(x, num_classes, dtype=jnp.float32):
    return _one_hot(x, num_classes=int(num_classes), dtype=np.dtype(dtype))


# --------------------------------------------------------------------- linear algebra


@primitive("matmul")
def matmul(a, b):
    return jnp.matmul(a, b)


def _matmul_vjp_a(g, out, a, b):
    if ndim_of(a) < 2 or ndim_of(b) < 2:
        return matmul.derived_vjp_rule(0)(g, out, a, b)
    return _unbroadcast(matmul(g, swapaxes(b, -1, -2)), shape_of(a))


def _matmul_vjp_b(g, out, a, b):
    if ndim_of(a) < 2 or ndim_of(b) < 2:
        return matmul.derived_vjp_rule(1)(g, out, a, b)
    return _unbroadcast(matmul(swapaxes(a, -1, -2), g), shape_of(b))


matmul.defvjp(_matmul_vjp_a, _matmul_vjp_b)


@primitive("einsum")
def _einsum(*operands, spec):
    return jnp.einsum(spec, *operands)


def einsum(spec, *operands):
    return _einsum(*operands, spec=spec)


# ------------------------------------------------------------------- conv and pooling
# These use derived backward kernels (see Primitive.derived_vjp_rule).


@primitive("conv2d")
def _conv2d(x, w, *, strides, padding):
    return lax.conv_general_dilated(
        x, w, window_strides=strides, padding=padding, dimension_numbers=("NHWC", "HWIO", "NHWC")
    )


def conv2d(x, w, strides=(1, 1), padding="SAME"):
    """2-D convolution. x: NHWC, w: HWIO."""
    return _conv2d(x, w, strides=tuple(strides), padding=padding)


@primitive("max_pool2d")
def _max_pool2d(x, *, window, strides, padding):
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, *window, 1), (1, *strides, 1), padding)


@primitive("avg_pool2d")
def _avg_pool2d(x, *, window, strides, padding):
    summed = lax.reduce_window(x, 0.0, lax.add, (1, *window, 1), (1, *strides, 1), padding)
    return summed / (window[0] * window[1])


def max_pool2d(x, window=(2, 2), strides=None, padding="VALID"):
    return _max_pool2d(x, window=tuple(window), strides=tuple(strides or window), padding=padding)


def avg_pool2d(x, window=(2, 2), strides=None, padding="VALID"):
    return _avg_pool2d(x, window=tuple(window), strides=tuple(strides or window), padding=padding)


# --------------------------------------------------------------------------- random
# Stateless RNG: the key is an ordinary input, so random ops are pure functions of
# their inputs. That is what makes them safe to cache, CSE and constant-fold.


@primitive("random_split", nondiff=True)
def _random_split(key, *, num):
    return jax.random.split(key, num)


@primitive("bernoulli", nondiff=True)
def _bernoulli(key, *, p, shape):
    return jax.random.bernoulli(key, p, shape)


def random_split(key, num=2):
    return _random_split(key, num=int(num))


def bernoulli(key, p, shape):
    return _bernoulli(key, p=float(p), shape=tuple(shape))


# ------------------------------------------------------------------------ composites


def square(x):
    return x * x


def clip(x, lo, hi):
    return minimum(maximum(x, lo), hi)


def logsumexp(x, axis=-1, keepdims=False):
    m = stop_gradient(max(x, axis, keepdims=True))
    out = log(sum(exp(x - m), axis, keepdims=True)) + m
    return out if keepdims else squeeze(out, axis)


def log_softmax(x, axis=-1):
    shifted = x - stop_gradient(max(x, axis, keepdims=True))
    return shifted - log(sum(exp(shifted), axis, keepdims=True))


def softmax(x, axis=-1):
    unnormalized = exp(x - stop_gradient(max(x, axis, keepdims=True)))
    return unnormalized / sum(unnormalized, axis, keepdims=True)


def gelu(x):
    """GELU, tanh approximation (same formula as jax.nn.gelu(approximate=True))."""
    cdf = 0.5 * (1.0 + tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x**3))))
    return x * cdf


# Constructors return concrete arrays even while tracing; graph mode captures them
# as consts (and the folding pass keeps small derived values constant too).


def zeros(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype)


def ones(shape, dtype=jnp.float32):
    return jnp.ones(shape, dtype)


def zeros_like(x):
    return jnp.zeros(shape_of(x), dtype_of(x))


def ones_like(x):
    return jnp.ones(shape_of(x), dtype_of(x))
