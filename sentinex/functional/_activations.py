import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Tuple

# Handling exports:
__all__ = ["relu",
           "heaviside",
           "leaky_relu",
           "elu",
           "selu",
           "sigmoid",
           "tanh",
           "softmax",
           "swish"]

def relu(x: Array,
         max_val: float = jnp.inf,
         min_val: float = 0.0) -> Array:
    """A Rectified Linear Unit computation function.

    Args:
        x (Array): The input array.
        max_val (float, optional): The maximum value of an output element. Defaults to jnp.inf.
        min_val (float, optional): The minimum value of an output element. Defaults to 0.0.
    
    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return leaky_relu(x=x, 
                      max_val=max_val, 
                      min_val=min_val, 
                      alpha = 0)

def heaviside(x: Array, threshold: float = 0.0) -> Array:
    """A vanilla heaviside function.

    Args:
        x (Array): The input array.
        threshold (float, optional): The threshold where the output elements transforms into 1. Defaults to 0.0.

    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jnp.float32(x > threshold)

def leaky_relu(x: Array,
               max_val: float = jnp.inf,
               min_val: float = 0.0,
               alpha: float = 1e-3) -> Array:
    """A Leaky Rectified Linear Unit computation function.

    Args:
        x (Array): The input array.
        max_val (float, optional): The maximum value of an output element. Defaults to jnp.inf.
        min_val (float, optional): The minimum value of an output element. Defaults to 0.0.
        alpha (float, optional): The slope of the leaky minimum. Defaults to 0.001.
    
    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    input_shape = jnp.shape(x)
    return jnp.minimum(jnp.zeros(input_shape) + max_val, jnp.maximum(jnp.zeros(input_shape) + min_val + x * alpha, x))

def elu(x: Array,
        alpha: float = 1e-3) -> Array:
    """An Exponential Linear Unit computation function.
    Thinly wraps around ``jax.nn.elu`` to provide type hints.

    Args:
        x (Array): The input array.
        alpha (float, optional): The alpha value. Defaults to 1e-3.

    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jax.nn.elu(x, alpha)

def selu(x: Array) -> Array:
    """A Scaled Exponential Linear Unit computation function.
    Thinly wraps around ``jax.nn.selu`` to provide type hints.
    
    Args:
        x (Array): The input array.
    
    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jax.nn.selu(x)

def sigmoid(x: Array) -> Array:
    """A Sigmoid activation function.
    Thinly wraps around ``jax.nn.sigmoid`` to provide type hints.

    Args:
        x (Array): The input array.

    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jax.nn.sigmoid(x)

def tanh(x: Array) -> Array:
    """A Hyperbolic Tangent activation function.
    Thinly wraps around ``jax.nn.tanh`` to provide type hints.
    
    Args:
        x (Array): The input array.
    
    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jax.nn.tanh(x)

def softmax(x: Array) -> Array:
    """A Softmax activation function.
    Thinly wraps around ``jax.nn.softmax`` to provide type hints.

    Args:
        x (Array): The input array.

    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jax.nn.softmax(x)

def swish(x: Array) -> Array:
    """A Swish activation function.
    Thinly wraps around ``jax.nn.swish`` to provide type hints.

    Args:
        x (Array): The input array.

    Returns:
        Array: An output tensor with adjusted elements, but same shape.
    """
    return jax.nn.swish(x)