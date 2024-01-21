from random import randint
from typing import Tuple

from jax.random import key, normal, uniform

__all__ = ["randn", "randu"]

def randn(shape: Tuple[int,...], seed: int = randint(0, 5)):
    """Returns an normal distribution tensor of the given shape.
    
    Args:
      shape (tuple): Specifies the shape of the output tensor.
      seed (int, optional): Specifies the seed to reproduce randomness. Defaults to a random integer from 0 to 5."""
    random_key = key(seed) # type: ignore
    return normal(random_key, shape)

def randu(shape: Tuple[int,...], seed: int = randint(0, 5), min_val: int | float = -1, max_val: int | float = 1):
    """Returns an uniform distribution tensor of the given shape.
    
    Args:
      shape (tuple): Specifies the shape of the output tensor.
      seed (int, optional): Specifies the seed to reproduce randomness. Defaults to a random integer from 0 to 5.
      min_val (float | int, optional): Specifies the minimum value of the uniform tensor element. Defaults to -1.
      max_val (float | int, optional): Specifies the maximum value of the uniform tensor element. Defaults to 1."""
    random_key = key(seed)
    return uniform(random_key, shape, minval=min_val, maxval=max_val)