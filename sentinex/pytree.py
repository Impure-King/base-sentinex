"""Any API for PyTree Manipulations"""
import jax
import jax.tree_util as tu
import numpy as np
from typing import Any
from functools import partial

__all__ = ["is_array", "is_jax_array", "is_array_like", "filter_pytree"
           , "partition", "combine"]

def is_array(x: Any):
    """Checks if the instance is an array.
    
    Args:
      x (Any): An object that can be verified as an array."""
    return isinstance(x, (jax.Array, np.ndarray))

def is_jax_array(x):
    """Check if the instance is an JAX array."""
    return isinstance(x, jax.Array)

def is_array_like(x):
    """Check if the instance is a JAX/Numpy array, or
    a Python float/int/bool/complex value."""
    return isinstance(x, (jax.Array,
                          np.ndarray,
                          float,
                          int,
                          bool,
                          complex))

def filter_pytree(pytree,
                  filter_method,
                  replace_val: Any = None,
                  reverse: bool = False,
                  is_leaf: Any = None):
    """Filters a pytree and returns nodes that satisfies the filter method.
    If reverse is True, then it returns nodes that don't satisfy the filter method."""
    
    reverse = bool(reverse)
    mask_tree = tu.tree_map(lambda x: True if filter_method(x) != reverse else False, pytree, is_leaf=is_leaf)
    leaves_tree = tu.tree_map(lambda x, mask: x if mask else replace_val, pytree, mask_tree, is_leaf=is_leaf)
    return leaves_tree

def partition(pytree,
              filter_method,
              replace_val: Any = None,
              is_leaf: Any = None):
    """Splits a pytree into two parts, one representing a tree that satisfies the filter method
    while the other doesn't."""
    leaves_tree = filter_pytree(pytree, filter_method, replace_val, is_leaf=is_leaf)
    aux_tree = filter_pytree(pytree, filter_method, replace_val, reverse=True, is_leaf=is_leaf)
    return leaves_tree, aux_tree

def _combine(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


def _is_none(x):
    return x is None


def combine(
    *pytrees, is_leaf = None
):
    """
    """
    if is_leaf is None:
        _is_leaf = _is_none
    else:
        _is_leaf = lambda x: _is_none(x) or is_leaf(x)

    return tu.tree_map(_combine, *pytrees, is_leaf=_is_leaf)


# Add some filter transformations eventually.