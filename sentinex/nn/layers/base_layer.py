from typing import Any, Callable, Optional
from jax import Array
from jax.tree_util import tree_map

from sentinex.module import Module
from sentinex.nn.initializers.base_initializer import Initializer

__all__ = ["Layer"]

class Layer(Module):
  """Base Layer class for all layers. It adds functionality to existing Module class,
  such as parameter registeration and auto-parameter building.

  Args:
      name (str, optional): The name of the layer. Defaults to "Layer".
  """
  def __init__(self, name: str = "Layer", **kwargs):
    super().__init__(name=name,
                     **kwargs)
    self.built = False


  def add_param(self,
                shape: tuple,
                initializer: Any) -> Array:
    """Adds a parameter by using the initializer to initialize and return a tensor of the specified shape.

    Args:
        shape (tuple): Specifies the shape of the attribute tensor.
        initializer (callable): An initializer that determines the content of the initialized tensors. 

    Returns:
        Array: The initialized tensor.
    """
    return initializer(shape)
  
  def build(self, *args, **kwargs):
    self.built = True
  
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    if not self.built:
        self.build(*args, **kwargs)
    return self.call(*args, **kwargs)
  
  def call(self):
      raise NotImplementedError

Layer.__module__ = "sentinex.nn"