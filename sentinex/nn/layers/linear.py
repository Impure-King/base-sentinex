from typing import Any

import jax

from sentinex.nn.activations.base_activations import Activation
from sentinex.nn.initializers.base_initializer import Initializer
from sentinex.nn.layers.base_layer import Layer
from equinox import filter_jit

__all__ = ["Dense", "Linear"]

class Dense(Layer):
  """A Basic Linear layer that applies a linear transformation to all the inputs.

  Args:
      units (int): Specifies the dimensionality of the output tensor.
      activation (Any, optional): An activation function that produces non-linearity. Defaults to None.
      use_bias (bool, optional): Specifies whether to use a bias or not. Defaults to True.
      kernel_initializer (str | Initializer, optional): Initializer for the kernel. Defaults to 'glorot_uniform'.
      bias_initializer (str | Initializer, optional): Initializer for the bias. Defaults to 'zeros'.
      kernel_regularizer (str | None, optional): Regularizer for the kernel. Defaults to None.
      bias_regularizer (str | None, optional): Regularizer for the bias. Defaults to None.
      kernel_constraint (str | None, optional): Constraint for the kernel. Defaults to None.
      bias_constraint (str | None, optional): Constraint for the bias. Defaults to None.
      name (str, optional): The name of the layer. Defaults to "Dense".

  Raises:
      ValueError: If incorrent types get passed in.
  """
  
  # Defining annotations:
  kernel: jax.Array
  def __init__(self, 
              units:int,
              activation:Any=None,
              use_bias:bool=True,
              kernel_initializer: str | Initializer = 'glorot_uniform',
              bias_initializer: str | Initializer = 'zeros',
              kernel_regularizer: str | None = None,
              bias_regularizer: str | None = None,
              kernel_constraint: str | None = None,
              bias_constraint: str | None = None,
              name:str = "Dense",
              **kwargs):
    super().__init__(name,
                     **kwargs)
    
    # Defining all the variables:
    self.units = units
    self.activation = activation or Activation.get_activation("none")
    self.use_bias = use_bias

    if isinstance(kernel_initializer, str):
      self.kernel_initializer = Initializer.get_initializers(kernel_initializer)
    elif isinstance(kernel_initializer, Initializer):
      self.kernel_initializer = kernel_initializer
    
    if isinstance(bias_initializer, str):
      self.bias_initializer = Initializer.get_initializers(bias_initializer)
    elif isinstance(bias_initializer, Initializer):
      self.bias_initializer = bias_initializer

    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint

    if isinstance(activation, str):
      self.activation = Activation.get_activation(activation)

    # Performing a few checks:
    if not isinstance(units, int):
      raise ValueError(f"Layer {self.name}"
                       f"Argument ``units`` must be an integer, not {type(units)}")
    elif units <= 0:
      raise ValueError(f"Layer {self.name}"
                       f"Argument ``units`` must be greater than 0. Current value {units}")

  def build(self, input, *args, **kwargs):
    """Builds the variables of the layer.

    Args:
        input (Array): A sample input to model the variables after.
    """
    super().build()
    input_shape = input.shape
    
    self.kernel = self.add_param((input_shape[-1], self.units),
                                 self.kernel_initializer)
    if self.use_bias:
      self.bias = self.add_param((self.units,),
                               self.bias_initializer)
      self.set_annotation("bias", jax.Array)


  def call(self, x):
    if not self.use_bias:
      return self.activation(x @ self.kernel)
    return self.activation(x @ self.kernel + self.bias)


class Linear(Layer):
  kernel: jax.Array
  def __init__(self,
              in_shape:int, 
              out_shape:int,
              activation:Any=None,
              use_bias:bool=True,
              kernel_initializer='glorot_uniform',
              bias_initializer='zeros',
              kernel_regularizer=None,
              bias_regularizer=None,
              kernel_constraint=None,
              bias_constraint=None,
              name:str = "Linear",
              *args,
              **kwargs):
    super().__init__(name,
                     *args,
                     **kwargs)
    
    # Defining all the variables:
    self.in_shape = in_shape
    self.out_shape = out_shape
    self.activation = activation or Activation.get_activation("none")
    self.use_bias = use_bias

    if isinstance(kernel_initializer, str):
      self.kernel_initializer = Initializer.get_initializers(kernel_initializer)
    elif isinstance(kernel_initializer, Initializer):
      self.kernel_initializer = kernel_initializer
    
    if isinstance(bias_initializer, str):
      self.bias_initializer = Initializer.get_initializers(bias_initializer)
    elif isinstance(bias_initializer, Initializer):
      self.bias_initializer = bias_initializer

    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.kernel_constraint = kernel_constraint
    self.bias_constraint = bias_constraint

    if isinstance(activation, str):
      self.activation = Activation.get_activation(activation)

    # Performing a few checks:
    if not isinstance(in_shape, int):
      raise ValueError(f"Layer {self.name}"
                       f"Argument ``in_shape`` must be an integer, not {type(in_shape)}")
    
    if not isinstance(out_shape, int):
      raise ValueError(f"Layer {self.name}"
                       f"Argument ``out_shape`` must be an integer, not {type(out_shape)}")

    elif in_shape <= 0:
      raise ValueError(f"Layer {self.name}"
                       f"Argument ``in_shape`` must be greater than 0. Current value {in_shape}")
    elif out_shape <= 0:
      raise ValueError(f"Layer {self.name}"
                       f"Argument ``out_shape`` must be greater than 0. Current value {out_shape}")

    # Creating the weights:
    self.kernel = self.add_param((self.in_shape, self.out_shape),
                                 self.kernel_initializer)
    if self.use_bias:
      self.bias = self.add_param((self.out_shape,),
                               self.bias_initializer)
      self.set_annotation("bias", jax.Array)


  def call(self, x):
    if not self.use_bias:
      return self.activation(x @ self.kernel)
    return self.activation(x @ self.kernel + self.bias)

Dense.__module__ = "sentinex.nn"
Linear.__module__ = "sentinex.nn"