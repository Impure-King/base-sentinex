from typing import Any, Tuple
import jax.numpy as jnp
from jax import lax
from .base_layer import Layer
from ..activations._activations import Activation

__all__ = ["Pool", "MaxPool", "MaxPool1D", "MaxPool2D", "MaxPool3D"]

class Pool(Layer):
    """A general N-D pooling layer.

        NOTE: Pooling layers is not generally differentiable,
        even if the pooling function is differentiable.

        Args:
            rank (int): The rank of the layer.
            pooling_fn (callable): The function that causes window reduction.
            init_val (_type_): An initial value that is passed to the pooling function.
            window_shape (Tuple[int, ...]): The size of the window that reduces the inputs.
            strides (Tuple[int, ...]): Specifies the translation of the window.
            padding (str, optional): The padding name. Defaults to "valid".
            activation (Activation | str, optional): The activation to apply before returning the outputs. Defaults to None.
            window_dilation (Tuple, optional): The dilation of the windows. Defaults to None.
            name (str, optional): The internal name of the layer. Defaults to "Pool".
    """
    def __init__(self,
                 rank: int,
                 pooling_fn,
                 init_val: Any, 
                 window_shape: Tuple[int, ...], 
                 strides: Tuple[int, ...],
                 padding: str = "valid",
                 activation: Activation | str | None = None,
                 window_dilation: Tuple[int, ...] | None = None,
                 name: str = "Pool",
                 **kwargs):
        super().__init__(name,
                         **kwargs)
        self.rank = rank
        self.pooling_fun = pooling_fn
        self.init_val = init_val
        self.window_shape = window_shape
        self.strides = strides
        self.padding = padding
        self.window_dilation = window_dilation or (1, ) * (self.rank + 2)
        self.activation = activation or Activation.get_activation("none")
        
        if isinstance(self.activation, str):
            self.activation = Activation.get_activation(self.activation.lower())
        
        if isinstance(padding, str):
            self.padding = padding.upper()
        
    def check_input_type(self):
        # toDo: Complete input validation
        pass

    def call(self, x):
        if jnp.ndim(x) < (self.rank + 2):
            raise ValueError(f"""Raised from {self.name}
                             Input dimensions must be {self.rank + 2}.
                             Current dimensions: {jnp.ndim(x)}.
                             """)
        inputs = x
        return lax.reduce_window(inputs,
                                 self.init_val,
                                 self.pooling_fun,
                                 (1,) + self.window_shape + (1, ),
                                 (1,) + self.strides + (1,),
                                 self.padding,
                                 self.window_dilation)

class MaxPool(Pool):
    def __init__(self, 
                 rank: int,  
                 window_shape: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 activation: Activation | str | None = None, 
                 window_dilation: Tuple[int, ...] | None = None, 
                 name: str = "MaxPool", 
                 **kwargs):
        super().__init__(rank, 
                         lax.max, 
                         -jnp.inf, 
                         window_shape, 
                         strides, 
                         padding, 
                         activation, 
                         window_dilation, 
                         name, 
                         **kwargs)

class MaxPool1D(Pool):
    def __init__(self,  
                 window_shape: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 activation: Activation | str | None = None, 
                 window_dilation: Tuple[int, ...] | None = None, 
                 name: str = "MaxPool1D", 
                 **kwargs):
        super().__init__(1, 
                         lax.max, 
                         -jnp.inf, 
                         window_shape, 
                         strides, 
                         padding, 
                         activation, 
                         window_dilation, 
                         name, 
                         **kwargs)

class MaxPool2D(Pool):
    def __init__(self,  
                 window_shape: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 activation: Activation | str | None = None, 
                 window_dilation: Tuple[int, ...] | None = None, 
                 name: str = "MaxPool2D", 
                 **kwargs):
        super().__init__(2, 
                         lax.max, 
                         -jnp.inf, 
                         window_shape, 
                         strides, 
                         padding, 
                         activation, 
                         window_dilation, 
                         name, 
                         **kwargs)

class MaxPool3D(Pool):
    def __init__(self,  
                 window_shape: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 activation: Activation | str | None = None, 
                 window_dilation: Tuple[int, ...] | None = None, 
                 name: str = "MaxPool3D", 
                 **kwargs):
        super().__init__(3, 
                         lax.max, 
                         -jnp.inf, 
                         window_shape, 
                         strides, 
                         padding, 
                         activation, 
                         window_dilation, 
                         name, 
                         **kwargs)