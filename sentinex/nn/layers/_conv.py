from typing import List, Tuple
from jaxtyping import Array
from .base_layer import Layer
from ..initializers._initializer import Initializer
from ..activations._general_activations import Activation
import jax
import jax.numpy as jnp

__all__ = ["Conv", "Conv1D", "Conv2D", "Conv3D"]

# Computes the dimension numbers:
def compute_dimension_numbers(rank: int):
    """A small wrapper that computes the convolutional dimension numbers in accordance to XLA N-D convolutions.

    NOTE: Only supports ``channels_last`` convolutions (aka ``NHWC`` convolution).
    Args:
        rank (int): The rank of the convolution.

    Returns:
        ConvDimensionNumbers: A special series of tuples that instructs the dimensional corresponse between the inputs, kernels, and outputs.
    """
    # Computing fixed weight shape:
    kernel_spec = tuple(range(rank+1, 1, -1)) + (0, 1)
    
    
    # Creating basic input dimension numbers:
    input_spec = [0, ] + list(range(rank + 1, 0, -1))
    input_spec[-2], input_spec[-1] = input_spec[-1], input_spec[-2]
    
    output_spec = input_spec
    
    # Converting all the dimensions into tuples for hashing:
    input_spec = tuple(input_spec)
    output_spec = tuple(output_spec)
    kernel_spec = tuple(kernel_spec)
    
    # toDo: Add support for various channel dimensions for N-D convolutions:
    return jax.lax.ConvDimensionNumbers(lhs_spec = input_spec,
                                        rhs_spec = kernel_spec,
                                        out_spec = output_spec) # Making the outputs channel's last.
    

class Conv(Layer):
    """A general N-D convolutional layer.

        Args:
            rank (int): The number of dimensions for the convolution operators.
            filters (int): The number of filters.
            kernel_size (Tuple[int, ...]): The shape of each filter.
            strides (Tuple[int, ...]): The translation step of each filter.
            padding (str, optional): _description_. Defaults to "valid".
            input_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            kernel_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.
            use_bias (bool, optional): _description_. Defaults to True.
            kernel_initializer (str | Initializer, optional): _description_. Defaults to "glorot_uniform".
            bias_initializer (str | Initializer, optional): _description_. Defaults to "zeros".
            kernel_regularizer (_type_, optional): _description_. Defaults to None.
            bias_regularizer (_type_, optional): _description_. Defaults to None.
            kernel_constraint (_type_, optional): _description_. Defaults to None.
            bias_constraint (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "ConvolutionND".
        """
    kernel: Array
    
    def __init__(self,
                 rank: int,
                 filters: int,
                 kernel_size: Tuple[int, ...],
                 strides: Tuple[int, ...],
                 padding: str = "valid",
                 input_dilation_rate: Tuple[int, ...] | None = None,
                 kernel_dilation_rate: Tuple[int, ...] | None = None,
                 activation = None,
                 use_bias: bool = True,
                 kernel_initializer: str | Initializer = "glorot_uniform",
                 bias_initializer: str | Initializer = "zeros",
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 name: str = "ConvolutionND", 
                 **kwargs) -> None:
        super().__init__(name, **kwargs)
        # toDo: Add some support for basic channel dimensions (for pytorch to sentinex conversion)?
        # toDo: Add some group support?
        # toDo: Add some mask support?
        self.rank = rank
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation or Activation.get_activation("none")
        self.use_bias = use_bias
        self.kernel_initializer = Initializer.get_initializers(kernel_initializer) if isinstance(kernel_initializer, str) else kernel_initializer
        self.bias_initializer = Initializer.get_initializers(bias_initializer) if isinstance(bias_initializer, str) else bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.input_dilation_rate = input_dilation_rate or (1, ) * self.rank
        self.kernel_dilation_rate = kernel_dilation_rate or (1, ) * self.rank
        self.check_input_type()
        self.check_shape_and_rank()
        
        if isinstance(activation, str):
            self.activation = Activation.get_activation(activation)
        
        if self.use_bias:
            self.set_annotation("bias", Array)
        
        # ToDo: Add input validation and then generate docstring.
        
    def check_input_type(self) -> None:
        """Checks the input types and assures correct inputs have been passed."""
        if not isinstance(self.rank, int):
            raise ValueError(f"""Raised from {self.name}.
                             Rank argument must be an integer, not {type(self.rank)}.""")
        
        if not isinstance(self.filters, int):
            raise ValueError(f"""Raised from {self.name}.
                             Rank argument must be an integer, not {type(self.filters)}.""")
        
        def check_dimension(x, message):
            if x < 0 or not isinstance(x, int):
                raise ValueError(f"""Raised from {self.name}.
                                 {message}""")
        jax.tree_map(lambda x: check_dimension(x, "Kernel size must be a tuple of non-negative integers."), self.kernel_size)
        jax.tree_map(lambda x: check_dimension(x, "Stride shape must be a tuple of non-negative integers."), self.strides)
        
        if isinstance(self.padding, str) and self.padding.lower() not in ["valid", "same"]:
            raise ValueError(f"""Raised from {self.name}.
                             Padding must only be "valid" or "same", unless a tuple of (low, high) integer pair is provided.
                             Current Value: {self.padding}""")
        elif isinstance(self.padding, str):
            self.padding = self.padding.upper()
        elif isinstance(self.padding, tuple):
            jax.tree_map(lambda x: check_dimension(x, "Padding must be a tuple of positive integers."), self.padding)

        if not isinstance(self.use_bias, bool):
            raise ValueError(f"""Raised from {self.name}.
                             Use_bias argument must be an boolean value.
                             Current value: {self.use_bias}""")
    
    def check_shape_and_rank(self) -> None:
        """Checks the input shapes and assures correct rank arguments have been passed."""
        if len(self.kernel_size) != self.rank:
            raise ValueError(f"""Raised from {self.name}.
                             Kernel_size argument must be {self.rank} dimension.
                             Current dimensions: {len(self.kernel_size)}""")
        
        if len(self.strides) != self.rank:
            raise ValueError(f"""Raised from {self.name}.
                             Strides argument must be {self.rank} dimension.
                             Current dimensions: {len(self.strides)}""")
        
        if len(self.input_dilation_rate) != self.rank:
            raise ValueError(f"""Raised from {self.name}.
                             Input_dilation_rate argument must be {self.rank} dimension.
                             Current dimensions: {len(self.input_dilation_rate)}""")
            
        if len(self.kernel_dilation_rate) != self.rank:
            raise ValueError(f"""Raised from {self.name}.
                             kernel_dilation_rate argument must be {self.rank} dimension.
                             Current dimensions: {len(self.kernel_dilation_rate)}""")
    
    def build(self, inputs: Array) -> None:
        """Builds the kernel and bias variables.

        Args:
            inputs (Array): The input array to build the variable from.
        """
        super().build()
        channel_dim = jnp.shape(inputs)[-1]
        output_shape = self.kernel_size + (channel_dim, self.filters)
        self.kernel = self.add_param(output_shape,
                                     self.kernel_initializer)
        self.bias = self.add_param((self.filters,),
                                   self.bias_initializer) # toDo: Add an alternative bias for different output channels.
        
        self.conv_numbers = compute_dimension_numbers(self.rank)
    
    def call(self, x: Array) -> Array:
        """Defines the control flow of the convolution layer.

        Args:
            x (Array): The inputs to convolve.

        Returns:
            Array: The convolved outputs.
        """
        if jnp.ndim(x) < (self.rank + 2):
            raise ValueError(f"""Raised from {self.name}.
                             Input array must be of dim {self.rank + 2}.
                             Current dimensions: {jnp.ndim(x)}""")
        inputs = jnp.float32(x)
        transformed_x = jax.lax.conv_general_dilated(inputs,
                                                     self.kernel,
                                                     self.strides,
                                                     self.padding,
                                                     self.input_dilation_rate,
                                                     self.kernel_dilation_rate,
                                                     self.conv_numbers)
        
        if self.use_bias:
            return transformed_x + self.bias
        return transformed_x



class Conv1D(Conv):
    """A general 1-D convolutional layer.

        Args:
            filters (int): The number of filters.
            kernel_size (Tuple[int, ...]): The shape of each filter.
            strides (Tuple[int, ...]): The translation step of each filter.
            padding (str, optional): _description_. Defaults to "valid".
            input_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            kernel_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.
            use_bias (bool, optional): _description_. Defaults to True.
            kernel_initializer (str | Initializer, optional): _description_. Defaults to "glorot_uniform".
            bias_initializer (str | Initializer, optional): _description_. Defaults to "zeros".
            kernel_regularizer (_type_, optional): _description_. Defaults to None.
            bias_regularizer (_type_, optional): _description_. Defaults to None.
            kernel_constraint (_type_, optional): _description_. Defaults to None.
            bias_constraint (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "Convolution1D".
        """
    def __init__(self, 
                 filters: int, 
                 kernel_size: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 input_dilation_rate: Tuple[int, ...] | None = None, 
                 kernel_dilation_rate: Tuple[int, ...] | None = None, 
                 activation=None, 
                 use_bias: bool = True, kernel_initializer: str | Initializer = "glorot_uniform", 
                 bias_initializer: str | Initializer = "zeros", 
                 kernel_regularizer=None, bias_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 name: str = "Convolution1D", 
                 **kwargs) -> None:
        super().__init__(1, 
                         filters, 
                         kernel_size, 
                         strides, 
                         padding, 
                         input_dilation_rate, 
                         kernel_dilation_rate, 
                         activation, 
                         use_bias, 
                         kernel_initializer, 
                         bias_initializer, 
                         kernel_regularizer, 
                         bias_regularizer, 
                         kernel_constraint, 
                         bias_constraint, 
                         name, 
                         **kwargs)

class Conv2D(Conv):
    """A general 2-D convolutional layer.

        Args:
            filters (int): The number of filters.
            kernel_size (Tuple[int, ...]): The shape of each filter.
            strides (Tuple[int, ...]): The translation step of each filter.
            padding (str, optional): _description_. Defaults to "valid".
            input_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            kernel_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.
            use_bias (bool, optional): _description_. Defaults to True.
            kernel_initializer (str | Initializer, optional): _description_. Defaults to "glorot_uniform".
            bias_initializer (str | Initializer, optional): _description_. Defaults to "zeros".
            kernel_regularizer (_type_, optional): _description_. Defaults to None.
            bias_regularizer (_type_, optional): _description_. Defaults to None.
            kernel_constraint (_type_, optional): _description_. Defaults to None.
            bias_constraint (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "Convolution2D".
        """
    def __init__(self, 
                 filters: int, 
                 kernel_size: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 input_dilation_rate: Tuple[int, ...] | None = None, 
                 kernel_dilation_rate: Tuple[int, ...] | None = None, 
                 activation=None, 
                 use_bias: bool = True, kernel_initializer: str | Initializer = "glorot_uniform", 
                 bias_initializer: str | Initializer = "zeros", 
                 kernel_regularizer=None, bias_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 name: str = "Convolution2D", 
                 **kwargs) -> None:
        super().__init__(2, 
                         filters, 
                         kernel_size, 
                         strides, 
                         padding, 
                         input_dilation_rate, 
                         kernel_dilation_rate, 
                         activation, 
                         use_bias, 
                         kernel_initializer, 
                         bias_initializer, 
                         kernel_regularizer, 
                         bias_regularizer, 
                         kernel_constraint, 
                         bias_constraint, 
                         name, 
                         **kwargs)

class Conv3D(Conv):
    """A general 3-D convolutional layer.

        Args:
            filters (int): The number of filters.
            kernel_size (Tuple[int, ...]): The shape of each filter.
            strides (Tuple[int, ...]): The translation step of each filter.
            padding (str, optional): _description_. Defaults to "valid".
            input_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            kernel_dilation_rate (Tuple[int, ...]): _description_. Defaults to None.
            activation (_type_, optional): _description_. Defaults to None.
            use_bias (bool, optional): _description_. Defaults to True.
            kernel_initializer (str | Initializer, optional): _description_. Defaults to "glorot_uniform".
            bias_initializer (str | Initializer, optional): _description_. Defaults to "zeros".
            kernel_regularizer (_type_, optional): _description_. Defaults to None.
            bias_regularizer (_type_, optional): _description_. Defaults to None.
            kernel_constraint (_type_, optional): _description_. Defaults to None.
            bias_constraint (_type_, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "Convolution3D".
        """
    def __init__(self, 
                 filters: int, 
                 kernel_size: Tuple[int, ...], 
                 strides: Tuple[int, ...], 
                 padding: str = "valid", 
                 input_dilation_rate: Tuple[int, ...] | None = None, 
                 kernel_dilation_rate: Tuple[int, ...] | None = None, 
                 activation=None, 
                 use_bias: bool = True, kernel_initializer: str | Initializer = "glorot_uniform", 
                 bias_initializer: str | Initializer = "zeros", 
                 kernel_regularizer=None, bias_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 name: str = "Convolution3D", 
                 **kwargs) -> None:
        super().__init__(3, 
                         filters, 
                         kernel_size, 
                         strides, 
                         padding, 
                         input_dilation_rate, 
                         kernel_dilation_rate, 
                         activation, 
                         use_bias, 
                         kernel_initializer, 
                         bias_initializer, 
                         kernel_regularizer, 
                         bias_regularizer, 
                         kernel_constraint, 
                         bias_constraint, 
                         name, 
                         **kwargs)

# class Conv2D(Conv):
#     def __init__(self, filters: int, 
#                  kernel_size: Tuple[int, ...], 
#                  strides: Tuple[int, ...], 
#                 #  dilation_rate: Tuple[int, ...], 
#                  padding: str = "valid", 
#                  in_channel_dim: int = -1, 
#                  data_format: str | None = "channels_last", 
#                  groups=1, 
#                  activation=None, use_bias: bool = True, 
#                  kernel_initializer: str | Initializer = "glorot_uniform", 
#                  bias_initializer: str | Initializer = "zeros", 
#                  kernel_regularizer=None, bias_regularizer=None, 
#                  kernel_constraint=None, bias_constraint=None, 
#                  name: str = "Layer", **kwargs) -> None:
#         super().__init__(2, 
#                          filters, 
#                          kernel_size, 
#                          strides, 
#                          padding, 
#                          in_channel_dim, 
#                          data_format, 
#                          groups, 
#                          activation, 
#                          use_bias, 
#                          kernel_initializer, 
#                          bias_initializer, 
#                          kernel_regularizer, 
#                          bias_regularizer, 
#                          kernel_constraint, 
#                          bias_constraint, 
#                          name, **kwargs)