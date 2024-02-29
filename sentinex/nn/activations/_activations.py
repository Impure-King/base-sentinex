from random import randint
from typing import Any, Self

import jax
import jax.numpy as jnp
from jax.numpy import float32, maximum, minimum, zeros_like
from jax.random import PRNGKey, uniform
from jaxtyping import Array

import sentinex.functional as F
from sentinex.module import Module

# Handling exports:
__all__ = ["Activation",
           "ReLU",
           "Heaviside",
           "LeakyReLU",
           "RandomReLU",
           "ELU",
           "SELU",
           "Sigmoid",
           "Tanh",
           "Swish",
           "Softplus",
           "Softmax",
           "SiLU",
           "Mish"]


class Activation(Module):
    """A superclass that provides prebuilt functionality for ``sentinex`` compatible activation classes.
    To set trainable attributes, just the attribute name and type as a  field.
    
    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    class ReLU(nn.Activation):
        min_lim: float # sets min_lim as a trainable attribute/leave of the class.
        def __init__(self, min_lim, **kwargs):
            super().__init__(**kwargs) # Defines the name and performs various optimizations
            self.min_lim = min_lim
        
        def call(self, x):
            return sx.array_max(x, sx.zeros(x.shape) + self.min_lim)
    ```

    Args:
        name (str, optional): The hidden name of activation instance. Defaults to "Activation".

    NOTE: Don't override ``self.__call__`` to ensure compatibility. Define control flow in ``self.call`` instead.
    """
    
    def __init__(self, name: str = "Activation", **kwargs) -> None:
        super().__init__(name,
                         **kwargs)
        pass

    @classmethod
    def get_activation(cls, name: str) -> Self:
        """Returns an activation function, when given a string name.
        Largely used internally with ``sentinex.nn.Layer`` subclasses.
        
        Example:
        ```python
        import sentinex as sx
        from sentinex import nn
        
        # Dummy variable for testing:
        dummy_x = sx.tensor([1, -1, 0, 2])
        
        # Retrieving vanilla ReLU Activation:
        activation_name = "relu"
        activation = nn.Activation.get_activation(activation_name)
        
        print(activation(dummy_x)) # [1, 0, 0, 2]
        ```

        Args:
            name (str): The name of the activation function desired.

        Returns:
            Self: The corresponding activation function requested.
        """
        # List of activations:
        __activations = {
            "none": lambda x: x,
            "relu": ReLU(),
            "heaviside": Heaviside(),
            "random_relu": RandomReLU(),
            "leaky_relu": LeakyReLU(),
            "elu": ELU(),
            "selu": SELU(),
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
            "swish": Swish(),
            "softplus": Softplus(),
            "silu": SiLU(),
            "softmax": Softmax(),
            "mish": Mish()
        }
        return __activations[str(name).lower()]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)
    
    @jax.named_scope("sentinex.nn.Activation")    
    def call(self, *args: Any, **kwargs: Any) -> Any:
        return NotImplementedError


class ReLU(Activation):
    """A Rectified Linear Unit function.
    
    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    # Dummy variable for testing:
    dummy_x = sx.tensor([1, -1, 2, 100, 99])
    
    # Initializing the object:
    activation = nn.ReLU(max_value = 99,
                         threshold = 2)
    
    print(activation(dummy_x)) # [2 2 2 99 99]
    ```
    
    Args:
        max_value (float, optional): Specifies the maximum value of the output element. Defaults to None.
        threshold (float, optional): Specifies the minimum value of the output element. Defaults to 0.0.
        name (str, optional): The internal name of the activation instance. Defaults to "ReLU".
    """
    
    def __init__(self,
                 max_value: float | None = None,
                 threshold: float = 0.0,
                 name: str = "ReLU",
                 **kwargs) -> None:
        super().__init__(name=name,
                         **kwargs)
        self.max_value = max_value or jnp.inf
        self.threshold = threshold

    @jax.named_scope("sentinex.nn.ReLU")
    def call(self, x: Array) -> Array:
        return F.relu(x=x, 
                      max_val=self.max_value, 
                      min_val=self.threshold)


class Heaviside(Activation):
    """A vanilla Heaviside function that classifies each element into 1s or 0s,
    based on a certain threshold.

    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    # Dummy variable for testing:
    dummy_x = sx.tensor([1, -1, 2, 100, 99])
    
    # Defining the activation function:
    activation = nn.Heaviside(threshold=3)
    
    print(activation(dummy_x)) # [0. 0. 0. 1. 1.]
    ```
    
    Args:
        threshold (float, optional): The threshold to exceed. Defaults to 0.0.
        name (str, optional): The internal name of the activation instance. Defaults to "Heaviside".
    """
    def __init__(self,
                 threshold: float = 0.0,
                 name: str = "Heaviside",
                 **kwargs) -> None:
        super().__init__(name=name,
                         **kwargs)
        self.threshold = threshold

    @jax.named_scope("sentinex.nn.Heaviside")
    def call(self, x: Array) -> Array:
        return F.heaviside(x=x, 
                           threshold=self.threshold)


class LeakyReLU(Activation):
    """A Leaky Rectified Linear Unit.
    
    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    # Dummy variable for testing:
    dummy_x = sx.tensor([1, -1, 2, 100, 99])
    
    # Defining the activation function:
    activation = nn.LeakyReLU(alpha=1e-3, max_value=99)
    
    print(activation(dummy_x)) # [1 -0.001 2. 99 99]
    ```
    
    Args:
        alpha (float, optional): The slope of the leaky minimum. Defaults to 1e-3.
        max_value (float | None, optional): The maximum value of the output element. Defaults to None.
        threshold (float, optional): The minimum value of the output element. Defaults to 0.0.
        name (str, optional): The internal name of the activation instance. Defaults to "LeakyReLU".
    """
    def __init__(self,
                 alpha: float = 1e-3,
                 max_value: float | None = None,
                 threshold: float = 0.0,
                 name: str = "LeakyReLU",
                 **kwargs) -> None:
        super().__init__(name=name,
                         **kwargs)
        self.alpha = alpha
        self.max_value = max_value or jnp.inf
        self.threshold = threshold

    @jax.named_scope("sentinex.nn.LeakyReLU")
    def call(self, x: Array) -> Array:
        return F.leaky_relu(x = x,
                            max_val=self.max_value,
                            min_val=self.threshold,
                            alpha=self.alpha)


class RandomReLU(LeakyReLU):
    """A Leaky Rectified Linear Unit where the alpha is choosen randomly between a range.

    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    # Dummy variable for testing:
    dummy_x = sx.tensor([1, -1, 2, 100, 99])

    # Defining the activation function:
    activation = nn.RandomReLU(alpha_min=1e-3, alpha_max=1, max_value=99)

    print(activation(dummy_x)) # [1. -0.39  2. 99. 99.]
    ```
    
    Args:
        alpha_min (float, optional): The minimum value of the alpha. Defaults to 0.001.
        alpha_max (float, optional): The maximum value of the alpha. Defaults to 0.3.
        name (str, optional): The internal name of the activation instance. Defaults to "RandomReLU".
        seed (int, optional): The reproducibility of the randomness. Defaults to randint(1, 100).
    """
    def __init__(self,
                 alpha_min: float = 0.001,
                 alpha_max: float = 0.3,
                 name: str = "RandomReLU",
                 seed: int = randint(1, 100),
                 **kwargs) -> None:
        if alpha_min >= alpha_max:
            raise ValueError(f"Raised from {name}.\n",
                             f"Argument alpha_min must be strictly less than alpha_max.")
            
        # Assigning random alpha to superclass.
        alpha = float(uniform(PRNGKey(seed), (1,),
                      minval=alpha_min, maxval=alpha_max))
        super().__init__(alpha=alpha,
                         name=name,
                         **kwargs)


class ELU(Activation):
    """An Exponential Linear Unit.

    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    # Dummy variable for testing:
    dummy_x = sx.tensor([1, -1, 2, 100, 99])

    # Defining the activation function:
    activation = nn.ELU(alpha=1e3)

    print(activation(dummy_x)) # [1. -632.12 2. 100. 99.]
    ```
    
    Args:
        alpha (float, optional): _description_. Defaults to 1e-3.
        name (str, optional): The internal name of the activation instance. Defaults to "ELU".
    """
    def __init__(self,
                 alpha: float = 1e-3,
                 name: str = "ELU",
                 **kwargs) -> None:
        super().__init__(name=name,
                         **kwargs)
        self.alpha = alpha

    @jax.named_scope("sentinex.nn.ELU")
    def call(self, x: Array) -> Array:
        return F.elu(x=x, 
                     alpha=self.alpha)


class SELU(Activation):
    """A Scaled Exponential Linear Unit.
    
    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    # Dummy variable for testing:
    dummy_x = sx.tensor([1, -1, 2, 100, 99])

    # Defining the activation function:
    activation = nn.SELU()

    print(activation(dummy_x)) # [1.05 -1.10 2.1 104.99 103.95]
    ```

    Args:
        name (str, optional): The internal name of the activation instance. Defaults to "SELU".
    """
    def __init__(self, 
                 name: str = "SELU", 
                 **kwargs) -> None:
        super().__init__(name=name, 
                         **kwargs)

    @jax.named_scope("sentinex.nn.SELU")
    def call(self, x: Array) -> Array:
        return F.selu(x=x)


class Sigmoid(Activation):
    """A Sigmoid computation function.

    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    ```
    
    Args:
        name (str, optional): The internal name of the activation instance. Defaults to "Sigmoid".
    """
    def __init__(self, 
                 name: str = "Sigmoid", 
                 **kwargs) -> None:
        super().__init__(name, 
                         **kwargs)

    def call(self, x: Array) -> Array:
        return F.sigmoid(x=x)


class Swish(Activation):
    """A Swish computation function.
    
    Example:
    ```python
    import sentinex as sx
    from sentinex import nn
    
    ```
    
    Args:
        beta (float, optional): __description__. Defaults to 1.702.
        trainable (bool, optional): Dictates where to mark ``beta`` as a trainable variable or not. Defaults to False.
        name (str, optional): The internal name of the activation instance. Defaults to "Swish".    
    """
    def __init__(self, 
                 name: str = "Swish", 
                 **kwargs) -> None:
        super().__init__(name, 
                         **kwargs)

    def call(self, x: Array) -> Array:
        return F.swish(x)


class Tanh(Activation):
    def __init__(self, name: str = "Tanh", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def call(self, x: Array) -> Array:
        return F.tanh(x)


class SiLU(Swish):
    def __init__(self, beta: float = 1, name: str = "SiLU", **kwargs) -> None:
        super().__init__(beta=beta, 
                         name=name, 
                         **kwargs)


class Softmax(Activation):
    def __init__(self, name: str = "Softmax", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def call(self, x):
        return jnp.exp(x)/jnp.sum(jnp.exp(x))


class Softplus(Activation):
    def __init__(self, name: str = "Softplus", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def call(self, x):
        return jnp.log(1 + jnp.exp(x))


class Mish(Activation):
    def __init__(self, name: str = "Mish", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.softplus = Softplus()
        self.tanh = Tanh()

    def call(self, x):
        return x*self.tanh(self.softplus(x))


Activation.__module__ = "sentinex.nn"
ReLU.__module__ = "sentinex.nn"
Heaviside.__module__ = "sentinex.nn"
LeakyReLU.__module__ = "sentinex.nn"
RandomReLU.__module__ = "sentinex.nn"
ELU.__module__ = "sentinex.nn"
SELU.__module__ = "sentinex.nn"
Sigmoid.__module__ = "sentinex.nn"
Swish.__module__ = "sentinex.nn"
Tanh.__module__ = "sentinex.nn"
SiLU.__module__ = "sentinex.nn"
Softmax.__module__ = "sentinex.nn"
Softplus.__module__ = "sentinex.nn"
Mish.__module__ = "sentinex.nn"
