from typing import Any

import jax

from sentinex.module import Module
import inspect

__all__ = ["Model", "Sequential"]


class Model(Module):
    """The base superclass for all models. It provides all the
    ``Module`` class' functionality, but also provides a build
    method to initialize parameters before hand.

    Args:
        name (str, optional): The implicit name of the instance. Defaults to "Model".
        dynamic (bool, optional): Specifies whether to jit (False) or not jit (True) the ``__call__`` method. Defaults to False.
    """
    def __init__(self,
                 dynamic: bool = False,
                 name: str = "Model",
                 **kwargs) -> None:
        super().__init__(dynamic=dynamic,
                         name=name,
                         **kwargs)
        self.built = False

    def build(self, *args, **kwargs):
        """Builds the model parameters by calling
        the model. Disables jit during execution.
        """
        self.built = True
        with jax.disable_jit():
            self.__call__(*args, **kwargs)

    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the model and performs computation.
        Don't override this method and instead define control flow
        in ``call`` method.

        Returns:
            Any: Returns the outputs of the model.
        """
        # Checks whether the model is built:
        if not self.built:
            self.build(*args, **kwargs)
            
        return self.call(*args, **kwargs)
    
    def call(self, *args, **kwargs):
        return NotImplementedError


class Sequential(Model):
    """A Sequential model that defines control of layers in
    sequentially manners. It stacks all the layers as a list
    and then iterates through while passing the inputs.

    Args:
        layers (list, optional): The list of layers to compute the outputs with. Defaults to list().
        name (str, optional): The implicit name of the instance. Defaults to "Sequential".
    """
    layers: list
    def __init__(self, layers: list = list(), name: str = "Sequential", **kwargs) -> None:
        super().__init__(
            name=name, 
            **kwargs)
        self.layers = layers

    def add(self, module: Module) -> None:
        """Appends a module to the layer list.

        Args:
            module (Module): The module to append.

        Raises:
            ValueError: Occurs when the user tries to append the model to it's own list.
        """
        if module == self:
            raise ValueError(f"Originates from ``{self.name}``.\n"
                             "Don't add a model to itself.")
        self.layers.append(module)

    def call(self, x: Any, training=False) -> Any:
        """Computes the outputs, when given an input.

        Args:
            x (Any): An input to compute with.

        Returns:
            Any: The output of the computation.
        """
        for layer in self.layers:
            if "training" in dict(inspect.signature(layer.call).parameters):
                x = layer(x, training)
            else:
                x = layer(x)
        return x


Model.__module__ = "sentinex.nn"
Sequential.__module__ = "sentinex.nn"
