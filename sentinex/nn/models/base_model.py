from typing import Any

import jax

from sentinex.module import Module

__all__ = ["Model", "Sequential"]


class Model(Module):
    """The main superclass for all Models."""

    def __init__(self,
                 name="Model",
                 *args,
                 **kwargs):
        super().__init__(name=name,
                         *args,
                         **kwargs)
        self.built = False

    def build(self, x, *args, **kwargs):
        self.built = True
        with jax.disable_jit():
            self.__call__(x)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.built:
            self.build(*args, **kwargs)
        return self.call(*args, **kwargs)
    
    def call(self, *args, **kwargs):
        return NotImplementedError


class Sequential(Model):
    layers: list
    def __init__(self, layers: list = list(), name="Sequential", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.layers = layers

    def add(self, module: Module):
        if module == self:
            raise ValueError(f"Originates from ``{self.name}``.\n"
                             "Don't add a model to itself.")
        self.layers.append(module)

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


Model.__module__ = "sentinex.nn"
Sequential.__module__ = "sentinex.nn"
