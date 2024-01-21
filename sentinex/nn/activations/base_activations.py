from random import randint
from typing import Any, Optional

import jax.numpy as jnp
from jax.numpy import float32, maximum, minimum, zeros_like
from jax.random import PRNGKey, uniform

from sentinex.module import Module

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

    def __init__(self, name: str | None = "Activation") -> None:
        super().__init__(name)

    @classmethod
    def get_activation(cls, name: str):
        cls.__activations = {
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
        activations = cls.__activations
        return activations[name.lower()]


class ReLU(Activation):
    def __init__(self,
                 max_value=None,
                 negative_slope=0.0,
                 threshold=0.0,
                 name="ReLU",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)
        self.max_value = max_value or jnp.inf
        self.negative_slope = negative_slope
        self.threshold = threshold

    def __call__(self, x):
        return minimum(maximum(x, zeros_like(x) + self.threshold), zeros_like(x) + self.max_value)


class Heaviside(Activation):
    def __init__(self,
                 threshold=0.0,
                 name="Heaviside",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)
        self.threshold = threshold

    def __call__(self, x):
        return float32(x > self.threshold)


class LeakyReLU(Activation):
    def __init__(self,
                 alpha=1e-3,
                 name: str | None = "LeakyReLU",
                 **kwargs) -> None:
        super().__init__(name,
                         **kwargs)
        self.alpha = alpha

    def __call__(self, x):
        return maximum(x, zeros_like(x) + x * self.alpha)


class RandomReLU(LeakyReLU):
    def __init__(self,
                 alpha_min=0.001,
                 alpha_max=0.3,
                 name: str | None = "RandomReLU",
                 seed: int = randint(1, 100),
                 **kwargs) -> None:
        alpha = float(uniform(PRNGKey(seed), (1,),
                      minval=alpha_min, maxval=alpha_max))
        super().__init__(alpha=alpha,
                         name=name,
                         **kwargs)


class ELU(Activation):
    def __init__(self,
                 alpha: float = 1e-3,
                 name: str | None = "ELU",
                 seed: int = randint(1, 100),
                 **kwargs) -> None:
        super().__init__(
            name=name,
            **kwargs
        )
        self.alpha = alpha
        self.key = PRNGKey(seed)

    def __call__(self, x):
        return jnp.where(x >= 0, x, self.alpha * (jnp.exp(x) - 1))


class SELU(ELU):
    def __init__(self, name: str | None = "SELU", seed: int = randint(1, 100), **kwargs) -> None:
        super().__init__(alpha=1.67, name=name, seed=seed, **kwargs)

    def __call__(self, x):
        return 1.05 * super().__call__(x)


class Sigmoid(Activation):
    def __init__(self, name: str | None = "Sigmoid", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def __call__(self, x):
        return 1/(1 + jnp.exp(x * -1))


class Swish(Activation):
    def __init__(self, beta: float = 1.702, trainable = False, name: str | None = "Swish", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.sigmoid = Sigmoid()
        self.beta = beta
        if trainable:
            self.set_annotation('beta', float)

    def __call__(self, x):
        return x * self.sigmoid(float(self.beta) * x)


class Tanh(Activation):
    def __init__(self, name: str | None = "Tanh", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def __call__(self, x):
        return jnp.tanh(x)


class SiLU(Swish):
    def __init__(self, beta: float = 1, name: str | None = "SiLU", **kwargs) -> None:
        super().__init__(beta, name, **kwargs)


class Softmax(Activation):
    def __init__(self, name: str | None = "Softmax", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def __call__(self, x):
        return jnp.exp(x)/jnp.sum(jnp.exp(x))


class Softplus(Activation):
    def __init__(self, name: str | None = "Softplus", **kwargs) -> None:
        super().__init__(name, **kwargs)

    def __call__(self, x):
        return jnp.log(1 + jnp.exp(x))


class Mish(Activation):
    def __init__(self, name: str | None = "Mish", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.softplus = Softplus()
        self.tanh = Tanh()

    def __call__(self, x):
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
