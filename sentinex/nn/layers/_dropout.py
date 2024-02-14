from .base_layer import Layer

import jax
import jax.numpy as jnp
import random

__all__ = ["Dropout"]

class Dropout(Layer):
    def __init__(self, dropout_rate,
                 name="Dropout",
                 **kwargs):
        super().__init__(name=name,
                         **kwargs)
        if not 0 <= dropout_rate <= 1:
            raise ValueError(f"Raised from {self.name}.\n",
                             f"Dropout Rate must be between 0 and 1.")
        self.dropout_rate = float(dropout_rate)
    
    def call(self, x, training=False):
        if not training or not self.dropout_rate:
            return x
        # Creating a basic array to hardmard product with:
        key = jax.random.key(random.randint(0, 100))
        bernouli_rate = 1 - self.dropout_rate
        bernouli_array = jnp.float32(jax.random.bernoulli(key, bernouli_rate, jnp.shape(x)))
        return bernouli_array * x
        
        
Dropout.__module__ = "sentinex.nn"