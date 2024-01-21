from sentinex.nn.losses.base_losses import Loss

import optax
import jax.numpy as jnp

class SparseCategoricalCrossentropy(Loss):
    def __init__(self, from_logits = False, name="SparseCategoricalCrossentropy", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        return optax.softmax_cross_entropy_with_integer_labels(y_pred, y_true)