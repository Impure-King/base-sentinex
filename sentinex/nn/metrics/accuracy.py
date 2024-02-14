from ._metric import Metric
import jax.numpy as jnp

class SparseCategoricalAccuracy(Metric):
    
    def __init__(self, name="SparseCategoricalAccuracy", **kwargs):
        super().__init__(name, 
                         **kwargs)
    
    def call(self, y_true, y_pred):
        y_pred = jnp.argmax(y_pred, axis=-1)
        return jnp.mean(y_pred == y_true)