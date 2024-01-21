import jax.numpy as jnp

from sentinex.module import Module

__all__ = ["Loss", "MeanSquaredError", "MeanAbsoluteError"]

class Loss(Module):
  def __init__(self, name="Loss", **kwargs):
    super().__init__(name=name,
                     **kwargs)
    pass

class MeanSquaredError(Loss):
  def __call__(self, y_true, y_pred):
    return jnp.mean(jnp.square(y_pred - y_true))
  
class MeanAbsoluteError(Loss):
  def __call__(self, y_true, y_pred):
    return jnp.abs(jnp.mean(y_pred - y_true))

Loss.__module__ = "sentinex.nn"
MeanSquaredError.__module__ = "sentinex.nn"
MeanAbsoluteError.__module__ = "sentinex.nn"