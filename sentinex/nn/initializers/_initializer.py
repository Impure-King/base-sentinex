import jax
import jax.numpy as jnp
from jax.numpy import float32

from sentinex.module import Module

__all__ = ["Initializer",
           "RandomNormal",
           "GlorotNormal",
           "HeNormal",
           "RandomUniform",
           "GlorotUniform",
           "HeUniform",
           "VarianceScaling",
           "Zeros"]

__mod__= "sentinex.nn"

class Initializer(Module):
    def __init__(self, name="Initializer", **kwargs):
        super().__init__(name=name,
                        **kwargs)
    
    @classmethod
    def get_initializers(cls, name:str):
        activations = {
                "zeros": Zeros(),
                "glorot_uniform": GlorotUniform(),
                "glorot_normal": GlorotNormal(),
                "random_uniform": RandomUniform(),
                "random_normal": RandomNormal(),
                "lecun_normal": LecunNormal(),
                "lecun_uniform": LecunUniform(),
                "he_normal": HeNormal(),
                "he_uniform": HeUniform()
            }
        return activations[name.lower()]

class VarianceScaling(Initializer):
    def __init__(self, 
                 scale=1.0, 
                 mode='fan_in', 
                 distribution='truncated_normal', 
                 seed=0,
                 name="VarianceScaling",  
                **kwargs):
        super().__init__(name=name,
                        **kwargs)
        
        modes = {
            "fan_in": self.fan_in,
            "fan_out": self.fan_out,
            "fan_avg": self.fan_avg,
        }
        self.scale = scale
        if not mode in modes:
            raise ValueError(f"Originates from {self.name}.\n"
                            "Incorrect mode specified.")
        
        self.fan = modes[mode.lower()]
        distribution = distribution.lower()
        if distribution in ["truncated_normal", "untruncated_normal"]:
            self.compute_matrix = jax.random.normal
        elif distribution == "uniform":
            self.compute_matrix = jax.random.uniform
            self.scale *= 3
        else:
            raise ValueError(f"Originates from {self.name}.\n"
                            "Incorrect distribution specified.")
        self.distribution = distribution
        self.key = jax.random.PRNGKey(seed)

    def fan_in(self, scale, shape):
        fan_in = jnp.prod(shape[:-1])
        return jnp.sqrt(scale/fan_in)

    def fan_out(self, scale, shape):
        fan_out = jnp.prod(shape[1:])
        return jnp.sqrt(scale/fan_out)
    
    def fan_avg(self, scale, shape):
        fan_avg = (jnp.prod(shape[:-1]) + jnp.prod(shape[1:]))/2
        return jnp.sqrt(scale/fan_avg)

    def  __call__(self, shape):
        shape = jnp.asarray(shape)
        computed_std_r = self.fan(self.scale, shape)
        if self.distribution == "uniform":
            return self.compute_matrix(self.key, shape, minval=-1*computed_std_r, maxval=computed_std_r)
        return computed_std_r * self.compute_matrix(self.key, shape)

class GlorotUniform(VarianceScaling):
    def __init__(self, seed=0, name="GlorotUniform", **kwargs):
        super().__init__(scale=1.0,
                         mode="fan_avg",
                         distribution="uniform",
                         name=name, 
                         seed=seed, **kwargs)


class GlorotNormal(VarianceScaling):
    def __init__(self, seed=0, name="GlorotNormal", **kwargs):
        super().__init__(scale=1.0,
                         mode="fan_avg",
                         distribution="untruncated_normal",
                         name=name, 
                         seed=seed, **kwargs)


class RandomUniform(Initializer):
    def __init__(self, seed=0, name="RandomUniform", **kwargs):
        super().__init__(name, **kwargs)
        key = jax.random.PRNGKey(seed)
        self.key, key = jax.random.split(key, 2)

    def  __call__(self, shape):
        return jax.random.uniform(self.key, shape=shape)


class RandomNormal(Initializer):
    def __init__(self, seed=0, name="RandomNormal", **kwargs):
        super().__init__(name, **kwargs)
        key = jax.random.PRNGKey(seed)
        self.key, key = jax.random.split(key, 2)

    def  __call__(self, shape):
        return jax.random.normal(self.key, shape=shape)



class LecunUniform(VarianceScaling):
    def __init__(self, seed=0, name="LecunUniform", **kwargs):
        super().__init__(scale=1.0,
                         mode="fan_in",
                         distribution="uniform",
                         name=name, 
                         seed=seed, **kwargs)
    

class LecunNormal(VarianceScaling):
    def __init__(self, seed=0, name="LecunNormal", **kwargs):
        super().__init__(scale=1.0,
                         mode="fan_in",
                         distribution="untruncated_normal",
                         name=name, 
                         seed=seed, **kwargs)


class HeUniform(VarianceScaling):
    def __init__(self, seed=0, name="HeUniform", **kwargs):
        super().__init__(scale=2.0,
                         mode="fan_in",
                         distribution="uniform",
                         name=name, 
                         seed=seed, **kwargs)


class HeNormal(VarianceScaling):
    def __init__(self, seed=0, name="HeNormal", **kwargs):
        super().__init__(scale=2.0,
                         mode="fan_in",
                         distribution="untruncated_normal",
                         name=name, 
                         seed=seed, **kwargs)


class Zeros(Initializer):
    def __init__(self, dtype=float32, name="Zeros", **kwargs):
        super().__init__(name, **kwargs)
        self.dtype = dtype

    def  __call__(self, shape):
        return jnp.zeros(shape, self.dtype)

# Fixing inspections

Initializer.__module__ = __mod__
GlorotNormal.__module__ = __mod__
GlorotUniform.__module__ = __mod__
RandomNormal.__module__ = __mod__
RandomUniform.__module__ = __mod__
LecunNormal.__module__ = __mod__
LecunUniform.__module__ = __mod__
HeNormal.__module__ = __mod__
HeUniform.__module__ = __mod__
Zeros.__module__ = __mod__