# Sentinex - A full-fledged Deep Learning Library based on JAX.



## Introduction

Sentinex is a deep learning library that was built on top of JAX and aims to promote simplicity with fast speeds. This guide is a brief walkthrough the main features and helps the user get started with the library.

### Contents
* [Core Features](#core-features)
* [Layers](#layers)
* [Models](#models)
* [Other Examples](#other-examples)
* [Current gimmicks](#current-gimmicks)
* [Installation](#installation)
* [Citations](#citations)
* [Reference documentation](#reference-documentation)

### Core Features

Sentinex has a basic api that allows for the creation and manipulation of various types of arrays. It offers direct JAX functions with little to no modifications. However, some of the functions are modified to provide user ease-of-use and follow similar conventions to bigger deep learning libraries.

Some Examples Include:

```python
import sentinex as sx

# Creates normal random distribution array.
array1 = sx.randn((3,), seed=0) # Array([ 1.8160863 , -0.48262316,  0.33988908], dtype=float32)

# Creates uniform random distribution array.
array2 = sx.randu((3,), seed=0) # Array([ 0.93064284, -0.3706367 ,  0.26605988], dtype=float32)

```


In this case, the random array methods have been reworked to require no JAX PRNG key (it generates one internally), but rather uses an integer to track psuedo-random number generation. Even the ``seed`` argument is optional, since it can be supplied by default (however, best practice dictates to set the number as shown above).

The library also has test utilities that provide a cleaner output, when testing for device setups and availability of accelerations. 


```python
import sentinex as sx

# Tests whether JAX can see the GPU:
print(sx.config.is_device_available(device="gpu")) # True if GPU is available and is setup with proper drivers. 

# Lists all the devices of the type specified:
print(sx.config.list_devices("gpu"))
```

There is even ``sx.config.set_devices("gpu")`` that changes the global accelerator in use.

Lastly, Sentinex has a varied amount of functions like activation functions, initializers, and matrix operations that have been reworked to be more mathematically accurate and provide options to the users.

### Layers

Some of the most basic, yet comprehensive parts of an AI models are its layers. Each layer generally holds __weights__ or a __set of transformations__ that help transform the inputs to the desired output. Let us create Pytorch's Linear layer that performs the following mathematical transformation: $$y = xA^T + B$$

```python
import sentinex as sx
from sentinex import nn # The module containing superclasses suited for neural networks.

class Linear(nn.Layer):
    # Defining annotations to tell that "kernel" is a trainable variable
    kernel: sx.Array
    
    # Omitting "bias", since it is dependent on use_bias boolean
    
    def __init__(self, in_features, out_features, use_bias:bool = True, name: str = "Layer", **kwargs):
        super().__init__(name, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        # Creating kernel and bias, since inputs are already given:
        self.kernel = sx.randn((out_features, in_features))
        
        if self.use_bias:
            self.bias = sx.zeros((out_features,))
            self.set_annotation("bias", sx.Array) # Adds bias to trainable variables if created.
    
    # We can skip build step, since our weights aren't input shape dependent:
    
    # No need of a params argument (unlike TensorWrap), since "self" tracks all parameters
    @sx.filter_jit # Applying Equinox's filter jit.
    def call(self, inputs):
        """Defines the control flow."""
        if not self.use_bias:
            return inputs @ self.kernel.T
        
        return inputs @ self.kernel.T + self.bias

# Creating the layer:
layer = Linear(1, 2)

x = sx.randn((3, 1), seed = 0)

print(layer(x).shape) # (3, 2)
```

Here we utilize the ``sentinex.nn.Layer`` superclass, since it provides all the PyTree handling and imparts compatibility of the class with the rest of the library. With this superclass, our class functions can utilize object-oriented properties, while being JIT compiled and having __no pure functions__ (self.kernel and self.bias wasn't an argument, so the ``call`` function is using intermediaries making the function __impure__).

In case of more complex layers, where building the weights are delayed, we can override the ``build`` attribute to build our weights with input shapes. Check out our ``sentinex.nn.Dense`` layer's source code for those details. For now, we can use premade layers that handle all the inputs, have more detailed errors, and have more options for finetuning.

```python
import sentinex as sx
from sentinex import nn

layer = nn.Dense(units=2, activations='relu', use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros")

x = sx.randn((3, 1), seed = 0)

print(layer(x).shape) # (3, 2)
```

### Models:

Sentinex offers two ways of creating models. One is a simple Sequential class that just stacks the layers in feedforward manner.

```python
import sentinex as sx
from sentinex import nn

modified_multi_layer_perceptron = nn.Sequential([
    nn.Dense(1, activation='tanh'),
    nn.Dense(1, activation='tanh')
])

# Creating inputs:
x = sx.randn((100, 1), seed=0)

# Building model weights:
modified_multi_layer_perceptron.build(x)

print(modified_multi_layer_perceptron) # Viewing the weights and model structure

print(modified_multi_layer_perceptron(x).shape) # Viewing output shape
```

However, for more flexibility, there is a subclassing API available as well.

```python
import sentinex as sx
from sentinex import nn

class Sequential(nn.Model):
    layers: list # Defining where our trainable layers are stored.
    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs) # May pass extra information to nn.Model __init__ function
        self.layers = layers

    # Avoid JIT, since there is a loop. JIT the layers instead (Dense layers are jitted by default)
    def call(self, x):
        """Defines the control flow."""
        for layer in self.layers:
            x = layer(x)
        return x

modified_multi_layer_perceptron = Sequential([
    nn.Dense(1, activation='tanh'),
    nn.Dense(1, activation='tanh')
])

# Creating inputs:
x = sx.randn((100, 1), seed=0)

# Building model weights:
modified_multi_layer_perceptron.build(x)

print(modified_multi_layer_perceptron) # Viewing the weights and model structure

print(modified_multi_layer_perceptron(x).shape) # Viewing output shape
```

Through ``nn.Model``, we can skip parameter initialization logic and just focus on the ``call`` method to give a custom order of transformations. Hence, some subclassing and annotations is all one needs to define more complex models, even if they require multiple inputs or outputs.

### Other Examples

1) Custom Loss Functions
```python
import sentinex as sx
from sentinex import nn

class MAELoss(nn.Loss):
    def __init__(self, name="Mean Absolute Error", **kwargs) -> None:
        super().__init__(name, kwargs)
        # Can add other attributes for more dynamic loss functions.

    @sx.filter_jit
    def call(self, y_true, y_pred):
        return sx.mean(sx.abs(y_true - y_pred))

loss_fn = MAELoss()

array = sx.randn((100, 100))
loss_fn(array, array)
 ```

We can use a normal function as well, since this loss function is a __pure function__. Normally, classes are more helpful for Categorical Crossentropy.
```python
import sentinex as sx
from sentinex import nn

@sx.filter_jit
def mae(y_pred, y_true):
    return sx.mean(sx.abs(y_pred - y_true))

array = sx.randn((100, 100))
print(mae(array, array))
```
2) Custom Activations:
```python 
import sentinex as sx
from sentinex import nn

class ReLU(nn.Activation):
    def __init__(self, max_value, name="ReLU", **kwargs) -> None:
        super().__init__(name = "ReLU", **kwargs)
        self.max_value = max_value

    @sx.filter_jit
    def call(self, inputs):
        sx.array_max(sx.array_min(a, self.max_value), 0)
        

activation = ReLU(1)
print(activation(sx.array([-1, 2, 0.1, 3]))) # [0, 1, 0.1, 1]
```

3) Wacky Custom Modules:
If none of my superclasses fit the need, one can use ``sentinex.Module`` to create a PyTree that can fit nearly all other cases. ``sentinex.Module`` is the superclass of all neural network specific superclasses like ``sentinex.nn.Model``, ``sentinex.nn.Layer``, ``sentinex.nn.Activation``, ``sentinex.nn.Initializer``, and etc.

Hence, it is last recommended superclass, only applicable when the idea doesn't fit the scope of typical AI research.

### Current Gimmicks
There are some gimmicks, due to the nature of classes and JAX's stateless nature of programming. 
1. Current models are all compiled by JAX's internal jit, so any error may remain a bit more cryptic than PyTorchs. However, this problem is still being worked on. We have provided ``dynamic`` keyword arguments for most superclasses, so one has the option to jit compile or not (``dynamic = True`` helps debugging).

2. Graph execution is currently not available, which means that all exported models can only be deployed within a python environment.

3. PyPI version is facing some errors in accessing all the functions, so I plan to release a patch that can fix import errors. For now, please download from the source (this GitHub repository).



### Installation (Install from Source recommended for now)

The device installation of Sentinex depends on its backend, being JAX. Thus, our normal install will be covering only the cpu version. For gpu version, please check [JAX](https://github.com/google/jax)'s documentation.

```bash
pip install --upgrade pip
pip install --upgrade sentinex
```

On Linux, it is often necessary to first update `pip` to a version that supports
`manylinux2014` wheels. Also note that for Linux, we currently release wheels for `x86_64` architectures only, other architectures require building from source. Trying to pip install with other Linux architectures may lead to `jaxlib` not being installed alongside `jax`, although `jax` may successfully install (but fail at runtime). 
**These `pip` installations do not work with Windows, and may fail silently; see
[above](#installation).**

**Note**

If any problems occur with cuda installation, please visit the [JAX](https://github.com/google/jax#installation) github page, in order to understand the problem with lower API installation.

## Citations

This project have been heavily inspired by __TensorFlow__ and once again, is built on the open-source machine learning XLA framework __JAX__. Therefore, I recognize the authors of JAX and TensorFlow for the exceptional work they have done and understand that my library doesn't profit in any sort of way, since it is merely an add-on to the already existing community. __Equinox__ by Patrick Kidger and __JAX Dataloader__ by Hangzhi Guo have graciously allowed me to use some of their functions and their documentation has also aided the development of this project.

```
@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}

@article{kidger2021equinox,
    author={Patrick Kidger and Cristian Garcia},
    title={{E}quinox: neural networks in {JAX} via callable {P}y{T}rees and filtered transformations},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}
```
## Reference documentation

For details about the Sentinex API, see the
[main documentation] (coming soon!)

For details about JAX, see the
[reference documentation](https://jax.readthedocs.io/).

For documentation on TensorFlow API, see the
[API documentation](https://www.tensorflow.org/api_docs/python/tf)