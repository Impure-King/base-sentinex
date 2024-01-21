"""Sentinex's Neural Network API"""

from jax.nn import *
# from optax import *

from sentinex.nn.activations.base_activations import (ELU, SELU, Activation,
                                                      Heaviside, LeakyReLU,
                                                      Mish, RandomReLU, ReLU,
                                                      Sigmoid, SiLU, Softmax,
                                                      Softplus, Swish)
from sentinex.nn.initializers.base_initializer import (GlorotNormal,
                                                       GlorotUniform, HeNormal,
                                                       HeUniform, Initializer,
                                                       LecunNormal,
                                                       LecunUniform,
                                                       RandomNormal,
                                                       RandomUniform,
                                                       VarianceScaling, Zeros)
from sentinex.nn.layers.base_layer import Layer
from sentinex.nn.layers.linear import Dense, Linear
from sentinex.nn.layers.non_trainable import Flatten
from sentinex.nn.losses.base_losses import (Loss, MeanAbsoluteError,
                                            MeanSquaredError)
from sentinex.nn.losses.categorical import SparseCategoricalCrossentropy
from sentinex.nn.models.base_model import Model, Sequential
from sentinex.nn.models.train_state import TrainState
from sentinex.nn.optimizers.base_optimizers import (SGD, OptaxAdam,
                                                    OptaxOptimizer, OptaxSGD,
                                                    Optimizer)
