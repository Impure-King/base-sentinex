"""Sentinex's Neural Network API"""

import optax
from jax.nn import *

from sentinex.nn.activations._general_activations import *
from sentinex.nn.initializers._initializer import *
from sentinex.nn.layers._conv import *
from sentinex.nn.layers._pool import *
from sentinex.nn.layers._linear import *
from sentinex.nn.layers._dropout import *
from sentinex.nn.layers.base_layer import Layer
from sentinex.nn.layers.non_trainable import Flatten
from sentinex.nn.losses.base_losses import (Loss, MeanAbsoluteError,
                                            MeanSquaredError)
from sentinex.nn.metrics.accuracy import *
from sentinex.nn.metrics._metric import *
from sentinex.nn.losses.categorical import SparseCategoricalCrossentropy
from sentinex.nn.models._models import *
from sentinex.nn.models.train_state import TrainState
from sentinex.nn.optimizers.base_optimizers import (SGD, OptaxAdam,
                                                    OptaxOptimizer, OptaxSGD,
                                                    Optimizer)
