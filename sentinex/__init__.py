"""Sentinex's Base API"""

# JAX based imports:

from jax import *
from jax.numpy import *
from equinox import *

global_list_deletion = ["nn",
                        "experimental",
                        "core",
                        "arange",
                        "maximum",
                        "max",
                        "device_put",
                        "array",
                        "Module",
                        "experimental",
                        "modelzoo",
                        "pytree",
                        "train_utils"]

for i in global_list_deletion:
    if i in globals():
        del globals()[i]

import jax_dataloader as data
from jax.numpy import arange as range
from jax.numpy import array as tensor
from jax.numpy import max as array_max
from jax.numpy import maximum as multi_array_max
from jax.numpy import min as array_min
from jax.numpy import minimum as multi_array_min

# Sentinex-based imports:
from sentinex import core, modelzoo, nn, pytree, train_utils
from sentinex.core import config
from sentinex.core.config import device_put
from sentinex.core.custom_ops import randn, randu
from sentinex.module import Module
from sentinex.version import __version__
