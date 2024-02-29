"""Sentinex's Base API"""

# JAX based imports:
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from equinox import *
from jax import *
from jax.numpy import *

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


# Importing jax_dataloader data api:
import jax_dataloader as data

# Renaming several imports
from jax.numpy import arange as range
from jax.numpy import array as tensor
from jax.numpy import max as elem_max
from jax.numpy import maximum as array_max
from jax.numpy import min as elem_min
from jax.numpy import minimum as array_min

# Sentinex-based imports:
from sentinex import modelzoo, nn, pytree, train_utils, functional
from sentinex.core import config
from sentinex.core.config import device_put
from sentinex.core.custom_ops import *
from sentinex.module import *
from sentinex.version import __version__
