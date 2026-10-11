"""The raw JAX baselines compute the same function as the Sentinex models.

Without this, a benchmark gap could come from the models differing, not the
framework.
"""

import os
import sys

import jax
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
import workloads  # noqa: E402


@pytest.mark.parametrize("name", ["mlp", "cnn", "transformer"])
def test_raw_baseline_matches_sentinex(name):
    w = workloads.WORKLOADS[name]()
    params = w.init_params()
    x, y = w.make_batch(0)
    np.testing.assert_allclose(w.raw_forward(params, x), w.model.apply(params, x), rtol=1e-4, atol=1e-5)

    raw_state, raw_step, raw_loss = w.raw_step()
    sx_state, sx_step, sx_loss = w.sentinex_step("jit")
    for seed in range(3):
        batch = w.make_batch(seed)
        raw_out, sx_out = raw_step(raw_state, batch), sx_step(sx_state, batch)
        np.testing.assert_allclose(raw_loss(raw_out), sx_loss(sx_out), rtol=1e-4)
        raw_state, sx_state = raw_out[0], sx_out[0]
    jax.block_until_ready((raw_state, sx_state))
