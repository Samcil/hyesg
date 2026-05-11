"""Shared test fixtures for hyesg."""

import jax
import pytest

# Enable float64 for all tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX PRNG key for tests."""
    return jax.random.PRNGKey(42)
