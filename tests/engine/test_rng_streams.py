"""Tests for PRNG stream manager and C# seed derivation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.engine.rng import PRNGStreamManager


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


class TestPRNGStreamManager:
    """Tests for PRNGStreamManager C# seed derivation."""

    def test_normals_key_is_master_seed(self) -> None:
        """Normals key is derived directly from the master seed."""
        mgr = PRNGStreamManager(42)
        expected = jax.random.PRNGKey(42)
        assert jnp.array_equal(mgr.normals_key, expected)

    def test_copula_seed_formula(self) -> None:
        """Copula seed = seed * 1000003 + 13."""
        seed = 42
        mgr = PRNGStreamManager(seed)
        copula_seed = (seed * 1000003 + 13) & 0xFFFFFFFF
        expected = jax.random.PRNGKey(copula_seed)
        assert jnp.array_equal(mgr.copula_key, expected)

    def test_chi2_seed_formula(self) -> None:
        """Chi-squared seed = seed * (-104723) - 1000003."""
        seed = 42
        mgr = PRNGStreamManager(seed)
        chi2_seed = (seed * (-104723) - 1000003) & 0xFFFFFFFF
        expected = jax.random.PRNGKey(chi2_seed)
        assert jnp.array_equal(mgr.chi_squared_key, expected)

    def test_three_streams_are_different(self) -> None:
        """All three streams produce different keys."""
        mgr = PRNGStreamManager(42)
        assert not jnp.array_equal(mgr.normals_key, mgr.copula_key)
        assert not jnp.array_equal(mgr.normals_key, mgr.chi_squared_key)
        assert not jnp.array_equal(mgr.copula_key, mgr.chi_squared_key)


# ---------------------------------------------------------------------------
# Trial key derivation
# ---------------------------------------------------------------------------


class TestTrialKeys:
    """Tests for get_trial_keys / fold_in."""

    def test_fold_in_produces_unique_keys(self) -> None:
        """Different trial IDs produce different keys."""
        mgr = PRNGStreamManager(42)
        k0 = mgr.get_trial_keys(0)
        k1 = mgr.get_trial_keys(1)
        # All three streams should differ between trials
        for a, b in zip(k0, k1, strict=True):
            assert not jnp.array_equal(a, b)

    def test_fold_in_reproducible(self) -> None:
        """Same trial ID always produces the same keys."""
        mgr = PRNGStreamManager(42)
        k_first = mgr.get_trial_keys(5)
        k_second = mgr.get_trial_keys(5)
        for a, b in zip(k_first, k_second, strict=True):
            assert jnp.array_equal(a, b)

    def test_different_master_seeds_different_streams(self) -> None:
        """Different master seeds produce entirely different streams."""
        mgr_a = PRNGStreamManager(42)
        mgr_b = PRNGStreamManager(99)
        ka = mgr_a.get_trial_keys(0)
        kb = mgr_b.get_trial_keys(0)
        for a, b in zip(ka, kb, strict=True):
            assert not jnp.array_equal(a, b)

    def test_seed_overflow_handled(self) -> None:
        """Large seeds wrap correctly to 32-bit."""
        # A large seed that would overflow 32-bit without masking
        mgr = PRNGStreamManager(2**31 - 1)
        keys = mgr.get_trial_keys(0)
        # Should not raise; keys should be valid
        assert len(keys) == 3
        for k in keys:
            assert k.shape == (2,)  # JAX PRNGKey shape
