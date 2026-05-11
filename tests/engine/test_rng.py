"""Tests for the RNG subsystem."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hyesg.engine.rng import (
    create_rng_keys,
    generate_shocks,
    generate_trial_shocks,
    split_shocks,
)

# ---------------------------------------------------------------------------
# Key hierarchy
# ---------------------------------------------------------------------------


class TestCreateRngKeys:
    """Tests for ``create_rng_keys``."""

    def test_shape(self) -> None:
        """Keys array has shape (n_regimes, n_trials, 2)."""
        keys = create_rng_keys(seed=0, n_trials=5, n_regimes=3)
        assert keys.shape == (3, 5, 2)

    def test_determinism(self) -> None:
        """Same seed produces identical keys."""
        a = create_rng_keys(seed=42, n_trials=4, n_regimes=2)
        b = create_rng_keys(seed=42, n_trials=4, n_regimes=2)
        assert jnp.array_equal(a, b)

    def test_different_seed_differs(self) -> None:
        """Different seeds produce different keys."""
        a = create_rng_keys(seed=0, n_trials=4, n_regimes=2)
        b = create_rng_keys(seed=1, n_trials=4, n_regimes=2)
        assert not jnp.array_equal(a, b)

    def test_regime_keys_differ(self) -> None:
        """Each regime receives a distinct key."""
        keys = create_rng_keys(seed=7, n_trials=3, n_regimes=4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert not jnp.array_equal(keys[i], keys[j])

    def test_trial_keys_differ_within_regime(self) -> None:
        """Trial keys within a single regime are distinct."""
        keys = create_rng_keys(seed=7, n_trials=5, n_regimes=1)
        trial_keys = keys[0]  # shape (5, 2)
        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.array_equal(trial_keys[i], trial_keys[j])


# ---------------------------------------------------------------------------
# Shock generation (single trial)
# ---------------------------------------------------------------------------


class TestGenerateShocks:
    """Tests for ``generate_shocks``."""

    def test_shape(self) -> None:
        """Output has shape (n_steps, n_shocks)."""
        key = jax.random.PRNGKey(0)
        shocks = generate_shocks(key, n_steps=10, n_shocks=3)
        assert shocks.shape == (10, 3)

    def test_determinism(self) -> None:
        """Same key → same shocks."""
        key = jax.random.PRNGKey(99)
        a = generate_shocks(key, n_steps=20, n_shocks=4)
        b = generate_shocks(key, n_steps=20, n_shocks=4)
        assert jnp.array_equal(a, b)

    def test_different_key_differs(self) -> None:
        """Different keys → different shocks."""
        a = generate_shocks(jax.random.PRNGKey(0), n_steps=10, n_shocks=3)
        b = generate_shocks(jax.random.PRNGKey(1), n_steps=10, n_shocks=3)
        assert not jnp.array_equal(a, b)

    def test_dtype_float64(self) -> None:
        """Shocks use float64 when x64 is enabled."""
        key = jax.random.PRNGKey(0)
        shocks = generate_shocks(key, n_steps=5, n_shocks=2)
        assert shocks.dtype == jnp.float64

    def test_approximate_standard_normal(self) -> None:
        """Large sample is approximately N(0,1) (moment check)."""
        key = jax.random.PRNGKey(42)
        shocks = generate_shocks(key, n_steps=50_000, n_shocks=1)
        mean = float(jnp.mean(shocks))
        std = float(jnp.std(shocks))
        assert abs(mean) < 0.02, f"Mean too far from 0: {mean}"
        assert abs(std - 1.0) < 0.02, f"Std too far from 1: {std}"


# ---------------------------------------------------------------------------
# Shock splitting
# ---------------------------------------------------------------------------


class TestSplitShocks:
    """Tests for ``split_shocks``."""

    def test_split_correct_shapes(self) -> None:
        """Each sub-array has the expected number of columns."""
        raw = jnp.ones((10, 5))
        parts = split_shocks(raw, shock_sizes=[1, 2, 1, 1])
        assert len(parts) == 4
        assert parts[0].shape == (10, 1)
        assert parts[1].shape == (10, 2)
        assert parts[2].shape == (10, 1)
        assert parts[3].shape == (10, 1)

    def test_split_preserves_data(self) -> None:
        """Concatenating the splits recovers the original array."""
        key = jax.random.PRNGKey(0)
        raw = jax.random.normal(key, shape=(8, 6))
        sizes = [2, 3, 1]
        parts = split_shocks(raw, shock_sizes=sizes)
        reconstructed = jnp.concatenate(parts, axis=1)
        assert jnp.allclose(reconstructed, raw)

    def test_single_model(self) -> None:
        """A single model gets the entire array."""
        raw = jnp.ones((5, 4))
        parts = split_shocks(raw, shock_sizes=[4])
        assert len(parts) == 1
        assert parts[0].shape == (5, 4)


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


class TestGenerateTrialShocks:
    """Tests for ``generate_trial_shocks``."""

    def test_shape(self) -> None:
        """Output has shape (n_trials, n_steps, n_shocks)."""
        shocks = generate_trial_shocks(seed=0, n_trials=8, n_steps=10, n_shocks=3)
        assert shocks.shape == (8, 10, 3)

    def test_determinism(self) -> None:
        """Same seed → identical shocks."""
        a = generate_trial_shocks(seed=42, n_trials=4, n_steps=5, n_shocks=2)
        b = generate_trial_shocks(seed=42, n_trials=4, n_steps=5, n_shocks=2)
        assert jnp.array_equal(a, b)

    def test_different_seed_differs(self) -> None:
        """Different seeds → different shocks."""
        a = generate_trial_shocks(seed=0, n_trials=4, n_steps=5, n_shocks=2)
        b = generate_trial_shocks(seed=1, n_trials=4, n_steps=5, n_shocks=2)
        assert not jnp.array_equal(a, b)

    def test_trials_independent(self) -> None:
        """Each trial receives different shocks."""
        shocks = generate_trial_shocks(seed=0, n_trials=10, n_steps=5, n_shocks=2)
        # Compare first two trials — they must differ
        assert not jnp.array_equal(shocks[0], shocks[1])

    def test_large_batch_statistics(self) -> None:
        """1000 trials × 100 steps × 5 shocks: mean ≈ 0, std ≈ 1."""
        shocks = generate_trial_shocks(seed=7, n_trials=1000, n_steps=100, n_shocks=5)
        assert shocks.shape == (1000, 100, 5)
        mean = float(jnp.mean(shocks))
        std = float(jnp.std(shocks))
        assert abs(mean) < 0.01, f"Mean too far from 0: {mean}"
        assert abs(std - 1.0) < 0.01, f"Std too far from 1: {std}"

    def test_dtype_float64(self) -> None:
        """Batch shocks use float64."""
        shocks = generate_trial_shocks(seed=0, n_trials=2, n_steps=3, n_shocks=1)
        assert shocks.dtype == jnp.float64
