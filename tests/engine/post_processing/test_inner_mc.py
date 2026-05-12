"""Tests for inner Monte Carlo utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.engine.post_processing.inner_mc import (
    MemoizingSkipForward,
    double_antithetic_paths,
)


# ---------------------------------------------------------------------------
# MemoizingSkipForward
# ---------------------------------------------------------------------------


class TestMemoizingSkipForward:
    def test_forward_returns_array(self):
        msf = MemoizingSkipForward(outer_time=1.0, inner_n_trials=50)
        key = jax.random.PRNGKey(0)
        state = jnp.float64(1.0)
        out = msf.forward(state, dt=0.25, key=key)
        assert out.shape == (50,)

    def test_memoises_keys(self):
        msf = MemoizingSkipForward(outer_time=0.5, inner_n_trials=20)
        key = jax.random.PRNGKey(1)
        state = jnp.float64(0.5)
        out1 = msf.forward(state, dt=0.1, key=key)
        out2 = msf.forward(state, dt=0.1, key=key)
        # Same dt → same cached keys → same shocks → same result
        assert jnp.allclose(out1, out2)

    def test_different_dt_different_cache(self):
        msf = MemoizingSkipForward(outer_time=0.0, inner_n_trials=30)
        key = jax.random.PRNGKey(2)
        state = jnp.float64(1.0)
        out1 = msf.forward(state, dt=0.1, key=key)
        out2 = msf.forward(state, dt=0.2, key=key)
        # Different dt → different cache entry
        assert not jnp.allclose(out1, out2)

    def test_properties(self):
        msf = MemoizingSkipForward(outer_time=2.5, inner_n_trials=100)
        assert msf.outer_time == 2.5
        assert msf.inner_n_trials == 100

    def test_positive_dt_variance(self):
        msf = MemoizingSkipForward(outer_time=0.0, inner_n_trials=500)
        key = jax.random.PRNGKey(3)
        state = jnp.float64(0.0)
        out = msf.forward(state, dt=1.0, key=key)
        # Variance of N(0, sqrt(dt)) with dt=1 should be ~1
        assert 0.5 < float(jnp.var(out)) < 2.0


# ---------------------------------------------------------------------------
# double_antithetic_paths
# ---------------------------------------------------------------------------


class TestDoubleAntitheticPaths:
    def test_output_shape(self):
        key = jax.random.PRNGKey(10)
        out = double_antithetic_paths(key, n_outer=5, n_inner=3, n_steps=8)
        assert out.shape == (10, 6, 8)  # 2*5, 2*3, 8

    def test_antithetic_symmetry_outer(self):
        key = jax.random.PRNGKey(11)
        n_outer, n_inner, n_steps = 4, 2, 6
        out = double_antithetic_paths(key, n_outer, n_inner, n_steps)
        # For a fixed inner index, first/second half of outer dim differ
        # only by the outer antithetic sign.  Their mean across the
        # outer dimension should cancel the outer component.
        mean_all_outer = jnp.mean(out, axis=0)
        # The outer z and -z cancel → residual is only the inner part
        # Check the outer mean equals the inner-only contribution
        k1, k2 = jax.random.split(key)
        inner_z = jax.random.normal(k2, shape=(n_inner, n_steps))
        inner_all = jnp.concatenate([inner_z, -inner_z], axis=0)
        assert jnp.allclose(mean_all_outer, inner_all, atol=1e-6)

    def test_antithetic_symmetry_inner(self):
        key = jax.random.PRNGKey(12)
        n_outer, n_inner, n_steps = 3, 5, 4
        out = double_antithetic_paths(key, n_outer, n_inner, n_steps)
        # Mean across inner dim cancels inner antithetic component
        mean_all_inner = jnp.mean(out, axis=1)
        k1, k2 = jax.random.split(key)
        outer_z = jax.random.normal(k1, shape=(n_outer, n_steps))
        outer_all = jnp.concatenate([outer_z, -outer_z], axis=0)
        assert jnp.allclose(mean_all_inner, outer_all, atol=1e-6)

    def test_small_case(self):
        key = jax.random.PRNGKey(13)
        out = double_antithetic_paths(key, n_outer=1, n_inner=1, n_steps=1)
        assert out.shape == (2, 2, 1)

    def test_different_keys_different_paths(self):
        out1 = double_antithetic_paths(jax.random.PRNGKey(0), 2, 2, 3)
        out2 = double_antithetic_paths(jax.random.PRNGKey(1), 2, 2, 3)
        assert not jnp.allclose(out1, out2)
