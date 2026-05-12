"""Tests for the BondIssuer default monitoring system."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.credit.bond_issuer import BondIssuer, IssuerState
from hyesg.models.credit.recovery import FaceValueRecovery, NoRecovery

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def recovery() -> FaceValueRecovery:
    """Standard face-value recovery strategy."""
    return FaceValueRecovery(recovery_rate=0.4)


@pytest.fixture
def issuer(recovery: FaceValueRecovery) -> BondIssuer:
    """Standard bond issuer."""
    return BondIssuer(
        alpha=0.5,
        sigma=0.1,
        initial_intensity=0.03,
        recovery_strategy=recovery,
        is_master=True,
    )


@pytest.fixture
def key() -> jax.Array:
    """Standard PRNG key."""
    return jax.random.PRNGKey(42)


# ─── Initialisation ───


class TestBondIssuerInit:
    """Tests for BondIssuer construction and state init."""

    def test_init_state_threshold_in_unit_interval(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Uniform threshold should be in [0, 1]."""
        state = issuer.init_state(key)
        threshold = float(state.uniform_threshold)
        assert 0.0 <= threshold <= 1.0

    def test_init_state_survival_one(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Initial survival probability = 1."""
        state = issuer.init_state(key)
        assert float(state.survival_prob) == pytest.approx(1.0, abs=1e-12)

    def test_init_state_cum_intensity_zero(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Initial cumulative intensity = 0."""
        state = issuer.init_state(key)
        assert float(state.cum_intensity) == pytest.approx(0.0, abs=1e-12)

    def test_init_state_not_defaulted(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Initially not defaulted."""
        state = issuer.init_state(key)
        assert float(state.has_defaulted) == pytest.approx(0.0, abs=1e-12)

    def test_init_state_default_time_inf(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Initial default time = inf."""
        state = issuer.init_state(key)
        assert jnp.isinf(state.default_time)

    def test_different_keys_give_different_thresholds(
        self, issuer: BondIssuer
    ) -> None:
        """Different keys should produce different thresholds."""
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)
        s1 = issuer.init_state(key1)
        s2 = issuer.init_state(key2)
        assert float(s1.uniform_threshold) != pytest.approx(
            float(s2.uniform_threshold), abs=1e-10
        )


# ─── Update ───


class TestBondIssuerUpdate:
    """Tests for BondIssuer.update."""

    def test_cum_intensity_accumulates(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Cumulative intensity should increase with positive intensity."""
        state = issuer.init_state(key)
        intensity = jnp.array(0.05, dtype=jnp.float64)
        dt = 0.25
        new_state = issuer.update(state, intensity, dt, 0.0)
        expected_cum = 0.05 * 0.25
        assert float(new_state.cum_intensity) == pytest.approx(
            expected_cum, abs=1e-12
        )

    def test_survival_decreases(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Survival should decrease when intensity is positive."""
        state = issuer.init_state(key)
        intensity = jnp.array(0.05, dtype=jnp.float64)
        new_state = issuer.update(state, intensity, 0.25, 0.0)
        assert float(new_state.survival_prob) < 1.0

    def test_survival_formula(
        self, issuer: BondIssuer, key: jax.Array
    ) -> None:
        """Survival = exp(-cum_intensity)."""
        state = issuer.init_state(key)
        intensity = jnp.array(0.1, dtype=jnp.float64)
        new_state = issuer.update(state, intensity, 1.0, 0.0)
        expected = jnp.exp(-0.1)
        assert float(new_state.survival_prob) == pytest.approx(
            float(expected), abs=1e-12
        )

    def test_default_detected_when_survival_below_threshold(
        self, recovery: FaceValueRecovery
    ) -> None:
        """Default triggers when survival drops below threshold."""
        issuer = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
        )
        # Create state with very high threshold (close to 1) to force default
        state = IssuerState(
            cum_intensity=jnp.array(0.0, dtype=jnp.float64),
            survival_prob=jnp.array(1.0, dtype=jnp.float64),
            uniform_threshold=jnp.array(0.999, dtype=jnp.float64),
            has_defaulted=jnp.array(0.0, dtype=jnp.float64),
            default_time=jnp.array(jnp.inf, dtype=jnp.float64),
        )
        # Large intensity → low survival → default
        intensity = jnp.array(5.0, dtype=jnp.float64)
        new_state = issuer.update(state, intensity, 1.0, 0.5)
        assert float(new_state.has_defaulted) == pytest.approx(1.0, abs=1e-12)

    def test_default_time_recorded(
        self, recovery: FaceValueRecovery
    ) -> None:
        """Default time should be recorded on first default."""
        issuer = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
        )
        state = IssuerState(
            cum_intensity=jnp.array(0.0, dtype=jnp.float64),
            survival_prob=jnp.array(1.0, dtype=jnp.float64),
            uniform_threshold=jnp.array(0.999, dtype=jnp.float64),
            has_defaulted=jnp.array(0.0, dtype=jnp.float64),
            default_time=jnp.array(jnp.inf, dtype=jnp.float64),
        )
        intensity = jnp.array(5.0, dtype=jnp.float64)
        new_state = issuer.update(state, intensity, 1.0, 2.5)
        assert float(new_state.default_time) == pytest.approx(2.5, abs=1e-12)

    def test_no_re_default_after_first(
        self, recovery: FaceValueRecovery
    ) -> None:
        """Once defaulted, has_defaulted stays 1 and default_time frozen."""
        issuer = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
        )
        # Already defaulted state
        state = IssuerState(
            cum_intensity=jnp.array(3.0, dtype=jnp.float64),
            survival_prob=jnp.array(0.05, dtype=jnp.float64),
            uniform_threshold=jnp.array(0.5, dtype=jnp.float64),
            has_defaulted=jnp.array(1.0, dtype=jnp.float64),
            default_time=jnp.array(1.0, dtype=jnp.float64),
        )
        intensity = jnp.array(0.01, dtype=jnp.float64)
        new_state = issuer.update(state, intensity, 0.25, 5.0)
        assert float(new_state.has_defaulted) == pytest.approx(1.0, abs=1e-12)
        assert float(new_state.default_time) == pytest.approx(1.0, abs=1e-12)


# ─── Master/Slave ───


class TestBondIssuerMasterSlave:
    """Tests for master/slave threshold sharing."""

    def test_slave_uses_external_threshold(
        self, recovery: FaceValueRecovery, key: jax.Array
    ) -> None:
        """Slave issuer should use externally supplied threshold."""
        master = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
            is_master=True,
        )
        master_state = master.init_state(key)

        slave = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
            is_master=False,
        )
        slave_state = slave.init_state_with_threshold(
            master_state.uniform_threshold
        )
        assert float(slave_state.uniform_threshold) == pytest.approx(
            float(master_state.uniform_threshold), abs=1e-12
        )

    def test_master_slave_same_default_timing(
        self, recovery: FaceValueRecovery, key: jax.Array
    ) -> None:
        """Master and slave with same threshold default at same time."""
        master = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
            is_master=True,
        )
        master_state = master.init_state(key)

        slave = BondIssuer(
            alpha=0.5,
            sigma=0.1,
            initial_intensity=0.03,
            recovery_strategy=recovery,
            is_master=False,
        )
        slave_state = slave.init_state_with_threshold(
            master_state.uniform_threshold
        )

        # Apply same intensity to both
        intensity = jnp.array(0.5, dtype=jnp.float64)
        for i in range(20):
            t = i * 0.25
            master_state = master.update(master_state, intensity, 0.25, t)
            slave_state = slave.update(slave_state, intensity, 0.25, t)

        # Both should have same default status
        assert float(master_state.has_defaulted) == pytest.approx(
            float(slave_state.has_defaulted), abs=1e-12
        )
