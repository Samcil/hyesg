"""Tests for the CIR-driven stochastic volatility model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.core.registry import clear_registry, get_model
from hyesg.core.types import VolState
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.models.volatility.cir_vol import CIRVolatility

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
MU = 0.04
SIGMA = 0.1
V0 = 0.04


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the CIRVolatility model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.volatility.cir_vol as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def model() -> CIRVolatility:
    """Standard CIRVolatility model with constant mu."""
    return CIRVolatility(alpha=ALPHA, sigma=SIGMA, v0=V0, mu=MU)


class TestCIRVolInit:
    """Tests for CIRVolatility construction and metadata."""

    def test_name(self, model: CIRVolatility) -> None:
        assert model.name == "cir_vol"

    def test_custom_name(self) -> None:
        m = CIRVolatility(alpha=ALPHA, sigma=SIGMA, v0=V0, name="my_vol")
        assert m.name == "my_vol"

    def test_n_shocks(self, model: CIRVolatility) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: CIRVolatility) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("cir_vol_z",)

    def test_registry(self) -> None:
        """CIRVolatility should be retrievable from registry."""
        cls = get_model("cir_vol")
        assert cls.__name__ == "CIRVolatility"


class TestCIRVolInitState:
    """Tests for init_state."""

    def test_state_type(self, model: CIRVolatility) -> None:
        state = model.init_state()
        assert isinstance(state, VolState)
        assert hasattr(state, "variance")
        assert hasattr(state, "volatility")

    def test_initial_variance(self, model: CIRVolatility) -> None:
        state = model.init_state()
        assert float(state.variance) == pytest.approx(V0, abs=1e-12)

    def test_initial_volatility(self, model: CIRVolatility) -> None:
        state = model.init_state()
        expected_vol = float(jnp.sqrt(jnp.array(V0)))
        assert float(state.volatility) == pytest.approx(expected_vol, abs=1e-12)

    def test_zero_initial(self) -> None:
        m = CIRVolatility(alpha=ALPHA, sigma=SIGMA, v0=0.0, mu=MU)
        state = m.init_state()
        assert float(state.variance) == pytest.approx(0.0, abs=1e-12)
        assert float(state.volatility) == pytest.approx(0.0, abs=1e-12)


class TestCIRVolStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: CIRVolatility) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "Variance" in outputs
        assert "Sigma" in outputs

    def test_zero_vol_of_vol_deterministic(self) -> None:
        """With σ=0, variance follows the ODE: V → μ."""
        m = CIRVolatility(alpha=ALPHA, sigma=0.0, v0=0.02, mu=MU)
        state = m.init_state()
        dt = 0.01
        n_steps = 5000
        for i in range(n_steps):
            shocks = jnp.array([1.0])  # shock irrelevant when sigma=0
            state, _ = m.step(state, i * dt, dt, shocks, {})
        # Should converge to mu
        assert float(state.variance) == pytest.approx(MU, abs=1e-6)

    def test_zero_shock_drift_toward_mu(self) -> None:
        """With zero shock and V0 < μ, variance should increase."""
        m = CIRVolatility(alpha=ALPHA, sigma=SIGMA, v0=0.01, mu=MU)
        state = m.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = m.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.variance) > float(state.variance)

    def test_floor_diffusion(self) -> None:
        """Diffusion uses √max(0, V), preventing NaN from negative V."""
        # Force a negative variance in the state
        state = VolState(
            variance=jnp.array(-0.01, dtype=jnp.float64),
            volatility=jnp.array(0.0, dtype=jnp.float64),
        )
        m = CIRVolatility(alpha=ALPHA, sigma=SIGMA, v0=V0, mu=MU)
        shocks = jnp.array([1.0])
        new_state, outputs = m.step(state, 0.0, 0.25, shocks, {})
        # Should not be NaN
        assert jnp.isfinite(new_state.variance)
        assert jnp.isfinite(outputs["Sigma"])

    def test_volatility_output_is_sqrt_variance(self, model: CIRVolatility) -> None:
        """output['volatility'] = √max(0, V)."""
        state = model.init_state()
        shocks = jnp.array([0.5])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        expected = float(jnp.sqrt(outputs["Variance"]))
        assert float(outputs["Sigma"]) == pytest.approx(expected, abs=1e-12)

    def test_variance_output_floored(self, model: CIRVolatility) -> None:
        """output['variance'] = max(0, V)."""
        state = model.init_state()
        # Large negative shock to force negative raw variance
        shocks = jnp.array([-20.0])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert float(outputs["Variance"]) >= 0.0

    def test_state_variance_can_be_negative(self, model: CIRVolatility) -> None:
        """The raw state variance can go negative (Euler artefact)."""
        state = model.init_state()
        shocks = jnp.array([-20.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # Raw variance might be negative, but outputs are floored
        # Just check it doesn't crash
        assert jnp.isfinite(new_state.variance)


@pytest.mark.slow
class TestCIRVolStatistical:
    """Statistical tests over many trials."""

    def test_mean_reverts_to_mu(self) -> None:
        """Over many trials, E[V(∞)] ≈ μ."""
        n_trials = 5000
        n_steps = 200
        dt = 0.25
        alpha = 1.0  # Faster mean reversion for test
        mu = 0.04
        sigma = 0.05
        v0 = 0.02

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, n_trials)

        def simulate_one(rng_key):
            m = CIRVolatility(alpha=alpha, sigma=sigma, v0=v0, mu=mu)
            state = m.init_state()
            shocks_all = jax.random.normal(rng_key, shape=(n_steps,))
            for i in range(n_steps):
                s = jnp.array([shocks_all[i]])
                state, _ = m.step(state, i * dt, dt, s, {})
            return state.variance

        final_variances = jnp.array([float(simulate_one(k)) for k in keys])
        mean_v = float(jnp.mean(final_variances))
        std_err = float(jnp.std(final_variances)) / jnp.sqrt(n_trials)

        # E[V(∞)] ≈ μ within 3σ/√N
        assert abs(mean_v - mu) < 3 * std_err, (
            f"E[V] = {mean_v:.6f}, μ = {mu}, SE = {std_err:.6f}"
        )


class TestCIRVolTimeDependentMu:
    """Tests for time-dependent μ(t) via ParametricCurve."""

    def test_constant_curve_matches_constant_mu(self) -> None:
        """ConstantCurve(mu) should behave identically to constant mu."""
        curve = ConstantCurve(MU)
        m_const = CIRVolatility(alpha=ALPHA, sigma=SIGMA, v0=V0, mu=MU)
        m_curve = CIRVolatility(
            alpha=ALPHA, sigma=SIGMA, v0=V0, mu_curve=curve
        )

        state_const = m_const.init_state()
        state_curve = m_curve.init_state()

        shocks = jnp.array([0.7])
        _, out_const = m_const.step(state_const, 1.0, 0.25, shocks, {})
        _, out_curve = m_curve.step(state_curve, 1.0, 0.25, shocks, {})

        assert float(out_const["Variance"]) == pytest.approx(
            float(out_curve["Variance"]), abs=1e-12
        )
        assert float(out_const["Sigma"]) == pytest.approx(
            float(out_curve["Sigma"]), abs=1e-12
        )

    def test_time_dependent_mu_uses_curve(self) -> None:
        """Different mu values at different times produce different results."""
        from hyesg.math.curves.primitives import LinearCurve

        # μ(t) = 0.04 + 0.01*t, so μ(0) = 0.04, μ(10) = 0.14
        curve = LinearCurve(slope=0.01, intercept=0.04)
        m = CIRVolatility(alpha=ALPHA, sigma=0.0, v0=V0, mu_curve=curve)

        state = m.init_state()
        shocks = jnp.array([0.0])

        # At t=0, mu=0.04 (same as V0) → no drift
        _, out_t0 = m.step(state, 0.0, 0.25, shocks, {})

        # At t=10, mu=0.14 → large positive drift
        _, out_t10 = m.step(state, 10.0, 0.25, shocks, {})

        # Variance at t=10 should be larger than at t=0 (more drift toward
        # higher mu)
        assert float(out_t10["Variance"]) > float(out_t0["Variance"])
