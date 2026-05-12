"""Tests for the credit default intensity model."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import CreditParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.models.credit.credit import Credit

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
MU = 0.02
SIGMA = 0.1
INITIAL_INTENSITY = 0.03
RECOVERY_RATE = 0.4


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the Credit model for each test."""
    clear_registry()
    import hyesg.models.credit.credit as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def params() -> CreditParams:
    """Standard credit parameters."""
    return CreditParams(
        alpha=ALPHA,
        mu=MU,
        sigma=SIGMA,
        initial_intensity=INITIAL_INTENSITY,
        recovery_rate=RECOVERY_RATE,
    )


@pytest.fixture
def model(params: CreditParams) -> Credit:
    """Standard credit model instance."""
    return Credit(params)


# ─── Construction and metadata ───


class TestCreditInit:
    """Tests for Credit model construction and metadata."""

    def test_name(self, model: Credit) -> None:
        assert model.name == "credit"

    def test_custom_name(self, params: CreditParams) -> None:
        m = Credit(params, name="my_credit")
        assert m.name == "my_credit"

    def test_n_shocks(self, model: Credit) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: Credit) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("credit_z",)

    def test_registry(self) -> None:
        """Credit should be retrievable from registry."""
        cls = get_model("credit")
        assert cls.__name__ == "Credit"

    def test_recovery_rate(self, model: Credit) -> None:
        assert model.recovery_rate == pytest.approx(RECOVERY_RATE)


# ─── Init state ───


class TestCreditInitState:
    """Tests for init_state."""

    def test_state_type(self, model: Credit) -> None:
        state = model.init_state()
        assert hasattr(state, "intensity")
        assert hasattr(state, "cum_intensity")
        assert hasattr(state, "has_defaulted")

    def test_initial_values(self, model: Credit) -> None:
        state = model.init_state()
        assert float(state.intensity) == pytest.approx(INITIAL_INTENSITY, abs=1e-12)
        assert float(state.cum_intensity) == pytest.approx(0.0, abs=1e-12)
        assert float(state.has_defaulted) == pytest.approx(0.0, abs=1e-12)


# ─── Step function ───


class TestCreditStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: Credit) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "Intensity" in outputs
        assert "SurvivalProbability" in outputs
        assert "CumIntensity" in outputs

    def test_zero_shock_drift(self, model: Credit) -> None:
        """With zero shock, intensity should drift toward mu."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # initial_intensity > mu, so intensity should decrease
        assert float(new_state.intensity) < float(state.intensity)

    def test_intensity_non_negative(self, model: Credit) -> None:
        """Intensity should remain non-negative after flooring."""
        state = model.init_state()
        # Large negative shock
        shocks = jnp.array([-20.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.intensity) >= 0.0

    def test_cum_intensity_increasing(self, model: Credit) -> None:
        """Cumulative intensity should increase."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.cum_intensity) > 0.0

    def test_survival_decreasing_over_steps(self, model: Credit) -> None:
        """Survival probability should decrease across time steps."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        survivals = []
        for i in range(10):
            state, outputs = model.step(state, i * 0.25, 0.25, shocks, {})
            survivals.append(float(outputs["SurvivalProbability"]))
        # Each survival should be < previous
        for i in range(1, len(survivals)):
            assert survivals[i] < survivals[i - 1]

    def test_survival_in_unit_interval(self, model: Credit) -> None:
        """Survival probability should be in (0, 1]."""
        state = model.init_state()
        shocks = jnp.array([0.5])
        for i in range(20):
            state, outputs = model.step(state, i * 0.25, 0.25, shocks, {})
            s = float(outputs["SurvivalProbability"])
            assert 0.0 < s <= 1.0

    def test_has_defaulted_unchanged(self, model: Credit) -> None:
        """has_defaulted should remain 0 through normal stepping."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.has_defaulted) == pytest.approx(0.0)


# ─── Protocol methods ───


class TestCreditProtocol:
    """Tests for CreditModel protocol methods."""

    def test_default_intensity(self, model: Credit) -> None:
        state = model.init_state()
        lam = model.default_intensity(state, 0.0)
        assert float(lam) == pytest.approx(INITIAL_INTENSITY, abs=1e-12)

    def test_has_defaulted_method(self, model: Credit) -> None:
        state = model.init_state()
        assert float(model.has_defaulted(state)) == pytest.approx(0.0)

    def test_survival_probability_at_t(self, model: Credit) -> None:
        """S(t, t) = 1."""
        state = model.init_state()
        surv = model.survival_probability(state, 0.0, 0.0)
        assert float(surv) == pytest.approx(1.0, abs=1e-10)

    def test_survival_probability_positive(self, model: Credit) -> None:
        """S(t, T) ∈ (0, 1) for T > t."""
        state = model.init_state()
        for T in [1.0, 5.0, 10.0]:
            surv = model.survival_probability(state, 0.0, T)
            assert 0.0 < float(surv) < 1.0

    def test_survival_decreasing_in_T(self, model: Credit) -> None:
        """S(0, T) should decrease with T."""
        state = model.init_state()
        s1 = float(model.survival_probability(state, 0.0, 1.0))
        s5 = float(model.survival_probability(state, 0.0, 5.0))
        s10 = float(model.survival_probability(state, 0.0, 10.0))
        assert s1 > s5 > s10


# ─── Credit spread ───


class TestCreditSpread:
    """Tests for credit spread calculation."""

    def test_spread_positive(self, model: Credit) -> None:
        """Credit spread should be positive for realistic parameters."""
        state = model.init_state()
        spread = model.credit_spread(state, 0.0, 5.0)
        assert float(spread) > 0.0

    def test_spread_reasonable(self, model: Credit) -> None:
        """Spread should be in a reasonable range."""
        state = model.init_state()
        spread = float(model.credit_spread(state, 0.0, 5.0))
        # For typical params, spread ≈ mu level
        assert 0.001 < spread < 0.5


# ─── Parameters validation ───


class TestCreditParams:
    """Tests for CreditParams validation."""

    def test_valid_params(self) -> None:
        p = CreditParams(alpha=0.5, mu=0.02, sigma=0.1)
        assert p.alpha == 0.5
        assert p.mu == 0.02
        assert p.sigma == 0.1
        assert p.initial_intensity == 0.01
        assert p.recovery_rate == 0.4
        assert p.recovery_type == "face_value"

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            CreditParams(alpha=0.0, mu=0.02, sigma=0.1)

    def test_mu_non_negative(self) -> None:
        with pytest.raises(Exception):
            CreditParams(alpha=0.5, mu=-0.01, sigma=0.1)

    def test_sigma_non_negative(self) -> None:
        with pytest.raises(Exception):
            CreditParams(alpha=0.5, mu=0.02, sigma=-0.1)

    def test_recovery_rate_bounds(self) -> None:
        with pytest.raises(Exception):
            CreditParams(alpha=0.5, mu=0.02, sigma=0.1, recovery_rate=1.5)
        with pytest.raises(Exception):
            CreditParams(alpha=0.5, mu=0.02, sigma=0.1, recovery_rate=-0.1)

    def test_frozen(self) -> None:
        p = CreditParams(alpha=0.5, mu=0.02, sigma=0.1)
        with pytest.raises(Exception):
            p.alpha = 1.0


# ─── Multi-step simulation ───


class TestCreditSimulation:
    """Integration-style tests with multi-step simulation."""

    def test_many_steps_intensity_mean_reverts(self, model: Credit) -> None:
        """Over many zero-shock steps, intensity should approach mu."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        for i in range(200):
            state, _ = model.step(state, i * 0.25, 0.25, shocks, {})
        # After 50 years with no shocks, intensity should be near mu
        assert float(state.intensity) == pytest.approx(MU, abs=0.005)

    def test_many_steps_survival_approaches_zero(self, model: Credit) -> None:
        """Over many steps, survival probability should decrease toward 0."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        for i in range(400):
            state, outputs = model.step(state, i * 0.25, 0.25, shocks, {})
        # After 100 years with mu=0.02, S ≈ exp(-mu*T) ≈ exp(-2) ≈ 0.135
        final_surv = float(outputs["SurvivalProbability"])
        assert final_surv < 0.2
