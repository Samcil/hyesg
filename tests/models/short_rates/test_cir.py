"""Tests for the CIR short rate model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import CIRParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.models.short_rates.cir import CIR

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
MU = 0.03
SIGMA = 0.1
X0 = 0.02


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the CIR model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.cir as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def params() -> CIRParams:
    """Standard CIR parameters."""
    return CIRParams(alpha=ALPHA, mu=MU, sigma=SIGMA, initial_value=X0)


@pytest.fixture
def model(params: CIRParams) -> CIR:
    """Standard CIR model instance."""
    return CIR(params)


class TestCIRInit:
    """Tests for CIR model construction and metadata."""

    def test_name(self, model: CIR) -> None:
        assert model.name == "cir"

    def test_custom_name(self, params: CIRParams) -> None:
        m = CIR(params, name="my_cir")
        assert m.name == "my_cir"

    def test_n_shocks(self, model: CIR) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: CIR) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("cir_z",)

    def test_registry(self) -> None:
        """CIR should be retrievable from registry."""
        cls = get_model("cir")
        assert cls.__name__ == "CIR"


class TestCIRInitState:
    """Tests for init_state."""

    def test_state_type(self, model: CIR) -> None:
        state = model.init_state()
        assert hasattr(state, "x")
        assert hasattr(state, "state_var")
        assert hasattr(state, "short_rate")

    def test_initial_values(self, model: CIR) -> None:
        state = model.init_state()
        assert float(state.x) == pytest.approx(X0, abs=1e-12)
        assert float(state.state_var) == pytest.approx(X0, abs=1e-12)
        assert float(state.short_rate) == pytest.approx(X0, abs=1e-12)

    def test_zero_initial(self) -> None:
        params = CIRParams(alpha=ALPHA, mu=MU, sigma=SIGMA, initial_value=0.0)
        m = CIR(params)
        state = m.init_state()
        assert float(state.x) == pytest.approx(0.0, abs=1e-12)
        assert float(state.state_var) == pytest.approx(0.0, abs=1e-12)


class TestCIRStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: CIR) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "short_rate" in outputs

    def test_zero_shock_drift(self, model: CIR) -> None:
        """With zero shock, state should drift toward mu."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # x0 < mu, so x should increase
        assert float(new_state.x) > float(state.x)

    def test_state_var_non_negative(self, model: CIR) -> None:
        """state_var should always be >= 0."""
        state = model.init_state()
        # Large negative shock
        shocks = jnp.array([-10.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.state_var) >= 0.0

    def test_short_rate_matches_state_var(self, model: CIR) -> None:
        """short_rate should equal state_var (no phi shift in basic CIR)."""
        state = model.init_state()
        shocks = jnp.array([1.5])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.short_rate) == pytest.approx(
            float(new_state.state_var), abs=1e-12
        )


class TestCIRAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, model: CIR) -> None:
        state = model.init_state()
        r = model.short_rate(state)
        assert float(r) == pytest.approx(X0, abs=1e-12)

    def test_zcb_at_zero_tau(self, model: CIR) -> None:
        """P(t, t) = 1."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-12)

    def test_zcb_in_unit_interval(self, model: CIR) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p = model.zcb_price(state, 0.0, mat)
            assert 0.0 < float(p) < 1.0

    def test_zcb_decreasing_with_maturity(self, model: CIR) -> None:
        """Longer maturity → lower ZCB price."""
        state = model.init_state()
        p1 = float(model.zcb_price(state, 0.0, 1.0))
        p5 = float(model.zcb_price(state, 0.0, 5.0))
        p10 = float(model.zcb_price(state, 0.0, 10.0))
        assert p1 > p5 > p10

    def test_spot_rate_positive(self, model: CIR) -> None:
        """Spot rate should be positive."""
        state = model.init_state()
        r_val = model.spot_rate(state, 0.0, 5.0)
        assert float(r_val) > 0.0

    def test_forward_rate_at_zero(self, model: CIR) -> None:
        """f(t, t) = r(t)."""
        state = model.init_state()
        f = model.forward_rate(state, 0.0, 0.0)
        assert float(f) == pytest.approx(X0, abs=1e-10)

    def test_forward_rate_positive(self, model: CIR) -> None:
        """Forward rate should be positive."""
        state = model.init_state()
        for mat in [0.5, 1.0, 5.0]:
            f = model.forward_rate(state, 0.0, mat)
            assert float(f) > 0.0

    def test_swap_rate_positive(self, model: CIR) -> None:
        """Swap rate should be positive."""
        state = model.init_state()
        s_val = model.swap_rate(state, 0.0, 5.0, 0.5)
        assert float(s_val) > 0.0

    def test_swap_rate_reasonable(self, model: CIR) -> None:
        """Swap rate should be in a reasonable range near the short rate."""
        state = model.init_state()
        s_val = float(model.swap_rate(state, 0.0, 5.0, 1.0))
        # Should be in the general range of mu and x0
        assert 0.001 < s_val < 0.2


class TestCIRStochasticProcess:
    """Tests for analytic_a and analytic_b."""

    def test_analytic_a_at_zero(self, model: CIR) -> None:
        """A(0) = 1."""
        a_val = model.analytic_a(0.0)
        assert float(a_val) == pytest.approx(1.0, abs=1e-12)

    def test_analytic_b_at_zero(self, model: CIR) -> None:
        """B(0) = 0."""
        b_val = model.analytic_b(0.0)
        assert float(b_val) == pytest.approx(0.0, abs=1e-12)

    def test_zcb_matches_ab(self, model: CIR) -> None:
        """P(τ, x) = A(τ) · exp(-B(τ) · x)."""
        state = model.init_state()
        tau = 5.0
        p = float(model.zcb_price(state, 0.0, tau))
        a_val = float(model.analytic_a(tau))
        b_val = float(model.analytic_b(tau))
        expected = a_val * float(jnp.exp(-b_val * state.state_var))
        assert p == pytest.approx(expected, abs=1e-12)
