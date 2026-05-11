"""Tests for the Vasicek short rate model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import OUParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.models.short_rates.vasicek import Vasicek

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
MU = 0.03
SIGMA = 0.01
X0 = 0.02


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the Vasicek model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.vasicek as mod

    importlib.reload(mod)
    yield
    clear_registry()


@pytest.fixture
def params() -> OUParams:
    """Standard Vasicek parameters."""
    return OUParams(
        alpha=ALPHA,
        mu=MU,
        sigma=SIGMA,
        initial_value=X0,
        model_type="vasicek",
    )


@pytest.fixture
def model(params: OUParams) -> Vasicek:
    """Standard Vasicek model instance."""
    return Vasicek(params)


class TestVasicekInit:
    """Tests for Vasicek model construction and metadata."""

    def test_name(self, model: Vasicek) -> None:
        assert model.name == "vasicek"

    def test_custom_name(self, params: OUParams) -> None:
        m = Vasicek(params, name="my_vasicek")
        assert m.name == "my_vasicek"

    def test_n_shocks(self, model: Vasicek) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: Vasicek) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True

    def test_registry(self) -> None:
        """Vasicek should be retrievable from registry."""
        cls = get_model("vasicek")
        assert cls.__name__ == "Vasicek"

    def test_wrong_model_type(self) -> None:
        """Should reject non-vasicek model_type."""
        params = OUParams(alpha=ALPHA, sigma=SIGMA, model_type="g1pp")
        with pytest.raises(ValueError, match="model_type='vasicek'"):
            Vasicek(params)


class TestVasicekInitState:
    """Tests for init_state."""

    def test_initial_values(self, model: Vasicek) -> None:
        state = model.init_state()
        assert float(state.x) == pytest.approx(X0, abs=1e-12)
        assert float(state.short_rate) == pytest.approx(X0, abs=1e-12)


class TestVasicekStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: Vasicek) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "short_rate" in outputs

    def test_zero_shock_drift(self, model: Vasicek) -> None:
        """With zero shock, state should drift toward mu."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # x0 < mu, so x should increase
        assert float(new_state.x) > float(state.x)

    def test_can_go_negative(self, model: Vasicek) -> None:
        """Vasicek short rates can go negative."""
        state = model.init_state()
        # Large negative shock
        shocks = jnp.array([-100.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.short_rate) < 0.0


class TestVasicekAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, model: Vasicek) -> None:
        state = model.init_state()
        r = model.short_rate(state)
        assert float(r) == pytest.approx(X0, abs=1e-12)

    def test_zcb_at_zero_tau(self, model: Vasicek) -> None:
        """P(t, t) = 1."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-12)

    def test_zcb_in_unit_interval(self, model: Vasicek) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p = model.zcb_price(state, 0.0, mat)
            assert 0.0 < float(p) < 1.0

    def test_zcb_decreasing_with_maturity(self, model: Vasicek) -> None:
        """Longer maturity → lower ZCB price."""
        state = model.init_state()
        p1 = float(model.zcb_price(state, 0.0, 1.0))
        p5 = float(model.zcb_price(state, 0.0, 5.0))
        p10 = float(model.zcb_price(state, 0.0, 10.0))
        assert p1 > p5 > p10

    def test_spot_rate_positive(self, model: Vasicek) -> None:
        """Spot rate should be positive for positive short rate."""
        state = model.init_state()
        r_val = model.spot_rate(state, 0.0, 5.0)
        assert float(r_val) > 0.0

    def test_forward_rate_at_zero(self, model: Vasicek) -> None:
        """f(t, t) should equal r(t).

        From the analytic formula:
            f(t,T) = μ + (r - μ)e^{-ατ} - (σ²/2α²)(1-e^{-ατ})²
        At τ=0: f = μ + (r - μ) = r.
        """
        state = model.init_state()
        f = model.forward_rate(state, 0.0, 0.0)
        assert float(f) == pytest.approx(X0, abs=1e-10)

    def test_forward_rate_long_run(self, model: Vasicek) -> None:
        """f(t, T) → μ - σ²/(2α²) as T → ∞."""
        state = model.init_state()
        f = model.forward_rate(state, 0.0, 100.0)
        long_run = MU - SIGMA**2 / (2.0 * ALPHA**2)
        assert float(f) == pytest.approx(long_run, abs=1e-6)

    def test_forward_agrees_with_numerical(self, model: Vasicek) -> None:
        """Analytic forward should match -d/dT ln P."""
        state = model.init_state()
        mat = 5.0
        eps = 1e-6
        ln_p_plus = float(jnp.log(model.zcb_price(state, 0.0, mat + eps)))
        ln_p_minus = float(jnp.log(model.zcb_price(state, 0.0, mat - eps)))
        numerical = -(ln_p_plus - ln_p_minus) / (2.0 * eps)
        analytic = float(model.forward_rate(state, 0.0, mat))
        assert analytic == pytest.approx(numerical, rel=1e-5)

    def test_swap_rate_positive(self, model: Vasicek) -> None:
        """Swap rate should be positive."""
        state = model.init_state()
        s_val = model.swap_rate(state, 0.0, 5.0, 0.5)
        assert float(s_val) > 0.0


class TestVasicekStochasticProcess:
    """Tests for analytic_a and analytic_b."""

    def test_analytic_a_at_zero(self, model: Vasicek) -> None:
        """A(0) = 1."""
        a_val = model.analytic_a(0.0)
        assert float(a_val) == pytest.approx(1.0, abs=1e-12)

    def test_analytic_b_at_zero(self, model: Vasicek) -> None:
        """B(0) = 0."""
        b_val = model.analytic_b(0.0)
        assert float(b_val) == pytest.approx(0.0, abs=1e-12)

    def test_zcb_matches_ab(self, model: Vasicek) -> None:
        """P(τ, r) = A(τ) · exp(-B(τ) · r)."""
        state = model.init_state()
        tau = 5.0
        p = float(model.zcb_price(state, 0.0, tau))
        a_val = float(model.analytic_a(tau))
        b_val = float(model.analytic_b(tau))
        expected = a_val * float(jnp.exp(-b_val * state.short_rate))
        assert p == pytest.approx(expected, abs=1e-12)
