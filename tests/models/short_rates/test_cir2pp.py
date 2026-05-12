"""Tests for the CIR2++ (two-factor market-fitted CIR) short rate model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import CIRParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.math.cir_formulas import cir_A, cir_B, cir_forward_rate
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.models.short_rates.cir2pp import CIR2PlusPlus

jax.config.update("jax_enable_x64", True)

# Standard test parameters — factor 1
ALPHA1 = 0.5
MU1 = 0.02
SIGMA1 = 0.08
X10 = 0.015

# Standard test parameters — factor 2
ALPHA2 = 0.3
MU2 = 0.01
SIGMA2 = 0.06
X20 = 0.005

FLAT_RATE = 0.04


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the CIR2++ model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.cir2pp as mod

    importlib.reload(mod)
    yield
    clear_registry()


class FlatForwardCurve(ParametricCurve):
    """Flat forward rate curve for testing."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def evaluate(self, x: float) -> float:
        return self._rate


class SlopedForwardCurve(ParametricCurve):
    """Upward-sloping forward rate curve for testing."""

    def __init__(self, base: float, slope: float) -> None:
        self._base = base
        self._slope = slope

    def evaluate(self, x: float) -> float:
        return self._base + self._slope * x


@pytest.fixture
def params1() -> CIRParams:
    """CIR parameters for factor 1."""
    return CIRParams(alpha=ALPHA1, mu=MU1, sigma=SIGMA1, initial_value=X10)


@pytest.fixture
def params2() -> CIRParams:
    """CIR parameters for factor 2."""
    return CIRParams(alpha=ALPHA2, mu=MU2, sigma=SIGMA2, initial_value=X20)


@pytest.fixture
def market_curve() -> FlatForwardCurve:
    """Flat forward curve at 4%."""
    return FlatForwardCurve(FLAT_RATE)


@pytest.fixture
def model(
    params1: CIRParams, params2: CIRParams, market_curve: FlatForwardCurve
) -> CIR2PlusPlus:
    """Standard CIR2++ model instance."""
    return CIR2PlusPlus(params1, params2, market_curve)


class TestCIR2PPInit:
    """Tests for CIR2++ model construction and metadata."""

    def test_name(self, model: CIR2PlusPlus) -> None:
        assert model.name == "cir2pp"

    def test_custom_name(
        self,
        params1: CIRParams,
        params2: CIRParams,
        market_curve: FlatForwardCurve,
    ) -> None:
        m = CIR2PlusPlus(params1, params2, market_curve, name="my_cir2pp")
        assert m.name == "my_cir2pp"

    def test_n_shocks(self, model: CIR2PlusPlus) -> None:
        assert model.n_shocks == 2

    def test_shock_config(self, model: CIR2PlusPlus) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 2
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("cir2pp_z1", "cir2pp_z2")

    def test_registry(self) -> None:
        """CIR2++ should be retrievable from registry."""
        cls = get_model("cir2pp")
        assert cls.__name__ == "CIR2PlusPlus"


class TestCIR2PPInitState:
    """Tests for init_state."""

    def test_state_type(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        assert hasattr(state, "x1")
        assert hasattr(state, "x2")
        assert hasattr(state, "state_var1")
        assert hasattr(state, "state_var2")
        assert hasattr(state, "short_rate")

    def test_initial_x_values(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        assert float(state.x1) == pytest.approx(X10, abs=1e-12)
        assert float(state.x2) == pytest.approx(X20, abs=1e-12)
        assert float(state.state_var1) == pytest.approx(X10, abs=1e-12)
        assert float(state.state_var2) == pytest.approx(X20, abs=1e-12)

    def test_short_rate_includes_phi(self, model: CIR2PlusPlus) -> None:
        """Short rate = x₁₀ + x₂₀ + φ(0), not just x₁₀ + x₂₀."""
        state = model.init_state()
        # φ(0) = f_market(0) - f_CIR1(0;x₁₀) - f_CIR2(0;x₂₀)
        # f_CIR(0;x₀) = x₀ (forward rate at τ=0 equals the state)
        # φ(0) = FLAT_RATE - X10 - X20
        expected_phi_0 = FLAT_RATE - X10 - X20
        expected_short_rate = X10 + X20 + expected_phi_0
        assert float(state.short_rate) == pytest.approx(expected_short_rate, abs=1e-8)

    def test_short_rate_equals_market_forward_at_zero(
        self, model: CIR2PlusPlus
    ) -> None:
        """At t=0, the short rate should equal the market forward rate f(0,0)."""
        state = model.init_state()
        # r(0) = x₁₀ + x₂₀ + φ(0) = f_market(0)
        assert float(state.short_rate) == pytest.approx(FLAT_RATE, abs=1e-8)


class TestCIR2PPPhi:
    """Tests for the phi shift function."""

    def test_phi_makes_forward_match(
        self,
        params1: CIRParams,
        params2: CIRParams,
        market_curve: FlatForwardCurve,
    ) -> None:
        """φ(t) should make the combined model match the market forward curve."""
        model = CIR2PlusPlus(params1, params2, market_curve)
        for t in [0.0, 0.5, 1.0, 5.0, 10.0]:
            phi_t = float(model._phi(t))
            f_cir1_t = float(cir_forward_rate(t, X10, ALPHA1, MU1, SIGMA1))
            f_cir2_t = float(cir_forward_rate(t, X20, ALPHA2, MU2, SIGMA2))
            f_market_t = market_curve.evaluate(t)
            # phi(t) + f_CIR1 + f_CIR2 should equal f_market
            assert phi_t >= 0.0
            assert phi_t + f_cir1_t + f_cir2_t == pytest.approx(f_market_t, abs=1e-6)

    def test_phi_non_negative_clamping(self) -> None:
        """Phi should be clamped to 0 when slightly negative."""
        # Use a market curve below the sum of CIR forward rates
        low_curve = FlatForwardCurve((X10 + X20) * 0.99)
        p1 = CIRParams(alpha=ALPHA1, mu=MU1, sigma=SIGMA1, initial_value=X10)
        p2 = CIRParams(alpha=ALPHA2, mu=MU2, sigma=SIGMA2, initial_value=X20)
        model = CIR2PlusPlus(p1, p2, low_curve)
        phi_0 = float(model._phi(0.0))
        assert phi_0 >= 0.0

    def test_phi_clamps_negative_to_zero(self) -> None:
        """Phi should clamp negative values to zero (JIT-safe)."""
        very_low_curve = FlatForwardCurve(0.001)
        p1 = CIRParams(alpha=ALPHA1, mu=MU1, sigma=SIGMA1, initial_value=X10)
        p2 = CIRParams(alpha=ALPHA2, mu=MU2, sigma=SIGMA2, initial_value=X20)
        model = CIR2PlusPlus(p1, p2, very_low_curve)
        phi_val = model._phi(0.0)
        assert float(phi_val) >= 0.0


class TestCIR2PPStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0, 0.0])
        new_state, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "ShortRate" in outputs

    def test_zero_shock_drift(self, model: CIR2PlusPlus) -> None:
        """With zero shocks, both CIR factors should drift toward their means."""
        state = model.init_state()
        shocks = jnp.array([0.0, 0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # x10 < mu1, so x1 should increase
        assert float(new_state.x1) > float(state.x1)
        # x20 < mu2, so x2 should increase
        assert float(new_state.x2) > float(state.x2)

    def test_state_vars_non_negative(self, model: CIR2PlusPlus) -> None:
        """state_var1 and state_var2 should always be >= 0."""
        state = model.init_state()
        shocks = jnp.array([-10.0, -10.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.state_var1) >= 0.0
        assert float(new_state.state_var2) >= 0.0

    def test_short_rate_includes_phi(self, model: CIR2PlusPlus) -> None:
        """Short rate should include phi shift."""
        state = model.init_state()
        shocks = jnp.array([0.5, -0.3])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        phi_at_dt = float(model._phi(0.25))
        expected_r = float(new_state.state_var1) + float(new_state.state_var2) + phi_at_dt
        assert float(new_state.short_rate) == pytest.approx(expected_r, abs=1e-12)

    def test_factors_evolve_independently(self, model: CIR2PlusPlus) -> None:
        """Each factor should only respond to its own shock."""
        state = model.init_state()

        # Shock only factor 1
        shocks_f1 = jnp.array([1.0, 0.0])
        state_f1, _ = model.step(state, 0.0, 0.25, shocks_f1, {})

        # Shock only factor 2
        shocks_f2 = jnp.array([0.0, 1.0])
        state_f2, _ = model.step(state, 0.0, 0.25, shocks_f2, {})

        # Factor 1 should differ between runs, factor 2 unchanged
        assert float(state_f1.x1) != pytest.approx(float(state_f2.x1), abs=1e-6)
        # Factor 2 with zero shock should be the same in f1 run as
        # factor 2 with zero shock is in a zero-shock run
        shocks_zero = jnp.array([0.0, 0.0])
        state_zero, _ = model.step(state, 0.0, 0.25, shocks_zero, {})
        assert float(state_f1.x2) == pytest.approx(float(state_zero.x2), abs=1e-12)
        assert float(state_f2.x1) == pytest.approx(float(state_zero.x1), abs=1e-12)


class TestCIR2PPZCB:
    """Tests for ZCB pricing — the critical FROM TIME 0 formulation."""

    def test_zcb_at_zero_tau(self, model: CIR2PlusPlus) -> None:
        """P(t, t) = 1."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-10)

    def test_zcb_in_unit_interval(self, model: CIR2PlusPlus) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p = model.zcb_price(state, 0.0, mat)
            assert 0.0 < float(p) < 1.0

    def test_zcb_decreasing_with_maturity(self, model: CIR2PlusPlus) -> None:
        """Longer maturity → lower ZCB price."""
        state = model.init_state()
        p1 = float(model.zcb_price(state, 0.0, 1.0))
        p5 = float(model.zcb_price(state, 0.0, 5.0))
        p10 = float(model.zcb_price(state, 0.0, 10.0))
        assert p1 > p5 > p10

    def test_zcb_at_t0_matches_market_discount(
        self,
        params1: CIRParams,
        params2: CIRParams,
        market_curve: FlatForwardCurve,
    ) -> None:
        """At t=0, P(0,T) should match the market discount factor.

        For a flat forward curve at rate r: P(0,T) = exp(-r·T).
        """
        model = CIR2PlusPlus(params1, params2, market_curve)
        state = model.init_state()
        for T in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            p_model = float(model.zcb_price(state, 0.0, T))
            p_market = float(jnp.exp(-FLAT_RATE * T))
            assert p_model == pytest.approx(p_market, abs=1e-5), (
                f"ZCB mismatch at T={T}: model={p_model:.8f}, market={p_market:.8f}"
            )

    def test_zcb_with_sloped_curve(self) -> None:
        """ZCB should work correctly with a non-flat forward curve."""
        curve = SlopedForwardCurve(base=0.03, slope=0.002)
        p1 = CIRParams(alpha=ALPHA1, mu=MU1, sigma=SIGMA1, initial_value=X10)
        p2 = CIRParams(alpha=ALPHA2, mu=MU2, sigma=SIGMA2, initial_value=X20)
        model = CIR2PlusPlus(p1, p2, curve)
        state = model.init_state()

        for T in [1.0, 5.0, 10.0]:
            p = float(model.zcb_price(state, 0.0, T))
            assert 0.0 < p < 1.0


class TestCIR2PPAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        r = model.short_rate(state)
        assert float(r) == pytest.approx(FLAT_RATE, abs=1e-8)

    def test_spot_rate_positive(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        r_val = model.spot_rate(state, 0.0, 5.0)
        assert float(r_val) > 0.0

    def test_spot_rate_near_flat_rate(self, model: CIR2PlusPlus) -> None:
        """For a flat forward curve, spot rate should equal the flat rate."""
        state = model.init_state()
        r_val = float(model.spot_rate(state, 0.0, 5.0))
        assert r_val == pytest.approx(FLAT_RATE, abs=1e-3)

    def test_forward_rate_at_zero(self, model: CIR2PlusPlus) -> None:
        """f(0, 0) should be close to the market forward rate f(0,0)."""
        state = model.init_state()
        f = float(model.forward_rate(state, 0.0, 0.0))
        assert f == pytest.approx(FLAT_RATE, abs=1e-3)

    def test_forward_rate_positive(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        for mat in [0.5, 1.0, 5.0]:
            f = model.forward_rate(state, 0.0, mat)
            assert float(f) > 0.0

    def test_swap_rate_positive(self, model: CIR2PlusPlus) -> None:
        state = model.init_state()
        s_val = model.swap_rate(state, 0.0, 5.0, 0.5)
        assert float(s_val) > 0.0

    def test_swap_rate_reasonable(self, model: CIR2PlusPlus) -> None:
        """Swap rate should be in a reasonable range near the flat rate."""
        state = model.init_state()
        s_val = float(model.swap_rate(state, 0.0, 5.0, 1.0))
        assert 0.01 < s_val < 0.10


class TestCIR2PPStochasticProcess:
    """Tests for analytic_a and analytic_b."""

    def test_analytic_a_at_zero(self, model: CIR2PlusPlus) -> None:
        """A(0) = 1 for both factors."""
        assert float(model.analytic_a(0.0, factor=1)) == pytest.approx(1.0, abs=1e-12)
        assert float(model.analytic_a(0.0, factor=2)) == pytest.approx(1.0, abs=1e-12)

    def test_analytic_b_at_zero(self, model: CIR2PlusPlus) -> None:
        """B(0) = 0 for both factors."""
        assert float(model.analytic_b(0.0, factor=1)) == pytest.approx(0.0, abs=1e-12)
        assert float(model.analytic_b(0.0, factor=2)) == pytest.approx(0.0, abs=1e-12)

    def test_analytic_a_positive(self, model: CIR2PlusPlus) -> None:
        """A(τ) should be positive for all τ > 0 for both factors."""
        for tau in [0.5, 1.0, 5.0, 10.0]:
            assert float(model.analytic_a(tau, factor=1)) > 0.0
            assert float(model.analytic_a(tau, factor=2)) > 0.0

    def test_analytic_b_positive(self, model: CIR2PlusPlus) -> None:
        """B(τ) should be positive for all τ > 0 for both factors."""
        for tau in [0.5, 1.0, 5.0, 10.0]:
            assert float(model.analytic_b(tau, factor=1)) > 0.0
            assert float(model.analytic_b(tau, factor=2)) > 0.0

    def test_factors_have_different_coefficients(self, model: CIR2PlusPlus) -> None:
        """Factor 1 and factor 2 should have different A/B due to different params."""
        tau = 5.0
        a1 = float(model.analytic_a(tau, factor=1))
        a2 = float(model.analytic_a(tau, factor=2))
        assert a1 != pytest.approx(a2, abs=1e-6)

        b1 = float(model.analytic_b(tau, factor=1))
        b2 = float(model.analytic_b(tau, factor=2))
        assert b1 != pytest.approx(b2, abs=1e-6)


class TestCIR2PPMarketFit:
    """Integration tests verifying market curve fitting."""

    def test_flat_curve_fit_at_multiple_maturities(self) -> None:
        """CIR2++ should match flat market curve discount factors closely."""
        rate = 0.05
        curve = FlatForwardCurve(rate)
        p1 = CIRParams(alpha=0.3, mu=0.02, sigma=0.06, initial_value=0.015)
        p2 = CIRParams(alpha=0.6, mu=0.01, sigma=0.04, initial_value=0.010)
        model = CIR2PlusPlus(p1, p2, curve)
        state = model.init_state()

        for T in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
            p_model = float(model.zcb_price(state, 0.0, T))
            p_market = float(jnp.exp(-rate * T))
            assert p_model == pytest.approx(p_market, abs=1e-4), (
                f"Market fit failed at T={T}"
            )

    def test_sloped_curve_fit(self) -> None:
        """CIR2++ should match a sloped market curve at t=0."""
        curve = SlopedForwardCurve(base=0.03, slope=0.001)
        p1 = CIRParams(alpha=0.5, mu=0.02, sigma=0.08, initial_value=0.015)
        p2 = CIRParams(alpha=0.3, mu=0.01, sigma=0.06, initial_value=0.010)
        model = CIR2PlusPlus(p1, p2, curve)
        state = model.init_state()

        for T in [1.0, 5.0, 10.0]:
            p_model = float(model.zcb_price(state, 0.0, T))
            # For linear forward f(t) = base + slope*t:
            # ∫₀ᵀ f(t)dt = base*T + slope*T²/2
            integral = 0.03 * T + 0.001 * T**2 / 2.0
            p_market = float(jnp.exp(-integral))
            assert p_model == pytest.approx(p_market, abs=1e-3), (
                f"Sloped curve fit failed at T={T}: "
                f"model={p_model:.8f}, market={p_market:.8f}"
            )

    def test_different_params_all_match_market(self) -> None:
        """Different CIR parameter sets should all match the same market curve."""
        rate = 0.04
        curve = FlatForwardCurve(rate)
        T = 5.0
        p_market = float(jnp.exp(-rate * T))

        param_pairs = [
            (
                CIRParams(alpha=0.2, mu=0.015, sigma=0.05, initial_value=0.010),
                CIRParams(alpha=0.4, mu=0.010, sigma=0.04, initial_value=0.005),
            ),
            (
                CIRParams(alpha=0.8, mu=0.025, sigma=0.10, initial_value=0.020),
                CIRParams(alpha=0.5, mu=0.015, sigma=0.08, initial_value=0.015),
            ),
            (
                CIRParams(alpha=1.0, mu=0.020, sigma=0.08, initial_value=0.015),
                CIRParams(alpha=0.6, mu=0.010, sigma=0.05, initial_value=0.008),
            ),
        ]
        for p1, p2 in param_pairs:
            model = CIR2PlusPlus(p1, p2, curve)
            state = model.init_state()
            p_model = float(model.zcb_price(state, 0.0, T))
            assert p_model == pytest.approx(p_market, abs=1e-4), (
                f"Fit failed with alpha1={p1.alpha}, alpha2={p2.alpha}: "
                f"model={p_model:.8f}, market={p_market:.8f}"
            )


class TestCIR2PPProtocol:
    """Tests that CIR2PlusPlus implements InterestRateModel protocol."""

    def test_has_required_attributes(self, model: CIR2PlusPlus) -> None:
        """Model should have all InterestRateModel attributes and methods."""
        assert hasattr(model, "name")
        assert hasattr(model, "n_shocks")
        assert hasattr(model, "shock_config")
        assert hasattr(model, "init_state")
        assert hasattr(model, "step")
        assert hasattr(model, "short_rate")
        assert hasattr(model, "zcb_price")
        assert hasattr(model, "spot_rate")
        assert hasattr(model, "forward_rate")
        assert hasattr(model, "swap_rate")
        assert hasattr(model, "analytic_a")
        assert hasattr(model, "analytic_b")

    def test_state_is_cir2state(self, model: CIR2PlusPlus) -> None:
        """State should be CIR2State type."""
        from hyesg.core.types import CIR2State

        state = model.init_state()
        assert isinstance(state, CIR2State)
