"""Tests for the CIR++ (market-fitted CIR) short rate model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import CIRParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.math.cir_formulas import cir_A, cir_B, cir_forward_rate, cir_zcb_price
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.models.short_rates.cirpp import CIRPlusPlus

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA = 0.5
MU = 0.03
SIGMA = 0.1
X0 = 0.02
FLAT_RATE = 0.04


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the CIR++ model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.cirpp as mod

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
def params() -> CIRParams:
    """Standard CIR parameters."""
    return CIRParams(alpha=ALPHA, mu=MU, sigma=SIGMA, initial_value=X0)


@pytest.fixture
def market_curve() -> FlatForwardCurve:
    """Flat forward curve at 4%."""
    return FlatForwardCurve(FLAT_RATE)


@pytest.fixture
def model(params: CIRParams, market_curve: FlatForwardCurve) -> CIRPlusPlus:
    """Standard CIR++ model instance."""
    return CIRPlusPlus(params, market_curve)


class TestCIRPPInit:
    """Tests for CIR++ model construction and metadata."""

    def test_name(self, model: CIRPlusPlus) -> None:
        assert model.name == "cirpp"

    def test_custom_name(self, params: CIRParams, market_curve: FlatForwardCurve) -> None:
        m = CIRPlusPlus(params, market_curve, name="my_cirpp")
        assert m.name == "my_cirpp"

    def test_n_shocks(self, model: CIRPlusPlus) -> None:
        assert model.n_shocks == 1

    def test_shock_config(self, model: CIRPlusPlus) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 1
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert cfg.names == ("cirpp_z",)

    def test_registry(self) -> None:
        """CIR++ should be retrievable from registry."""
        cls = get_model("cirpp")
        assert cls.__name__ == "CIRPlusPlus"


class TestCIRPPInitState:
    """Tests for init_state."""

    def test_state_type(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        assert hasattr(state, "x")
        assert hasattr(state, "state_var")
        assert hasattr(state, "short_rate")

    def test_initial_x(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        assert float(state.x) == pytest.approx(X0, abs=1e-12)
        assert float(state.state_var) == pytest.approx(X0, abs=1e-12)

    def test_short_rate_includes_phi(self, model: CIRPlusPlus) -> None:
        """Short rate = x₀ + φ(0), not just x₀."""
        state = model.init_state()
        # φ(0) = f_market(0) - f_CIR(0; x₀)
        # f_CIR(0; x₀) = x₀ (forward rate at τ=0 equals the state)
        # φ(0) = FLAT_RATE - X0
        expected_phi_0 = FLAT_RATE - X0
        expected_short_rate = X0 + expected_phi_0
        assert float(state.short_rate) == pytest.approx(expected_short_rate, abs=1e-8)

    def test_short_rate_equals_market_forward_at_zero(
        self, model: CIRPlusPlus
    ) -> None:
        """At t=0, the short rate should equal the market forward rate f(0,0)."""
        state = model.init_state()
        # r(0) = x₀ + φ(0) = x₀ + f_market(0) - f_CIR(0; x₀) = f_market(0)
        assert float(state.short_rate) == pytest.approx(FLAT_RATE, abs=1e-8)


class TestCIRPPPhi:
    """Tests for the phi shift function."""

    def test_phi_makes_forward_match(
        self, params: CIRParams, market_curve: FlatForwardCurve
    ) -> None:
        """φ(t) should make the model match the market forward curve at t=0."""
        model = CIRPlusPlus(params, market_curve)
        for t in [0.0, 0.5, 1.0, 5.0, 10.0]:
            phi_t = float(model._phi(t))
            f_cir_t = float(
                cir_forward_rate(t, X0, ALPHA, MU, SIGMA)
            )
            f_market_t = market_curve.evaluate(t)
            # phi(t) + f_CIR(0,t;x0) should equal f_market(0,t)
            # But phi is clamped to 0 if negative, so check pre-clamping
            # For flat rate above CIR forward, phi should be positive
            assert phi_t >= 0.0
            assert phi_t + f_cir_t == pytest.approx(f_market_t, abs=1e-6)

    def test_phi_non_negative_clamping(self) -> None:
        """Phi should be clamped to 0 when slightly negative."""
        # Use a market curve below the CIR forward rate to get negative phi
        # CIR forward at t=0 is x0, so use market rate below x0
        low_curve = FlatForwardCurve(X0 * 0.99)
        params = CIRParams(alpha=ALPHA, mu=MU, sigma=SIGMA, initial_value=X0)
        model = CIRPlusPlus(params, low_curve)
        phi_0 = float(model._phi(0.0))
        assert phi_0 >= 0.0

    def test_phi_warning_on_large_negative(self) -> None:
        """Phi should warn when significantly negative."""
        # Market rate well below CIR forward
        very_low_curve = FlatForwardCurve(0.001)
        params = CIRParams(alpha=ALPHA, mu=MU, sigma=SIGMA, initial_value=X0)
        with pytest.warns(UserWarning, match="significantly negative"):
            model = CIRPlusPlus(params, very_low_curve)
            _ = model._phi(0.0)


class TestCIRPPStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "short_rate" in outputs

    def test_zero_shock_drift(self, model: CIRPlusPlus) -> None:
        """With zero shock, CIR factor x should drift toward mu."""
        state = model.init_state()
        shocks = jnp.array([0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # x0 < mu, so x should increase
        assert float(new_state.x) > float(state.x)

    def test_state_var_non_negative(self, model: CIRPlusPlus) -> None:
        """state_var should always be >= 0."""
        state = model.init_state()
        shocks = jnp.array([-10.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.state_var) >= 0.0

    def test_short_rate_includes_phi(self, model: CIRPlusPlus) -> None:
        """Short rate should include phi shift (unlike plain CIR)."""
        state = model.init_state()
        shocks = jnp.array([0.5])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        # short_rate should differ from state_var by phi(t+dt)
        phi_at_dt = float(model._phi(0.25))
        expected_r = float(new_state.state_var) + phi_at_dt
        assert float(new_state.short_rate) == pytest.approx(expected_r, abs=1e-12)


class TestCIRPPZCB:
    """Tests for ZCB pricing — the critical FROM TIME 0 formulation."""

    def test_zcb_at_zero_tau(self, model: CIRPlusPlus) -> None:
        """P(t, t) = 1."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-10)

    def test_zcb_in_unit_interval(self, model: CIRPlusPlus) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p = model.zcb_price(state, 0.0, mat)
            assert 0.0 < float(p) < 1.0

    def test_zcb_decreasing_with_maturity(self, model: CIRPlusPlus) -> None:
        """Longer maturity → lower ZCB price."""
        state = model.init_state()
        p1 = float(model.zcb_price(state, 0.0, 1.0))
        p5 = float(model.zcb_price(state, 0.0, 5.0))
        p10 = float(model.zcb_price(state, 0.0, 10.0))
        assert p1 > p5 > p10

    def test_zcb_at_t0_matches_market_discount(
        self, params: CIRParams, market_curve: FlatForwardCurve
    ) -> None:
        """At t=0, P(0,T) should match the market discount factor.

        For a flat forward curve at rate r: P(0,T) = exp(-r·T).
        """
        model = CIRPlusPlus(params, market_curve)
        state = model.init_state()
        for T in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            p_model = float(model.zcb_price(state, 0.0, T))
            p_market = float(jnp.exp(-FLAT_RATE * T))
            assert p_model == pytest.approx(p_market, abs=1e-6), (
                f"ZCB mismatch at T={T}: model={p_model:.8f}, market={p_market:.8f}"
            )

    def test_zcb_uses_from_time_0_not_tau(
        self, params: CIRParams, market_curve: FlatForwardCurve
    ) -> None:
        """Verify the FROM TIME 0 formulation gives different results than A(T-t).

        The CIR++ formula uses A(0,T)/A(0,t), NOT A(T-t).
        These should differ when t > 0.
        """
        model = CIRPlusPlus(params, market_curve)
        state = model.init_state()
        # Evolve to a non-zero time
        shocks = jnp.array([0.5])
        state_1, _ = model.step(state, 0.0, 1.0, shocks, {})

        t = 1.0
        T = 5.0
        tau = T - t

        # FROM TIME 0 formula (correct)
        p_correct = float(model.zcb_price(state_1, t, T))

        # Plain CIR A(τ), B(τ) formula (incorrect for CIR++)
        p_plain_cir = float(cir_zcb_price(tau, state_1.state_var, ALPHA, MU, SIGMA))

        # These should differ because the CIR++ phi shift changes the bond price
        assert p_correct != pytest.approx(p_plain_cir, abs=1e-4), (
            f"FROM TIME 0 and plain CIR should differ: "
            f"correct={p_correct:.8f}, plain={p_plain_cir:.8f}"
        )

    def test_zcb_with_sloped_curve(self) -> None:
        """ZCB should work correctly with a non-flat forward curve."""
        curve = SlopedForwardCurve(base=0.03, slope=0.002)
        params = CIRParams(alpha=ALPHA, mu=MU, sigma=SIGMA, initial_value=X0)
        model = CIRPlusPlus(params, curve)
        state = model.init_state()

        for T in [1.0, 5.0, 10.0]:
            p = float(model.zcb_price(state, 0.0, T))
            assert 0.0 < p < 1.0


class TestCIRPPAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        r = model.short_rate(state)
        assert float(r) == pytest.approx(FLAT_RATE, abs=1e-8)

    def test_spot_rate_positive(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        r_val = model.spot_rate(state, 0.0, 5.0)
        assert float(r_val) > 0.0

    def test_spot_rate_near_flat_rate(self, model: CIRPlusPlus) -> None:
        """For a flat forward curve, spot rate should equal the flat rate."""
        state = model.init_state()
        r_val = float(model.spot_rate(state, 0.0, 5.0))
        assert r_val == pytest.approx(FLAT_RATE, abs=1e-3)

    def test_forward_rate_at_zero(self, model: CIRPlusPlus) -> None:
        """f(0, 0) should be close to the market forward rate f(0,0)."""
        state = model.init_state()
        f = float(model.forward_rate(state, 0.0, 0.0))
        assert f == pytest.approx(FLAT_RATE, abs=1e-3)

    def test_forward_rate_positive(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        for mat in [0.5, 1.0, 5.0]:
            f = model.forward_rate(state, 0.0, mat)
            assert float(f) > 0.0

    def test_swap_rate_positive(self, model: CIRPlusPlus) -> None:
        state = model.init_state()
        s_val = model.swap_rate(state, 0.0, 5.0, 0.5)
        assert float(s_val) > 0.0

    def test_swap_rate_reasonable(self, model: CIRPlusPlus) -> None:
        """Swap rate should be in a reasonable range near the flat rate."""
        state = model.init_state()
        s_val = float(model.swap_rate(state, 0.0, 5.0, 1.0))
        assert 0.01 < s_val < 0.10


class TestCIRPPStochasticProcess:
    """Tests for analytic_a and analytic_b."""

    def test_analytic_a_at_zero(self, model: CIRPlusPlus) -> None:
        """A(0) = 1."""
        a_val = model.analytic_a(0.0)
        assert float(a_val) == pytest.approx(1.0, abs=1e-12)

    def test_analytic_b_at_zero(self, model: CIRPlusPlus) -> None:
        """B(0) = 0."""
        b_val = model.analytic_b(0.0)
        assert float(b_val) == pytest.approx(0.0, abs=1e-12)

    def test_analytic_a_positive(self, model: CIRPlusPlus) -> None:
        """A(τ) should be positive for all τ > 0."""
        for tau in [0.5, 1.0, 5.0, 10.0]:
            assert float(model.analytic_a(tau)) > 0.0

    def test_analytic_b_positive(self, model: CIRPlusPlus) -> None:
        """B(τ) should be positive for all τ > 0."""
        for tau in [0.5, 1.0, 5.0, 10.0]:
            assert float(model.analytic_b(tau)) > 0.0


class TestCIRPPMarketFit:
    """Integration tests verifying market curve fitting."""

    def test_flat_curve_fit_at_multiple_maturities(self) -> None:
        """CIR++ should match flat market curve discount factors closely."""
        rate = 0.05
        curve = FlatForwardCurve(rate)
        params = CIRParams(alpha=0.3, mu=0.04, sigma=0.08, initial_value=0.03)
        model = CIRPlusPlus(params, curve)
        state = model.init_state()

        for T in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
            p_model = float(model.zcb_price(state, 0.0, T))
            p_market = float(jnp.exp(-rate * T))
            assert p_model == pytest.approx(p_market, abs=1e-5), (
                f"Market fit failed at T={T}"
            )

    def test_sloped_curve_fit(self) -> None:
        """CIR++ should match a sloped market curve at t=0."""
        curve = SlopedForwardCurve(base=0.03, slope=0.001)
        params = CIRParams(alpha=0.5, mu=0.03, sigma=0.1, initial_value=0.02)
        model = CIRPlusPlus(params, curve)
        state = model.init_state()

        # Compute expected P(0,T) by numerical integration of the forward curve
        for T in [1.0, 5.0, 10.0]:
            p_model = float(model.zcb_price(state, 0.0, T))
            # For linear forward f(t) = base + slope*t:
            # ∫₀ᵀ f(t)dt = base*T + slope*T²/2
            integral = 0.03 * T + 0.001 * T**2 / 2.0
            p_market = float(jnp.exp(-integral))
            assert p_model == pytest.approx(p_market, abs=1e-4), (
                f"Sloped curve fit failed at T={T}: "
                f"model={p_model:.8f}, market={p_market:.8f}"
            )

    def test_different_params_all_match_market(self) -> None:
        """Different CIR parameters should all match the same market curve."""
        rate = 0.04
        curve = FlatForwardCurve(rate)
        T = 5.0
        p_market = float(jnp.exp(-rate * T))

        param_sets = [
            CIRParams(alpha=0.2, mu=0.02, sigma=0.05, initial_value=0.01),
            CIRParams(alpha=0.8, mu=0.05, sigma=0.15, initial_value=0.04),
            CIRParams(alpha=1.0, mu=0.03, sigma=0.10, initial_value=0.03),
        ]
        for p in param_sets:
            model = CIRPlusPlus(p, curve)
            state = model.init_state()
            p_model = float(model.zcb_price(state, 0.0, T))
            assert p_model == pytest.approx(p_market, abs=1e-4), (
                f"Fit failed with alpha={p.alpha}, mu={p.mu}: "
                f"model={p_model:.8f}, market={p_market:.8f}"
            )
