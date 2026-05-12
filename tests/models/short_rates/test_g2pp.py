"""Tests for the G2++ (two-factor Gaussian) short rate model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import G2PPParams
from hyesg.core.registry import clear_registry, get_model
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.models.short_rates.g2pp import G2PP

jax.config.update("jax_enable_x64", True)

# Standard test parameters
ALPHA1 = 0.5
SIGMA1 = 0.01
ALPHA2 = 0.8
SIGMA2 = 0.015
RHO = 0.3
FLAT_RATE = 0.03


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register the G2PP model for each test."""
    clear_registry()
    import importlib

    import hyesg.models.short_rates.g2pp as mod

    importlib.reload(mod)
    yield
    clear_registry()


class FlatForwardCurve(ParametricCurve):
    """Flat forward rate curve for testing."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def evaluate(self, x: float) -> float:
        return self._rate


@pytest.fixture
def params() -> G2PPParams:
    """Standard G2++ parameters."""
    return G2PPParams(
        alpha1=ALPHA1,
        sigma1=SIGMA1,
        alpha2=ALPHA2,
        sigma2=SIGMA2,
        rho=RHO,
    )


@pytest.fixture
def params_uncorrelated() -> G2PPParams:
    """G2++ parameters with zero correlation."""
    return G2PPParams(
        alpha1=ALPHA1,
        sigma1=SIGMA1,
        alpha2=ALPHA2,
        sigma2=SIGMA2,
        rho=0.0,
    )


@pytest.fixture
def market_curve() -> FlatForwardCurve:
    """Flat forward curve at 3%."""
    return FlatForwardCurve(FLAT_RATE)


@pytest.fixture
def model(params: G2PPParams, market_curve: FlatForwardCurve) -> G2PP:
    """Standard G2++ model instance."""
    return G2PP(params, market_curve)


@pytest.fixture
def model_uncorrelated(
    params_uncorrelated: G2PPParams, market_curve: FlatForwardCurve
) -> G2PP:
    """G2++ model with ρ=0."""
    return G2PP(params_uncorrelated, market_curve, name="g2pp_uncorr")


class TestG2PPParams:
    """Tests for G2PPParams validation."""

    def test_valid_params(self) -> None:
        p = G2PPParams(
            alpha1=0.5, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=0.3
        )
        assert p.alpha1 == 0.5
        assert p.rho == 0.3

    def test_frozen(self) -> None:
        p = G2PPParams(
            alpha1=0.5, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=0.0
        )
        with pytest.raises(Exception):
            p.alpha1 = 1.0  # type: ignore[misc]

    def test_alpha_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            G2PPParams(alpha1=0.0, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=0.0)

    def test_sigma_can_be_zero(self) -> None:
        p = G2PPParams(
            alpha1=0.5, sigma1=0.0, alpha2=0.8, sigma2=0.0, rho=0.0
        )
        assert p.sigma1 == 0.0

    def test_rho_bounds(self) -> None:
        G2PPParams(alpha1=0.5, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=1.0)
        G2PPParams(alpha1=0.5, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=-1.0)
        with pytest.raises(Exception):
            G2PPParams(
                alpha1=0.5, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=1.1
            )

    def test_default_initial_values(self) -> None:
        p = G2PPParams(
            alpha1=0.5, sigma1=0.01, alpha2=0.8, sigma2=0.02, rho=0.0
        )
        assert p.x1_initial == 0.0
        assert p.x2_initial == 0.0


class TestG2PPInit:
    """Tests for G2++ model construction and metadata."""

    def test_name(self, model: G2PP) -> None:
        assert model.name == "g2pp"

    def test_custom_name(
        self, params: G2PPParams, market_curve: FlatForwardCurve
    ) -> None:
        m = G2PP(params, market_curve, name="my_g2pp")
        assert m.name == "my_g2pp"

    def test_n_shocks(self, model: G2PP) -> None:
        assert model.n_shocks == 2

    def test_shock_config(self, model: G2PP) -> None:
        cfg = model.shock_config
        assert cfg.n_shocks == 2
        assert cfg.distribution == "normal"
        assert cfg.correlate is True
        assert len(cfg.names) == 2

    def test_registry(self) -> None:
        """G2PP should be retrievable from registry."""
        cls = get_model("g2pp")
        assert cls.__name__ == "G2PP"


class TestG2PPInitState:
    """Tests for init_state."""

    def test_initial_x1(self, model: G2PP) -> None:
        state = model.init_state()
        assert float(state.x1) == pytest.approx(0.0, abs=1e-12)

    def test_initial_x2(self, model: G2PP) -> None:
        state = model.init_state()
        assert float(state.x2) == pytest.approx(0.0, abs=1e-12)

    def test_initial_short_rate(self, model: G2PP) -> None:
        """r(0) = x₁(0) + x₂(0) + φ(0), and φ(0) = f(0,0) = flat_rate."""
        state = model.init_state()
        assert float(state.short_rate) == pytest.approx(FLAT_RATE, abs=1e-10)

    def test_custom_initial_values(
        self, market_curve: FlatForwardCurve
    ) -> None:
        params = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=RHO,
            x1_initial=0.01,
            x2_initial=-0.005,
        )
        m = G2PP(params, market_curve, name="g2pp_custom")
        state = m.init_state()
        assert float(state.x1) == pytest.approx(0.01, abs=1e-12)
        assert float(state.x2) == pytest.approx(-0.005, abs=1e-12)


class TestG2PPStep:
    """Tests for the Euler step."""

    def test_output_keys(self, model: G2PP) -> None:
        state = model.init_state()
        shocks = jnp.array([0.0, 0.0])
        _, outputs = model.step(state, 0.0, 0.25, shocks, {})
        assert "ShortRate" in outputs

    def test_zero_shock(self, model: G2PP) -> None:
        """With zero shocks and x₁=x₂=0, both should stay near zero."""
        state = model.init_state()
        shocks = jnp.array([0.0, 0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.x1) == pytest.approx(0.0, abs=1e-10)
        assert float(new_state.x2) == pytest.approx(0.0, abs=1e-10)

    def test_positive_shock_increases_x1(self, model: G2PP) -> None:
        """A positive shock to factor 1 should increase x₁."""
        state = model.init_state()
        shocks = jnp.array([2.0, 0.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.x1) > 0.0

    def test_positive_shock_increases_x2(self, model: G2PP) -> None:
        """A positive shock to factor 2 should increase x₂."""
        state = model.init_state()
        shocks = jnp.array([0.0, 2.0])
        new_state, _ = model.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.x2) > 0.0

    def test_mean_reversion(self, model: G2PP) -> None:
        """Factors should mean-revert toward zero."""
        from hyesg.core.types import G2State

        x1_init = jnp.array(0.05, dtype=jnp.float64)
        x2_init = jnp.array(0.03, dtype=jnp.float64)
        state = G2State(
            x1=x1_init,
            x2=x2_init,
            short_rate=x1_init + x2_init + jnp.array(FLAT_RATE),
        )
        shocks = jnp.array([0.0, 0.0])
        new_state, _ = model.step(state, 1.0, 0.25, shocks, {})
        # Both should be pulled toward zero
        assert abs(float(new_state.x1)) < abs(float(state.x1))
        assert abs(float(new_state.x2)) < abs(float(state.x2))

    def test_shocks_independent(self, model: G2PP) -> None:
        """Shocks to factor 1 should not affect factor 2 directly."""
        state = model.init_state()
        shocks_z1 = jnp.array([1.0, 0.0])
        shocks_z2 = jnp.array([0.0, 1.0])

        new1, _ = model.step(state, 0.0, 0.25, shocks_z1, {})
        new2, _ = model.step(state, 0.0, 0.25, shocks_z2, {})

        # z1 shock moves x1 but not x2
        assert float(new1.x1) != 0.0
        assert float(new1.x2) == pytest.approx(0.0, abs=1e-12)
        # z2 shock moves x2 but not x1
        assert float(new2.x2) != 0.0
        assert float(new2.x1) == pytest.approx(0.0, abs=1e-12)


class TestG2PPAnalytics:
    """Tests for ShortRateModel analytics."""

    def test_short_rate_accessor(self, model: G2PP) -> None:
        state = model.init_state()
        r_val = model.short_rate(state)
        assert float(r_val) == pytest.approx(FLAT_RATE, abs=1e-10)

    def test_zcb_at_zero_tau(self, model: G2PP) -> None:
        """P(t, t) = 1."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 0.0)
        assert float(p) == pytest.approx(1.0, abs=1e-10)

    def test_zcb_matches_market_at_t0(self, model: G2PP) -> None:
        """At t=0, x₁=x₂=0: P(0,T) should match the market ZCB price.

        For flat forward rate f:
            P₀(T) = exp(-f·T)
        """
        state = model.init_state()
        for mat in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]:
            p_model = float(model.zcb_price(state, 0.0, mat))
            p_market = float(jnp.exp(-FLAT_RATE * mat))
            assert p_model == pytest.approx(p_market, rel=1e-6), (
                f"ZCB mismatch at T={mat}: model={p_model}, market={p_market}"
            )

    def test_zcb_in_unit_interval(self, model: G2PP) -> None:
        """P(t, T) ∈ (0, 1) for T > t with positive rates."""
        state = model.init_state()
        for mat in [1.0, 5.0, 10.0]:
            p = model.zcb_price(state, 0.0, mat)
            assert 0.0 < float(p) < 1.0

    def test_zcb_decreasing_with_maturity(self, model: G2PP) -> None:
        """Longer maturity → lower ZCB price."""
        state = model.init_state()
        p1 = float(model.zcb_price(state, 0.0, 1.0))
        p5 = float(model.zcb_price(state, 0.0, 5.0))
        p10 = float(model.zcb_price(state, 0.0, 10.0))
        assert p1 > p5 > p10

    def test_spot_rate_near_flat(self, model: G2PP) -> None:
        """Spot rate should be near the flat forward rate at t=0."""
        state = model.init_state()
        r_val = model.spot_rate(state, 0.0, 5.0)
        assert float(r_val) == pytest.approx(FLAT_RATE, rel=0.05)

    def test_forward_rate_near_flat(self, model: G2PP) -> None:
        """Forward rate should be near the flat market rate at t=0."""
        state = model.init_state()
        f = model.forward_rate(state, 0.0, 5.0)
        assert float(f) == pytest.approx(FLAT_RATE, rel=0.05)

    def test_swap_rate_positive(self, model: G2PP) -> None:
        """Swap rate should be positive."""
        state = model.init_state()
        s_val = model.swap_rate(state, 0.0, 5.0, 0.5)
        assert float(s_val) > 0.0

    def test_swap_rate_near_flat(self, model: G2PP) -> None:
        """With flat curve, swap rate ≈ flat_rate."""
        state = model.init_state()
        s_val = float(model.swap_rate(state, 0.0, 5.0, 1.0))
        assert s_val == pytest.approx(FLAT_RATE, rel=0.1)


class TestG2PPPhi:
    """Tests for the phi shift function."""

    def test_phi_at_zero(self, model: G2PP) -> None:
        """φ(0) = f(0,0) since all exponential terms vanish."""
        phi_0 = float(model._phi(0.0))
        assert phi_0 == pytest.approx(FLAT_RATE, abs=1e-10)

    def test_phi_includes_cross_term(
        self, market_curve: FlatForwardCurve
    ) -> None:
        """φ should differ when ρ≠0 vs ρ=0 for t>0."""
        params_corr = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=0.5,
        )
        params_uncorr = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=0.0,
        )
        m_corr = G2PP(params_corr, market_curve, name="g2pp_c")
        m_uncorr = G2PP(params_uncorr, market_curve, name="g2pp_u")

        phi_corr = float(m_corr._phi(5.0))
        phi_uncorr = float(m_uncorr._phi(5.0))
        assert phi_corr != pytest.approx(phi_uncorr, abs=1e-10)

    def test_phi_cross_term_sign(
        self, market_curve: FlatForwardCurve
    ) -> None:
        """Positive ρ should increase φ relative to ρ=0."""
        params_pos = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=0.5,
        )
        params_zero = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=0.0,
        )
        m_pos = G2PP(params_pos, market_curve, name="g2pp_p")
        m_zero = G2PP(params_zero, market_curve, name="g2pp_z")

        assert float(m_pos._phi(5.0)) > float(m_zero._phi(5.0))


class TestG2PPZeroCorrDegeneration:
    """When ρ=0, G2++ should degenerate to sum of two independent G1++."""

    def test_zcb_rho_zero_vs_independent_factors(
        self, model_uncorrelated: G2PP, market_curve: FlatForwardCurve
    ) -> None:
        """With ρ=0, ZCB should match product of two independent G1++ ZCBs."""
        from hyesg.config.params import OUParams
        from hyesg.models.short_rates.g1pp import G1PP

        g1_params1 = OUParams(
            alpha=ALPHA1, sigma=SIGMA1, model_type="g1pp"
        )
        g1_params2 = OUParams(
            alpha=ALPHA2, sigma=SIGMA2, model_type="g1pp"
        )

        g1_1 = G1PP(g1_params1, market_curve, name="g1pp_1")
        g1_2 = G1PP(g1_params2, market_curve, name="g1pp_2")

        state_g2 = model_uncorrelated.init_state()
        state_g1_1 = g1_1.init_state()
        state_g1_2 = g1_2.init_state()

        for mat in [1.0, 5.0, 10.0]:
            p_g2 = float(model_uncorrelated.zcb_price(state_g2, 0.0, mat))
            p_market = float(jnp.exp(-FLAT_RATE * mat))

            p_g1_1 = float(g1_1.zcb_price(state_g1_1, 0.0, mat))
            p_g1_2 = float(g1_2.zcb_price(state_g1_2, 0.0, mat))

            # Both should match market at t=0
            assert p_g2 == pytest.approx(p_market, rel=1e-6), (
                f"G2++ ρ=0 ZCB mismatch at T={mat}"
            )
            assert p_g1_1 == pytest.approx(p_market, rel=1e-6)
            assert p_g1_2 == pytest.approx(p_market, rel=1e-6)

    def test_v_squared_rho_zero_is_sum(self) -> None:
        """V²(ρ=0) = V₁² + V₂² (no cross-term)."""
        from hyesg.math.gaussian_helpers import variance_integral_ou
        from hyesg.models.short_rates.g2pp import _v_squared_full

        tau = 5.0
        v2_full = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, 0.0, tau)
        )
        v2_1 = float(variance_integral_ou(SIGMA1, ALPHA1, tau))
        v2_2 = float(variance_integral_ou(SIGMA2, ALPHA2, tau))
        assert v2_full == pytest.approx(v2_1 + v2_2, rel=1e-12)


class TestG2PPVSquared:
    """Tests for the full V² calculation."""

    def test_v_squared_at_zero_tau(self) -> None:
        """V²(τ=0) = 0."""
        from hyesg.models.short_rates.g2pp import _v_squared_full

        v2 = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, RHO, 0.0)
        )
        assert v2 == pytest.approx(0.0, abs=1e-12)

    def test_v_squared_positive_for_positive_tau(self) -> None:
        """V² > 0 for τ > 0 with positive vol."""
        from hyesg.models.short_rates.g2pp import _v_squared_full

        v2 = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, RHO, 5.0)
        )
        assert v2 > 0.0

    def test_v_squared_increases_with_positive_rho(self) -> None:
        """Positive ρ increases V² relative to ρ=0."""
        from hyesg.models.short_rates.g2pp import _v_squared_full

        tau = 5.0
        v2_zero = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, 0.0, tau)
        )
        v2_pos = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, 0.5, tau)
        )
        assert v2_pos > v2_zero

    def test_v_squared_decreases_with_negative_rho(self) -> None:
        """Negative ρ decreases V² relative to ρ=0."""
        from hyesg.models.short_rates.g2pp import _v_squared_full

        tau = 5.0
        v2_zero = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, 0.0, tau)
        )
        v2_neg = float(
            _v_squared_full(SIGMA1, ALPHA1, SIGMA2, ALPHA2, -0.5, tau)
        )
        assert v2_neg < v2_zero


class TestG2PPCorrelationEffect:
    """Tests that correlation ρ affects bond prices correctly."""

    def test_zcb_different_for_different_rho(
        self, market_curve: FlatForwardCurve
    ) -> None:
        """ZCB prices at t>0 should differ for different ρ values."""
        from hyesg.core.types import G2State

        params_pos = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=0.5,
        )
        params_neg = G2PPParams(
            alpha1=ALPHA1,
            sigma1=SIGMA1,
            alpha2=ALPHA2,
            sigma2=SIGMA2,
            rho=-0.5,
        )

        m_pos = G2PP(params_pos, market_curve, name="g2pp_rp")
        m_neg = G2PP(params_neg, market_curve, name="g2pp_rn")

        # With non-zero x values, ρ affects ZCB through V² cross-term
        x1 = jnp.array(0.01, dtype=jnp.float64)
        x2 = jnp.array(-0.005, dtype=jnp.float64)
        sr = x1 + x2 + jnp.array(FLAT_RATE)
        state = G2State(x1=x1, x2=x2, short_rate=sr)

        p_pos = float(m_pos.zcb_price(state, 1.0, 6.0))
        p_neg = float(m_neg.zcb_price(state, 1.0, 6.0))

        assert p_pos != pytest.approx(p_neg, rel=1e-6)

    def test_zcb_at_t0_independent_of_rho(
        self, market_curve: FlatForwardCurve
    ) -> None:
        """At t=0, x₁=x₂=0, ZCB should match market regardless of ρ."""
        for rho_val in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            params = G2PPParams(
                alpha1=ALPHA1,
                sigma1=SIGMA1,
                alpha2=ALPHA2,
                sigma2=SIGMA2,
                rho=rho_val,
            )
            m = G2PP(params, market_curve, name=f"g2pp_{rho_val}")
            state = m.init_state()
            for mat in [1.0, 5.0, 10.0]:
                p_model = float(m.zcb_price(state, 0.0, mat))
                p_market = float(jnp.exp(-FLAT_RATE * mat))
                assert p_model == pytest.approx(p_market, rel=1e-6), (
                    f"ρ={rho_val}, T={mat}: model={p_model}, market={p_market}"
                )


class TestG2PPEdgeCases:
    """Edge case tests."""

    def test_zero_volatility(self, market_curve: FlatForwardCurve) -> None:
        """With σ₁=σ₂=0, model is deterministic."""
        params = G2PPParams(
            alpha1=ALPHA1,
            sigma1=0.0,
            alpha2=ALPHA2,
            sigma2=0.0,
            rho=0.0,
        )
        m = G2PP(params, market_curve, name="g2pp_zv")
        state = m.init_state()

        # Step with any shocks should not change x values
        shocks = jnp.array([5.0, 5.0])
        new_state, _ = m.step(state, 0.0, 0.25, shocks, {})
        assert float(new_state.x1) == pytest.approx(0.0, abs=1e-12)
        assert float(new_state.x2) == pytest.approx(0.0, abs=1e-12)

    def test_small_dt(self, model: G2PP) -> None:
        """Small dt should work without numerical issues."""
        state = model.init_state()
        shocks = jnp.array([0.5, -0.3])
        new_state, _ = model.step(state, 0.0, 1e-6, shocks, {})
        assert jnp.isfinite(new_state.short_rate)

    def test_large_maturity(self, model: G2PP) -> None:
        """Large maturities should produce positive, finite ZCBs."""
        state = model.init_state()
        p = model.zcb_price(state, 0.0, 50.0)
        assert jnp.isfinite(p)
        assert float(p) > 0.0

    def test_extreme_rho(self, market_curve: FlatForwardCurve) -> None:
        """ρ=±1 should still produce valid results."""
        for rho_val in [-1.0, 1.0]:
            params = G2PPParams(
                alpha1=ALPHA1,
                sigma1=SIGMA1,
                alpha2=ALPHA2,
                sigma2=SIGMA2,
                rho=rho_val,
            )
            m = G2PP(params, market_curve, name=f"g2pp_r{rho_val}")
            state = m.init_state()
            p = m.zcb_price(state, 0.0, 5.0)
            assert jnp.isfinite(p)
            assert float(p) > 0.0
