"""Tests for calibration objective functions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.calibration.objectives import (
    cir_curve_objective,
    cir_curve_objective_direct,
    credit_spread_objective,
    credit_spread_objective_direct,
    credit_spread_from_survival,
    credit_survival_probability,
    ou_curve_objective,
    ou_curve_objective_direct,
    ou_zcb_price,
)
from hyesg.math.cir_formulas import cir_zcb_price

jax.config.update("jax_enable_x64", True)


# ── Helpers / fixtures ──────────────────────────────────────────────


@pytest.fixture()
def tenors() -> jax.Array:
    return jnp.array([1.0, 2.0, 5.0, 10.0, 20.0], dtype=jnp.float64)


@pytest.fixture()
def cir_params() -> dict:
    return {"alpha": 0.5, "mu": 0.04, "sigma": 0.1}


@pytest.fixture()
def ou_params() -> dict:
    return {"alpha": 0.3, "mu": 0.03, "sigma": 0.01}


@pytest.fixture()
def credit_params() -> dict:
    return {"alpha": 0.5, "mu": 0.02, "sigma": 0.05, "lambda0": 0.01}


# ── CIR objective tests ────────────────────────────────────────────


class TestCIRObjective:
    """Test CIR curve fitting objective."""

    def test_zero_residuals_at_true_params(self, tenors, cir_params):
        """When we generate prices from CIR and fit CIR, residuals ~ 0."""
        alpha, mu, sigma = cir_params["alpha"], cir_params["mu"], cir_params["sigma"]
        target_prices = cir_zcb_price(tenors, mu, alpha, mu, sigma)

        params = jnp.array([alpha, mu, sigma], dtype=jnp.float64)
        residuals = cir_curve_objective_direct(
            params, target_prices=target_prices, tenors=tenors
        )
        assert jnp.allclose(residuals, 0.0, atol=1e-10)

    def test_softplus_at_true_params(self, tenors, cir_params):
        """Softplus-transformed variant produces zero at softplus-inverse of true."""
        alpha, mu, sigma = cir_params["alpha"], cir_params["mu"], cir_params["sigma"]
        target_prices = cir_zcb_price(tenors, mu, alpha, mu, sigma)

        # Inverse softplus: log(exp(x) - 1)
        inv_sp = lambda x: jnp.log(jnp.exp(jnp.asarray(x, dtype=jnp.float64)) - 1.0)
        raw = jnp.array([inv_sp(alpha), inv_sp(mu), inv_sp(sigma)])
        residuals = cir_curve_objective(
            raw, target_prices=target_prices, tenors=tenors
        )
        assert jnp.allclose(residuals, 0.0, atol=1e-8)

    def test_nonzero_residuals_wrong_params(self, tenors, cir_params):
        alpha, mu, sigma = cir_params["alpha"], cir_params["mu"], cir_params["sigma"]
        target_prices = cir_zcb_price(tenors, mu, alpha, mu, sigma)

        wrong_params = jnp.array([0.1, 0.08, 0.2], dtype=jnp.float64)
        residuals = cir_curve_objective_direct(
            wrong_params, target_prices=target_prices, tenors=tenors
        )
        assert jnp.any(jnp.abs(residuals) > 1e-4)

    def test_residuals_shape(self, tenors, cir_params):
        alpha, mu, sigma = cir_params["alpha"], cir_params["mu"], cir_params["sigma"]
        target_prices = cir_zcb_price(tenors, mu, alpha, mu, sigma)

        params = jnp.array([alpha, mu, sigma], dtype=jnp.float64)
        residuals = cir_curve_objective_direct(
            params, target_prices=target_prices, tenors=tenors
        )
        assert residuals.shape == tenors.shape

    def test_differentiable(self, tenors, cir_params):
        """Objective is differentiable via JAX grad."""
        alpha, mu, sigma = cir_params["alpha"], cir_params["mu"], cir_params["sigma"]
        target_prices = cir_zcb_price(tenors, mu, alpha, mu, sigma)

        def scalar_obj(params):
            r = cir_curve_objective_direct(
                params, target_prices=target_prices, tenors=tenors
            )
            return jnp.sum(r**2)

        params = jnp.array([alpha, mu, sigma], dtype=jnp.float64)
        grad = jax.grad(scalar_obj)(params)
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_direct_positivity_enforced(self, tenors):
        """Direct variant does not enforce positivity — caller responsibility."""
        target_prices = jnp.ones(5) * 0.95
        params = jnp.array([0.5, 0.04, 0.1], dtype=jnp.float64)
        residuals = cir_curve_objective_direct(
            params, target_prices=target_prices, tenors=tenors
        )
        assert jnp.all(jnp.isfinite(residuals))


# ── OU objective tests ──────────────────────────────────────────────


class TestOUObjective:
    """Test OU/Vasicek curve fitting objective."""

    def test_zero_residuals_at_true_params(self, tenors, ou_params):
        alpha, mu, sigma = ou_params["alpha"], ou_params["mu"], ou_params["sigma"]
        target_prices = ou_zcb_price(
            tenors,
            jnp.asarray(mu),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(sigma),
        )

        params = jnp.array([alpha, mu, sigma], dtype=jnp.float64)
        residuals = ou_curve_objective_direct(
            params, target_prices=target_prices, tenors=tenors
        )
        assert jnp.allclose(residuals, 0.0, atol=1e-10)

    def test_softplus_at_true_params(self, tenors, ou_params):
        alpha, mu, sigma = ou_params["alpha"], ou_params["mu"], ou_params["sigma"]
        target_prices = ou_zcb_price(
            tenors,
            jnp.asarray(mu),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(sigma),
        )

        # OU softplus only on alpha and sigma (not mu)
        inv_sp = lambda x: jnp.log(jnp.exp(jnp.asarray(x, dtype=jnp.float64)) - 1.0)
        raw = jnp.array([inv_sp(alpha), mu, inv_sp(sigma)])
        residuals = ou_curve_objective(
            raw, target_prices=target_prices, tenors=tenors
        )
        assert jnp.allclose(residuals, 0.0, atol=1e-8)

    def test_residuals_shape(self, tenors, ou_params):
        alpha, mu, sigma = ou_params["alpha"], ou_params["mu"], ou_params["sigma"]
        target_prices = ou_zcb_price(
            tenors,
            jnp.asarray(mu),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(sigma),
        )

        params = jnp.array([alpha, mu, sigma], dtype=jnp.float64)
        residuals = ou_curve_objective_direct(
            params, target_prices=target_prices, tenors=tenors
        )
        assert residuals.shape == tenors.shape

    def test_nonzero_residuals_wrong_params(self, tenors, ou_params):
        alpha, mu, sigma = ou_params["alpha"], ou_params["mu"], ou_params["sigma"]
        target_prices = ou_zcb_price(
            tenors,
            jnp.asarray(mu),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(sigma),
        )
        wrong = jnp.array([1.0, 0.1, 0.05], dtype=jnp.float64)
        residuals = ou_curve_objective_direct(
            wrong, target_prices=target_prices, tenors=tenors
        )
        assert jnp.any(jnp.abs(residuals) > 1e-4)

    def test_differentiable(self, tenors, ou_params):
        alpha, mu, sigma = ou_params["alpha"], ou_params["mu"], ou_params["sigma"]
        target_prices = ou_zcb_price(
            tenors,
            jnp.asarray(mu),
            jnp.asarray(alpha),
            jnp.asarray(mu),
            jnp.asarray(sigma),
        )

        def scalar_obj(params):
            r = ou_curve_objective_direct(
                params, target_prices=target_prices, tenors=tenors
            )
            return jnp.sum(r**2)

        params = jnp.array([alpha, mu, sigma], dtype=jnp.float64)
        grad = jax.grad(scalar_obj)(params)
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))


# ── OU ZCB pricing tests ───────────────────────────────────────────


class TestOUZCBPrice:
    def test_unit_price_at_zero_maturity(self):
        """P(0,0) = 1."""
        t = jnp.array([0.0])
        p = ou_zcb_price(t, jnp.asarray(0.05), jnp.asarray(0.5), jnp.asarray(0.05), jnp.asarray(0.01))
        assert jnp.allclose(p, 1.0, atol=1e-10)

    def test_decreasing_with_maturity(self):
        """Prices decrease with maturity for positive rates."""
        tenors = jnp.array([1.0, 5.0, 10.0, 20.0])
        prices = ou_zcb_price(
            tenors,
            jnp.asarray(0.05),
            jnp.asarray(0.5),
            jnp.asarray(0.05),
            jnp.asarray(0.01),
        )
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1]

    def test_prices_in_01(self):
        tenors = jnp.array([0.5, 1.0, 2.0, 5.0, 10.0])
        prices = ou_zcb_price(
            tenors,
            jnp.asarray(0.03),
            jnp.asarray(0.3),
            jnp.asarray(0.03),
            jnp.asarray(0.01),
        )
        assert jnp.all(prices > 0.0)
        assert jnp.all(prices <= 1.0)


# ── Credit objective tests ─────────────────────────────────────────


class TestCreditObjective:
    """Test credit spread fitting objective."""

    def test_survival_probability_one_at_zero(self):
        """Q(0) = 1."""
        t = jnp.array([0.0])
        q = credit_survival_probability(t, 0.5, 0.02, 0.05, 0.01)
        assert jnp.allclose(q, 1.0, atol=1e-10)

    def test_survival_probability_decreasing(self, tenors):
        q = credit_survival_probability(tenors, 0.5, 0.02, 0.05, 0.01)
        for i in range(len(q) - 1):
            assert q[i] >= q[i + 1]

    def test_survival_probability_in_01(self, tenors):
        q = credit_survival_probability(tenors, 0.5, 0.02, 0.05, 0.01)
        assert jnp.all(q >= 0.0)
        assert jnp.all(q <= 1.0)

    def test_spread_positive(self, tenors):
        q = credit_survival_probability(tenors, 0.5, 0.02, 0.05, 0.01)
        # Avoid T=0
        pos_tenors = tenors[tenors > 0.0]
        q_pos = q[tenors > 0.0]
        s = credit_spread_from_survival(pos_tenors, q_pos, 0.4)
        assert jnp.all(s > 0.0)

    def test_zero_residuals_at_true_params(self, tenors, credit_params):
        alpha = credit_params["alpha"]
        mu = credit_params["mu"]
        sigma = credit_params["sigma"]
        lambda0 = credit_params["lambda0"]
        recovery = 0.4

        q = credit_survival_probability(tenors, alpha, mu, sigma, lambda0)
        target_spreads = credit_spread_from_survival(tenors, q, recovery)

        params = jnp.array([alpha, mu, sigma, lambda0], dtype=jnp.float64)
        residuals = credit_spread_objective_direct(
            params,
            target_spreads=target_spreads,
            tenors=tenors,
            recovery_rate=recovery,
        )
        assert jnp.allclose(residuals, 0.0, atol=1e-10)

    def test_softplus_at_true_params(self, tenors, credit_params):
        alpha = credit_params["alpha"]
        mu = credit_params["mu"]
        sigma = credit_params["sigma"]
        lambda0 = credit_params["lambda0"]
        recovery = 0.4

        q = credit_survival_probability(tenors, alpha, mu, sigma, lambda0)
        target_spreads = credit_spread_from_survival(tenors, q, recovery)

        inv_sp = lambda x: jnp.log(jnp.exp(jnp.asarray(x, dtype=jnp.float64)) - 1.0)
        raw = jnp.array([inv_sp(alpha), inv_sp(mu), inv_sp(sigma), inv_sp(lambda0)])
        residuals = credit_spread_objective(
            raw,
            target_spreads=target_spreads,
            tenors=tenors,
            recovery_rate=recovery,
        )
        assert jnp.allclose(residuals, 0.0, atol=1e-8)

    def test_residuals_shape(self, tenors, credit_params):
        alpha = credit_params["alpha"]
        mu = credit_params["mu"]
        sigma = credit_params["sigma"]
        lambda0 = credit_params["lambda0"]

        q = credit_survival_probability(tenors, alpha, mu, sigma, lambda0)
        target_spreads = credit_spread_from_survival(tenors, q, 0.4)

        params = jnp.array([alpha, mu, sigma, lambda0], dtype=jnp.float64)
        residuals = credit_spread_objective_direct(
            params,
            target_spreads=target_spreads,
            tenors=tenors,
            recovery_rate=0.4,
        )
        assert residuals.shape == tenors.shape

    def test_nonzero_residuals_wrong_params(self, tenors, credit_params):
        alpha = credit_params["alpha"]
        mu = credit_params["mu"]
        sigma = credit_params["sigma"]
        lambda0 = credit_params["lambda0"]

        q = credit_survival_probability(tenors, alpha, mu, sigma, lambda0)
        target_spreads = credit_spread_from_survival(tenors, q, 0.4)

        wrong = jnp.array([1.0, 0.1, 0.2, 0.05], dtype=jnp.float64)
        residuals = credit_spread_objective_direct(
            wrong,
            target_spreads=target_spreads,
            tenors=tenors,
            recovery_rate=0.4,
        )
        assert jnp.any(jnp.abs(residuals) > 1e-4)

    def test_differentiable(self, tenors, credit_params):
        alpha = credit_params["alpha"]
        mu = credit_params["mu"]
        sigma = credit_params["sigma"]
        lambda0 = credit_params["lambda0"]

        q = credit_survival_probability(tenors, alpha, mu, sigma, lambda0)
        target_spreads = credit_spread_from_survival(tenors, q, 0.4)

        def scalar_obj(params):
            r = credit_spread_objective_direct(
                params,
                target_spreads=target_spreads,
                tenors=tenors,
                recovery_rate=0.4,
            )
            return jnp.sum(r**2)

        params = jnp.array([alpha, mu, sigma, lambda0], dtype=jnp.float64)
        grad = jax.grad(scalar_obj)(params)
        assert grad.shape == (4,)
        assert jnp.all(jnp.isfinite(grad))

    def test_higher_recovery_lower_spread(self, tenors):
        """Higher recovery → lower credit spread."""
        q = credit_survival_probability(tenors, 0.5, 0.02, 0.05, 0.01)
        s_low = credit_spread_from_survival(tenors, q, 0.2)
        s_high = credit_spread_from_survival(tenors, q, 0.6)
        assert jnp.all(s_low > s_high)
