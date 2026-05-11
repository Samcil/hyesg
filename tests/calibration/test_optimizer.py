"""Tests for calibration optimizers (LevenbergMarquardt, ScipyMinimize)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.calibration.optimizer import (
    LevenbergMarquardt,
    LevenbergMarquardtConfig,
    ScipyMinimize,
)
from hyesg.calibration.protocols import Optimizer
from hyesg.calibration.result import OptimizationResult

jax.config.update("jax_enable_x64", True)


# ── Helpers / fixtures ──────────────────────────────────────────────


def rosenbrock_residuals(params: jax.Array, **_kwargs) -> jax.Array:
    """Rosenbrock function as a sum of squared residuals.

    f(x,y) = (1-x)^2 + 100*(y - x^2)^2
    Residuals: r1 = 1 - x, r2 = 10*(y - x^2)
    """
    x, y = params[0], params[1]
    return jnp.array([1.0 - x, 10.0 * (y - x**2)])


def quadratic_residuals(params: jax.Array, **_kwargs) -> jax.Array:
    """Simple quadratic residuals: r_i = params_i - target_i."""
    target = jnp.array([1.0, 2.0, 3.0])
    return params - target


def linear_residuals(
    params: jax.Array, *, target: jax.Array, **_kwargs
) -> jax.Array:
    """Linear residuals: r_i = a * x_i + b - y_i."""
    a, b = params[0], params[1]
    x = jnp.arange(len(target), dtype=jnp.float64)
    return a * x + b - target


def exponential_residuals(
    params: jax.Array, *, target: jax.Array, x_data: jax.Array, **_kwargs
) -> jax.Array:
    """y = a * exp(b * x) residuals."""
    a, b = params[0], params[1]
    return a * jnp.exp(b * x_data) - target


@pytest.fixture()
def lm_optimizer() -> LevenbergMarquardt:
    return LevenbergMarquardt()


@pytest.fixture()
def lm_tight() -> LevenbergMarquardt:
    return LevenbergMarquardt(
        LevenbergMarquardtConfig(max_iterations=500, tol_grad=1e-14, tol_param=1e-14)
    )


@pytest.fixture()
def scipy_optimizer() -> ScipyMinimize:
    return ScipyMinimize()


# ── LevenbergMarquardtConfig tests ─────────────────────────────────


class TestLevenbergMarquardtConfig:
    def test_default_values(self):
        cfg = LevenbergMarquardtConfig()
        assert cfg.max_iterations == 200
        assert cfg.tol_grad == 1e-10
        assert cfg.damping_init == 1e-3
        assert cfg.damping_increase == 10.0
        assert cfg.damping_decrease == 0.1

    def test_custom_values(self):
        cfg = LevenbergMarquardtConfig(
            max_iterations=50, tol_grad=1e-6, damping_init=0.1
        )
        assert cfg.max_iterations == 50
        assert cfg.tol_grad == 1e-6
        assert cfg.damping_init == 0.1

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(LevenbergMarquardtConfig())


# ── LevenbergMarquardt tests ───────────────────────────────────────


class TestLevenbergMarquardt:
    """Test LM optimizer convergence on standard problems."""

    def test_protocol_compliance(self):
        assert isinstance(LevenbergMarquardt(), Optimizer)

    def test_returns_optimization_result(self, lm_optimizer):
        x0 = jnp.array([0.5, 0.5])
        result = lm_optimizer.minimize(rosenbrock_residuals, x0)
        assert isinstance(result, OptimizationResult)

    def test_rosenbrock_convergence(self, lm_tight):
        x0 = jnp.array([-1.0, 1.0])
        result = lm_tight.minimize(rosenbrock_residuals, x0)
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([1.0, 1.0]), atol=1e-4)

    def test_rosenbrock_objective_near_zero(self, lm_tight):
        x0 = jnp.array([0.0, 0.0])
        result = lm_tight.minimize(rosenbrock_residuals, x0)
        assert result.objective_value < 1e-8

    def test_quadratic_exact(self, lm_optimizer):
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = lm_optimizer.minimize(quadratic_residuals, x0)
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([1.0, 2.0, 3.0]), atol=1e-6)

    def test_linear_fit(self, lm_optimizer):
        target = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=jnp.float64)
        x0 = jnp.array([1.0, 0.0])
        result = lm_optimizer.minimize(linear_residuals, x0, target=target)
        # y = 2x + 1
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([2.0, 1.0]), atol=1e-5)

    def test_exponential_fit(self, lm_tight):
        # y = 2 * exp(0.5 * x)
        x_data = jnp.linspace(0.0, 2.0, 10)
        target = 2.0 * jnp.exp(0.5 * x_data)
        x0 = jnp.array([1.0, 0.1])
        result = lm_tight.minimize(
            exponential_residuals, x0, target=target, x_data=x_data
        )
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([2.0, 0.5]), atol=1e-4)

    def test_already_at_minimum(self, lm_optimizer):
        x0 = jnp.array([1.0, 2.0, 3.0])
        result = lm_optimizer.minimize(quadratic_residuals, x0)
        assert result.converged
        assert result.objective_value < 1e-14
        assert result.n_iterations <= 2  # should converge immediately

    def test_residuals_stored(self, lm_optimizer):
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = lm_optimizer.minimize(quadratic_residuals, x0)
        assert result.residuals is not None
        assert result.residuals.shape == (3,)

    def test_n_iterations_positive(self, lm_optimizer):
        x0 = jnp.array([0.0, 0.0])
        result = lm_optimizer.minimize(rosenbrock_residuals, x0)
        assert result.n_iterations > 0

    def test_custom_config(self):
        cfg = LevenbergMarquardtConfig(max_iterations=5, tol_grad=1e-4)
        opt = LevenbergMarquardt(cfg)
        x0 = jnp.array([-1.0, 1.0])
        result = opt.minimize(rosenbrock_residuals, x0)
        assert result.n_iterations <= 5

    def test_max_iter_respected(self):
        cfg = LevenbergMarquardtConfig(max_iterations=3)
        opt = LevenbergMarquardt(cfg)
        x0 = jnp.array([-5.0, 5.0])
        result = opt.minimize(rosenbrock_residuals, x0)
        assert result.n_iterations <= 3


# ── ScipyMinimize tests ────────────────────────────────────────────


class TestScipyMinimize:
    """Test SciPy optimizer wrapper."""

    def test_protocol_compliance(self):
        assert isinstance(ScipyMinimize(), Optimizer)

    def test_returns_optimization_result(self, scipy_optimizer):
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = scipy_optimizer.minimize(quadratic_residuals, x0)
        assert isinstance(result, OptimizationResult)

    def test_quadratic_convergence(self, scipy_optimizer):
        x0 = jnp.array([10.0, -5.0, 0.0])
        result = scipy_optimizer.minimize(quadratic_residuals, x0)
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([1.0, 2.0, 3.0]), atol=1e-3)

    def test_linear_fit(self, scipy_optimizer):
        target = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=jnp.float64)
        x0 = jnp.array([1.0, 0.0])
        result = scipy_optimizer.minimize(linear_residuals, x0, target=target)
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([2.0, 1.0]), atol=1e-3)

    def test_rosenbrock_convergence(self, scipy_optimizer):
        x0 = jnp.array([0.0, 0.0])
        result = scipy_optimizer.minimize(rosenbrock_residuals, x0)
        assert result.converged
        assert jnp.allclose(result.params, jnp.array([1.0, 1.0]), atol=1e-2)

    def test_custom_method(self):
        opt = ScipyMinimize(method="Nelder-Mead")
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = opt.minimize(quadratic_residuals, x0)
        assert jnp.allclose(result.params, jnp.array([1.0, 2.0, 3.0]), atol=1e-2)

    def test_custom_options(self):
        opt = ScipyMinimize(options={"maxiter": 5})
        x0 = jnp.array([-5.0, 5.0])
        result = opt.minimize(rosenbrock_residuals, x0)
        # May not converge with only 5 iterations
        assert isinstance(result, OptimizationResult)

    def test_n_iterations_nonneg(self, scipy_optimizer):
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = scipy_optimizer.minimize(quadratic_residuals, x0)
        assert result.n_iterations >= 0

    def test_objective_value_nonneg(self, scipy_optimizer):
        x0 = jnp.array([0.0, 0.0, 0.0])
        result = scipy_optimizer.minimize(quadratic_residuals, x0)
        assert result.objective_value >= 0.0


# ── Cross-optimizer consistency ────────────────────────────────────


class TestOptimizerConsistency:
    """Both optimizers should agree on simple problems."""

    def test_quadratic_agree(self, lm_optimizer, scipy_optimizer):
        x0 = jnp.array([0.0, 0.0, 0.0])
        lm = lm_optimizer.minimize(quadratic_residuals, x0)
        sp = scipy_optimizer.minimize(quadratic_residuals, x0)
        assert jnp.allclose(lm.params, sp.params, atol=1e-3)

    def test_linear_agree(self, lm_optimizer, scipy_optimizer):
        target = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=jnp.float64)
        x0 = jnp.array([1.0, 0.0])
        lm = lm_optimizer.minimize(linear_residuals, x0, target=target)
        sp = scipy_optimizer.minimize(linear_residuals, x0, target=target)
        assert jnp.allclose(lm.params, sp.params, atol=1e-3)
