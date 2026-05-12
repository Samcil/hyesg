"""Tests for RobustLevenbergMarquardt optimizer."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from jax import Array

from hyesg.calibration.optimizer import (
    RobustLevenbergMarquardt,
    RobustLevenbergMarquardtConfig,
)
from hyesg.calibration.protocols import Optimizer
from hyesg.calibration.result import OptimizationResult


# ── Helper functions ───────────────────────────────────────────────


def rosenbrock_residuals(x: Array) -> Array:
    """Rosenbrock function as a 2-element residual vector."""
    return jnp.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def quadratic_residuals(x: Array) -> Array:
    """Simple quadratic: minimum at x=[1, 2]."""
    return jnp.array([x[0] - 1.0, x[1] - 2.0])


def sphere_residuals(x: Array) -> Array:
    """Sphere function: minimum at origin."""
    return x


# ── Protocol conformance ──────────────────────────────────────────


class TestRobustLMProtocol:
    def test_implements_optimizer(self) -> None:
        opt = RobustLevenbergMarquardt()
        assert isinstance(opt, Optimizer)


# ── Config ─────────────────────────────────────────────────────────


class TestRobustLMConfig:
    def test_defaults(self) -> None:
        cfg = RobustLevenbergMarquardtConfig()
        assert cfg.max_iter == 200
        assert cfg.heuristic_trials == 50
        assert cfg.heuristic_seed == 1

    def test_custom_config(self) -> None:
        cfg = RobustLevenbergMarquardtConfig(max_iter=10, heuristic_trials=5)
        opt = RobustLevenbergMarquardt(config=cfg)
        assert opt.config.max_iter == 10
        assert opt.config.heuristic_trials == 5


# ── Unconstrained optimisation ─────────────────────────────────────


class TestRobustLMUnconstrained:
    def test_simple_quadratic(self) -> None:
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=10)
        )
        x0 = jnp.array([5.0, 5.0])
        result = opt.minimize(quadratic_residuals, x0)
        assert isinstance(result, OptimizationResult)
        assert result.params[0] == pytest.approx(1.0, abs=1e-4)
        assert result.params[1] == pytest.approx(2.0, abs=1e-4)

    def test_sphere(self) -> None:
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=10)
        )
        x0 = jnp.array([3.0, -4.0, 2.0])
        result = opt.minimize(sphere_residuals, x0)
        assert jnp.allclose(result.params, 0.0, atol=1e-4)
        assert result.converged

    def test_rosenbrock(self) -> None:
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=30, max_iter=500)
        )
        x0 = jnp.array([-1.0, 1.0])
        result = opt.minimize(rosenbrock_residuals, x0)
        assert result.params[0] == pytest.approx(1.0, abs=1e-3)
        assert result.params[1] == pytest.approx(1.0, abs=1e-3)


# ── Box-constrained optimisation ───────────────────────────────────


class TestRobustLMBoxConstraints:
    def test_bounds_respected(self) -> None:
        """Minimum is at [1,2] but upper bound restricts x[1] ≤ 1.5."""
        lower = jnp.array([-10.0, -10.0])
        upper = jnp.array([10.0, 1.5])
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=20)
        )
        x0 = jnp.array([5.0, 0.0])
        result = opt.minimize(quadratic_residuals, x0, bounds=(lower, upper))
        assert result.params[0] == pytest.approx(1.0, abs=1e-3)
        # x[1] should be clamped to upper bound
        assert float(result.params[1]) <= 1.5 + 1e-6

    def test_tight_bounds(self) -> None:
        """Tight bounds force solution near centre of box."""
        lower = jnp.array([0.9, 1.9])
        upper = jnp.array([1.1, 2.1])
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=10)
        )
        x0 = jnp.array([1.0, 2.0])
        result = opt.minimize(quadratic_residuals, x0, bounds=(lower, upper))
        assert 0.9 - 1e-6 <= float(result.params[0]) <= 1.1 + 1e-6
        assert 1.9 - 1e-6 <= float(result.params[1]) <= 2.1 + 1e-6


# ── Heuristic warm-start ──────────────────────────────────────────


class TestHeuristic:
    def test_heuristic_improves(self) -> None:
        """Heuristic should find a point better than a bad x0."""
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=100)
        )
        x0 = jnp.array([100.0, 100.0])
        x_heur = opt._basic_heuristic(quadratic_residuals, x0, bounds=None)
        cost_x0 = float(0.5 * jnp.sum(quadratic_residuals(x0) ** 2))
        cost_heur = float(0.5 * jnp.sum(quadratic_residuals(x_heur) ** 2))
        assert cost_heur < cost_x0

    def test_heuristic_reproducible(self) -> None:
        """Same seed → same result."""
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_seed=42, heuristic_trials=20)
        )
        x0 = jnp.array([5.0, 5.0])
        r1 = opt._basic_heuristic(quadratic_residuals, x0, bounds=None)
        r2 = opt._basic_heuristic(quadratic_residuals, x0, bounds=None)
        assert jnp.allclose(r1, r2)


# ── Jacobian ───────────────────────────────────────────────────────


class TestJacobian:
    def test_central_diff_accuracy(self) -> None:
        """Jacobian of quadratic_residuals is the identity matrix."""
        x = jnp.array([3.0, 7.0])
        J = RobustLevenbergMarquardt._central_differences_jacobian(
            quadratic_residuals, x
        )
        expected = jnp.eye(2)
        assert jnp.allclose(J, expected, atol=1e-5)

    def test_rosenbrock_jacobian_shape(self) -> None:
        x = jnp.array([1.0, 1.0])
        J = RobustLevenbergMarquardt._central_differences_jacobian(
            rosenbrock_residuals, x
        )
        assert J.shape == (2, 2)


# ── Fallback behaviour ────────────────────────────────────────────


class TestFallback:
    def test_lm_divergence_returns_heuristic(self) -> None:
        """If LM is set to 0 iterations, heuristic result is returned."""
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(
                max_iter=0, heuristic_trials=50, heuristic_seed=1
            )
        )
        x0 = jnp.array([10.0, 10.0])
        result = opt.minimize(quadratic_residuals, x0)
        # With 0 LM iterations, heuristic result should still be decent
        assert isinstance(result, OptimizationResult)


# ── Result fields ──────────────────────────────────────────────────


class TestResultFields:
    def test_result_has_residuals(self) -> None:
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=5)
        )
        result = opt.minimize(quadratic_residuals, jnp.array([5.0, 5.0]))
        assert result.residuals is not None
        assert result.residuals.shape == (2,)

    def test_result_has_gradient_norm(self) -> None:
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=5)
        )
        result = opt.minimize(quadratic_residuals, jnp.array([5.0, 5.0]))
        assert result.gradient_norm is not None

    def test_result_objective_value(self) -> None:
        opt = RobustLevenbergMarquardt(
            RobustLevenbergMarquardtConfig(heuristic_trials=5)
        )
        result = opt.minimize(quadratic_residuals, jnp.array([5.0, 5.0]))
        assert result.objective_value >= 0.0
        assert result.objective_value < 1e-4
