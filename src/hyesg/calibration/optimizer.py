"""Optimizer implementations for calibration.

Provides Levenberg-Marquardt (pure JAX) and SciPy-based optimizers
for fitting model parameters to market data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from hyesg.calibration.result import OptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class LevenbergMarquardtConfig:
    """Configuration for Levenberg-Marquardt optimizer.

    Attributes:
        max_iterations: Maximum number of LM iterations.
        tol_grad: Convergence tolerance on gradient norm.
        tol_param: Convergence tolerance on parameter change.
        tol_residual: Convergence tolerance on residual change.
        damping_init: Initial damping factor (λ).
        damping_increase: Factor to increase damping on rejection.
        damping_decrease: Factor to decrease damping on acceptance.
        min_damping: Minimum damping value.
        max_damping: Maximum damping value.
    """

    max_iterations: int = 200
    tol_grad: float = 1e-10
    tol_param: float = 1e-10
    tol_residual: float = 1e-10
    damping_init: float = 1e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.1
    min_damping: float = 1e-15
    max_damping: float = 1e10


class LevenbergMarquardt:
    """Levenberg-Marquardt optimizer using pure JAX.

    Implements Gauss-Newton with adaptive damping for nonlinear
    least-squares problems. Computes the Jacobian via ``jax.jacfwd``.

    The objective function must return a residual vector (not a scalar).
    The optimizer minimizes 0.5 * ||r(p)||^2.
    """

    def __init__(self, config: LevenbergMarquardtConfig | None = None) -> None:
        self.config = config or LevenbergMarquardtConfig()

    def minimize(
        self,
        fn: Any,
        x0: Array,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Minimize a residual function using Levenberg-Marquardt.

        Args:
            fn: Residual function r(params) -> Array. The optimizer
                minimizes 0.5 * ||r(params)||^2.
            x0: Initial parameter guess.
            **kwargs: Passed through to the residual function.

        Returns:
            Optimization result with fitted parameters.
        """
        cfg = self.config
        params = jnp.asarray(x0, dtype=jnp.float64)
        damping = jnp.asarray(cfg.damping_init, dtype=jnp.float64)

        residuals = fn(params, **kwargs)
        cost = jnp.asarray(0.5) * jnp.sum(residuals**2)

        converged = False
        n_iter = 0

        for i in range(cfg.max_iterations):
            n_iter = i + 1

            # Jacobian: J[i, j] = d r_i / d p_j
            J = jax.jacfwd(lambda p: fn(p, **kwargs))(params)

            # Gauss-Newton: (J^T J + λ I) δ = -J^T r
            JtJ = J.T @ J
            Jtr = J.T @ residuals
            grad_norm = jnp.max(jnp.abs(Jtr))

            if grad_norm < cfg.tol_grad:
                converged = True
                break

            # Solve with damping
            n_params = params.shape[0]
            A = JtJ + damping * jnp.eye(n_params)
            delta = jnp.linalg.solve(A, -Jtr)

            # Evaluate candidate
            params_new = params + delta
            residuals_new = fn(params_new, **kwargs)
            cost_new = jnp.asarray(0.5) * jnp.sum(residuals_new**2)

            # Gain ratio: actual reduction / predicted reduction
            predicted = -jnp.dot(delta, Jtr) - 0.5 * jnp.dot(delta, JtJ @ delta)
            actual = cost - cost_new
            rho = jnp.where(
                jnp.abs(predicted) > 1e-30,
                actual / predicted,
                jnp.asarray(0.0),
            )

            if rho > 0.0:
                # Accept step
                params = params_new
                residuals = residuals_new
                cost = cost_new
                damping = jnp.maximum(
                    damping * jnp.asarray(cfg.damping_decrease),
                    jnp.asarray(cfg.min_damping),
                )

                # Check parameter convergence
                param_change = jnp.max(jnp.abs(delta)) / jnp.maximum(
                    jnp.max(jnp.abs(params)), jnp.asarray(1.0)
                )
                if param_change < cfg.tol_param:
                    converged = True
                    break
            else:
                # Reject step, increase damping
                damping = jnp.minimum(
                    damping * jnp.asarray(cfg.damping_increase),
                    jnp.asarray(cfg.max_damping),
                )

        return OptimizationResult(
            params=params,
            objective_value=float(cost),
            n_iterations=n_iter,
            converged=converged,
            residuals=residuals,
            gradient_norm=float(grad_norm) if n_iter > 0 else None,
        )


class ScipyMinimize:
    """Optimizer wrapping ``scipy.optimize.minimize``.

    Not JIT-compatible but useful for comparison and validation.
    Supports any SciPy method (L-BFGS-B, Nelder-Mead, etc.).
    """

    def __init__(
        self,
        method: str = "L-BFGS-B",
        options: dict[str, Any] | None = None,
    ) -> None:
        self.method = method
        self.options = options or {"maxiter": 500, "ftol": 1e-12, "gtol": 1e-10}

    def minimize(
        self,
        fn: Any,
        x0: Array,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Minimize using SciPy.

        The objective function should return a scalar loss value,
        or a residual vector (which is converted to 0.5 * ||r||^2).

        Args:
            fn: Objective or residual function.
            x0: Initial parameter guess.
            **kwargs: Passed through to the objective function.

        Returns:
            Optimization result with fitted parameters.
        """
        from scipy.optimize import minimize as sp_minimize

        import numpy as np

        x0_np = np.asarray(x0, dtype=np.float64)

        def scalar_fn(p: np.ndarray) -> float:
            p_jax = jnp.asarray(p, dtype=jnp.float64)
            result = fn(p_jax, **kwargs)
            if result.ndim > 0:
                return float(0.5 * jnp.sum(result**2))
            return float(result)

        # Use JAX grad for gradient if available
        def grad_fn(p: np.ndarray) -> np.ndarray:
            p_jax = jnp.asarray(p, dtype=jnp.float64)

            def loss(pp: Array) -> Array:
                result = fn(pp, **kwargs)
                if result.ndim > 0:
                    return jnp.asarray(0.5) * jnp.sum(result**2)
                return result

            g = jax.grad(loss)(p_jax)
            return np.asarray(g, dtype=np.float64)

        result = sp_minimize(
            scalar_fn,
            x0_np,
            jac=grad_fn,
            method=self.method,
            options=self.options,
        )

        params_jax = jnp.asarray(result.x, dtype=jnp.float64)
        residuals = fn(params_jax, **kwargs)

        return OptimizationResult(
            params=params_jax,
            objective_value=float(result.fun),
            n_iterations=int(result.get("nit", 0)),
            converged=bool(result.success),
            residuals=residuals if residuals.ndim > 0 else None,
        )
