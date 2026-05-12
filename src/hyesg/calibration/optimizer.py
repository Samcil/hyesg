"""Optimizer implementations for calibration.

Provides Levenberg-Marquardt (pure JAX), a robust Madsen-Nielsen-Tingleff
variant with box constraints and heuristic warm-start, and a SciPy-based
optimizer for fitting model parameters to market data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from hyesg.calibration.result import OptimizationResult

if TYPE_CHECKING:
    from collections.abc import Callable

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


@dataclass
class RobustLevenbergMarquardtConfig:
    """Configuration for the robust Levenberg-Marquardt optimizer.

    Attributes:
        max_iter: Maximum LM iterations.
        tol: Convergence tolerance on cost reduction.
        tol_grad: Convergence tolerance on gradient norm.
        tol_param: Convergence tolerance on parameter change.
        heuristic_trials: Number of random perturbation trials
            in the BasicHeuristic warm-start phase.
        heuristic_seed: NumPy RNG seed for the heuristic search.
        damping_init: Initial Levenberg-Marquardt damping (λ).
        damping_increase: Factor to increase damping on rejection.
        damping_decrease: Factor to decrease damping on acceptance.
    """

    max_iter: int = 200
    tol: float = 1e-10
    tol_grad: float = 1e-10
    tol_param: float = 1e-10
    heuristic_trials: int = 50
    heuristic_seed: int = 1
    damping_init: float = 1e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.1


class RobustLevenbergMarquardt:
    """Madsen-Nielsen-Tingleff LM variant with box constraints.

    Two-phase optimisation:

    1. **BasicHeuristic** — random perturbation search around *x0*
       using a NumPy MT19937 PRNG (seed configurable).  Evaluates
       ``heuristic_trials`` candidates within the box and keeps the
       best.  This provides a warm-start that avoids poor local minima.
    2. **LM refinement** — standard Gauss-Newton with adaptive damping
       from the best heuristic solution.  Box constraints are enforced
       by clamping the step.

    If LM diverges or fails to improve, the heuristic solution is
    returned as a fallback.

    Args:
        config: Configuration dataclass.  Uses defaults if ``None``.
    """

    def __init__(
        self, config: RobustLevenbergMarquardtConfig | None = None
    ) -> None:
        self.config = config or RobustLevenbergMarquardtConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def minimize(
        self,
        objective: Callable[..., Array],
        x0: Array,
        bounds: tuple[Array, Array] | None = None,
        jacobian: Callable[..., Array] | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Minimize a residual function with optional box constraints.

        The *objective* must return a residual vector ``r(x)``; the
        optimizer minimises ``0.5 × ||r(x)||²``.

        Args:
            objective: Residual function ``r(params) → Array``.
            x0: Initial parameter guess.
            bounds: Optional ``(lower, upper)`` arrays for box
                constraints.  Both must have the same shape as *x0*.
            jacobian: Optional Jacobian function
                ``J(params) → Array``.  If ``None``, central
                differences are used.
            **kwargs: Forwarded to *objective*.

        Returns:
            Optimisation result with fitted parameters.
        """
        cfg = self.config
        x0 = jnp.asarray(x0, dtype=jnp.float64)

        # Phase 1 — heuristic warm-start
        x_best = self._basic_heuristic(objective, x0, bounds, **kwargs)
        r_best = objective(x_best, **kwargs)
        cost_best = float(0.5 * jnp.sum(r_best**2))

        # Phase 2 — LM refinement
        params = jnp.array(x_best)
        residuals = objective(params, **kwargs)
        cost = float(0.5 * jnp.sum(residuals**2))
        damping = cfg.damping_init

        converged = False
        n_iter = 0
        grad_norm_val: float | None = None

        for i in range(cfg.max_iter):
            n_iter = i + 1

            if jacobian is not None:
                jac = jacobian(params, **kwargs)
            else:
                jac = self._central_differences_jacobian(
                    lambda p: objective(p, **kwargs), params
                )

            jtj = jac.T @ jac
            jtr = jac.T @ residuals
            grad_norm_val = float(jnp.max(jnp.abs(jtr)))

            if grad_norm_val < cfg.tol_grad:
                converged = True
                break

            n_params = params.shape[0]
            lhs = jtj + damping * jnp.eye(n_params)
            delta = jnp.linalg.solve(lhs, -jtr)

            # Apply step with box clamping
            params_new = self._clamp(params + delta, bounds)

            residuals_new = objective(params_new, **kwargs)
            cost_new = float(0.5 * jnp.sum(residuals_new**2))

            if cost_new < cost:
                # Accept step
                params = params_new
                residuals = residuals_new
                cost = cost_new
                damping = max(damping * cfg.damping_decrease, 1e-15)

                param_change = float(
                    jnp.max(jnp.abs(delta))
                    / jnp.maximum(jnp.max(jnp.abs(params)), 1.0)
                )
                if param_change < cfg.tol_param:
                    converged = True
                    break

                if cost < cfg.tol:
                    converged = True
                    break
            else:
                # Reject step, increase damping
                damping = min(damping * cfg.damping_increase, 1e10)

        # Fallback: if LM result is worse than heuristic, keep heuristic
        if cost > cost_best:
            logger.info(
                "LM failed to improve on heuristic (%.4e > %.4e); "
                "returning heuristic solution",
                cost,
                cost_best,
            )
            return OptimizationResult(
                params=x_best,
                objective_value=cost_best,
                n_iterations=n_iter,
                converged=False,
                residuals=r_best,
                gradient_norm=grad_norm_val,
            )

        return OptimizationResult(
            params=params,
            objective_value=cost,
            n_iterations=n_iter,
            converged=converged,
            residuals=residuals,
            gradient_norm=grad_norm_val,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _basic_heuristic(
        self,
        objective: Callable[..., Array],
        x0: Array,
        bounds: tuple[Array, Array] | None,
        **kwargs: Any,
    ) -> Array:
        """Random perturbation search around *x0*.

        Uses a NumPy MT19937 PRNG for reproducibility.
        """
        cfg = self.config
        rng = np.random.RandomState(cfg.heuristic_seed)  # noqa: NPY002

        x_best = jnp.array(x0)
        r_best = objective(x_best, **kwargs)
        cost_best = float(0.5 * jnp.sum(r_best**2))

        for _ in range(cfg.heuristic_trials):
            # Perturb within bounds or ±50 % of x0
            perturbation = jnp.asarray(
                rng.uniform(-1.0, 1.0, size=x0.shape), dtype=jnp.float64
            )
            if bounds is not None:
                lower, upper = bounds
                candidate = lower + perturbation * 0.5 * (upper - lower) + 0.5 * (
                    upper + lower
                )
                candidate = jnp.clip(candidate, lower, upper)
            else:
                scale = jnp.maximum(jnp.abs(x0), jnp.ones_like(x0))
                candidate = x0 + perturbation * 0.5 * scale

            r_cand = objective(candidate, **kwargs)
            cost_cand = float(0.5 * jnp.sum(r_cand**2))

            if cost_cand < cost_best:
                x_best = candidate
                cost_best = cost_cand

        return x_best

    @staticmethod
    def _clamp(
        x: Array, bounds: tuple[Array, Array] | None
    ) -> Array:
        """Clamp *x* to lie within *bounds*."""
        if bounds is None:
            return x
        lower, upper = bounds
        return jnp.clip(x, lower, upper)

    @staticmethod
    def _central_differences_jacobian(
        f: Callable[[Array], Array],
        x: Array,
        eps: float = 1e-7,
    ) -> Array:
        """Compute Jacobian via central finite differences.

        Args:
            f: Function mapping params → residual vector.
            x: Point at which to evaluate the Jacobian.
            eps: Step size for finite differences.

        Returns:
            Jacobian array of shape ``(n_residuals, n_params)``.
        """
        n = x.shape[0]
        f0 = f(x)
        m = f0.shape[0]
        jac = jnp.zeros((m, n), dtype=jnp.float64)

        for j in range(n):
            e_j = jnp.zeros(n, dtype=jnp.float64).at[j].set(1.0)
            f_plus = f(x + eps * e_j)
            f_minus = f(x - eps * e_j)
            col = (f_plus - f_minus) / (2.0 * eps)
            jac = jac.at[:, j].set(col)

        return jac
