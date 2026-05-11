"""High-level calibration interface.

Provides the ``Calibrator`` class that orchestrates model fitting
using objective functions and optimizers. Supports CIR, OU/Vasicek,
and credit intensity models with single and multi-regime calibration.
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
from jax import Array

from hyesg.calibration.objectives import (
    cir_curve_objective_direct,
    credit_spread_objective_direct,
    ou_curve_objective_direct,
    ou_zcb_price,
)
from hyesg.calibration.optimizer import LevenbergMarquardt
from hyesg.calibration.result import CalibrationResult
from hyesg.config.params import CIRParams, CreditParams, OUParams
from hyesg.math.cir_formulas import cir_zcb_price
from hyesg.math.curves.protocol import ParametricCurve

logger = logging.getLogger(__name__)


def _curve_to_zcb_prices(
    curve: ParametricCurve,
    tenors: Array,
) -> Array:
    """Convert a yield/spot-rate curve to ZCB prices.

    Assumes the curve evaluates to continuously compounded spot rates:
        P(0,T) = exp(-r(T) * T)

    Args:
        curve: Parametric spot-rate curve.
        tenors: Maturities in years.

    Returns:
        Array of ZCB prices.
    """
    rates = jnp.array([curve.evaluate(float(t)) for t in tenors], dtype=jnp.float64)
    return jnp.exp(-rates * tenors)


class Calibrator:
    """High-level model calibrator.

    Wraps objective functions and an optimizer to provide a clean
    interface for fitting model parameters to market data.

    Args:
        optimizer: Optimizer to use. Defaults to LevenbergMarquardt.
        tenors: Default calibration tenors. If None, uses standard set.
    """

    def __init__(
        self,
        optimizer: Any | None = None,
        tenors: Array | None = None,
    ) -> None:
        self.optimizer = optimizer or LevenbergMarquardt()
        self.tenors = (
            tenors
            if tenors is not None
            else jnp.array(
                [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
                dtype=jnp.float64,
            )
        )

    def calibrate_cir(
        self,
        market_curve: ParametricCurve,
        initial_guess: CIRParams | None = None,
        tenors: Array | None = None,
    ) -> CalibrationResult:
        """Calibrate CIR model parameters to a market yield curve.

        Fits alpha, mu, sigma to match ZCB prices implied by the
        market curve at the specified tenors.

        Args:
            market_curve: Market spot-rate curve (r(T)).
            initial_guess: Starting CIR parameters.
            tenors: Override calibration tenors.

        Returns:
            Calibration result with fitted CIR parameters.
        """
        cal_tenors = tenors if tenors is not None else self.tenors
        target_prices = _curve_to_zcb_prices(market_curve, cal_tenors)

        if initial_guess is None:
            initial_guess = CIRParams(
                alpha=0.5, mu=0.05, sigma=0.1, initial_value=0.05
            )

        x0 = jnp.array(
            [initial_guess.alpha, initial_guess.mu, initial_guess.sigma],
            dtype=jnp.float64,
        )

        result = self.optimizer.minimize(
            cir_curve_objective_direct,
            x0,
            target_prices=target_prices,
            tenors=cal_tenors,
        )

        fitted_alpha = float(result.params[0])
        fitted_mu = float(result.params[1])
        fitted_sigma = float(result.params[2])

        # Compute model-implied prices for diagnostics
        model_prices = cir_zcb_price(
            cal_tenors, fitted_mu, fitted_alpha, fitted_mu, fitted_sigma
        )
        price_errors = model_prices - target_prices

        return CalibrationResult(
            params={
                "alpha": fitted_alpha,
                "mu": fitted_mu,
                "sigma": fitted_sigma,
            },
            residuals=result.residuals if result.residuals is not None else price_errors,
            objective_value=result.objective_value,
            n_iterations=result.n_iterations,
            converged=result.converged,
            diagnostics={
                "target_prices": target_prices,
                "model_prices": model_prices,
                "tenors": cal_tenors,
                "initial_guess": {
                    "alpha": initial_guess.alpha,
                    "mu": initial_guess.mu,
                    "sigma": initial_guess.sigma,
                },
            },
        )

    def calibrate_ou(
        self,
        market_curve: ParametricCurve,
        initial_guess: OUParams | None = None,
        tenors: Array | None = None,
    ) -> CalibrationResult:
        """Calibrate OU/Vasicek parameters to a market yield curve.

        Args:
            market_curve: Market spot-rate curve (r(T)).
            initial_guess: Starting OU parameters.
            tenors: Override calibration tenors.

        Returns:
            Calibration result with fitted OU parameters.
        """
        cal_tenors = tenors if tenors is not None else self.tenors
        target_prices = _curve_to_zcb_prices(market_curve, cal_tenors)

        if initial_guess is None:
            initial_guess = OUParams(alpha=0.5, mu=0.05, sigma=0.01, initial_value=0.05)

        x0 = jnp.array(
            [initial_guess.alpha, initial_guess.mu, initial_guess.sigma],
            dtype=jnp.float64,
        )

        result = self.optimizer.minimize(
            ou_curve_objective_direct,
            x0,
            target_prices=target_prices,
            tenors=cal_tenors,
        )

        fitted_alpha = float(result.params[0])
        fitted_mu = float(result.params[1])
        fitted_sigma = float(result.params[2])

        model_prices = ou_zcb_price(
            cal_tenors,
            jnp.asarray(fitted_mu),
            jnp.asarray(fitted_alpha),
            jnp.asarray(fitted_mu),
            jnp.asarray(fitted_sigma),
        )
        price_errors = model_prices - target_prices

        return CalibrationResult(
            params={
                "alpha": fitted_alpha,
                "mu": fitted_mu,
                "sigma": fitted_sigma,
            },
            residuals=result.residuals if result.residuals is not None else price_errors,
            objective_value=result.objective_value,
            n_iterations=result.n_iterations,
            converged=result.converged,
            diagnostics={
                "target_prices": target_prices,
                "model_prices": model_prices,
                "tenors": cal_tenors,
            },
        )

    def calibrate_credit(
        self,
        market_spreads: Array,
        tenors: Array | None = None,
        initial_guess: CreditParams | None = None,
        recovery_rate: float = 0.4,
    ) -> CalibrationResult:
        """Calibrate credit intensity model to CDS spreads.

        Args:
            market_spreads: Market CDS spreads at each tenor.
            tenors: Maturities for the spreads.
            initial_guess: Starting credit parameters.
            recovery_rate: Recovery rate assumption.

        Returns:
            Calibration result with fitted credit parameters.
        """
        cal_tenors = tenors if tenors is not None else self.tenors
        market_spreads = jnp.asarray(market_spreads, dtype=jnp.float64)

        if initial_guess is None:
            initial_guess = CreditParams(
                alpha=0.5,
                mu=0.02,
                sigma=0.05,
                initial_intensity=0.01,
                recovery_rate=recovery_rate,
            )

        x0 = jnp.array(
            [
                initial_guess.alpha,
                initial_guess.mu,
                initial_guess.sigma,
                initial_guess.initial_intensity,
            ],
            dtype=jnp.float64,
        )

        result = self.optimizer.minimize(
            credit_spread_objective_direct,
            x0,
            target_spreads=market_spreads,
            tenors=cal_tenors,
            recovery_rate=recovery_rate,
        )

        fitted_alpha = float(result.params[0])
        fitted_mu = float(result.params[1])
        fitted_sigma = float(result.params[2])
        fitted_lambda0 = float(result.params[3])

        return CalibrationResult(
            params={
                "alpha": fitted_alpha,
                "mu": fitted_mu,
                "sigma": fitted_sigma,
                "initial_intensity": fitted_lambda0,
            },
            residuals=result.residuals if result.residuals is not None else jnp.zeros(1),
            objective_value=result.objective_value,
            n_iterations=result.n_iterations,
            converged=result.converged,
            diagnostics={
                "target_spreads": market_spreads,
                "tenors": cal_tenors,
                "recovery_rate": recovery_rate,
            },
        )

    def calibrate_multi_regime(
        self,
        model_type: str,
        market_data: list[dict[str, Any]],
        n_regimes: int,
    ) -> list[CalibrationResult]:
        """Calibrate model parameters for multiple regimes.

        Each regime gets its own set of parameters fitted to the
        corresponding market data.

        Args:
            model_type: One of "cir", "ou", "credit".
            market_data: List of dicts, one per regime. Each dict
                should contain "curve" (ParametricCurve) or "spreads"
                (Array), and optionally "tenors" (Array).
            n_regimes: Number of regimes (must match len(market_data)).

        Returns:
            List of CalibrationResults, one per regime.

        Raises:
            ValueError: If model_type is unknown or data length mismatches.
        """
        if len(market_data) != n_regimes:
            raise ValueError(
                f"market_data length ({len(market_data)}) "
                f"!= n_regimes ({n_regimes})"
            )

        results: list[CalibrationResult] = []

        for i, data in enumerate(market_data):
            logger.info("Calibrating regime %d/%d (%s)", i + 1, n_regimes, model_type)
            tenors = data.get("tenors")

            if model_type == "cir":
                result = self.calibrate_cir(
                    market_curve=data["curve"],
                    initial_guess=data.get("initial_guess"),
                    tenors=tenors,
                )
            elif model_type == "ou":
                result = self.calibrate_ou(
                    market_curve=data["curve"],
                    initial_guess=data.get("initial_guess"),
                    tenors=tenors,
                )
            elif model_type == "credit":
                result = self.calibrate_credit(
                    market_spreads=data["spreads"],
                    tenors=tenors,
                    initial_guess=data.get("initial_guess"),
                    recovery_rate=data.get("recovery_rate", 0.4),
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type!r}")

            results.append(result)

        return results
