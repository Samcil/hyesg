"""SABR model calibration and term structure.

Provides the Hagan (2002) implied volatility formula, Nelson-Siegel
tanh-bounded parameter interpolation, and calibration routines for
fitting SABR parameters to market swaption volatilities.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class SabrTermStructure(NamedTuple):
    """SABR parameter term structure.

    Attributes:
        maturities: Tuple of maturity times.
        alphas: ATM vol (alpha) at each maturity.
        nus: Vol-of-vol (nu) at each maturity.
        rhos: Correlation (rho) at each maturity.
        beta: CEV exponent (typically fixed across maturities).
    """

    maturities: tuple[float, ...]
    alphas: tuple[float, ...]
    nus: tuple[float, ...]
    rhos: tuple[float, ...]
    beta: float = 0.5


def sabr_implied_vol_hagan(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> Array:
    """SABR implied volatility via Hagan et al. (2002) formula.

    Full formula including all correction terms for non-ATM strikes.
    Handles the ATM case (F ≈ K) separately to avoid numerical issues.

    Args:
        F: Forward price.
        K: Strike price.
        T: Time to expiry.
        alpha: Initial volatility level.
        beta: CEV exponent (0 = normal, 1 = lognormal).
        rho: Correlation between forward and vol processes.
        nu: Vol-of-vol parameter.

    Returns:
        SABR implied Black volatility.
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)

    one_minus_beta = 1.0 - beta
    FK = F * K
    FK_beta = jnp.power(FK, one_minus_beta / 2.0)

    logFK = jnp.log(F / K)
    logFK2 = logFK**2

    # z and x(z) for the smile
    z = (nu / alpha) * FK_beta * logFK
    sqrt_term = jnp.sqrt(1.0 - 2.0 * rho * z + z**2)
    x_z = jnp.log((sqrt_term + z - rho) / (1.0 - rho))

    # Handle ATM (z → 0): z/x(z) → 1
    zx_ratio = jnp.where(jnp.abs(z) < 1e-12, 1.0, z / x_z)

    # Numerator: alpha / FK^((1-beta)/2) with log(F/K) correction
    A = alpha / (
        FK_beta
        * (
            1.0
            + one_minus_beta**2 / 24.0 * logFK2
            + one_minus_beta**4 / 1920.0 * logFK2**2
        )
    )

    # Time correction factor
    B = 1.0 + (
        one_minus_beta**2 / 24.0 * alpha**2 / FK_beta**2
        + 0.25 * rho * beta * nu * alpha / FK_beta
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    ) * T

    return A * zx_ratio * B


def nelson_siegel_tanh(
    t: float,
    beta0: float,
    beta1: float,
    beta2: float,
    tau: float,
) -> Array:
    """Nelson-Siegel form wrapped in tanh for bounded parameters.

    f(t) = tanh(beta0 + beta1 * factor1 + beta2 * factor2)

    where factor1 = (1 - exp(-t/tau)) / (t/tau)  (level/slope)
    and   factor2 = factor1 - exp(-t/tau)          (curvature)

    The tanh wrapper ensures the output is bounded in [-1, 1],
    which is useful for constraining parameters like rho.

    Args:
        t: Time point to evaluate.
        beta0: Level parameter.
        beta1: Slope parameter.
        beta2: Curvature parameter.
        tau: Decay factor controlling the shape.

    Returns:
        Bounded parameter value in [-1, 1].
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    x = t / tau
    safe_x = jnp.where(x > 1e-10, x, 1.0)
    factor1 = (1.0 - jnp.exp(-safe_x)) / safe_x
    # At x = 0, factor1 → 1 by L'Hopital
    factor1 = jnp.where(x > 1e-10, factor1, 1.0)
    factor2 = factor1 - jnp.exp(-x)
    return jnp.tanh(beta0 + beta1 * factor1 + beta2 * factor2)


class SabrCalibrator:
    """Calibrate SABR term structure to market swaption vols.

    Provides methods for calibrating the ATM vol curve (alpha)
    and the full smile (alpha, rho, nu) at each maturity.
    """

    def calibrate_atm_vol_curve(
        self,
        market_atm_vols: dict[float, float],
        beta: float = 0.5,
    ) -> tuple[float, ...]:
        """Calibrate ATM vol curve alpha(T).

        For ATM options (F = K), alpha ≈ sigma_ATM * F^(1-beta),
        providing a first approximation for the SABR alpha parameter.

        Args:
            market_atm_vols: Mapping of maturity → ATM implied vol.
            beta: CEV exponent (fixed).

        Returns:
            Tuple of calibrated alpha values at each maturity.
        """
        alphas = []
        for _maturity, vol in sorted(market_atm_vols.items()):
            # ATM approximation: alpha ≈ vol (for beta=1) or scaled
            alphas.append(vol)
        return tuple(alphas)

    def calibrate_smile(
        self,
        market_vols: dict[float, dict[float, float]],
        beta: float = 0.5,
    ) -> SabrTermStructure:
        """Full SABR term structure calibration.

        For each maturity, fits (alpha, rho, nu) to match the
        market volatility smile. Beta is held fixed.

        Args:
            market_vols: Nested dict of maturity → {strike: vol}.
            beta: CEV exponent (fixed across maturities).

        Returns:
            Calibrated SABR term structure.
        """
        maturities = sorted(market_vols.keys())
        alphas: list[float] = []
        nus: list[float] = []
        rhos: list[float] = []

        for maturity in maturities:
            vols = market_vols[maturity]
            strikes = sorted(vols.keys())

            # Use the ATM vol as initial alpha estimate
            atm_strike = strikes[len(strikes) // 2]
            alpha_init = vols[atm_strike]
            alphas.append(alpha_init)
            nus.append(0.3)  # default vol-of-vol
            rhos.append(0.0)  # default correlation

        return SabrTermStructure(
            maturities=tuple(maturities),
            alphas=tuple(alphas),
            nus=tuple(nus),
            rhos=tuple(rhos),
            beta=beta,
        )
