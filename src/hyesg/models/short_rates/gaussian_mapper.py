"""Gaussian (OU) mapping of CIR process parameters.

Maps a CIR process to an equivalent Gaussian Ornstein-Uhlenbeck (OU)
process by linearising the volatility around the long-run mean:

    CIR:  dx = α(μ - x)dt + σ√x dW
    OU:   dy = -κ·y·dt + η·dW

where κ = α and η = σ√μ.

This mapping enables analytic ZCB pricing via the Vasicek/OU affine
framework when combining two CIR factors, and is used in the CIR2++
model for efficient bond pricing under the Gaussian approximation.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


class GaussianMapper:
    """Maps CIR process parameters to equivalent Gaussian (OU) process.

    The mapping linearises the CIR diffusion σ√x around the long-run
    mean μ, giving an OU process with:
        κ = α           (mean-reversion speed preserved)
        η = σ·√μ        (volatility at long-run mean)

    This is used for analytic ZCB pricing when combining two CIR
    factors under the Gaussian approximation.

    Args:
        alpha: CIR mean-reversion speed.
        mu: CIR long-run mean level.
        sigma: CIR volatility parameter.
    """

    def __init__(self, alpha: float, mu: float, sigma: float) -> None:
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma

    @property
    def kappa(self) -> float:
        """OU mean-reversion speed κ = α."""
        return self._alpha

    @property
    def eta(self) -> float:
        """OU volatility η = σ·√μ."""
        return self._sigma * jnp.sqrt(jnp.maximum(self._mu, 0.0))

    def ou_zcb_affine(self, t: float, T: float) -> tuple[Array, Array]:
        """Compute OU affine ZCB price coefficients A(t,T) and B(t,T).

        The ZCB price under the mapped OU process is:
            P(t,T) = exp(A(t,T) - B(t,T)·y(t))

        where:
            B(t,T) = (1 - exp(-κ(T-t))) / κ
            A(t,T) = (B - (T-t))(κ²μ - η²/2)/κ² - η²B²/(4κ)

        Note: Here μ is the long-run mean of the OU process (set to the
        CIR long-run mean for the mapping), and the A coefficient
        incorporates the risk-neutral drift correction.

        Args:
            t: Current time.
            T: Maturity time.

        Returns:
            Tuple of (A, B) affine coefficients as JAX arrays.
        """
        kappa = jnp.asarray(self._alpha, dtype=jnp.float64)
        eta = self.eta
        mu = jnp.asarray(self._mu, dtype=jnp.float64)
        tau = jnp.asarray(T - t, dtype=jnp.float64)

        safe_kappa = jnp.where(jnp.abs(kappa) > 1e-30, kappa, 1e-30)

        B = (1.0 - jnp.exp(-kappa * tau)) / safe_kappa

        # A = (B - tau) * (kappa^2 * mu - eta^2/2) / kappa^2 - eta^2 * B^2 / (4*kappa)
        drift_term = (kappa**2 * mu - eta**2 / 2.0) / safe_kappa**2
        A = (B - tau) * drift_term - eta**2 * B**2 / (4.0 * safe_kappa)

        return A, B


class ExactGaussianMapper:
    """Maps CIR to OU using variance-matching at a reference state.

    Instead of linearising at the long-run mean μ, this mapper matches
    the instantaneous variance of the CIR diffusion σ²·x at a given
    reference state x_ref:

        κ = α
        η = σ·√x_ref

    When x_ref = μ this reduces to the standard ``GaussianMapper``.
    When x_ref = E[x(t)] = μ + (x₀-μ)e^{-αt}, this provides a
    time-dependent mapping that is more accurate at short horizons.

    Args:
        alpha: CIR mean-reversion speed.
        mu: CIR long-run mean level.
        sigma: CIR volatility parameter.
        x_ref: Reference state for volatility matching.
    """

    def __init__(
        self,
        alpha: float,
        mu: float,
        sigma: float,
        x_ref: float | None = None,
    ) -> None:
        self._alpha = alpha
        self._mu = mu
        self._sigma = sigma
        self._x_ref = x_ref if x_ref is not None else mu

    @property
    def kappa(self) -> float:
        """OU mean-reversion speed κ = α."""
        return self._alpha

    @property
    def eta(self) -> float:
        """OU volatility η = σ·√x_ref."""
        return self._sigma * jnp.sqrt(jnp.maximum(self._x_ref, 0.0))

    @property
    def x_ref(self) -> float:
        """Reference state for the mapping."""
        return self._x_ref

    def ou_zcb_affine(self, t: float, T: float) -> tuple[Array, Array]:
        """Compute OU affine ZCB price coefficients A(t,T) and B(t,T).

        Same formula as GaussianMapper but with η = σ·√x_ref.
        """
        kappa = jnp.asarray(self._alpha, dtype=jnp.float64)
        eta = self.eta
        mu = jnp.asarray(self._mu, dtype=jnp.float64)
        tau = jnp.asarray(T - t, dtype=jnp.float64)

        safe_kappa = jnp.where(jnp.abs(kappa) > 1e-30, kappa, 1e-30)

        B = (1.0 - jnp.exp(-kappa * tau)) / safe_kappa

        drift_term = (kappa**2 * mu - eta**2 / 2.0) / safe_kappa**2
        A = (B - tau) * drift_term - eta**2 * B**2 / (4.0 * safe_kappa)

        return A, B
