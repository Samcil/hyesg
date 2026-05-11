"""Longstaff-Schwartz Least-Squares Monte Carlo pricer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from hyesg.models.lsmc.basis import laguerre_basis, polynomial_basis

if TYPE_CHECKING:
    from collections.abc import Callable

    from hyesg.engine.output import SimulationResult


@dataclass
class LSMCConfig:
    """Configuration for the LSMC pricer.

    Attributes:
        basis: Basis function family, "polynomial" or "laguerre".
        degree: Polynomial degree for the regression basis.
        exercise_type: Option exercise style — "european", "american",
            or "bermudan".
    """

    basis: str = "polynomial"
    degree: int = 3
    exercise_type: str = "american"


@dataclass
class LSMCResult:
    """Results from LSMC pricing.

    Attributes:
        price: Option price (mean of discounted optimal cash flows).
        std_error: Monte Carlo standard error of the price estimate.
        exercise_boundary: Critical stock price at each exercise date,
            shape (n_exercise_dates,). None for European options.
        optimal_stopping: Optimal exercise time index per path,
            shape (n_paths,). None for European options.
        continuation_values: Estimated continuation values,
            shape (n_paths, n_exercise_dates). None unless requested.
    """

    price: float
    std_error: float
    exercise_boundary: Array | None = None
    optimal_stopping: Array | None = None
    continuation_values: Array | None = None


class LSMCPricer:
    """Longstaff-Schwartz Least-Squares Monte Carlo option pricer.

    The algorithm works in two passes:

    1. **Forward pass** — asset price paths are provided (or extracted
       from a ``SimulationResult``).
    2. **Backward pass** — starting from the terminal date and moving
       backwards, at each exercise date the continuation value is
       estimated by regressing discounted future cash flows on a set
       of basis functions of the current state.  Paths where immediate
       exercise is worth more than the estimated continuation value
       are marked for early exercise.
    3. **Pricing** — the mean of the discounted optimal cash flows
       gives the option price.

    Note:
        The backward pass is **not** JIT-compiled because it involves
        dynamic boolean indexing (in-the-money filtering).  The
        regression (``jnp.linalg.lstsq``) and payoff calculations
        use JAX operations for GPU acceleration where available.
    """

    def __init__(self, config: LSMCConfig | None = None) -> None:
        self._config = config or LSMCConfig()
        if self._config.basis == "laguerre":
            self._basis_fn = laguerre_basis
        else:
            self._basis_fn = polynomial_basis

    def price(
        self,
        paths: Array,
        payoff_fn: Callable[[Array, float], Array],
        strike: float,
        discount_factors: Array,
        exercise_dates: Array | None = None,
    ) -> LSMCResult:
        """Price an option using the LSMC algorithm.

        Args:
            paths: Asset prices along simulated paths,
                shape (n_paths, n_steps).
            payoff_fn: Callable ``(spots, strike) -> payoffs`` returning
                exercise values for each path.
            strike: Option strike price.
            discount_factors: Discount factor from time 0 to each step,
                shape (n_steps,).  ``discount_factors[t]`` is D(0, t).
            exercise_dates: Step indices at which exercise is allowed.
                For American options this is all dates (default).
                For European options, only the final date.
                For Bermudan options, the specified subset.

        Returns:
            An ``LSMCResult`` with the option price and diagnostics.
        """
        n_paths, n_steps = paths.shape

        exercise_mask = self._build_exercise_mask(n_steps, exercise_dates)

        if self._config.exercise_type == "european":
            return self._price_european(
                paths, payoff_fn, strike, discount_factors
            )

        return self._backward_pass(
            paths, payoff_fn, strike, discount_factors, exercise_mask
        )

    def price_from_simulation(
        self,
        result: SimulationResult,
        asset_model: str,
        rate_model: str,
        payoff_fn: Callable[[Array, float], Array],
        strike: float,
        exercise_dates: Array | None = None,
    ) -> LSMCResult:
        """Price using a ``SimulationResult`` directly.

        Extracts asset paths from ``result.select(asset_model, "level")``
        and short rates from ``result.select(rate_model, "short_rate")``.

        Args:
            result: Simulation output containing model paths.
            asset_model: Model name for the underlying asset (e.g. "equity").
            rate_model: Model name for the discount rate (e.g. "nominal").
            payoff_fn: Callable ``(spots, strike) -> payoffs``.
            strike: Option strike price.
            exercise_dates: Optional exercise date indices (see ``price``).

        Returns:
            An ``LSMCResult`` with the option price and diagnostics.
        """
        paths = result.select(asset_model, "level")
        short_rates = result.select(rate_model, "short_rate")

        dt = float(result.time_grid[1] - result.time_grid[0])
        discount_factors = self._build_discount_factors(short_rates, dt)

        # Use path-averaged discount factors for the regression
        mean_df = jnp.mean(discount_factors, axis=0)

        return self.price(
            paths=paths,
            payoff_fn=payoff_fn,
            strike=strike,
            discount_factors=mean_df,
            exercise_dates=exercise_dates,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_exercise_mask(
        self, n_steps: int, exercise_dates: Array | None
    ) -> Array:
        """Create a boolean mask indicating exercisable timesteps.

        Args:
            n_steps: Total number of timesteps.
            exercise_dates: Indices where exercise is allowed, or None
                for American style (all dates except t=0).

        Returns:
            Boolean array of shape (n_steps,).
        """
        if exercise_dates is not None:
            mask = jnp.zeros(n_steps, dtype=bool)
            mask = mask.at[exercise_dates].set(True)
            return mask
        # American: every step except t=0 is exercisable
        mask = jnp.ones(n_steps, dtype=bool)
        mask = mask.at[0].set(False)
        return mask

    def _price_european(
        self,
        paths: Array,
        payoff_fn: Callable[[Array, float], Array],
        strike: float,
        discount_factors: Array,
    ) -> LSMCResult:
        """Price a European option (exercise only at maturity).

        Args:
            paths: Asset price paths, shape (n_paths, n_steps).
            payoff_fn: Payoff function.
            strike: Strike price.
            discount_factors: Discount factors, shape (n_steps,).

        Returns:
            LSMCResult for the European option.
        """
        n_paths = paths.shape[0]
        terminal_payoff = payoff_fn(paths[:, -1], strike)
        discounted = terminal_payoff * discount_factors[-1]
        price = float(jnp.mean(discounted))
        std_error = float(jnp.std(discounted) / jnp.sqrt(n_paths))
        return LSMCResult(price=price, std_error=std_error)

    def _backward_pass(
        self,
        paths: Array,
        payoff_fn: Callable[[Array, float], Array],
        strike: float,
        discount_factors: Array,
        exercise_mask: Array,
    ) -> LSMCResult:
        """Run the Longstaff-Schwartz backward induction.

        This is intentionally **not** JIT-compiled because the
        in-the-money filtering produces dynamic shapes that are
        incompatible with ``jax.jit``.

        Args:
            paths: Asset price paths, shape (n_paths, n_steps).
            payoff_fn: Payoff function.
            strike: Strike price.
            discount_factors: D(0, t) for each step, shape (n_steps,).
            exercise_mask: Boolean mask of exercisable steps.

        Returns:
            LSMCResult with price, standard error, exercise boundary,
            and optimal stopping times.
        """
        n_paths, n_steps = paths.shape

        # Cash flow received by each path (undiscounted, at exercise time)
        cash_flows = payoff_fn(paths[:, -1], strike)
        exercise_time = jnp.full(n_paths, n_steps - 1, dtype=jnp.int32)

        # Track exercise boundary (critical stock price per date)
        boundary_prices: list[float] = []
        boundary_times: list[int] = []

        for t in range(n_steps - 2, 0, -1):
            if not bool(exercise_mask[t]):
                continue

            exercise_value = payoff_fn(paths[:, t], strike)
            itm = exercise_value > 0  # in-the-money mask

            n_itm = int(jnp.sum(itm))
            if n_itm < self._config.degree + 1:
                continue

            # Discount cash flows from exercise_time back to time t
            # For each path, df_ratio = D(0,t) / D(0, exercise_time[i])
            # continuation_at_t = cash_flows * D(0, exercise_time) / D(0, t)
            # But we want continuation at t, so:
            # value_at_t = cash_flows * D(0, exercise_time) / D(0, t)
            # Actually: discounted_to_t = cash_flows * df[exercise_time] / df[t]
            # where df[s] = D(0, s)
            df_ratio = discount_factors[exercise_time] / discount_factors[t]
            continuation_at_t = cash_flows * df_ratio

            # Regression on ITM paths only
            itm_indices = jnp.where(itm, size=n_itm)[0]
            x_itm = paths[itm_indices, t]
            y_itm = continuation_at_t[itm_indices]

            basis_matrix = self._basis_fn(x_itm, self._config.degree)
            coeffs, _, _, _ = jnp.linalg.lstsq(basis_matrix, y_itm, rcond=None)
            continuation_est = basis_matrix @ coeffs

            # Exercise where immediate value exceeds estimated continuation
            do_exercise = exercise_value[itm_indices] > continuation_est

            # Update cash flows and exercise times for exercising paths
            exercising = itm_indices[do_exercise]
            cash_flows = cash_flows.at[exercising].set(
                exercise_value[exercising]
            )
            exercise_time = exercise_time.at[exercising].set(t)

            # Record exercise boundary (max stock price at which we exercise)
            if int(jnp.sum(do_exercise)) > 0:
                boundary_price = float(jnp.max(x_itm[do_exercise]))
                boundary_prices.append(boundary_price)
                boundary_times.append(t)

        # Final price: discount each path's cash flow from its exercise time
        discounted_cashflows = cash_flows * discount_factors[exercise_time]
        price = float(jnp.mean(discounted_cashflows))
        std_error = float(
            jnp.std(discounted_cashflows) / jnp.sqrt(n_paths)
        )

        # Build exercise boundary array (sorted by time)
        exercise_boundary = None
        if boundary_times:
            sorted_idx = sorted(range(len(boundary_times)),
                                key=lambda i: boundary_times[i])
            exercise_boundary = jnp.array(
                [boundary_prices[i] for i in sorted_idx]
            )

        return LSMCResult(
            price=price,
            std_error=std_error,
            exercise_boundary=exercise_boundary,
            optimal_stopping=exercise_time,
        )

    @staticmethod
    def _build_discount_factors(short_rates: Array, dt: float) -> Array:
        """Build discount factors from short rate paths.

        D(0, t_i) = exp(-sum(r_j * dt, j=0..i-1))

        Args:
            short_rates: Short rates, shape (n_trials, n_steps).
            dt: Timestep size in years.

        Returns:
            Discount factors, shape (n_trials, n_steps).
        """
        cum_rates = jnp.cumsum(short_rates * dt, axis=1)
        return jnp.exp(-cum_rates)
