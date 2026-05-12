"""Concrete post-processor implementations.

Provides 14 processor types that each conform to the ``PostProcessor``
protocol.  All array operations use ``jax.numpy`` so processors remain
JIT-friendly when used inside traced code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable

    from hyesg.engine.post_processing.protocol import SimulationResults


# ---------------------------------------------------------------------------
# 1. CustomProcessor
# ---------------------------------------------------------------------------


class CustomProcessor:
    """Apply a user-defined function to simulation results.

    Args:
        fn: Callable that accepts and returns ``SimulationResults``.
    """

    def __init__(self, fn: Callable[[SimulationResults], SimulationResults]) -> None:
        self._fn = fn

    def process(self, results: SimulationResults) -> SimulationResults:
        return self._fn(results)


# ---------------------------------------------------------------------------
# 2. PathStatisticsProcessor
# ---------------------------------------------------------------------------


class PathStatisticsProcessor:
    """Compute mean, std and percentiles across trials for every path.

    Results are stored in ``metadata["statistics"]``.

    Args:
        percentiles: Tuple of percentile levels in [0, 1].
    """

    def __init__(
        self,
        percentiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> None:
        self._percentiles = percentiles

    def process(self, results: SimulationResults) -> SimulationResults:
        stats: dict[str, Any] = {}
        for model_name, arr in results.paths.items():
            arr = jnp.asarray(arr)
            model_stats: dict[str, Any] = {
                "mean": jnp.mean(arr, axis=0),
                "std": jnp.std(arr, axis=0),
            }
            for p in self._percentiles:
                q = jnp.float64(p * 100.0)
                model_stats[f"p{int(p * 100)}"] = jnp.percentile(arr, q, axis=0)
            stats[model_name] = model_stats

        new_meta = {**results.metadata, "statistics": stats}
        return results.model_copy(update={"metadata": new_meta})


# ---------------------------------------------------------------------------
# 3. PercentileExtractionProcessor
# ---------------------------------------------------------------------------


class PercentileExtractionProcessor:
    """Extract specific percentiles from paths and add them as new entries.

    Args:
        percentiles: Tuple of percentile levels in [0, 1].
        models: Optional list of model names to process.  ``None`` means all.
    """

    def __init__(
        self,
        percentiles: tuple[float, ...],
        models: list[str] | None = None,
    ) -> None:
        self._percentiles = percentiles
        self._models = models

    def process(self, results: SimulationResults) -> SimulationResults:
        new_paths = dict(results.paths)
        targets = (
            self._models if self._models is not None
            else list(results.paths.keys())
        )
        for model_name in targets:
            if model_name not in results.paths:
                continue
            arr = jnp.asarray(results.paths[model_name])
            for p in self._percentiles:
                q = jnp.float64(p * 100.0)
                key = f"{model_name}_p{int(p * 100)}"
                new_paths[key] = jnp.percentile(arr, q, axis=0)
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 4. ConditionalExpectationProcessor
# ---------------------------------------------------------------------------


class ConditionalExpectationProcessor:
    """Compute E[target | condition_fn(condition_field)] for each target model.

    Args:
        condition_model: Model whose paths are tested.
        condition_field: Unused directly — condition is applied to paths.
        condition_fn: Boolean-valued callable applied element-wise to the
            condition model's array; returns a boolean mask over trials.
        target_models: List of model names for which to compute the
            conditional expectation.
    """

    def __init__(
        self,
        condition_model: str,
        condition_field: str,
        condition_fn: Callable,
        target_models: list[str],
    ) -> None:
        self._condition_model = condition_model
        self._condition_field = condition_field
        self._condition_fn = condition_fn
        self._target_models = target_models

    def process(self, results: SimulationResults) -> SimulationResults:
        cond_arr = jnp.asarray(results.paths[self._condition_model])
        # mask shape: (n_trials,) — True for trials meeting condition
        mask = self._condition_fn(cond_arr)
        # Ensure mask is 1-D (one value per trial)
        if mask.ndim > 1:
            mask = jnp.all(mask, axis=tuple(range(1, mask.ndim)))

        new_meta = dict(results.metadata)
        cond_stats: dict[str, Any] = {}
        for tgt in self._target_models:
            if tgt not in results.paths:
                continue
            arr = jnp.asarray(results.paths[tgt])
            # Weighted mean using mask
            weight = mask.astype(arr.dtype)
            n_valid = jnp.maximum(jnp.sum(weight), jnp.float64(1.0))
            cond_mean = jnp.sum(arr * weight[:, None], axis=0) / n_valid
            cond_stats[tgt] = cond_mean

        new_meta["conditional_expectations"] = cond_stats
        return results.model_copy(update={"metadata": new_meta})


# ---------------------------------------------------------------------------
# 5. PortfolioAggregationProcessor
# ---------------------------------------------------------------------------


class PortfolioAggregationProcessor:
    """Aggregate model paths into a portfolio-level weighted sum.

    Args:
        portfolio_weights: Mapping of model_name -> weight.
    """

    def __init__(self, portfolio_weights: dict[str, float]) -> None:
        self._weights = portfolio_weights

    def process(self, results: SimulationResults) -> SimulationResults:
        first_key = next(iter(self._weights))
        portfolio = jnp.zeros_like(jnp.asarray(results.paths[first_key]))
        for model_name, w in self._weights.items():
            if model_name in results.paths:
                weighted = jnp.float64(w) * jnp.asarray(results.paths[model_name])
                portfolio = portfolio + weighted

        new_paths = {**results.paths, "portfolio": portfolio}
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 6. FeeDeductionProcessor
# ---------------------------------------------------------------------------


class FeeDeductionProcessor:
    """Deduct annual management fees (in bps) from gross return paths.

    Fees are applied as a multiplicative factor per timestep, assuming
    paths represent cumulative growth factors.

    Args:
        fee_schedule: Mapping of model_name -> fee in basis points.
    """

    def __init__(self, fee_schedule: dict[str, float]) -> None:
        self._fee_schedule = fee_schedule

    def process(self, results: SimulationResults) -> SimulationResults:
        new_paths = dict(results.paths)
        dt = jnp.float64(1.0)
        if results.time_grid is not None:
            tg = jnp.asarray(results.time_grid)
            if tg.shape[0] > 1:
                dt = tg[1] - tg[0]

        for model_name, fee_bps in self._fee_schedule.items():
            if model_name not in results.paths:
                continue
            arr = jnp.asarray(results.paths[model_name])
            fee_rate = jnp.float64(fee_bps) / jnp.float64(10000.0)
            # Compound fee deduction per step
            fee_factor = (jnp.float64(1.0) - fee_rate) ** dt
            n_steps = arr.shape[1] if arr.ndim > 1 else arr.shape[0]
            step_indices = jnp.arange(n_steps)
            factors = fee_factor ** step_indices
            new_paths[model_name] = arr * factors
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 7. InflationAdjustmentProcessor
# ---------------------------------------------------------------------------


class InflationAdjustmentProcessor:
    """Deflate nominal paths by a simulated inflation index.

    Args:
        inflation_model: Name of the model whose paths are the inflation index.
        base_year: Index into the time grid used as the base for deflation.
    """

    def __init__(self, inflation_model: str = "inflation", base_year: int = 0) -> None:
        self._inflation_model = inflation_model
        self._base_year = base_year

    def process(self, results: SimulationResults) -> SimulationResults:
        if self._inflation_model not in results.paths:
            return results

        infl = jnp.asarray(results.paths[self._inflation_model])
        # Normalise to base_year
        if infl.ndim > 1:
            base = infl[:, self._base_year : self._base_year + 1]
        else:
            base = infl[self._base_year : self._base_year + 1]
        deflator = infl / jnp.maximum(base, jnp.float64(1e-12))

        new_paths = dict(results.paths)
        for model_name, arr in results.paths.items():
            if model_name == self._inflation_model:
                continue
            arr = jnp.asarray(arr)
            safe_deflator = jnp.maximum(deflator, jnp.float64(1e-12))
            new_paths[f"{model_name}_real"] = arr / safe_deflator
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 8. AnnualisationProcessor
# ---------------------------------------------------------------------------


class AnnualisationProcessor:
    """Convert sub-annual paths to annual frequency.

    Args:
        frequency: Number of sub-annual periods per year (e.g. 12 for monthly).
        method: ``"geometric"`` or ``"arithmetic"``.
    """

    def __init__(self, frequency: int = 12, method: str = "geometric") -> None:
        self._frequency = frequency
        self._method = method

    def process(self, results: SimulationResults) -> SimulationResults:
        new_paths = dict(results.paths)
        for model_name, arr in results.paths.items():
            arr = jnp.asarray(arr)
            n_steps = arr.shape[-1]
            n_annual = n_steps // self._frequency
            if n_annual == 0:
                continue

            trimmed = arr[..., : n_annual * self._frequency]
            if arr.ndim == 2:
                reshaped = trimmed.reshape(arr.shape[0], n_annual, self._frequency)
            else:
                reshaped = trimmed.reshape(n_annual, self._frequency)

            if self._method == "geometric":
                annual = jnp.prod(reshaped, axis=-1)
            else:
                annual = jnp.mean(reshaped, axis=-1)

            new_paths[f"{model_name}_annual"] = annual
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 9. CurrencyConversionProcessor
# ---------------------------------------------------------------------------


class CurrencyConversionProcessor:
    """Convert foreign-currency paths to a base currency using an FX model.

    Args:
        fx_model: Name of the FX rate model in ``paths``.
        target_models: List of models to convert.
    """

    def __init__(self, fx_model: str, target_models: list[str]) -> None:
        self._fx_model = fx_model
        self._target_models = target_models

    def process(self, results: SimulationResults) -> SimulationResults:
        if self._fx_model not in results.paths:
            return results

        fx = jnp.asarray(results.paths[self._fx_model])
        new_paths = dict(results.paths)
        for model_name in self._target_models:
            if model_name not in results.paths:
                continue
            arr = jnp.asarray(results.paths[model_name])
            new_paths[f"{model_name}_base_ccy"] = arr * fx
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 10. OutputFormattingProcessor
# ---------------------------------------------------------------------------


class OutputFormattingProcessor:
    """Format results for export (rounding, labelling).

    Args:
        format: Output format label (``"table"``, ``"csv"``).
        decimal_places: Number of decimal places for rounding.
    """

    def __init__(self, format: str = "table", decimal_places: int = 6) -> None:
        self._format = format
        self._decimal_places = decimal_places

    def process(self, results: SimulationResults) -> SimulationResults:
        factor = jnp.float64(10.0 ** self._decimal_places)
        new_paths = {}
        for model_name, arr in results.paths.items():
            arr = jnp.asarray(arr)
            new_paths[model_name] = jnp.round(arr * factor) / factor

        new_meta = {
            **results.metadata,
            "output_format": self._format,
            "decimal_places": self._decimal_places,
        }
        return results.model_copy(update={"paths": new_paths, "metadata": new_meta})


# ---------------------------------------------------------------------------
# 11. EstimateFilterProcessor
# ---------------------------------------------------------------------------


class EstimateFilterProcessor:
    """LSMC estimate filter — zero out paths below a threshold.

    Args:
        filter_model: Model whose paths are tested against the threshold.
        threshold: Minimum acceptable value.
    """

    def __init__(self, filter_model: str, threshold: float = 0.0) -> None:
        self._filter_model = filter_model
        self._threshold = threshold

    def process(self, results: SimulationResults) -> SimulationResults:
        if self._filter_model not in results.paths:
            return results

        arr = jnp.asarray(results.paths[self._filter_model])
        mask = arr >= jnp.float64(self._threshold)

        new_paths = dict(results.paths)
        for model_name, val in results.paths.items():
            val = jnp.asarray(val)
            new_paths[model_name] = jnp.where(mask, val, jnp.float64(0.0))
        return results.model_copy(update={"paths": new_paths})


# ---------------------------------------------------------------------------
# 12. LSMCRegressionProcessor
# ---------------------------------------------------------------------------


class LSMCRegressionProcessor:
    """Least-Squares Monte Carlo regression on raw paths.

    Fits polynomial basis functions to continuation values at each
    exercise date and stores the fitted coefficients.

    Args:
        basis_degree: Degree of the polynomial basis.
        n_exercise_dates: Number of exercise opportunities.
    """

    def __init__(self, basis_degree: int = 3, n_exercise_dates: int = 12) -> None:
        self._basis_degree = basis_degree
        self._n_exercise_dates = n_exercise_dates

    def process(self, results: SimulationResults) -> SimulationResults:
        new_meta = dict(results.metadata)
        lsmc_results: dict[str, Any] = {}

        for model_name, arr in results.paths.items():
            arr = jnp.asarray(arr)
            if arr.ndim < 2:
                continue

            n_trials, n_steps = arr.shape
            step_size = max(1, n_steps // self._n_exercise_dates)
            exercise_indices = jnp.arange(0, n_steps, step_size)

            coefficients: list[Any] = []
            for idx in exercise_indices:
                x = arr[:, int(idx)]
                # Build polynomial basis
                basis = jnp.stack(
                    [x ** jnp.float64(d) for d in range(self._basis_degree + 1)],
                    axis=-1,
                )
                # Use terminal value as simple continuation proxy
                y = arr[:, -1]
                # Least-squares fit
                coeff, *_ = jnp.linalg.lstsq(basis, y, rcond=None)
                coefficients.append(coeff)

            lsmc_results[model_name] = {
                "coefficients": jnp.stack(coefficients),
                "exercise_indices": exercise_indices,
            }

        new_meta["lsmc"] = lsmc_results
        return results.model_copy(update={"metadata": new_meta})


# ---------------------------------------------------------------------------
# 13. SABRCalibrationProcessor
# ---------------------------------------------------------------------------


class SABRCalibrationProcessor:
    """Calibrate SABR parameters from simulated paths.

    Estimates alpha, rho, and nu from path statistics at each
    (expiry, tenor) point.

    Args:
        expiries: List of option expiries.
        tenors: List of swap tenors.
    """

    def __init__(self, expiries: list[float], tenors: list[float]) -> None:
        self._expiries = expiries
        self._tenors = tenors

    def process(self, results: SimulationResults) -> SimulationResults:
        new_meta = dict(results.metadata)
        sabr_params: dict[str, Any] = {}

        for model_name, arr in results.paths.items():
            arr = jnp.asarray(arr)
            if arr.ndim < 2:
                continue
            vol = jnp.std(jnp.log(jnp.maximum(arr, jnp.float64(1e-12))), axis=0)
            sabr_params[model_name] = {
                "implied_vol": vol,
                "expiries": self._expiries,
                "tenors": self._tenors,
                "alpha": jnp.mean(vol),
                "rho": jnp.float64(0.0),
                "nu": jnp.std(vol),
            }

        new_meta["sabr_calibration"] = sabr_params
        return results.model_copy(update={"metadata": new_meta})


# ---------------------------------------------------------------------------
# 14. EquilibriumSwapRateProcessor
# ---------------------------------------------------------------------------


class EquilibriumSwapRateProcessor:
    """Calculate a fair swap rate from simulated discount-factor paths.

    Assumes paths represent zero-coupon bond prices P(0, T).

    Args:
        tenor: Swap tenor in years.
        fixed_frequency: Number of fixed-leg payments per year.
    """

    def __init__(self, tenor: float = 10.0, fixed_frequency: int = 2) -> None:
        self._tenor = tenor
        self._fixed_frequency = fixed_frequency

    def process(self, results: SimulationResults) -> SimulationResults:
        new_paths = dict(results.paths)
        n_payments = int(self._tenor * self._fixed_frequency)

        for model_name, arr in results.paths.items():
            arr = jnp.asarray(arr)
            if arr.ndim < 2:
                continue
            n_steps = arr.shape[-1]
            if n_steps < n_payments:
                continue

            step = max(1, n_steps // n_payments)
            indices = jnp.arange(step, n_steps, step)[:n_payments]
            if indices.shape[0] == 0:
                continue

            # Annuity = sum of discount factors at payment dates
            discount_factors = arr[:, indices]
            annuity = jnp.sum(discount_factors, axis=-1)
            # Swap rate = (1 - P(0,T)) / annuity
            p_terminal = arr[:, -1]
            swap_rate = (jnp.float64(1.0) - p_terminal) / jnp.maximum(
                annuity, jnp.float64(1e-12)
            )
            new_paths[f"{model_name}_swap_rate"] = swap_rate

        return results.model_copy(update={"paths": new_paths})
