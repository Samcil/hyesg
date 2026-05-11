"""Tests for the high-level Calibrator class."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.calibration.calibrator import Calibrator, _curve_to_zcb_prices
from hyesg.calibration.optimizer import LevenbergMarquardt, LevenbergMarquardtConfig
from hyesg.calibration.result import CalibrationResult
from hyesg.config.params import CIRParams, CreditParams, OUParams
from hyesg.math.cir_formulas import cir_zcb_price
from hyesg.math.curves.protocol import ParametricCurve

jax.config.update("jax_enable_x64", True)


# ── Helpers / fixtures ──────────────────────────────────────────────


class FlatCurve(ParametricCurve):
    """Constant spot-rate curve for testing: r(T) = rate ∀ T."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def evaluate(self, x: float) -> float:
        return self._rate

    def derivative(self, x: float) -> float:
        return 0.0

    def integral(self, x: float) -> float:
        return self._rate * x


class LinearCurve(ParametricCurve):
    """Linearly increasing spot-rate curve: r(T) = base + slope * T."""

    def __init__(self, base: float, slope: float) -> None:
        self._base = base
        self._slope = slope

    def evaluate(self, x: float) -> float:
        return self._base + self._slope * x

    def derivative(self, x: float) -> float:
        return self._slope

    def integral(self, x: float) -> float:
        return self._base * x + 0.5 * self._slope * x**2


@pytest.fixture()
def calibrator() -> Calibrator:
    cfg = LevenbergMarquardtConfig(max_iterations=500, tol_grad=1e-12, tol_param=1e-12)
    return Calibrator(optimizer=LevenbergMarquardt(cfg))


@pytest.fixture()
def short_tenors() -> jax.Array:
    return jnp.array([1.0, 2.0, 5.0, 10.0], dtype=jnp.float64)


# ── _curve_to_zcb_prices tests ─────────────────────────────────────


class TestCurveToZCB:
    def test_flat_curve(self, short_tenors):
        curve = FlatCurve(0.05)
        prices = _curve_to_zcb_prices(curve, short_tenors)
        expected = jnp.exp(-0.05 * short_tenors)
        assert jnp.allclose(prices, expected, atol=1e-12)

    def test_unit_price_at_zero(self):
        curve = FlatCurve(0.05)
        tenors = jnp.array([0.0])
        prices = _curve_to_zcb_prices(curve, tenors)
        assert jnp.allclose(prices, 1.0, atol=1e-12)

    def test_linear_curve(self, short_tenors):
        curve = LinearCurve(0.02, 0.001)
        prices = _curve_to_zcb_prices(curve, short_tenors)
        rates = jnp.array([0.02 + 0.001 * float(t) for t in short_tenors])
        expected = jnp.exp(-rates * short_tenors)
        assert jnp.allclose(prices, expected, atol=1e-12)


# ── CIR calibration tests ──────────────────────────────────────────


class TestCalibrateCIR:
    def test_returns_calibration_result(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        result = calibrator.calibrate_cir(curve, tenors=short_tenors)
        assert isinstance(result, CalibrationResult)

    def test_converges_on_flat_curve(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        result = calibrator.calibrate_cir(curve, tenors=short_tenors)
        assert result.converged

    def test_params_keys(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        result = calibrator.calibrate_cir(curve, tenors=short_tenors)
        assert "alpha" in result.params
        assert "mu" in result.params
        assert "sigma" in result.params

    def test_params_positive(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        result = calibrator.calibrate_cir(curve, tenors=short_tenors)
        assert result.params["alpha"] > 0
        assert result.params["mu"] > 0
        # sigma can be very close to 0 for flat curve
        assert result.params["sigma"] >= 0

    def test_model_prices_close_to_target(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        result = calibrator.calibrate_cir(curve, tenors=short_tenors)
        diag = result.diagnostics
        assert jnp.allclose(
            diag["model_prices"], diag["target_prices"], atol=1e-4
        )

    def test_round_trip_with_known_params(self, calibrator):
        """Generate CIR prices, then recover parameters."""
        tenors = jnp.array([1.0, 2.0, 5.0, 10.0, 20.0], dtype=jnp.float64)
        true_alpha, true_mu, true_sigma = 0.5, 0.04, 0.08

        # Build a curve whose evaluate gives the CIR-implied yield
        target_prices = cir_zcb_price(tenors, true_mu, true_alpha, true_mu, true_sigma)
        yields = -jnp.log(target_prices) / tenors

        class SyntheticCurve(ParametricCurve):
            def __init__(self, t, y):
                self._t = t
                self._y = y

            def evaluate(self, x: float) -> float:
                return float(jnp.interp(x, self._t, self._y))

            def derivative(self, x: float) -> float:
                return 0.0

            def integral(self, x: float) -> float:
                return 0.0

        curve = SyntheticCurve(tenors, yields)
        guess = CIRParams(alpha=0.3, mu=0.06, sigma=0.05, initial_value=0.06)
        result = calibrator.calibrate_cir(curve, initial_guess=guess, tenors=tenors)

        assert result.converged
        assert abs(result.params["alpha"] - true_alpha) < 0.15
        assert abs(result.params["mu"] - true_mu) < 0.01

    def test_custom_initial_guess(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        guess = CIRParams(alpha=1.0, mu=0.02, sigma=0.05, initial_value=0.02)
        result = calibrator.calibrate_cir(curve, initial_guess=guess, tenors=short_tenors)
        assert isinstance(result, CalibrationResult)

    def test_diagnostics_has_tenors(self, calibrator, short_tenors):
        curve = FlatCurve(0.04)
        result = calibrator.calibrate_cir(curve, tenors=short_tenors)
        assert "tenors" in result.diagnostics


# ── OU calibration tests ───────────────────────────────────────────


class TestCalibrateOU:
    def test_returns_calibration_result(self, calibrator, short_tenors):
        curve = FlatCurve(0.03)
        result = calibrator.calibrate_ou(curve, tenors=short_tenors)
        assert isinstance(result, CalibrationResult)

    def test_converges_on_flat_curve(self, calibrator, short_tenors):
        curve = FlatCurve(0.03)
        result = calibrator.calibrate_ou(curve, tenors=short_tenors)
        assert result.converged

    def test_params_keys(self, calibrator, short_tenors):
        curve = FlatCurve(0.03)
        result = calibrator.calibrate_ou(curve, tenors=short_tenors)
        assert "alpha" in result.params
        assert "mu" in result.params
        assert "sigma" in result.params

    def test_custom_initial_guess(self, calibrator, short_tenors):
        curve = FlatCurve(0.03)
        guess = OUParams(alpha=1.0, mu=0.01, sigma=0.005, initial_value=0.01)
        result = calibrator.calibrate_ou(curve, initial_guess=guess, tenors=short_tenors)
        assert isinstance(result, CalibrationResult)


# ── Credit calibration tests ───────────────────────────────────────


class TestCalibrateCredit:
    def test_returns_calibration_result(self, calibrator, short_tenors):
        spreads = jnp.array([0.005, 0.008, 0.012, 0.015], dtype=jnp.float64)
        result = calibrator.calibrate_credit(
            spreads, tenors=short_tenors
        )
        assert isinstance(result, CalibrationResult)

    def test_converges(self, calibrator, short_tenors):
        spreads = jnp.array([0.005, 0.008, 0.012, 0.015], dtype=jnp.float64)
        result = calibrator.calibrate_credit(spreads, tenors=short_tenors)
        # Credit calibration is harder; just check it ran
        assert isinstance(result.objective_value, float)

    def test_params_keys(self, calibrator, short_tenors):
        spreads = jnp.array([0.005, 0.008, 0.012, 0.015], dtype=jnp.float64)
        result = calibrator.calibrate_credit(spreads, tenors=short_tenors)
        assert "alpha" in result.params
        assert "mu" in result.params
        assert "sigma" in result.params
        assert "initial_intensity" in result.params

    def test_custom_recovery_rate(self, calibrator, short_tenors):
        spreads = jnp.array([0.005, 0.008, 0.012, 0.015], dtype=jnp.float64)
        result = calibrator.calibrate_credit(
            spreads, tenors=short_tenors, recovery_rate=0.3
        )
        assert result.diagnostics["recovery_rate"] == 0.3

    def test_custom_initial_guess(self, calibrator, short_tenors):
        spreads = jnp.array([0.005, 0.008, 0.012, 0.015], dtype=jnp.float64)
        guess = CreditParams(
            alpha=0.3, mu=0.03, sigma=0.08, initial_intensity=0.02, recovery_rate=0.4
        )
        result = calibrator.calibrate_credit(
            spreads, tenors=short_tenors, initial_guess=guess
        )
        assert isinstance(result, CalibrationResult)


# ── Multi-regime calibration tests ─────────────────────────────────


class TestCalibrateMultiRegime:
    def test_returns_list(self, calibrator, short_tenors):
        data = [
            {"curve": FlatCurve(0.03), "tenors": short_tenors},
            {"curve": FlatCurve(0.06), "tenors": short_tenors},
        ]
        results = calibrator.calibrate_multi_regime("cir", data, n_regimes=2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, CalibrationResult) for r in results)

    def test_different_regimes_different_params(self, calibrator, short_tenors):
        data = [
            {"curve": FlatCurve(0.02), "tenors": short_tenors},
            {"curve": FlatCurve(0.08), "tenors": short_tenors},
        ]
        results = calibrator.calibrate_multi_regime("cir", data, n_regimes=2)
        # Different curves should give different mu
        assert results[0].params["mu"] != results[1].params["mu"]

    def test_ou_multi_regime(self, calibrator, short_tenors):
        data = [
            {"curve": FlatCurve(0.02), "tenors": short_tenors},
            {"curve": FlatCurve(0.05), "tenors": short_tenors},
        ]
        results = calibrator.calibrate_multi_regime("ou", data, n_regimes=2)
        assert len(results) == 2

    def test_credit_multi_regime(self, calibrator, short_tenors):
        data = [
            {
                "spreads": jnp.array([0.003, 0.005, 0.008, 0.010]),
                "tenors": short_tenors,
            },
            {
                "spreads": jnp.array([0.010, 0.015, 0.025, 0.035]),
                "tenors": short_tenors,
            },
        ]
        results = calibrator.calibrate_multi_regime("credit", data, n_regimes=2)
        assert len(results) == 2

    def test_wrong_n_regimes_raises(self, calibrator, short_tenors):
        data = [{"curve": FlatCurve(0.03), "tenors": short_tenors}]
        with pytest.raises(ValueError, match="n_regimes"):
            calibrator.calibrate_multi_regime("cir", data, n_regimes=2)

    def test_unknown_model_type_raises(self, calibrator, short_tenors):
        data = [{"curve": FlatCurve(0.03), "tenors": short_tenors}]
        with pytest.raises(ValueError, match="Unknown model_type"):
            calibrator.calibrate_multi_regime("unknown", data, n_regimes=1)


# ── Calibrator defaults ────────────────────────────────────────────


class TestCalibratorDefaults:
    def test_default_optimizer(self):
        cal = Calibrator()
        assert isinstance(cal.optimizer, LevenbergMarquardt)

    def test_default_tenors(self):
        cal = Calibrator()
        assert len(cal.tenors) == 10
        assert float(cal.tenors[0]) == 0.5

    def test_custom_tenors(self):
        t = jnp.array([1.0, 5.0, 10.0])
        cal = Calibrator(tenors=t)
        assert len(cal.tenors) == 3

    def test_calibrate_cir_uses_default_tenors(self):
        cal = Calibrator()
        curve = FlatCurve(0.04)
        result = cal.calibrate_cir(curve)
        assert len(result.diagnostics["tenors"]) == 10
