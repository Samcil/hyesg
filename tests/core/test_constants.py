"""Tests for hyesg.core.constants."""

from __future__ import annotations

from hyesg.core import constants


class TestConstants:
    """Verify all constants have correct types and values."""

    def test_black_iv_tolerance(self) -> None:
        assert isinstance(constants.BLACK_IV_TOLERANCE, float)
        assert constants.BLACK_IV_TOLERANCE == 1e-15

    def test_time_epsilon(self) -> None:
        assert isinstance(constants.TIME_EPSILON, float)
        assert constants.TIME_EPSILON == 1e-7

    def test_time_places(self) -> None:
        assert isinstance(constants.TIME_PLACES, int)
        assert constants.TIME_PLACES == 7

    def test_compounding_precision(self) -> None:
        assert isinstance(constants.COMPOUNDING_PRECISION, float)
        assert constants.COMPOUNDING_PRECISION == 1e-8

    def test_yield_curve_epsilon(self) -> None:
        assert isinstance(constants.YIELD_CURVE_EPSILON, float)
        assert constants.YIELD_CURVE_EPSILON == 1e-12

    def test_parametric_curve_epsilon(self) -> None:
        assert isinstance(constants.PARAMETRIC_CURVE_EPSILON, float)
        assert constants.PARAMETRIC_CURVE_EPSILON == 1e-8

    def test_volatility_precision(self) -> None:
        assert isinstance(constants.VOLATILITY_PRECISION, float)
        assert constants.VOLATILITY_PRECISION == 1e-8

    def test_numerical_derivative_h(self) -> None:
        assert isinstance(constants.NUMERICAL_DERIVATIVE_H, float)
        assert constants.NUMERICAL_DERIVATIVE_H == 1e-4

    def test_portfolio_brent_tol(self) -> None:
        assert isinstance(constants.PORTFOLIO_BRENT_TOL, float)
        assert constants.PORTFOLIO_BRENT_TOL == 1e-8

    def test_portfolio_brent_iters(self) -> None:
        assert isinstance(constants.PORTFOLIO_BRENT_ITERS, int)
        assert constants.PORTFOLIO_BRENT_ITERS == 100

    def test_min_transaction(self) -> None:
        assert isinstance(constants.MIN_TRANSACTION, float)
        assert constants.MIN_TRANSACTION == 1e-6

    def test_min_value(self) -> None:
        assert isinstance(constants.MIN_VALUE, float)
        assert constants.MIN_VALUE == 1e-8

    def test_shorting_test(self) -> None:
        assert isinstance(constants.SHORTING_TEST, float)
        assert constants.SHORTING_TEST == 1e-12

    def test_cir_zero_vol_threshold(self) -> None:
        assert isinstance(constants.CIR_ZERO_VOL_THRESHOLD, float)
        assert constants.CIR_ZERO_VOL_THRESHOLD == 1e-7

    def test_cir_zero_h_threshold(self) -> None:
        assert isinstance(constants.CIR_ZERO_H_THRESHOLD, float)
        assert constants.CIR_ZERO_H_THRESHOLD == 1e-8

    def test_default_rng_seed(self) -> None:
        assert isinstance(constants.DEFAULT_RNG_SEED, int)
        assert constants.DEFAULT_RNG_SEED == 27

    def test_all_constants_positive(self) -> None:
        """All numerical tolerances should be positive."""
        for name in dir(constants):
            if name.startswith("_"):
                continue
            val = getattr(constants, name)
            if isinstance(val, (int, float)):
                assert val > 0, f"{name} should be positive"
