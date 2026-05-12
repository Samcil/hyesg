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
        """All numerical tolerances should be positive (except regime seeds)."""
        negative_ok = {
            "REGIME_CHI2_SEED_FACTOR",
            "REGIME_CHI2_SEED_OFFSET",
        }
        for name in dir(constants):
            if name.startswith("_"):
                continue
            val = getattr(constants, name)
            if isinstance(val, (int, float)):
                if name in negative_ok:
                    continue
                assert val > 0, f"{name} should be positive"

    # --- New constants ---

    def test_gauss_kronrod_bounds_tol(self) -> None:
        assert constants.GAUSS_KRONROD_BOUNDS_TOL == 1e-15

    def test_gauss_kronrod_default_tol(self) -> None:
        assert constants.GAUSS_KRONROD_DEFAULT_TOL == 1e-8

    def test_gauss_kronrod_max_depth(self) -> None:
        assert isinstance(constants.GAUSS_KRONROD_MAX_DEPTH, int)
        assert constants.GAUSS_KRONROD_MAX_DEPTH == 20

    def test_time_consistency_round(self) -> None:
        assert constants.TIME_CONSISTENCY_ROUND == 1e-15

    def test_time_consistency_places(self) -> None:
        assert isinstance(constants.TIME_CONSISTENCY_PLACES, int)
        assert constants.TIME_CONSISTENCY_PLACES == 15

    def test_initial_yc_target_coincidence_tol(self) -> None:
        assert constants.INITIAL_YC_TARGET_COINCIDENCE_TOL == 1e-5

    def test_akima_csv_round_places(self) -> None:
        assert isinstance(constants.AKIMA_CSV_ROUND_PLACES, int)
        assert constants.AKIMA_CSV_ROUND_PLACES == 12

    def test_limit_epsilon(self) -> None:
        assert constants.LIMIT_EPSILON == 1e-8

    def test_lm_default_max_iter(self) -> None:
        assert isinstance(constants.LM_DEFAULT_MAX_ITER, int)
        assert constants.LM_DEFAULT_MAX_ITER == 50

    def test_lm_default_tol(self) -> None:
        assert constants.LM_DEFAULT_TOL == 1e-8

    def test_lm_default_damping(self) -> None:
        assert constants.LM_DEFAULT_DAMPING == 0.01

    def test_bond_ytm_tol(self) -> None:
        assert constants.BOND_YTM_TOL == 1e-10

    def test_bond_ytm_max_iter(self) -> None:
        assert isinstance(constants.BOND_YTM_MAX_ITER, int)
        assert constants.BOND_YTM_MAX_ITER == 100

    def test_credit_recovery_quadrature_tol(self) -> None:
        assert constants.CREDIT_RECOVERY_QUADRATURE_TOL == 1e-8

    def test_regime_trial_ordering_seed_factor(self) -> None:
        assert isinstance(
            constants.REGIME_TRIAL_ORDERING_SEED_FACTOR, int
        )
        assert constants.REGIME_TRIAL_ORDERING_SEED_FACTOR == 1000003

    def test_regime_copula_seed_offset(self) -> None:
        assert isinstance(constants.REGIME_COPULA_SEED_OFFSET, int)
        assert constants.REGIME_COPULA_SEED_OFFSET == 13

    def test_regime_chi2_seed_factor(self) -> None:
        assert isinstance(constants.REGIME_CHI2_SEED_FACTOR, int)
        assert constants.REGIME_CHI2_SEED_FACTOR == -104723

    def test_regime_chi2_seed_offset(self) -> None:
        assert isinstance(constants.REGIME_CHI2_SEED_OFFSET, int)
        assert constants.REGIME_CHI2_SEED_OFFSET == -1000003
