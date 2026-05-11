"""Tests for hyesg.config.params."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from hyesg.config.params import (
    CIRParams,
    CopulaType,
    GBMParams,
    OUParams,
    PhiConfig,
    RebalanceStrategy,
    RecoveryType,
)


class TestCIRParams:
    def test_valid_construction(self) -> None:
        p = CIRParams(alpha=0.1, mu=0.03, sigma=0.05, initial_value=0.03)
        assert p.alpha == 0.1
        assert p.mu == 0.03

    def test_feller_satisfied(self) -> None:
        """No warning when Feller condition is met."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CIRParams(
                alpha=1.0,
                mu=0.05,
                sigma=0.1,
                initial_value=0.05,
            )

    def test_feller_warning(self) -> None:
        """Warning when Feller condition violated."""
        with pytest.warns(UserWarning, match="Feller"):
            CIRParams(
                alpha=0.01,
                mu=0.001,
                sigma=0.5,
                initial_value=0.01,
            )

    def test_feller_strict_raises(self) -> None:
        """Error when strict_feller and condition violated."""
        with pytest.raises(ValidationError, match="Feller"):
            CIRParams(
                alpha=0.01,
                mu=0.001,
                sigma=0.5,
                initial_value=0.01,
                strict_feller=True,
            )

    def test_negative_alpha_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CIRParams(
                alpha=-0.1,
                mu=0.03,
                sigma=0.05,
                initial_value=0.03,
            )

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CIRParams(
                alpha=0.1,
                mu=0.03,
                sigma=-0.05,
                initial_value=0.03,
            )

    def test_negative_initial_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CIRParams(
                alpha=0.1,
                mu=0.03,
                sigma=0.05,
                initial_value=-0.01,
            )

    def test_json_roundtrip(self) -> None:
        p = CIRParams(alpha=0.1, mu=0.03, sigma=0.05, initial_value=0.03)
        json_str = p.model_dump_json()
        p2 = CIRParams.model_validate_json(json_str)
        assert p == p2


class TestOUParams:
    def test_valid_vasicek(self) -> None:
        p = OUParams(alpha=0.1, mu=0.02, sigma=0.01, initial_value=0.02)
        assert p.model_type == "vasicek"

    def test_g1pp_mu_zero(self) -> None:
        """G1++ requires mu=0."""
        p = OUParams(
            alpha=0.1,
            mu=0.0,
            sigma=0.01,
            model_type="g1pp",
        )
        assert p.mu == 0.0

    def test_g1pp_nonzero_mu_raises(self) -> None:
        with pytest.raises(ValidationError, match="mu must be 0"):
            OUParams(
                alpha=0.1,
                mu=0.05,
                sigma=0.01,
                model_type="g1pp",
            )

    def test_g2pp_nonzero_mu_raises(self) -> None:
        with pytest.raises(ValidationError, match="mu must be 0"):
            OUParams(
                alpha=0.1,
                mu=0.01,
                sigma=0.01,
                model_type="g2pp",
            )

    def test_negative_alpha_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OUParams(alpha=-0.1, sigma=0.01)

    def test_json_roundtrip(self) -> None:
        p = OUParams(alpha=0.5, mu=0.02, sigma=0.01)
        json_str = p.model_dump_json()
        p2 = OUParams.model_validate_json(json_str)
        assert p == p2


class TestGBMParams:
    def test_valid(self) -> None:
        p = GBMParams(sigma=0.2, initial_value=100.0)
        assert p.sigma == 0.2

    def test_zero_initial_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GBMParams(sigma=0.2, initial_value=0.0)

    def test_negative_sigma_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GBMParams(sigma=-0.1, initial_value=1.0)


class TestPhiConfig:
    def test_default(self) -> None:
        p = PhiConfig()
        assert p.source == "analytic"
        assert p.curve_params is None

    def test_with_curve_params(self) -> None:
        p = PhiConfig(
            source="calibrated_curve",
            curve_params={"a": 1.0, "b": 2.0},
        )
        assert p.curve_params is not None


class TestEnums:
    def test_copula_values(self) -> None:
        assert CopulaType.GAUSSIAN == "gaussian"
        assert CopulaType.STUDENT_T == "student_t"

    def test_rebalance_values(self) -> None:
        assert RebalanceStrategy.FIXED == "fixed"
        assert RebalanceStrategy.BUY_AND_HOLD == "buy_and_hold"

    def test_recovery_values(self) -> None:
        assert RecoveryType.FACE == "face_value"
        assert RecoveryType.NONE == "no_recovery"


class TestFrozenModels:
    """Test that parameter models are immutable."""

    def test_cir_params_frozen(self) -> None:
        p = CIRParams(alpha=0.1, mu=0.03, sigma=0.05, initial_value=0.02)
        with pytest.raises(Exception):  # noqa: B017
            p.alpha = 0.5  # type: ignore[misc]

    def test_ou_params_frozen(self) -> None:
        from hyesg.config.params import OUParams

        p = OUParams(alpha=0.1, mu=0.0, sigma=0.05, initial_value=0.0)
        with pytest.raises(Exception):  # noqa: B017
            p.sigma = 1.0  # type: ignore[misc]

    def test_gbm_params_frozen(self) -> None:
        from hyesg.config.params import GBMParams

        p = GBMParams(sigma=0.2, initial_value=100.0)
        with pytest.raises(Exception):  # noqa: B017
            p.sigma = 0.5  # type: ignore[misc]
