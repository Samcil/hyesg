"""Tests for hyesg.config.calibration_params and related modules."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hyesg.config.calibration_builder import CalibrationParametersBuilder
from hyesg.config.calibration_params import (
    CIR2PPStructuralParams,
    CalibrationParameters,
    CorrelationSpec,
    CreditCalibrationParams,
    EquityCalibrationParams,
    FXCalibrationParams,
    G2PPStructuralParams,
    RegimeDefinition,
    YieldCurveSpec,
)
from hyesg.market_data.readers import read_correlation_csv, read_yield_curve_csv

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# RegimeDefinition
# ---------------------------------------------------------------------------


class TestRegimeDefinition:
    def test_valid_regime(self) -> None:
        r = RegimeDefinition(name="Strong", trials=2500, weight=0.5)
        assert r.name == "Strong"
        assert r.trials == 2500
        assert r.weight == 0.5

    def test_regime_with_overrides(self) -> None:
        r = RegimeDefinition(
            name="Weak",
            trials=1000,
            weight=0.2,
            overrides={"alpha": 0.05, "sigma": 0.1},
        )
        assert r.overrides["alpha"] == 0.05
        assert r.overrides["sigma"] == 0.1

    def test_zero_trials_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegimeDefinition(name="Bad", trials=0)

    def test_negative_trials_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RegimeDefinition(name="Bad", trials=-10)

    def test_weight_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            RegimeDefinition(name="Bad", trials=100, weight=1.5)

    def test_frozen(self) -> None:
        r = RegimeDefinition(name="Test", trials=100)
        with pytest.raises(Exception):  # noqa: B017
            r.name = "Other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# YieldCurveSpec
# ---------------------------------------------------------------------------


class TestYieldCurveSpec:
    def test_valid_curve(self) -> None:
        yc = YieldCurveSpec(
            knots=(1.0, 2.0, 5.0),
            spot_rates=(0.04, 0.041, 0.042),
        )
        assert len(yc.knots) == 3
        assert yc.extrapolation == "flat"

    def test_knots_rates_length_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            YieldCurveSpec(
                knots=(1.0, 2.0, 5.0),
                spot_rates=(0.04, 0.041),
            )

    def test_default_knots(self) -> None:
        rates = tuple(0.04 for _ in range(15))
        yc = YieldCurveSpec(spot_rates=rates)
        assert len(yc.knots) == 15

    def test_frozen(self) -> None:
        yc = YieldCurveSpec(knots=(1.0,), spot_rates=(0.04,))
        with pytest.raises(Exception):  # noqa: B017
            yc.extrapolation = "linear"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EquityCalibrationParams
# ---------------------------------------------------------------------------


class TestEquityCalibrationParams:
    def test_valid(self) -> None:
        p = EquityCalibrationParams(dividend_yield=0.03, volatility=0.18)
        assert p.market_price_of_risk == 0.0

    def test_negative_volatility_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EquityCalibrationParams(dividend_yield=0.03, volatility=-0.1)

    def test_negative_dividend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EquityCalibrationParams(dividend_yield=-0.01, volatility=0.1)


# ---------------------------------------------------------------------------
# FXCalibrationParams
# ---------------------------------------------------------------------------


class TestFXCalibrationParams:
    def test_valid(self) -> None:
        p = FXCalibrationParams(spot_rate=1.25, volatility=0.10)
        assert p.spot_rate == 1.25

    def test_zero_spot_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FXCalibrationParams(spot_rate=0.0, volatility=0.1)

    def test_negative_vol_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FXCalibrationParams(spot_rate=1.0, volatility=-0.05)


# ---------------------------------------------------------------------------
# CreditCalibrationParams
# ---------------------------------------------------------------------------


class TestCreditCalibrationParams:
    def test_valid(self) -> None:
        p = CreditCalibrationParams(
            initial_intensity=0.01,
            alpha=0.5,
            sigma=0.1,
        )
        assert p.recovery_rate == 0.4

    def test_custom_recovery(self) -> None:
        p = CreditCalibrationParams(
            initial_intensity=0.01,
            alpha=0.5,
            sigma=0.1,
            recovery_rate=0.6,
        )
        assert p.recovery_rate == 0.6

    def test_invalid_recovery_rate(self) -> None:
        with pytest.raises(ValidationError):
            CreditCalibrationParams(
                initial_intensity=0.01,
                alpha=0.5,
                sigma=0.1,
                recovery_rate=1.5,
            )

    def test_zero_alpha_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CreditCalibrationParams(
                initial_intensity=0.01,
                alpha=0.0,
                sigma=0.1,
            )


# ---------------------------------------------------------------------------
# CIR2PPStructuralParams
# ---------------------------------------------------------------------------


class TestCIR2PPStructuralParams:
    def test_valid(self) -> None:
        p = CIR2PPStructuralParams(
            factor1_alpha=0.1,
            factor1_sigma=0.05,
            factor2_alpha=0.2,
            factor2_sigma=0.03,
        )
        assert p.blending_alpha == 0.5

    def test_zero_alpha_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CIR2PPStructuralParams(
                factor1_alpha=0.0,
                factor1_sigma=0.05,
                factor2_alpha=0.2,
                factor2_sigma=0.03,
            )


# ---------------------------------------------------------------------------
# G2PPStructuralParams
# ---------------------------------------------------------------------------


class TestG2PPStructuralParams:
    def test_valid(self) -> None:
        p = G2PPStructuralParams(
            alpha1=0.1,
            sigma1=0.01,
            alpha2=0.2,
            sigma2=0.02,
            rho=-0.3,
        )
        assert p.rho == -0.3

    def test_rho_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            G2PPStructuralParams(
                alpha1=0.1,
                sigma1=0.01,
                alpha2=0.2,
                sigma2=0.02,
                rho=1.5,
            )


# ---------------------------------------------------------------------------
# CorrelationSpec
# ---------------------------------------------------------------------------


class TestCorrelationSpec:
    def test_valid_symmetric(self) -> None:
        c = CorrelationSpec(
            labels=("A", "B"),
            matrix=((1.0, 0.5), (0.5, 1.0)),
        )
        assert c.labels == ("A", "B")

    def test_non_symmetric_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not symmetric"):
            CorrelationSpec(
                labels=("A", "B"),
                matrix=((1.0, 0.5), (0.3, 1.0)),
            )

    def test_wrong_row_count(self) -> None:
        with pytest.raises(ValidationError, match="rows"):
            CorrelationSpec(
                labels=("A", "B"),
                matrix=((1.0, 0.5),),
            )

    def test_wrong_column_count(self) -> None:
        with pytest.raises(ValidationError, match="entries"):
            CorrelationSpec(
                labels=("A", "B"),
                matrix=((1.0,), (0.5, 1.0)),
            )


# ---------------------------------------------------------------------------
# CalibrationParameters
# ---------------------------------------------------------------------------


class TestCalibrationParameters:
    def test_defaults(self) -> None:
        p = CalibrationParameters()
        assert p.seed == 27
        assert p.inverse_dt == 12
        assert p.horizon == 100
        assert p.trials == 5000
        assert p.regimes == ()

    def test_full_construction(self) -> None:
        p = CalibrationParameters(
            seed=42,
            inverse_dt=4,
            horizon=50,
            trials=3000,
            regimes=(
                RegimeDefinition(name="Strong", trials=1500, weight=0.5),
                RegimeDefinition(name="Weak", trials=1500, weight=0.5),
            ),
            nominal_curves={
                "GBP": YieldCurveSpec(
                    knots=(1.0, 5.0, 10.0),
                    spot_rates=(0.04, 0.042, 0.043),
                ),
            },
            equity_params={
                "FTSE": EquityCalibrationParams(
                    dividend_yield=0.03,
                    volatility=0.18,
                ),
            },
            fx_params={
                "GBPUSD": FXCalibrationParams(
                    spot_rate=1.25,
                    volatility=0.10,
                ),
            },
            credit_params={
                "AAA": CreditCalibrationParams(
                    initial_intensity=0.005,
                    alpha=0.5,
                    sigma=0.08,
                ),
            },
            inflation_targets={"GBP": 0.02},
            cir2pp_structural=CIR2PPStructuralParams(
                factor1_alpha=0.1,
                factor1_sigma=0.05,
                factor2_alpha=0.2,
                factor2_sigma=0.03,
            ),
            g2pp_structural=G2PPStructuralParams(
                alpha1=0.1,
                sigma1=0.01,
                alpha2=0.2,
                sigma2=0.02,
                rho=-0.3,
            ),
        )
        assert p.trials == 3000
        assert len(p.regimes) == 2
        assert "GBP" in p.nominal_curves
        assert p.cir2pp_structural is not None
        assert p.g2pp_structural is not None

    def test_zero_horizon_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CalibrationParameters(horizon=0)

    def test_negative_trials_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CalibrationParameters(trials=-1)

    def test_frozen(self) -> None:
        p = CalibrationParameters()
        with pytest.raises(Exception):  # noqa: B017
            p.seed = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# JSON / dict serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_json_roundtrip(self) -> None:
        original = CalibrationParameters(
            seed=42,
            trials=3000,
            regimes=(
                RegimeDefinition(name="R1", trials=1000, weight=0.3),
            ),
            nominal_curves={
                "GBP": YieldCurveSpec(
                    knots=(1.0, 5.0),
                    spot_rates=(0.04, 0.042),
                ),
            },
        )
        json_str = original.model_dump_json()
        restored = CalibrationParameters.model_validate_json(json_str)
        assert restored == original
        assert restored.regimes[0].name == "R1"

    def test_dict_roundtrip(self) -> None:
        original = CalibrationParameters(
            seed=99,
            equity_params={
                "FTSE": EquityCalibrationParams(
                    dividend_yield=0.03,
                    volatility=0.18,
                ),
            },
        )
        data = original.model_dump()
        restored = CalibrationParameters.model_validate(data)
        assert restored == original

    def test_regime_overrides_roundtrip(self) -> None:
        regime = RegimeDefinition(
            name="Special",
            trials=500,
            overrides={"alpha": 0.1, "nested": {"a": 1}},
        )
        json_str = regime.model_dump_json()
        restored = RegimeDefinition.model_validate_json(json_str)
        assert restored.overrides == regime.overrides

    def test_correlation_roundtrip(self) -> None:
        c = CorrelationSpec(
            labels=("A", "B"),
            matrix=((1.0, 0.5), (0.5, 1.0)),
        )
        data = c.model_dump()
        restored = CorrelationSpec.model_validate(data)
        assert restored == c


# ---------------------------------------------------------------------------
# CalibrationParametersBuilder
# ---------------------------------------------------------------------------


class TestCalibrationParametersBuilder:
    def test_basic_build(self) -> None:
        params = CalibrationParametersBuilder().build()
        assert params.seed == 27
        assert params.trials == 5000

    def test_fluent_chain(self) -> None:
        params = (
            CalibrationParametersBuilder()
            .with_seed(42)
            .with_horizon(50, inverse_dt=4)
            .with_trials(3000)
            .with_regime("Strong", trials=2000, weight=0.6)
            .with_regime("Weak", trials=1000, weight=0.4)
            .with_equity("FTSE", dividend_yield=0.03, volatility=0.18)
            .with_fx("GBPUSD", spot_rate=1.25, volatility=0.10)
            .with_credit(
                "AAA",
                initial_intensity=0.005,
                alpha=0.5,
                sigma=0.08,
            )
            .build()
        )
        assert params.seed == 42
        assert params.horizon == 50
        assert params.inverse_dt == 4
        assert params.trials == 3000
        assert len(params.regimes) == 2
        assert params.regimes[0].name == "Strong"
        assert "FTSE" in params.equity_params
        assert "GBPUSD" in params.fx_params
        assert "AAA" in params.credit_params

    def test_with_curves(self) -> None:
        yc = YieldCurveSpec(knots=(1.0, 5.0), spot_rates=(0.04, 0.042))
        params = (
            CalibrationParametersBuilder()
            .with_nominal_curve("GBP", yc)
            .with_real_curve("GBP", yc)
            .build()
        )
        assert "GBP" in params.nominal_curves
        assert "GBP" in params.real_curves

    def test_with_correlation(self) -> None:
        c = CorrelationSpec(
            labels=("A", "B"),
            matrix=((1.0, 0.5), (0.5, 1.0)),
        )
        params = (
            CalibrationParametersBuilder()
            .with_correlation("main", c)
            .build()
        )
        assert "main" in params.correlation_specs

    def test_invalid_equity_raises(self) -> None:
        with pytest.raises(ValidationError):
            (
                CalibrationParametersBuilder()
                .with_equity("BAD", dividend_yield=-1.0, volatility=0.1)
                .build()
            )


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------


class TestReadYieldCurveCSV:
    def test_read_nominal(self) -> None:
        yc = read_yield_curve_csv(FIXTURES / "nominal_curve.csv")
        assert len(yc.knots) == 9
        assert len(yc.spot_rates) == 9
        assert yc.knots[0] == 1.0
        assert yc.spot_rates[0] == pytest.approx(0.04)

    def test_read_real_with_maturity_header(self) -> None:
        yc = read_yield_curve_csv(FIXTURES / "real_curve.csv")
        assert len(yc.knots) == 7
        assert yc.spot_rates[0] == pytest.approx(0.01)

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_yield_curve_csv(FIXTURES / "nonexistent.csv")


class TestReadCorrelationCSV:
    def test_read_correlation(self) -> None:
        c = read_correlation_csv(FIXTURES / "correlation.csv")
        assert c.labels == ("NomZ1", "NomZ2", "RealX1", "RealX2")
        assert len(c.matrix) == 4
        # Check diagonal
        assert c.matrix[0][0] == 1.0
        assert c.matrix[3][3] == 1.0

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_correlation_csv(FIXTURES / "nonexistent.csv")
