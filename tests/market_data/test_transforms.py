"""Tests for market data transforms."""

from __future__ import annotations

from pathlib import Path

from hyesg.config.models import ModelConfig, SimulationConfig
from hyesg.market_data.file_provider import FileMarketData
from hyesg.market_data.snapshot import MarketDataSnapshot
from hyesg.market_data.transforms import to_simulation_curves
from hyesg.math.curves.primitives import ConstantCurve
from hyesg.math.curves.protocol import ParametricCurve

FIXTURES = Path(__file__).parent / "fixtures"


def _make_config(*models: ModelConfig) -> SimulationConfig:
    """Helper to build a minimal SimulationConfig."""
    return SimulationConfig(name="test", models=list(models))


# ── Nominal model mapping ────────────────────────────────────


class TestNominalModelMapping:
    """to_simulation_curves maps nominal models to zero curves."""

    def test_cir2pp_gets_zero_curve(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="cir2pp", name="nominal", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "nominal" in result
        assert isinstance(result["nominal"], ParametricCurve)

    def test_g1pp_gets_zero_curve(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="g1pp", name="nom_g1", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "nom_g1" in result

    def test_vasicek_gets_zero_curve(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="vasicek", name="vas", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "vas" in result

    def test_cirpp_gets_zero_curve(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="cirpp", name="cp", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "cp" in result

    def test_cir_gets_zero_curve(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="cir", name="c", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "c" in result

    def test_default_currency_is_gbp(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="cir2pp", name="nom")
        )
        result = to_simulation_curves(snap, cfg)
        assert "nom" in result


# ── Real model mapping ───────────────────────────────────────


class TestRealModelMapping:
    """to_simulation_curves maps real models to inflation curves."""

    def test_g2pp_gets_inflation_curve(self) -> None:
        snap = MarketDataSnapshot(
            inflation_curves={"GBP": ConstantCurve(0.03)}
        )
        cfg = _make_config(
            ModelConfig(type="g2pp", name="real", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "real" in result

    def test_g2pp_missing_currency_skipped(self) -> None:
        snap = MarketDataSnapshot(
            inflation_curves={"USD": ConstantCurve(0.02)}
        )
        cfg = _make_config(
            ModelConfig(type="g2pp", name="real", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "real" not in result


# ── Credit model mapping ─────────────────────────────────────


class TestCreditModelMapping:
    """to_simulation_curves maps credit models to spread curves."""

    def test_credit_gets_spread_curve(self) -> None:
        snap = MarketDataSnapshot(
            credit_curves={"AAA": {"GBP": ConstantCurve(0.005)}}
        )
        cfg = _make_config(
            ModelConfig(
                type="credit",
                name="cred",
                params={"rating": "AAA", "currency": "GBP"},
            )
        )
        result = to_simulation_curves(snap, cfg)
        assert "cred" in result

    def test_credit_missing_rating_skipped(self) -> None:
        snap = MarketDataSnapshot(
            credit_curves={"AAA": {"GBP": ConstantCurve(0.005)}}
        )
        cfg = _make_config(
            ModelConfig(
                type="credit",
                name="cred",
                params={"rating": "CCC", "currency": "GBP"},
            )
        )
        result = to_simulation_curves(snap, cfg)
        assert "cred" not in result


# ── Unknown / mixed models ───────────────────────────────────


class TestUnknownModels:
    """Unknown model types are silently skipped."""

    def test_unknown_type_skipped(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="fca_equity", name="eq")
        )
        result = to_simulation_curves(snap, cfg)
        assert "eq" not in result

    def test_mixed_models(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)},
            inflation_curves={"GBP": ConstantCurve(0.03)},
        )
        cfg = _make_config(
            ModelConfig(type="cir2pp", name="nom", params={"currency": "GBP"}),
            ModelConfig(type="g2pp", name="real", params={"currency": "GBP"}),
            ModelConfig(type="fca_equity", name="eq"),
        )
        result = to_simulation_curves(snap, cfg)
        assert "nom" in result
        assert "real" in result
        assert "eq" not in result

    def test_empty_config(self) -> None:
        snap = MarketDataSnapshot()
        cfg = _make_config()
        result = to_simulation_curves(snap, cfg)
        assert result == {}


# ── Integration with FileMarketData ──────────────────────────


class TestTransformIntegration:
    """End-to-end: files → snapshot → curves."""

    def test_file_to_simulation_curves(self) -> None:
        provider = FileMarketData(FIXTURES)
        snap = MarketDataSnapshot.from_provider(
            provider,
            currencies=["GBP"],
            credit_ratings=["AAA"],
            credit_currencies=["GBP"],
        )
        cfg = _make_config(
            ModelConfig(type="cir2pp", name="nom", params={"currency": "GBP"}),
            ModelConfig(type="g2pp", name="real", params={"currency": "GBP"}),
            ModelConfig(
                type="credit",
                name="cred",
                params={"rating": "AAA", "currency": "GBP"},
            ),
        )
        result = to_simulation_curves(snap, cfg)
        assert len(result) == 3
        # Verify curves evaluate correctly
        assert 0.03 < result["nom"].evaluate(5.0) < 0.06
        assert 0.02 < result["real"].evaluate(5.0) < 0.04
        assert 0.001 < result["cred"].evaluate(5.0) < 0.01

    def test_case_insensitive_model_type(self) -> None:
        snap = MarketDataSnapshot(
            zero_curves={"GBP": ConstantCurve(0.04)}
        )
        cfg = _make_config(
            ModelConfig(type="CIR2PP", name="nom", params={"currency": "GBP"})
        )
        result = to_simulation_curves(snap, cfg)
        assert "nom" in result
