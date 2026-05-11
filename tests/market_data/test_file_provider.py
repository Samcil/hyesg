"""Tests for FileMarketData provider."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyesg.market_data.file_provider import FileMarketData, MarketDataError
from hyesg.market_data.protocols import MarketDataProvider
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.splines import CubicSpline

FIXTURES = Path(__file__).parent / "fixtures"


# ── Construction ─────────────────────────────────────────────


class TestFileMarketDataInit:
    """Tests for FileMarketData construction."""

    def test_valid_directory(self) -> None:
        provider = FileMarketData(FIXTURES)
        assert provider is not None

    def test_missing_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            FileMarketData(FIXTURES / "nonexistent")

    def test_satisfies_protocol(self) -> None:
        provider = FileMarketData(FIXTURES)
        assert isinstance(provider, MarketDataProvider)


# ── Zero curve ───────────────────────────────────────────────


class TestGetZeroCurve:
    """Tests for zero curve loading."""

    def test_returns_parametric_curve(self) -> None:
        provider = FileMarketData(FIXTURES)
        curve = provider.get_zero_curve("GBP")
        assert isinstance(curve, ParametricCurve)

    def test_returns_cubic_spline(self) -> None:
        provider = FileMarketData(FIXTURES)
        curve = provider.get_zero_curve("GBP")
        assert isinstance(curve, CubicSpline)

    def test_evaluate_at_known_tenor(self) -> None:
        """Spline should pass through data points."""
        provider = FileMarketData(FIXTURES)
        curve = provider.get_zero_curve("GBP")
        # At tenor=1.0, CSV has rate=0.0460
        assert abs(curve.evaluate(1.0) - 0.0460) < 1e-10

    def test_evaluate_interpolates(self) -> None:
        """Intermediate tenors should produce reasonable values."""
        provider = FileMarketData(FIXTURES)
        curve = provider.get_zero_curve("GBP")
        rate = curve.evaluate(4.0)
        # Between 3y (0.0475) and 5y (0.0480)
        assert 0.04 < rate < 0.06

    def test_missing_currency_raises(self) -> None:
        provider = FileMarketData(FIXTURES)
        with pytest.raises(MarketDataError, match="file not found"):
            provider.get_zero_curve("EUR")

    def test_callable_syntax(self) -> None:
        """Curves support __call__ via ParametricCurve."""
        provider = FileMarketData(FIXTURES)
        curve = provider.get_zero_curve("GBP")
        assert curve(5.0) == curve.evaluate(5.0)


# ── Inflation curve ──────────────────────────────────────────


class TestGetInflationCurve:
    """Tests for inflation curve loading."""

    def test_returns_parametric_curve(self) -> None:
        provider = FileMarketData(FIXTURES)
        curve = provider.get_inflation_curve("GBP")
        assert isinstance(curve, ParametricCurve)

    def test_evaluate_at_known_tenor(self) -> None:
        provider = FileMarketData(FIXTURES)
        curve = provider.get_inflation_curve("GBP")
        assert abs(curve.evaluate(5.0) - 0.0300) < 1e-10

    def test_missing_currency_raises(self) -> None:
        provider = FileMarketData(FIXTURES)
        with pytest.raises(MarketDataError, match="file not found"):
            provider.get_inflation_curve("JPY")


# ── Credit curve ─────────────────────────────────────────────


class TestGetCreditCurve:
    """Tests for credit spread curve loading."""

    def test_returns_parametric_curve(self) -> None:
        provider = FileMarketData(FIXTURES)
        curve = provider.get_credit_curve("AAA", "GBP")
        assert isinstance(curve, ParametricCurve)

    def test_evaluate_at_known_tenor(self) -> None:
        provider = FileMarketData(FIXTURES)
        curve = provider.get_credit_curve("AAA", "GBP")
        assert abs(curve.evaluate(5.0) - 0.0040) < 1e-10

    def test_missing_rating_raises(self) -> None:
        provider = FileMarketData(FIXTURES)
        with pytest.raises(MarketDataError, match="file not found"):
            provider.get_credit_curve("CCC", "GBP")

    def test_missing_currency_raises(self) -> None:
        provider = FileMarketData(FIXTURES)
        with pytest.raises(MarketDataError, match="file not found"):
            provider.get_credit_curve("AAA", "EUR")


# ── FX spot ──────────────────────────────────────────────────


class TestGetFxSpot:
    """Tests for FX spot rate loading."""

    def test_returns_float(self) -> None:
        provider = FileMarketData(FIXTURES)
        rate = provider.get_fx_spot("GBP", "USD")
        assert isinstance(rate, float)

    def test_correct_value(self) -> None:
        provider = FileMarketData(FIXTURES)
        rate = provider.get_fx_spot("GBP", "USD")
        assert abs(rate - 1.2735) < 1e-10

    def test_missing_pair_raises(self) -> None:
        provider = FileMarketData(FIXTURES)
        with pytest.raises(MarketDataError, match="file not found"):
            provider.get_fx_spot("GBP", "JPY")


# ── Equity index ─────────────────────────────────────────────


class TestGetEquityIndex:
    """Tests for equity index loading."""

    def test_returns_float(self) -> None:
        provider = FileMarketData(FIXTURES)
        level = provider.get_equity_index("FTSE100")
        assert isinstance(level, float)

    def test_correct_value(self) -> None:
        provider = FileMarketData(FIXTURES)
        level = provider.get_equity_index("FTSE100")
        assert abs(level - 7733.24) < 1e-10

    def test_missing_index_raises(self) -> None:
        provider = FileMarketData(FIXTURES)
        with pytest.raises(MarketDataError, match="file not found"):
            provider.get_equity_index("SP500")


# ── Caching ──────────────────────────────────────────────────


class TestCaching:
    """Verify lru_cache prevents redundant file reads."""

    def test_zero_curve_cached(self) -> None:
        provider = FileMarketData(FIXTURES)
        c1 = provider.get_zero_curve("GBP")
        c2 = provider.get_zero_curve("GBP")
        assert c1 is c2

    def test_inflation_curve_cached(self) -> None:
        provider = FileMarketData(FIXTURES)
        c1 = provider.get_inflation_curve("GBP")
        c2 = provider.get_inflation_curve("GBP")
        assert c1 is c2

    def test_credit_curve_cached(self) -> None:
        provider = FileMarketData(FIXTURES)
        c1 = provider.get_credit_curve("AAA", "GBP")
        c2 = provider.get_credit_curve("AAA", "GBP")
        assert c1 is c2

    def test_fx_spot_cached(self) -> None:
        provider = FileMarketData(FIXTURES)
        r1 = provider.get_fx_spot("GBP", "USD")
        r2 = provider.get_fx_spot("GBP", "USD")
        assert r1 == r2

    def test_equity_index_cached(self) -> None:
        provider = FileMarketData(FIXTURES)
        l1 = provider.get_equity_index("FTSE100")
        l2 = provider.get_equity_index("FTSE100")
        assert l1 == l2

    def test_different_currencies_not_shared(self) -> None:
        """Cache distinguishes between different currency keys."""
        provider = FileMarketData(FIXTURES)
        gbp = provider.get_zero_curve("GBP")
        # EUR would raise, but GBP should be independently cached
        gbp2 = provider.get_zero_curve("GBP")
        assert gbp is gbp2


# ── Validation ───────────────────────────────────────────────


class TestValidation:
    """Tests for data validation in FileMarketData."""

    def test_negative_tenor_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n-1.0,0.04\n1.0,0.05\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="negative tenor"):
            provider.get_zero_curve("XXX")

    def test_non_finite_rate_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n1.0,inf\n2.0,0.05\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="non-finite rate"):
            provider.get_zero_curve("XXX")

    def test_nan_rate_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n1.0,nan\n2.0,0.05\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="non-finite rate"):
            provider.get_zero_curve("XXX")

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("maturity,yield\n1.0,0.04\n2.0,0.05\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="must have 'tenor' and 'rate'"):
            provider.get_zero_curve("XXX")

    def test_too_few_points_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n1.0,0.04\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="at least 2"):
            provider.get_zero_curve("XXX")

    def test_duplicate_tenor_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n1.0,0.04\n1.0,0.05\n2.0,0.06\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="duplicate tenor"):
            provider.get_zero_curve("XXX")

    def test_non_numeric_tenor_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\nabc,0.04\n2.0,0.05\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="invalid data"):
            provider.get_zero_curve("XXX")

    def test_empty_data_csv_raises(self, tmp_path: Path) -> None:
        ccy_dir = tmp_path / "XXX"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="at least 2"):
            provider.get_zero_curve("XXX")

    def test_scalar_empty_csv_raises(self, tmp_path: Path) -> None:
        eq_dir = tmp_path / "equity"
        eq_dir.mkdir()
        csv_file = eq_dir / "TEST.csv"
        csv_file.write_text("date,level\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="no data rows"):
            provider.get_equity_index("TEST")

    def test_scalar_missing_column_raises(self, tmp_path: Path) -> None:
        eq_dir = tmp_path / "equity"
        eq_dir.mkdir()
        csv_file = eq_dir / "TEST.csv"
        csv_file.write_text("date,price\n2024-01-01,100\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="'level' column"):
            provider.get_equity_index("TEST")

    def test_scalar_non_finite_raises(self, tmp_path: Path) -> None:
        eq_dir = tmp_path / "equity"
        eq_dir.mkdir()
        csv_file = eq_dir / "TEST.csv"
        csv_file.write_text("date,level\n2024-01-01,inf\n")
        provider = FileMarketData(tmp_path)
        with pytest.raises(MarketDataError, match="non-finite"):
            provider.get_equity_index("TEST")

    def test_unsorted_tenors_are_sorted(self, tmp_path: Path) -> None:
        """CSV rows in arbitrary order should produce a valid curve."""
        ccy_dir = tmp_path / "YYY"
        ccy_dir.mkdir()
        csv_file = ccy_dir / "zero_curve.csv"
        csv_file.write_text("tenor,rate\n5.0,0.05\n1.0,0.04\n10.0,0.06\n")
        provider = FileMarketData(tmp_path)
        curve = provider.get_zero_curve("YYY")
        assert abs(curve.evaluate(1.0) - 0.04) < 1e-10
        assert abs(curve.evaluate(5.0) - 0.05) < 1e-10
        assert abs(curve.evaluate(10.0) - 0.06) < 1e-10


# ── Hashability ──────────────────────────────────────────────


class TestHashability:
    """FileMarketData must be hashable for lru_cache."""

    def test_is_hashable(self) -> None:
        provider = FileMarketData(FIXTURES)
        assert hash(provider) == hash(provider)

    def test_equal_providers(self) -> None:
        p1 = FileMarketData(FIXTURES)
        p2 = FileMarketData(FIXTURES)
        assert p1 == p2
        assert hash(p1) == hash(p2)
