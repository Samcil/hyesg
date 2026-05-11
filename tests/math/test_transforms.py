"""Tests for yield curve transformations."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves import ConstantCurve, NelsonSiegelCurve
from hyesg.math.transforms import (
    change_compounding,
    forward_to_inverse_zcbp,
    forward_to_spot,
    forward_to_zcbp,
    inverse_zcbp_to_forward,
    inverse_zcbp_to_spot,
    spot_to_forward,
    spot_to_zcbp,
    zcbp_to_forward,
    zcbp_to_spot,
)


class TestFlatForwardTransforms:
    """Tests with flat forward rate (simplest case)."""

    def test_forward_to_spot_flat(self) -> None:
        """Flat forward → spot should also be flat."""
        fwd = ConstantCurve(0.05)
        spot = forward_to_spot(fwd)
        for t in [1.0, 5.0, 10.0, 30.0]:
            assert spot.evaluate(t) == pytest.approx(0.05, rel=1e-4)

    def test_forward_to_zcbp_flat(self) -> None:
        """Flat forward → ZCB price = exp(-r*t)."""
        fwd = ConstantCurve(0.05)
        zcb = forward_to_zcbp(fwd)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(-0.05 * t)
            assert zcb.evaluate(t) == pytest.approx(expected, rel=1e-4)

    def test_forward_to_inverse_zcbp_flat(self) -> None:
        """Flat forward → accumulation = exp(r*t)."""
        fwd = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(0.05 * t)
            assert inv.evaluate(t) == pytest.approx(expected, rel=1e-4)

    def test_spot_to_forward_flat(self) -> None:
        """Flat spot → forward should also be flat."""
        spot = ConstantCurve(0.05)
        fwd = spot_to_forward(spot)
        for t in [1.0, 5.0, 10.0]:
            assert fwd.evaluate(t) == pytest.approx(0.05, abs=1e-4)

    def test_spot_to_zcbp_flat(self) -> None:
        """Flat spot → ZCB = exp(-r*t)."""
        spot = ConstantCurve(0.05)
        zcb = spot_to_zcbp(spot)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(-0.05 * t)
            assert zcb.evaluate(t) == pytest.approx(expected, rel=1e-4)


class TestRoundTrips:
    """Tests for round-trip consistency."""

    def test_forward_spot_roundtrip(self) -> None:
        """forward → spot → forward should recover original."""
        fwd_orig = ConstantCurve(0.05)
        spot = forward_to_spot(fwd_orig)
        fwd_recovered = spot_to_forward(spot)
        for t in [1.0, 5.0, 10.0]:
            assert fwd_recovered.evaluate(t) == pytest.approx(0.05, abs=1e-3)

    def test_forward_zcbp_roundtrip(self) -> None:
        """forward → zcbp → forward should recover original."""
        fwd_orig = ConstantCurve(0.05)
        zcb = forward_to_zcbp(fwd_orig)
        fwd_recovered = zcbp_to_forward(zcb)
        for t in [1.0, 5.0, 10.0]:
            assert fwd_recovered.evaluate(t) == pytest.approx(0.05, abs=1e-3)

    def test_zcbp_spot_roundtrip(self) -> None:
        """forward → zcbp → spot should agree with direct."""
        fwd = ConstantCurve(0.05)
        zcb = forward_to_zcbp(fwd)
        spot_via_zcb = zcbp_to_spot(zcb)
        spot_direct = forward_to_spot(fwd)
        for t in [1.0, 5.0, 10.0]:
            assert spot_via_zcb.evaluate(t) == pytest.approx(
                spot_direct.evaluate(t), abs=1e-3
            )

    def test_inverse_zcbp_roundtrip(self) -> None:
        """forward → inv_zcbp → forward should recover."""
        fwd_orig = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd_orig)
        fwd_recovered = inverse_zcbp_to_forward(inv)
        for t in [1.0, 5.0, 10.0]:
            assert fwd_recovered.evaluate(t) == pytest.approx(0.05, abs=1e-3)

    def test_inverse_zcbp_to_spot(self) -> None:
        """inv_zcbp → spot should agree with forward → spot."""
        fwd = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd)
        spot_via_inv = inverse_zcbp_to_spot(inv)
        for t in [1.0, 5.0, 10.0]:
            assert spot_via_inv.evaluate(t) == pytest.approx(0.05, rel=1e-3)


class TestNelsonSiegelTransforms:
    """Tests with Nelson-Siegel forward curve."""

    def test_ns_forward_to_spot(self) -> None:
        """Nelson-Siegel forward → spot at long end → beta0."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        spot = forward_to_spot(ns)
        # At long maturities, spot should approach beta0
        assert spot.evaluate(50.0) == pytest.approx(0.05, abs=0.01)

    def test_ns_zcbp_positive(self) -> None:
        """ZCB prices should be positive."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        zcb = forward_to_zcbp(ns)
        for t in [0.5, 1.0, 5.0, 10.0, 30.0]:
            assert zcb.evaluate(t) > 0.0

    def test_ns_zcbp_decreasing(self) -> None:
        """For positive rates, ZCB prices should decrease with t."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        zcb = forward_to_zcbp(ns)
        prev = zcb.evaluate(0.5)
        for t in [1.0, 5.0, 10.0, 30.0]:
            curr = zcb.evaluate(t)
            assert curr < prev
            prev = curr


class TestChangeCompounding:
    """Tests for compounding frequency conversion."""

    def test_identity(self) -> None:
        """Same convention should return same rate."""
        assert change_compounding(0.05, 0.0, 0.0) == pytest.approx(0.05)
        assert change_compounding(0.05, 1.0, 1.0) == pytest.approx(0.05)

    def test_continuous_to_annual(self) -> None:
        """Continuous 5% → annual."""
        annual = change_compounding(0.05, 0.0, 1.0)
        # exp(0.05) = 1 + r_annual → r_annual = e^0.05 - 1
        expected = math.exp(0.05) - 1.0
        assert annual == pytest.approx(expected)

    def test_annual_to_continuous(self) -> None:
        """Annual → continuous."""
        continuous = change_compounding(0.05, 1.0, 0.0)
        # (1.05) = e^r → r = ln(1.05)
        expected = math.log(1.05)
        assert continuous == pytest.approx(expected)

    def test_continuous_annual_roundtrip(self) -> None:
        """continuous → annual → continuous should recover."""
        r = 0.05
        annual = change_compounding(r, 0.0, 1.0)
        recovered = change_compounding(annual, 1.0, 0.0)
        assert recovered == pytest.approx(r)

    def test_continuous_semiannual_roundtrip(self) -> None:
        """continuous → semi-annual → continuous should recover."""
        r = 0.05
        semi = change_compounding(r, 0.0, 0.5)
        recovered = change_compounding(semi, 0.5, 0.0)
        assert recovered == pytest.approx(r)

    def test_annual_to_semiannual(self) -> None:
        """Annual to semi-annual."""
        r_ann = 0.10
        r_semi = change_compounding(r_ann, 1.0, 0.5)
        # (1.10) = (1 + r_semi * 0.5)^2
        # r_semi = 2 * (sqrt(1.10) - 1) / 0.5... actually
        # growth = (1 + 0.10)^1 = 1.10
        # (1 + r_semi * 0.5)^(1/0.5) = 1.10
        # (1 + r_semi * 0.5)^2 = 1.10
        # 1 + r_semi * 0.5 = sqrt(1.10)
        # r_semi = (sqrt(1.10) - 1) / 0.5
        expected = (math.sqrt(1.10) - 1.0) / 0.5
        assert r_semi == pytest.approx(expected)
