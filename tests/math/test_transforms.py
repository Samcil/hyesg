"""Tests for yield curve transformations."""

from __future__ import annotations

import math

import pytest

from hyesg.math.curves import ConstantCurve, NelsonSiegelCurve
from hyesg.math.seasonality import FourierSeasonalityAdjuster
from hyesg.math.transforms import (
    annually_compounded_to_inv_zcbp,
    change_compounding,
    continuously_compounded_to_zcbp,
    forward_to_inverse_zcbp,
    forward_to_spot,
    forward_to_zcbp,
    inverse_zcbp_to_forward,
    inverse_zcbp_to_spot,
    spot_to_forward,
    spot_to_inverse_zcbp,
    spot_to_zcbp,
    zcbp_to_forward,
    zcbp_to_spot,
)


class TestFlatForwardTransforms:
    """Tests with flat forward rate (simplest case)."""

    def test_forward_to_spot_flat(self) -> None:
        """Flat forward -> spot should also be flat."""
        fwd = ConstantCurve(0.05)
        spot = forward_to_spot(fwd)
        for t in [1.0, 5.0, 10.0, 30.0]:
            assert spot.evaluate(t) == pytest.approx(0.05, rel=1e-4)

    def test_forward_to_zcbp_flat(self) -> None:
        """Flat forward -> ZCB price = exp(-r*t)."""
        fwd = ConstantCurve(0.05)
        zcb = forward_to_zcbp(fwd)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(-0.05 * t)
            assert zcb.evaluate(t) == pytest.approx(expected, rel=1e-4)

    def test_forward_to_inverse_zcbp_flat(self) -> None:
        """Flat forward -> accumulation = exp(r*t)."""
        fwd = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(0.05 * t)
            assert inv.evaluate(t) == pytest.approx(expected, rel=1e-4)

    def test_spot_to_forward_flat(self) -> None:
        """Flat spot -> forward should also be flat."""
        spot = ConstantCurve(0.05)
        fwd = spot_to_forward(spot)
        for t in [1.0, 5.0, 10.0]:
            assert fwd.evaluate(t) == pytest.approx(0.05, abs=1e-4)

    def test_spot_to_zcbp_flat(self) -> None:
        """Flat spot -> ZCB = exp(-r*t)."""
        spot = ConstantCurve(0.05)
        zcb = spot_to_zcbp(spot)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(-0.05 * t)
            assert zcb.evaluate(t) == pytest.approx(expected, rel=1e-4)


class TestRoundTrips:
    """Tests for round-trip consistency."""

    def test_forward_spot_roundtrip(self) -> None:
        """forward -> spot -> forward should recover original."""
        fwd_orig = ConstantCurve(0.05)
        spot = forward_to_spot(fwd_orig)
        fwd_recovered = spot_to_forward(spot)
        for t in [1.0, 5.0, 10.0]:
            assert fwd_recovered.evaluate(t) == pytest.approx(0.05, abs=1e-3)

    def test_forward_zcbp_roundtrip(self) -> None:
        """forward -> zcbp -> forward should recover original."""
        fwd_orig = ConstantCurve(0.05)
        zcb = forward_to_zcbp(fwd_orig)
        fwd_recovered = zcbp_to_forward(zcb)
        for t in [1.0, 5.0, 10.0]:
            assert fwd_recovered.evaluate(t) == pytest.approx(0.05, abs=1e-3)

    def test_zcbp_spot_roundtrip(self) -> None:
        """forward -> zcbp -> spot should agree with direct."""
        fwd = ConstantCurve(0.05)
        zcb = forward_to_zcbp(fwd)
        spot_via_zcb = zcbp_to_spot(zcb)
        spot_direct = forward_to_spot(fwd)
        for t in [1.0, 5.0, 10.0]:
            assert spot_via_zcb.evaluate(t) == pytest.approx(
                spot_direct.evaluate(t), abs=1e-3
            )

    def test_inverse_zcbp_roundtrip(self) -> None:
        """forward -> inv_zcbp -> forward should recover."""
        fwd_orig = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd_orig)
        fwd_recovered = inverse_zcbp_to_forward(inv)
        for t in [1.0, 5.0, 10.0]:
            assert fwd_recovered.evaluate(t) == pytest.approx(0.05, abs=1e-3)

    def test_inverse_zcbp_to_spot(self) -> None:
        """inv_zcbp -> spot should agree with forward -> spot."""
        fwd = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd)
        spot_via_inv = inverse_zcbp_to_spot(inv)
        for t in [1.0, 5.0, 10.0]:
            assert spot_via_inv.evaluate(t) == pytest.approx(0.05, rel=1e-3)

    def test_forward_spot_zcbp_forward_chain(self) -> None:
        """forward -> spot -> zcbp -> forward = identity."""
        fwd = ConstantCurve(0.04)
        spot = forward_to_spot(fwd)
        zcb = spot_to_zcbp(spot)
        fwd_back = zcbp_to_forward(zcb)
        for t in [1.0, 5.0, 10.0, 20.0]:
            assert fwd_back.evaluate(t) == pytest.approx(
                fwd.evaluate(t), abs=1e-3
            )

    def test_spot_forward_zcbp_spot_chain(self) -> None:
        """spot -> forward -> zcbp -> spot = identity."""
        spot_orig = ConstantCurve(0.03)
        fwd = spot_to_forward(spot_orig)
        zcb = forward_to_zcbp(fwd)
        spot_back = zcbp_to_spot(zcb)
        for t in [1.0, 5.0, 10.0, 20.0]:
            assert spot_back.evaluate(t) == pytest.approx(
                spot_orig.evaluate(t), abs=1e-3
            )

    def test_inv_zcbp_spot_forward_inv_zcbp_chain(self) -> None:
        """inverse_zcbp -> spot -> forward -> inverse_zcbp = identity."""
        fwd_orig = ConstantCurve(0.06)
        inv_orig = forward_to_inverse_zcbp(fwd_orig)
        spot = inverse_zcbp_to_spot(inv_orig)
        fwd = spot_to_forward(spot)
        inv_back = forward_to_inverse_zcbp(fwd)
        for t in [1.0, 5.0, 10.0]:
            assert inv_back.evaluate(t) == pytest.approx(
                inv_orig.evaluate(t), rel=1e-3
            )

    def test_spot_to_inv_zcbp_roundtrip(self) -> None:
        """spot -> inv_zcbp -> spot should recover."""
        spot_orig = ConstantCurve(0.05)
        inv = spot_to_inverse_zcbp(spot_orig)
        spot_back = inverse_zcbp_to_spot(inv)
        for t in [1.0, 5.0, 10.0]:
            assert spot_back.evaluate(t) == pytest.approx(
                spot_orig.evaluate(t), abs=1e-3
            )


class TestNelsonSiegelTransforms:
    """Tests with Nelson-Siegel forward curve."""

    def test_ns_forward_to_spot(self) -> None:
        """Nelson-Siegel forward -> spot at long end -> beta0."""
        ns = NelsonSiegelCurve(0.05, -0.02, 0.01, 1.5)
        spot = forward_to_spot(ns)
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
        """Continuous 5%% -> annual."""
        annual = change_compounding(0.05, 0.0, 1.0)
        expected = math.exp(0.05) - 1.0
        assert annual == pytest.approx(expected)

    def test_annual_to_continuous(self) -> None:
        """Annual -> continuous."""
        continuous = change_compounding(0.05, 1.0, 0.0)
        expected = math.log(1.05)
        assert continuous == pytest.approx(expected)

    def test_continuous_annual_roundtrip(self) -> None:
        """continuous -> annual -> continuous should recover."""
        r = 0.05
        annual = change_compounding(r, 0.0, 1.0)
        recovered = change_compounding(annual, 1.0, 0.0)
        assert recovered == pytest.approx(r)

    def test_continuous_semiannual_roundtrip(self) -> None:
        """continuous -> semi-annual -> continuous should recover."""
        r = 0.05
        semi = change_compounding(r, 0.0, 0.5)
        recovered = change_compounding(semi, 0.5, 0.0)
        assert recovered == pytest.approx(r)

    def test_annual_to_semiannual(self) -> None:
        """Annual to semi-annual."""
        r_ann = 0.10
        r_semi = change_compounding(r_ann, 1.0, 0.5)
        expected = (math.sqrt(1.10) - 1.0) / 0.5
        assert r_semi == pytest.approx(expected)


class TestTransformAtZero:
    """Test that transform functions handle t=0 without division by zero."""

    def test_forward_to_spot_at_zero(self) -> None:
        """forward_to_spot(fwd).evaluate(0) should return fwd(0)."""
        fwd = ConstantCurve(0.05)
        spot = forward_to_spot(fwd)
        assert spot.evaluate(0.0) == pytest.approx(0.05, abs=1e-12)

    def test_forward_to_spot_at_zero_nelson_siegel(self) -> None:
        """forward_to_spot at t=0 for Nelson-Siegel curve."""
        ns = NelsonSiegelCurve(beta0=0.04, beta1=-0.01, beta2=0.02, tau=2.0)
        spot = forward_to_spot(ns)
        expected = ns.evaluate(0.0)
        assert spot.evaluate(0.0) == pytest.approx(expected, abs=1e-6)

    def test_zcbp_to_spot_at_zero(self) -> None:
        """zcbp_to_spot(P).evaluate(0) should be finite."""
        fwd = ConstantCurve(0.05)
        zcbp = forward_to_zcbp(fwd)
        spot = zcbp_to_spot(zcbp)
        assert spot.evaluate(0.0) == pytest.approx(0.05, abs=1e-4)

    def test_inverse_zcbp_to_spot_at_zero(self) -> None:
        """inverse_zcbp_to_spot at t=0 should be finite."""
        fwd = ConstantCurve(0.05)
        inv = forward_to_inverse_zcbp(fwd)
        spot = inverse_zcbp_to_spot(inv)
        assert spot.evaluate(0.0) == pytest.approx(0.05, abs=1e-4)

    def test_spot_to_inverse_zcbp_at_zero(self) -> None:
        """spot_to_inverse_zcbp at t=0 should return 1 (exp(0))."""
        spot = ConstantCurve(0.05)
        inv = spot_to_inverse_zcbp(spot)
        assert inv.evaluate(0.0) == pytest.approx(1.0, abs=1e-12)


class TestScalarHelpers:
    """Tests for continuously_compounded_to_zcbp and annually_compounded_to_inv_zcbp."""

    def test_cc_to_zcbp_at_zero(self) -> None:
        """At t=0, ZCB price should be 1."""
        assert continuously_compounded_to_zcbp(0.05, 0.0) == pytest.approx(1.0)

    def test_cc_to_zcbp_standard(self) -> None:
        """exp(-0.05 * 10) for standard case."""
        result = continuously_compounded_to_zcbp(0.05, 10.0)
        assert result == pytest.approx(math.exp(-0.5))

    def test_cc_to_zcbp_negative_rate(self) -> None:
        """Negative rate -> ZCB > 1."""
        result = continuously_compounded_to_zcbp(-0.03, 5.0)
        assert result == pytest.approx(math.exp(0.15))

    def test_ann_to_inv_zcbp_at_zero(self) -> None:
        """At t=0, accumulation should be 1."""
        assert annually_compounded_to_inv_zcbp(0.05, 0.0) == pytest.approx(1.0)

    def test_ann_to_inv_zcbp_standard(self) -> None:
        """(1.05)^10 for standard case."""
        result = annually_compounded_to_inv_zcbp(0.05, 10.0)
        assert result == pytest.approx(1.05**10)

    def test_ann_to_inv_zcbp_fractional_t(self) -> None:
        """Works with fractional time."""
        result = annually_compounded_to_inv_zcbp(0.08, 2.5)
        assert result == pytest.approx(1.08**2.5)


class TestSpotToInverseZcbp:
    """Tests for spot_to_inverse_zcbp."""

    def test_flat_spot_to_inv_zcbp(self) -> None:
        """Flat spot -> inv_zcbp = exp(r*t)."""
        spot = ConstantCurve(0.05)
        inv = spot_to_inverse_zcbp(spot)
        for t in [1.0, 5.0, 10.0]:
            expected = math.exp(0.05 * t)
            assert inv.evaluate(t) == pytest.approx(expected, rel=1e-4)

    def test_agrees_with_forward_path(self) -> None:
        """spot -> inv_zcbp should equal spot -> fwd -> inv_zcbp."""
        spot = ConstantCurve(0.04)
        inv_direct = spot_to_inverse_zcbp(spot)
        fwd = spot_to_forward(spot)
        inv_via_fwd = forward_to_inverse_zcbp(fwd)
        for t in [1.0, 5.0, 10.0]:
            assert inv_direct.evaluate(t) == pytest.approx(
                inv_via_fwd.evaluate(t), rel=1e-3
            )


class TestFourierSeasonalityAdjuster:
    """Tests for FourierSeasonalityAdjuster."""

    def test_zero_coefficients(self) -> None:
        """Zero coefficients -> zero adjustment."""
        adj = FourierSeasonalityAdjuster(
            coeffs_cos=(0.0, 0.0), coeffs_sin=(0.0, 0.0)
        )
        for t in [0.0, 0.25, 0.5, 1.0]:
            assert adj.adjustment(t) == pytest.approx(0.0, abs=1e-15)

    def test_single_cosine(self) -> None:
        """Single cosine term: a*cos(2*pi*t)."""
        adj = FourierSeasonalityAdjuster(
            coeffs_cos=(0.01,), coeffs_sin=(0.0,)
        )
        assert adj.adjustment(0.0) == pytest.approx(0.01, abs=1e-12)
        assert adj.adjustment(0.25) == pytest.approx(0.0, abs=1e-12)
        assert adj.adjustment(0.5) == pytest.approx(-0.01, abs=1e-12)

    def test_single_sine(self) -> None:
        """Single sine term: b*sin(2*pi*t)."""
        adj = FourierSeasonalityAdjuster(
            coeffs_cos=(0.0,), coeffs_sin=(0.02,)
        )
        assert adj.adjustment(0.0) == pytest.approx(0.0, abs=1e-12)
        assert adj.adjustment(0.25) == pytest.approx(0.02, abs=1e-12)

    def test_periodicity(self) -> None:
        """adjustment(t+1) == adjustment(t) for all t."""
        adj = FourierSeasonalityAdjuster(
            coeffs_cos=(0.01, -0.005), coeffs_sin=(0.008, 0.003)
        )
        for t in [0.0, 0.1, 0.33, 0.5, 0.75, 0.99]:
            assert adj.adjustment(t + 1.0) == pytest.approx(
                adj.adjustment(t), abs=1e-12
            )

    def test_adjust_curve(self) -> None:
        """Adjusted curve = base + adjustment."""
        adj = FourierSeasonalityAdjuster(
            coeffs_cos=(0.01,), coeffs_sin=(0.0,)
        )
        base = ConstantCurve(0.03)
        adjusted = adj.adjust_curve(base)
        for t in [0.0, 0.25, 0.5, 0.75]:
            expected = 0.03 + 0.01 * math.cos(2.0 * math.pi * t)
            assert adjusted.evaluate(t) == pytest.approx(expected, abs=1e-12)

    def test_mismatched_coeffs_raises(self) -> None:
        """Mismatched coefficient lengths should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            FourierSeasonalityAdjuster(
                coeffs_cos=(0.01, 0.02), coeffs_sin=(0.01,)
            )