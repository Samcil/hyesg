"""Tests for SplineIntensityTransform and IntensityTransform protocol."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.credit.intensity_transform import (
    IntensityTransform,
    ScaledIntensityTransform,
    SplineIntensityTransform,
)

# Enable float64
jax.config.update("jax_enable_x64", True)

# C# reference: 9-knot RN→RW spline
KNOT_XS = [0, 0.02, 0.06, 0.09, 0.12, 0.2, 0.5, 1, 10]
KNOT_YS = [
    0,
    0.00450450,
    0.03009029,
    0.05256794,
    0.07213938,
    0.12588947,
    0.33689402,
    0.70219276,
    7.70218303,
]


@pytest.fixture
def transform() -> SplineIntensityTransform:
    """Create a SplineIntensityTransform with production knots."""
    return SplineIntensityTransform(KNOT_XS, KNOT_YS)


class TestSplineIntensityTransform:
    """Tests for SplineIntensityTransform."""

    def test_zero_maps_to_zero(self, transform: SplineIntensityTransform) -> None:
        """Transform of zero intensity should be zero."""
        result = transform.transform(jnp.array(0.0))
        assert jnp.isclose(result, 0.0, atol=1e-12)

    def test_identity_at_small_values(
        self, transform: SplineIntensityTransform
    ) -> None:
        """Near-linear behaviour at small intensities (output ≈ input * slope)."""
        small_rn = jnp.array(0.001)
        result = transform.transform(small_rn)
        # At very small values the transform should give a small positive result
        assert float(result) > 0.0
        assert float(result) < float(small_rn)  # sub-linear

    def test_sub_linear_compression(
        self, transform: SplineIntensityTransform
    ) -> None:
        """Output should be less than input for high intensities."""
        high_rn = jnp.array(0.5)
        result = transform.transform(high_rn)
        assert float(result) < float(high_rn)  # sub-linear compression
        assert float(result) > 0.0

    def test_monotonically_increasing(
        self, transform: SplineIntensityTransform
    ) -> None:
        """Transform should be monotonically increasing."""
        xs = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
        results = [float(transform.transform(jnp.array(x))) for x in xs]
        for i in range(1, len(results)):
            assert results[i] >= results[i - 1], (
                f"Not monotone at index {i}: "
                f"f({xs[i-1]})={results[i-1]} > f({xs[i]})={results[i]}"
            )

    def test_known_knot_values(self, transform: SplineIntensityTransform) -> None:
        """Transform should exactly match at knot points."""
        for x, expected_y in zip(KNOT_XS, KNOT_YS):
            result = transform.transform(jnp.array(float(x)))
            assert jnp.isclose(result, expected_y, atol=1e-8), (
                f"At x={x}: expected {expected_y}, got {float(result)}"
            )

    def test_c_sharp_reference_values(
        self, transform: SplineIntensityTransform
    ) -> None:
        """Compare against known C# spline outputs at interior points."""
        # At knot 0.06 -> 0.03009029
        result_006 = transform.transform(jnp.array(0.06))
        assert jnp.isclose(result_006, 0.03009029, atol=1e-6)

        # At knot 0.2 -> 0.12588947
        result_02 = transform.transform(jnp.array(0.2))
        assert jnp.isclose(result_02, 0.12588947, atol=1e-6)

        # At knot 1.0 -> 0.70219276
        result_10 = transform.transform(jnp.array(1.0))
        assert jnp.isclose(result_10, 0.70219276, atol=1e-6)

    def test_flat_extrapolation_left(
        self, transform: SplineIntensityTransform
    ) -> None:
        """Negative inputs should use flat extrapolation (return y[0]=0)."""
        result = transform.transform(jnp.array(-1.0))
        assert jnp.isclose(result, 0.0, atol=1e-12)

    def test_flat_extrapolation_right(
        self, transform: SplineIntensityTransform
    ) -> None:
        """Inputs beyond max knot should use flat extrapolation."""
        result = transform.transform(jnp.array(100.0))
        assert jnp.isclose(result, KNOT_YS[-1], atol=1e-8)

    def test_protocol_compliance(self, transform: SplineIntensityTransform) -> None:
        """SplineIntensityTransform should satisfy IntensityTransform protocol."""
        assert isinstance(transform, IntensityTransform)

    def test_knot_properties(self, transform: SplineIntensityTransform) -> None:
        """Knot accessors should return correct values."""
        assert transform.knot_xs == KNOT_XS
        assert transform.knot_ys == KNOT_YS


class TestScaledIntensityTransform:
    """Tests for ScaledIntensityTransform."""

    def test_scale_factor(self) -> None:
        """Scaled transform should multiply base output by scale."""
        base = SplineIntensityTransform(KNOT_XS, KNOT_YS)
        scaled = ScaledIntensityTransform(base, 0.1)

        rn = jnp.array(0.2)
        base_result = base.transform(rn)
        scaled_result = scaled.transform(rn)

        assert jnp.isclose(scaled_result, base_result * 0.1, atol=1e-10)

    def test_scale_factor_property(self) -> None:
        """Scale factor property should return the configured value."""
        base = SplineIntensityTransform(KNOT_XS, KNOT_YS)
        scaled = ScaledIntensityTransform(base, 0.1)
        assert scaled.scale_factor == 0.1

    def test_protocol_compliance(self) -> None:
        """ScaledIntensityTransform should satisfy IntensityTransform protocol."""
        base = SplineIntensityTransform(KNOT_XS, KNOT_YS)
        scaled = ScaledIntensityTransform(base, 0.1)
        assert isinstance(scaled, IntensityTransform)
