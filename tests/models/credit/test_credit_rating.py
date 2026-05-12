"""Tests for CreditRating with dynamic re-rating."""

from __future__ import annotations

import jax
import pytest

from hyesg.core.enums import CreditClass
from hyesg.models.credit.credit_rating import CreditRating

jax.config.update("jax_enable_x64", True)


# ─── Initialisation ───


class TestCreditRatingInit:
    """Tests for CreditRating construction."""

    def test_initial_rating_preserved(self) -> None:
        """Initial rating should be accessible."""
        cr = CreditRating(CreditClass.AA)
        assert cr.rating == CreditClass.AA

    def test_initial_aaa(self) -> None:
        """Can initialise with AAA."""
        cr = CreditRating(CreditClass.AAA)
        assert cr.rating == CreditClass.AAA

    def test_initial_default(self) -> None:
        """Can initialise with DEFAULT."""
        cr = CreditRating(CreditClass.DEFAULT)
        assert cr.rating == CreditClass.DEFAULT


# ─── CreditClass ordering ───


class TestCreditClassOrdering:
    """Tests for CreditClass IntEnum ordering."""

    def test_aaa_highest(self) -> None:
        """AAA has the highest value."""
        assert CreditClass.AAA > CreditClass.AA
        assert CreditClass.AAA > CreditClass.DEFAULT

    def test_default_lowest(self) -> None:
        """DEFAULT has the lowest value."""
        assert CreditClass.DEFAULT < CreditClass.CCC
        assert CreditClass.DEFAULT < CreditClass.AAA

    def test_ordering_complete(self) -> None:
        """Full ordering: AAA > AA > A > BBB > BB > B > CCC > DEFAULT."""
        ratings = [
            CreditClass.DEFAULT,
            CreditClass.CCC,
            CreditClass.B,
            CreditClass.BB,
            CreditClass.BBB,
            CreditClass.A,
            CreditClass.AA,
            CreditClass.AAA,
        ]
        for i in range(len(ratings) - 1):
            assert ratings[i] < ratings[i + 1]

    def test_int_values(self) -> None:
        """Verify integer values."""
        assert CreditClass.AAA == 7
        assert CreditClass.DEFAULT == 0


# ─── Dynamic re-rating ───


class TestCreditRatingUpdate:
    """Tests for dynamic re-rating based on intensity."""

    @pytest.fixture
    def thresholds(self) -> dict[CreditClass, float]:
        """Standard intensity thresholds (ascending intensity → worse rating)."""
        return {
            CreditClass.DEFAULT: 0.20,
            CreditClass.CCC: 0.15,
            CreditClass.B: 0.10,
            CreditClass.BB: 0.05,
            CreditClass.BBB: 0.03,
            CreditClass.A: 0.02,
            CreditClass.AA: 0.01,
            CreditClass.AAA: 0.005,
        }

    def test_high_intensity_downgrades(
        self, thresholds: dict[CreditClass, float]
    ) -> None:
        """High intensity should result in low rating."""
        cr = CreditRating(CreditClass.AAA)
        result = cr.update_rating(0.25, thresholds)
        assert result == CreditClass.DEFAULT

    def test_low_intensity_upgrades(
        self, thresholds: dict[CreditClass, float]
    ) -> None:
        """Very low intensity should result in best rating."""
        cr = CreditRating(CreditClass.B)
        result = cr.update_rating(0.001, thresholds)
        assert result == CreditClass.AAA

    def test_rating_property_updated(
        self, thresholds: dict[CreditClass, float]
    ) -> None:
        """The rating property should reflect the update."""
        cr = CreditRating(CreditClass.AAA)
        cr.update_rating(0.12, thresholds)
        assert cr.rating.value <= CreditClass.B.value

    def test_mid_intensity_mid_rating(
        self, thresholds: dict[CreditClass, float]
    ) -> None:
        """Mid-range intensity should give mid-range rating."""
        cr = CreditRating(CreditClass.AAA)
        # intensity 0.06 >= BB threshold (0.05) but < B threshold (0.10)
        result = cr.update_rating(0.06, thresholds)
        assert result == CreditClass.BB
