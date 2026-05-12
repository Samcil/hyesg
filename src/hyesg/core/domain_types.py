"""Validated domain value types for hyesg."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from hyesg.core.enums import CreditClass, Liquidity, RecoveryType


class Maturity(BaseModel):
    """Bond/instrument maturity specification.

    Attributes:
        years: Time to maturity in years.
    """

    years: float = Field(ge=0.0, description="Time to maturity in years")

    def is_matured(self, current_time: float) -> bool:
        """Check if instrument has matured at current_time.

        Args:
            current_time: The current simulation time in years.

        Returns:
            True if the instrument has matured.
        """
        return current_time >= self.years


class Cashflow(BaseModel):
    """A single cash flow at a specific time.

    Attributes:
        time: Payment time in years.
        amount: Cash flow amount.
    """

    time: float = Field(ge=0.0, description="Payment time in years")
    amount: float = Field(description="Cash flow amount")


class ForwardRate(BaseModel):
    """Forward rate between two dates.

    Attributes:
        start: Start time in years.
        end: End time in years (must be after start).
        rate: The forward rate value.
    """

    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    rate: float

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: object) -> float:
        """Validate that end is strictly after start."""
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be after start")
        return v


class BondMetrics(BaseModel):
    """Computed bond analytics.

    Attributes:
        clean_price: Bond price excluding accrued interest.
        dirty_price: Bond price including accrued interest.
        accrued_interest: Interest accrued since last coupon.
        yield_to_maturity: Annualised yield to maturity.
        duration: Macaulay duration.
        convexity: Bond convexity.
        modified_duration: Modified duration.
    """

    clean_price: float
    dirty_price: float
    accrued_interest: float
    yield_to_maturity: float
    duration: float
    convexity: float
    modified_duration: float


class CapAndFloor(BaseModel):
    """Cap and floor specification for LPI-style instruments.

    Attributes:
        cap: Upper bound (e.g. 5%).
        floor: Lower bound (e.g. 0%).
    """

    cap: float = Field(default=0.05, description="Upper bound (e.g. 5%)")
    floor: float = Field(default=0.0, description="Lower bound (e.g. 0%)")

    @model_validator(mode="after")
    def cap_above_floor(self) -> CapAndFloor:
        """Validate that cap is at least as large as floor."""
        if self.cap < self.floor:
            raise ValueError("cap must be >= floor")
        return self


class CreditSpec(BaseModel):
    """Credit instrument specification.

    Attributes:
        rating: Credit rating class.
        recovery_type: Recovery model on default.
        recovery_rate: Recovery rate as a fraction [0, 1].
        liquidity: Liquidity classification.
    """

    rating: CreditClass
    recovery_type: RecoveryType = RecoveryType.FACE_VALUE
    recovery_rate: float = Field(default=0.4, ge=0.0, le=1.0)
    liquidity: Liquidity = Liquidity.HIGH
