"""Domain enumerations for hyesg."""

from __future__ import annotations

from enum import Enum, IntEnum


class CreditClass(IntEnum):
    """Credit rating classes ordered by quality (highest=7, lowest=0)."""

    AAA = 7
    AA = 6
    A = 5
    BBB = 4
    BB = 3
    B = 2
    CCC = 1
    DEFAULT = 0


class Liquidity(Enum):
    """Liquidity classification for bond portfolios (3-tier)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecoveryType(Enum):
    """Bond recovery model type on default."""

    FACE_VALUE = "face_value"
    MARKET_VALUE = "market_value"
    TREASURY_VALUE = "treasury_value"
    NO_RECOVERY = "no_recovery"


class PoissonDistributionType(Enum):
    """Poisson distribution method for jump processes."""

    EXACT = "exact"
    CONTINUOUS = "continuous"


class CompoundingConvention(Enum):
    """Interest rate compounding convention."""

    CONTINUOUS = "continuous"
    ANNUAL = "annual"
    SEMI_ANNUAL = "semi_annual"
    QUARTERLY = "quarterly"


class RebalancingStrategy(Enum):
    """Portfolio rebalancing strategy."""

    BUY_AND_HOLD = "buy_and_hold"
    PERIODIC = "periodic"
    THRESHOLD = "threshold"


class ExerciseType(Enum):
    """Option exercise type."""

    EUROPEAN = "european"
    AMERICAN = "american"
    BERMUDAN = "bermudan"
