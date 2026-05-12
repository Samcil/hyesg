"""Pre-built economy configurations matching the C# ESS calibration.

Provides builder functions for each economy zone:

- GBP (domestic)
- USD, EUR, JPY, EM, APAC (foreign)
"""

from hyesg.config.economies.foreign import (
    build_apac_economy,
    build_em_economy,
    build_eur_economy,
    build_jpy_economy,
    build_usd_economy,
)
from hyesg.config.economies.gbp import build_gbp_economy

__all__ = [
    "build_apac_economy",
    "build_em_economy",
    "build_eur_economy",
    "build_gbp_economy",
    "build_jpy_economy",
    "build_usd_economy",
]
