"""Mathematical functions and models for hyesg."""

from __future__ import annotations

from hyesg.math.bond_analytics import (
    BondMetrics,
    compute_bond_metrics,
    dv01,
    elapsed_coupons,
    future_coupons,
    macaulay_duration,
    modified_duration,
    next_coupon,
    yield_to_maturity,
)
from hyesg.math.bond_analytics import (
    convexity as discrete_convexity,
)
from hyesg.math.cir_formulas import (
    cir_A,
    cir_B,
    cir_bond_option,
    cir_expectation,
    cir_forward_rate,
    cir_h,
    cir_integral_phi,
    cir_phi_from_curves,
    cir_variance,
    cir_zcb_price,
)
from hyesg.math.g2pp_analytics import (
    G2PPAnalyticParams,
    forward_cpi,
    g2pp_zcb_price,
    il_zcb_price,
    yyiis_rate,
    zciis_rate,
)
from hyesg.math.gaussian_helpers import (
    b_func,
    b_over_dt,
    variance_integral_ou,
)
from hyesg.math.pricing import (
    black_call,
    black_implied_vol,
    black_put,
    bond_convexity,
    bond_duration,
    bond_price,
    bond_yield,
    sabr_implied_vol,
)
from hyesg.math.quadrature import gauss_kronrod_integrate
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

__all__ = [
    # CIR formulas
    "cir_A",
    "cir_B",
    "cir_bond_option",
    "cir_expectation",
    "cir_forward_rate",
    "cir_h",
    "cir_integral_phi",
    "cir_phi_from_curves",
    "cir_variance",
    "cir_zcb_price",
    # Quadrature
    "gauss_kronrod_integrate",
    # Gaussian helpers
    "b_func",
    "b_over_dt",
    "variance_integral_ou",
    # Pricing
    "black_call",
    "black_implied_vol",
    "black_put",
    "bond_convexity",
    "bond_duration",
    "bond_price",
    "bond_yield",
    "sabr_implied_vol",
    # Bond analytics (discrete compounding)
    "BondMetrics",
    "compute_bond_metrics",
    "discrete_convexity",
    "dv01",
    "elapsed_coupons",
    "future_coupons",
    "macaulay_duration",
    "modified_duration",
    "next_coupon",
    "yield_to_maturity",
    # Transforms
    "annually_compounded_to_inv_zcbp",
    "change_compounding",
    "continuously_compounded_to_zcbp",
    "forward_to_inverse_zcbp",
    "forward_to_spot",
    "forward_to_zcbp",
    "inverse_zcbp_to_forward",
    "inverse_zcbp_to_spot",
    "spot_to_forward",
    "spot_to_inverse_zcbp",
    "spot_to_zcbp",
    "zcbp_to_forward",
    "zcbp_to_spot",
    # Seasonality
    "FourierSeasonalityAdjuster",
    # G2++ analytics
    "G2PPAnalyticParams",
    "forward_cpi",
    "g2pp_zcb_price",
    "il_zcb_price",
    "yyiis_rate",
    "zciis_rate",
]
