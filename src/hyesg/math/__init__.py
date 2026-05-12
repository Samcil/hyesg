"""Mathematical functions and models for hyesg."""

from __future__ import annotations

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
from hyesg.math.gaussian_helpers import (
    b_func,
    b_over_dt,
    variance_integral_ou,
)
from hyesg.math.quadrature import gauss_kronrod_integrate
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
    # Transforms
    "change_compounding",
    "forward_to_inverse_zcbp",
    "forward_to_spot",
    "forward_to_zcbp",
    "inverse_zcbp_to_forward",
    "inverse_zcbp_to_spot",
    "spot_to_forward",
    "spot_to_zcbp",
    "zcbp_to_forward",
    "zcbp_to_spot",
]
