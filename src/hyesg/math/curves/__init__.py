"""ParametricCurve system for algebraic curve composition.

Provides a composable curve algebra where curves can be combined
using standard operators (+, -, *, /, **) and functional transforms
(exp, log, sin, cos, tanh, cap, floor, shift, differentiate, integrate).
"""

from __future__ import annotations

from hyesg.math.curves.blending import (
    ConstantExtrapolation,
    LinearBlend,
    PolynomialBlend,
    PolynomialBlendingCurve,
    PowerBlend,
    SmoothConstantExtrapolation,
)
from hyesg.math.curves.operators import (
    Added,
    Capped,
    Composed,
    Cos,
    Differentiated,
    Divided,
    Exp,
    Floored,
    Integrated,
    Log,
    Multiplied,
    Power,
    Rounded,
    ScalarMultiplied,
    Shifted,
    Sin,
    Tanh,
)
from hyesg.math.curves.parametric import (
    GeneralizedLogistic,
    NelsonSiegelCurve,
)
from hyesg.math.curves.primitives import (
    BlendedCurve,
    ConstantCurve,
    HorizontallyShiftedCurve,
    IdentityCurve,
    IntegratedOverFixedIntervalCurve,
    InverseParametricCurve,
    LinearCurve,
    PiecewiseConstantCurve,
    VerticallyShiftedCurve,
)
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.sabr import SabrAtmVolCurve, SabrNuCurve, SabrRhoCurve
from hyesg.math.curves.splines import AkimaCubicSpline, CubicSpline
from hyesg.math.curves.surface import ParametricSurface

__all__ = [
    # Protocol
    "ParametricCurve",
    # Primitives
    "ConstantCurve",
    "LinearCurve",
    "IdentityCurve",
    "PiecewiseConstantCurve",
    "HorizontallyShiftedCurve",
    "VerticallyShiftedCurve",
    "InverseParametricCurve",
    "BlendedCurve",
    "IntegratedOverFixedIntervalCurve",
    # Operators
    "Added",
    "Multiplied",
    "Divided",
    "ScalarMultiplied",
    "Composed",
    "Capped",
    "Floored",
    "Power",
    "Exp",
    "Log",
    "Sin",
    "Cos",
    "Tanh",
    "Differentiated",
    "Integrated",
    "Shifted",
    "Rounded",
    # Splines
    "CubicSpline",
    "AkimaCubicSpline",
    # Parametric
    "NelsonSiegelCurve",
    "GeneralizedLogistic",
    # Blending
    "LinearBlend",
    "PolynomialBlend",
    "PolynomialBlendingCurve",
    "PowerBlend",
    "ConstantExtrapolation",
    "SmoothConstantExtrapolation",
    # SABR
    "SabrAtmVolCurve",
    "SabrNuCurve",
    "SabrRhoCurve",
    # Surface
    "ParametricSurface",
]
