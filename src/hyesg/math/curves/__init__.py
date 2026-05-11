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
    ConstantCurve,
    IdentityCurve,
    LinearCurve,
)
from hyesg.math.curves.protocol import ParametricCurve
from hyesg.math.curves.splines import AkimaCubicSpline, CubicSpline

__all__ = [
    # Protocol
    "ParametricCurve",
    # Primitives
    "ConstantCurve",
    "LinearCurve",
    "IdentityCurve",
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
    "ConstantExtrapolation",
]
