"""Pre-built simulation configuration templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hyesg.config.builder import SimulationBuilder

if TYPE_CHECKING:
    from hyesg.config.models import SimulationConfig


def base_ess_template() -> SimulationConfig:
    """Create a base ESS simulation configuration template.

    Returns a standard Economic Scenario Simulator configuration
    with nominal rates (CIR2++), real rates (G2++), and inflation
    (FCA), suitable as a starting point for customisation.

    Returns:
        SimulationConfig with standard ESS structure.
    """
    return (
        SimulationBuilder("base_ess")
        .description("Base ESS simulation template")
        .time_grid(0.0, 100.0, "monthly")
        .add_model(
            "cir2pp",
            "nominal",
            alpha1=0.1,
            mu1=0.03,
            sigma1=0.05,
            alpha2=0.2,
            mu2=0.02,
            sigma2=0.03,
        )
        .add_model(
            "g2pp",
            "real",
            alpha1=0.05,
            mu1=0.0,
            sigma1=0.01,
            alpha2=0.1,
            mu2=0.0,
            sigma2=0.015,
        )
        .add_model(
            "fca",
            "inflation",
            dependencies=["nominal", "real"],
            sigma=0.02,
        )
        .correlate("nominal_z1", "nominal_z2", 0.0)
        .correlate("real_z1", "real_z2", -0.1)
        .correlate("nominal_z1", "real_z1", 0.3)
        .add_regime("r1", n_trials=1667, seed=27)
        .add_regime("r2", n_trials=1667, seed=28)
        .add_regime("r3", n_trials=1666, seed=29)
        .output_models("nominal", "real", "inflation")
        .build()
    )
