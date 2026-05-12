"""Default ESS simulation setup matching the C# engine exactly.

Provides ``build_default_ess_setup`` which constructs the standard
3-regime (Strong/Moderate/Weak) simulation with 5 000 trials, monthly
time steps, and 100-year horizon.
"""

from __future__ import annotations

from hyesg.config.simulation_setup import SimulationSetup, SimulationSetupBuilder


def build_default_ess_setup() -> SimulationSetup:
    """Build the default ESS simulation setup matching C# exactly.

    Constants:
        - seed = 27
        - InverseDt = 12 (monthly time steps)
        - Horizon = 100 years → 1200 time steps
        - Total trials = 5000
        - Regimes: Strong (2500), Moderate (1500), Weak (1000)
        - PRNG seeds: normals=27, copula=27000094, chi_squared=-3828524

    Returns:
        A validated :class:`SimulationSetup` with ESS defaults.
    """
    return (
        SimulationSetupBuilder()
        .seed(27)
        .time_grid(horizon=100, inverse_dt=12)
        .add_regime("Strong", trials=2500)
        .add_regime("Moderate", trials=1500)
        .add_regime("Weak", trials=1000)
        .build(validate=False)
    )
