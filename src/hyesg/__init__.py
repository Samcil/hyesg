"""hyesg — Python/JAX Economic Scenario Generator."""

__version__ = "0.1.0"

# Enable float64 for financial precision
import jax

jax.config.update("jax_enable_x64", True)

from hyesg.config.models import SimulationConfig
from hyesg.engine.output import SimulationResult
from hyesg.engine.simulator import Simulator


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a complete ESG simulation.

    One-line entry point: builds models, runs all regimes, returns results.

    Args:
        config: Complete simulation configuration.

    Returns:
        SimulationResult with outputs from all models and regimes.
    """
    sim = Simulator(config)
    if config.regimes:
        return sim.run_all_regimes()
    return sim.run()
