"""hyesg — Python/JAX Economic Scenario Generator."""

__version__ = "0.1.0"

# Enable float64 for financial precision
import jax

jax.config.update("jax_enable_x64", True)
