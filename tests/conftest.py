"""Shared test fixtures for hyesg."""

import jax
import pytest

# Enable float64 for all tests
jax.config.update("jax_enable_x64", True)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --run-slow CLI flag."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (skipped by default).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip @pytest.mark.slow tests unless --run-slow is passed."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def rng_key():
    """Provide a deterministic JAX PRNG key for tests."""
    return jax.random.PRNGKey(42)
