"""Integration tests for CreditSystemConfig and credit system assembly."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import (
    CreditClassConfig,
    CreditDynamicsParams,
    CreditSystemConfig,
    IntensityTransformConfig,
    LiquidityCIRParams,
    PooledBondParams,
)
from hyesg.models.credit.intensity_transform import SplineIntensityTransform
from hyesg.models.credit.liquidity import CIRLiquidityProcess

# Enable float64
jax.config.update("jax_enable_x64", True)

# C# reference knots
KNOT_XS = [0, 0.02, 0.06, 0.09, 0.12, 0.2, 0.5, 1, 10]
KNOT_YS = [
    0,
    0.00450450,
    0.03009029,
    0.05256794,
    0.07213938,
    0.12588947,
    0.33689402,
    0.70219276,
    7.70218303,
]

# C# reference credit class parameters
CREDIT_DYNAMICS = {
    "AAA": CreditDynamicsParams(
        x0=0.01, mu=0.0255, alpha=0.0241, sigma=0.0682,
        rw_expectation=0.0101, rw_variance=0.000103,
        phi_knots=[0, 0, 0, 0],
    ),
    "AA": CreditDynamicsParams(
        x0=0.015, mu=0.0388, alpha=0.0241, sigma=0.0842,
        rw_expectation=0.0155, rw_variance=0.000239,
        phi_knots=[0, 0, 0, 0],
    ),
    "A": CreditDynamicsParams(
        x0=0.02, mu=0.0562, alpha=0.0241, sigma=0.0884,
        rw_expectation=0.0224, rw_variance=0.000381,
        phi_knots=[0, 0, 0, 0],
    ),
    "BBB": CreditDynamicsParams(
        x0=0.03, mu=0.0777, alpha=0.0241, sigma=0.0928,
        rw_expectation=0.0309, rw_variance=0.000582,
        phi_knots=[0, 0, 0, 0],
    ),
    "BB": CreditDynamicsParams(
        x0=0.05, mu=0.1344, alpha=0.0241, sigma=0.0975,
        rw_expectation=0.0535, rw_variance=0.001109,
        phi_knots=[0, 0, 0, 0],
    ),
    "B": CreditDynamicsParams(
        x0=0.08, mu=0.2049, alpha=0.0241, sigma=0.1024,
        rw_expectation=0.0816, rw_variance=0.001864,
        phi_knots=[0, 0, 0, 0],
    ),
    "CCC": CreditDynamicsParams(
        x0=0.15, mu=0.4280, alpha=0.0241, sigma=0.1075,
        rw_expectation=0.1704, rw_variance=0.004293,
        phi_knots=[0, 0, 0, 0],
    ),
}

POOL_PARAMS = {
    "AAA": PooledBondParams(activation_rate=0.0133, initial_active_issuers=10, total_pool_size=12),
    "AA": PooledBondParams(activation_rate=0.0695, initial_active_issuers=10, total_pool_size=17),
    "A": PooledBondParams(activation_rate=0.1367, initial_active_issuers=10, total_pool_size=24),
    "BBB": PooledBondParams(activation_rate=0.2133, initial_active_issuers=10, total_pool_size=32),
    "BB": PooledBondParams(activation_rate=0.4076, initial_active_issuers=10, total_pool_size=51),
    "B": PooledBondParams(activation_rate=0.8793, initial_active_issuers=10, total_pool_size=98),
    "CCC": PooledBondParams(activation_rate=1.5428, initial_active_issuers=10, total_pool_size=166),
}


def _build_system_config() -> CreditSystemConfig:
    """Build a full CreditSystemConfig from C# reference data."""
    credit_classes = {}
    for name, dynamics in CREDIT_DYNAMICS.items():
        credit_classes[name] = CreditClassConfig(
            dynamics=dynamics,
            pool=POOL_PARAMS.get(name),
        )

    return CreditSystemConfig(
        credit_classes=credit_classes,
        rn_rw_transform=IntensityTransformConfig(
            knot_xs=KNOT_XS,
            knot_ys=KNOT_YS,
        ),
        recovery_rate=0.35,
        recovery_type="treasury",
        currencies=["GBP", "USD", "EUR"],
        liquidity_medium=LiquidityCIRParams(
            x0=0.04, mu=0.1, alpha=0.0225, sigma=0.1,
            rw_expectation=0.02, rw_variance=0.000200,
        ),
        liquidity_low=LiquidityCIRParams(
            x0=0.08, mu=0.3, alpha=0.0225, sigma=0.12,
            rw_expectation=0.04, rw_variance=0.000720,
        ),
        liquidity_rn_scale=0.1,
    )


class TestCreditSystemConfig:
    """Integration tests for credit system configuration."""

    def test_all_seven_classes(self) -> None:
        """System config should have all 7 credit classes."""
        config = _build_system_config()
        expected_classes = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC"}
        assert set(config.credit_classes.keys()) == expected_classes

    def test_three_liquidity_tiers(self) -> None:
        """System should support 3 liquidity tiers (High=None, Medium, Low)."""
        config = _build_system_config()
        # High tier has no liquidity process (None)
        # Medium and Low tiers are configured
        assert config.liquidity_medium is not None
        assert config.liquidity_low is not None
        assert config.liquidity_medium.x0 == 0.04
        assert config.liquidity_low.x0 == 0.08

    def test_rn_rw_transform_construction(self) -> None:
        """Should be able to construct a SplineIntensityTransform from config."""
        config = _build_system_config()
        transform = SplineIntensityTransform(
            config.rn_rw_transform.knot_xs,
            config.rn_rw_transform.knot_ys,
        )
        # Test a known knot value
        result = transform.transform(jnp.array(0.2))
        assert jnp.isclose(result, 0.12588947, atol=1e-6)

    def test_pool_parameters_per_class(self) -> None:
        """Each class should have correct pool parameters."""
        config = _build_system_config()

        aaa = config.credit_classes["AAA"]
        assert aaa.pool is not None
        assert aaa.pool.activation_rate == 0.0133
        assert aaa.pool.initial_active_issuers == 10
        assert aaa.pool.total_pool_size == 12

        ccc = config.credit_classes["CCC"]
        assert ccc.pool is not None
        assert ccc.pool.activation_rate == 1.5428
        assert ccc.pool.total_pool_size == 166

    def test_recovery_params(self) -> None:
        """Recovery rate and type should be set correctly."""
        config = _build_system_config()
        assert config.recovery_rate == 0.35
        assert config.recovery_type == "treasury"

    def test_currencies(self) -> None:
        """Should include GBP, USD, EUR."""
        config = _build_system_config()
        assert config.currencies == ["GBP", "USD", "EUR"]

    def test_liquidity_process_from_config(self) -> None:
        """Should be able to construct CIRLiquidityProcess from config."""
        config = _build_system_config()
        transform = SplineIntensityTransform(
            config.rn_rw_transform.knot_xs,
            config.rn_rw_transform.knot_ys,
        )

        assert config.liquidity_medium is not None
        medium = CIRLiquidityProcess(
            alpha=config.liquidity_medium.alpha,
            mu=config.liquidity_medium.mu,
            sigma=config.liquidity_medium.sigma,
            x0=config.liquidity_medium.x0,
            rn_transform=transform,
            scale_factor=config.liquidity_rn_scale,
            recovery_rate=config.liquidity_medium.liquidity_recovery_rate,
        )

        key = jax.random.PRNGKey(0)
        state = medium.init_state(key)
        assert jnp.isclose(state.intensity, 0.04, atol=1e-12)

    def test_dynamics_params_frozen(self) -> None:
        """CreditDynamicsParams should be immutable (frozen)."""
        params = CREDIT_DYNAMICS["AAA"]
        with pytest.raises(Exception):  # Pydantic frozen validation error
            params.mu = 0.999  # type: ignore[misc]

    def test_pool_validation(self) -> None:
        """Pool size must be >= initial active."""
        with pytest.raises(ValueError, match="total_pool_size"):
            PooledBondParams(
                activation_rate=0.1,
                initial_active_issuers=20,
                total_pool_size=5,
            )

    def test_transform_config_validation(self) -> None:
        """Transform knot arrays must have equal length."""
        with pytest.raises(ValueError, match="knot_xs length"):
            IntensityTransformConfig(
                knot_xs=[0, 1, 2],
                knot_ys=[0, 1],
            )

    def test_all_classes_have_shared_alpha(self) -> None:
        """All credit classes should share the same alpha (0.0241)."""
        config = _build_system_config()
        for name, cc in config.credit_classes.items():
            assert cc.dynamics.alpha == 0.0241, (
                f"{name} has alpha={cc.dynamics.alpha}, expected 0.0241"
            )

    def test_mu_increases_with_lower_rating(self) -> None:
        """Long-run mean intensity should increase with lower rating."""
        config = _build_system_config()
        order = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        mus = [config.credit_classes[name].dynamics.mu for name in order]
        for i in range(1, len(mus)):
            assert mus[i] > mus[i - 1], (
                f"mu for {order[i]} ({mus[i]}) should be > "
                f"mu for {order[i-1]} ({mus[i-1]})"
            )
