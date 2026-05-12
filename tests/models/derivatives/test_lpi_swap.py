"""Tests for LPI swap pricing."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.derivatives.lpi_swap import (
    EquilibriumSwapRateProcessor,
    LPISwapConfig,
    LPISwapPricer,
)

jax.config.update("jax_enable_x64", True)


class TestCappedFlooredInflation:
    """Tests for cap/floor clipping of inflation rates."""

    def test_clips_above_cap(self) -> None:
        """Rates above cap are clipped to cap."""
        pricer = LPISwapPricer()
        rates = jnp.array([0.06, 0.07, 0.10])
        result = pricer.capped_floored_inflation(rates, cap=0.05, floor=0.0)
        assert jnp.all(result <= 0.05)
        assert float(result[0]) == pytest.approx(0.05)

    def test_clips_below_floor(self) -> None:
        """Rates below floor are clipped to floor."""
        pricer = LPISwapPricer()
        rates = jnp.array([-0.02, -0.01, -0.05])
        result = pricer.capped_floored_inflation(rates, cap=0.05, floor=0.0)
        assert jnp.all(result >= 0.0)
        assert float(result[0]) == pytest.approx(0.0)

    def test_passthrough_in_range(self) -> None:
        """Rates within [floor, cap] pass through unchanged."""
        pricer = LPISwapPricer()
        rates = jnp.array([0.01, 0.02, 0.03])
        result = pricer.capped_floored_inflation(rates, cap=0.05, floor=0.0)
        assert jnp.allclose(result, rates)

    def test_custom_cap_floor(self) -> None:
        """Custom cap/floor bounds are respected."""
        pricer = LPISwapPricer()
        rates = jnp.array([-0.05, 0.10, 0.03])
        result = pricer.capped_floored_inflation(rates, cap=0.03, floor=-0.01)
        assert float(result[0]) == pytest.approx(-0.01)
        assert float(result[1]) == pytest.approx(0.03)
        assert float(result[2]) == pytest.approx(0.03)


class TestLPISwapPricer:
    """Tests for LPI swap pricing."""

    def test_flat_inflation_price(self) -> None:
        """With constant inflation within bounds, price is deterministic."""
        pricer = LPISwapPricer()
        config = LPISwapConfig(cap=0.05, floor=0.0, maturity=5.0, notional=1.0)
        n_trials = 100
        n_steps = 5
        # Constant 2% inflation, no discounting
        inflation_paths = jnp.full((n_trials, n_steps), 0.02)
        discount_factors = jnp.ones(n_steps)

        price = pricer.price_lpi_swap(inflation_paths, discount_factors, config)
        # Expected: (1.02^5 - 1) * 1.0
        expected = (1.02**5 - 1.0) * 1.0
        assert float(price) == pytest.approx(expected, rel=1e-6)

    def test_zero_inflation_zero_price(self) -> None:
        """With zero inflation, fair rate should be near zero."""
        pricer = LPISwapPricer()
        config = LPISwapConfig(cap=0.05, floor=-0.01, maturity=5.0)
        n_trials = 50
        n_steps = 5
        inflation_paths = jnp.zeros((n_trials, n_steps))
        discount_factors = jnp.ones(n_steps)

        price = pricer.price_lpi_swap(inflation_paths, discount_factors, config)
        assert float(price) == pytest.approx(0.0, abs=1e-10)

    def test_capped_inflation_reduces_price(self) -> None:
        """Capping high inflation should reduce price vs uncapped."""
        pricer = LPISwapPricer()
        n_trials = 50
        n_steps = 5
        inflation_paths = jnp.full((n_trials, n_steps), 0.08)
        discount_factors = jnp.ones(n_steps)

        config_wide = LPISwapConfig(cap=0.10, floor=0.0)
        config_tight = LPISwapConfig(cap=0.05, floor=0.0)

        price_wide = pricer.price_lpi_swap(
            inflation_paths, discount_factors, config_wide
        )
        price_tight = pricer.price_lpi_swap(
            inflation_paths, discount_factors, config_tight
        )
        assert float(price_tight) < float(price_wide)

    def test_notional_scaling(self) -> None:
        """Price should scale with notional."""
        pricer = LPISwapPricer()
        n_trials = 50
        n_steps = 5
        inflation_paths = jnp.full((n_trials, n_steps), 0.03)
        discount_factors = jnp.ones(n_steps)

        config1 = LPISwapConfig(notional=1.0)
        config2 = LPISwapConfig(notional=2.0)

        p1 = pricer.price_lpi_swap(inflation_paths, discount_factors, config1)
        p2 = pricer.price_lpi_swap(inflation_paths, discount_factors, config2)
        assert float(p2) == pytest.approx(2.0 * float(p1), rel=1e-10)


class TestEquilibriumSwapRateProcessor:
    """Tests for equilibrium swap rate processing."""

    def test_zero_premium_passthrough(self) -> None:
        """Zero premium → output equals input."""
        proc = EquilibriumSwapRateProcessor(liquidity_premium=0.0)
        result = proc.process(jnp.float64(0.03), market_rate=0.035)
        assert float(result) == pytest.approx(0.03, abs=1e-12)

    def test_positive_premium_adds(self) -> None:
        """Positive premium is added to MC rate."""
        proc = EquilibriumSwapRateProcessor(liquidity_premium=0.005)
        result = proc.process(jnp.float64(0.03), market_rate=0.035)
        assert float(result) == pytest.approx(0.035, abs=1e-12)

    def test_negative_premium(self) -> None:
        """Negative premium reduces the rate."""
        proc = EquilibriumSwapRateProcessor(liquidity_premium=-0.002)
        result = proc.process(jnp.float64(0.03), market_rate=0.028)
        assert float(result) == pytest.approx(0.028, abs=1e-12)
