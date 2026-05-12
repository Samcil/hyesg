"""Tests for FX forward pricing, currency hedging, and transaction costs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.models.exchange_rates.forward import (
    ConstantBidOfferSpread,
    FCAForwardPricer,
    FXForward,
)
from hyesg.models.exchange_rates.hedging import (
    CurrencyHedgedEquityRebalancer,
    HedgeState,
)

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════
# Forward Pricer Tests
# ═══════════════════════════════════════════════════


class TestFCAForwardPricer:
    """Tests for the covered interest rate parity forward pricer."""

    @pytest.fixture
    def pricer(self) -> FCAForwardPricer:
        return FCAForwardPricer()

    def test_forward_rate_parity(self, pricer: FCAForwardPricer) -> None:
        """F = S × P_f / P_d — basic CIP relationship."""
        spot = jnp.array(1.5, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.97, dtype=jnp.float64)

        fwd = pricer.forward_rate(spot, p_d, p_f)
        expected = 1.5 * 0.97 / 0.95
        assert jnp.isclose(fwd, expected, atol=1e-12)

    def test_forward_equals_spot_when_rates_equal(
        self, pricer: FCAForwardPricer
    ) -> None:
        """When P_d = P_f, forward = spot (no interest rate differential)."""
        spot = jnp.array(1.25, dtype=jnp.float64)
        p = jnp.array(0.90, dtype=jnp.float64)

        fwd = pricer.forward_rate(spot, p, p)
        assert jnp.isclose(fwd, spot, atol=1e-12)

    def test_forward_greater_than_spot_when_domestic_rate_higher(
        self, pricer: FCAForwardPricer
    ) -> None:
        """Higher domestic rate → P_d < P_f → F > S."""
        spot = jnp.array(1.0, dtype=jnp.float64)
        # Domestic rate higher → lower domestic ZCB price
        p_d = jnp.array(0.90, dtype=jnp.float64)
        p_f = jnp.array(0.95, dtype=jnp.float64)

        fwd = pricer.forward_rate(spot, p_d, p_f)
        assert fwd > spot

    def test_forward_less_than_spot_when_foreign_rate_higher(
        self, pricer: FCAForwardPricer
    ) -> None:
        """Higher foreign rate → P_f < P_d → F < S."""
        spot = jnp.array(1.0, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.90, dtype=jnp.float64)

        fwd = pricer.forward_rate(spot, p_d, p_f)
        assert fwd < spot

    def test_mark_to_market_at_inception(
        self, pricer: FCAForwardPricer
    ) -> None:
        """MTM = 0 when strike equals current forward rate."""
        spot = jnp.array(1.5, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.97, dtype=jnp.float64)

        fwd = pricer.forward_rate(spot, p_d, p_f)
        mtm = pricer.mark_to_market(spot, p_d, p_f, strike=fwd)
        assert jnp.isclose(mtm, 0.0, atol=1e-12)

    def test_mark_to_market_in_the_money(
        self, pricer: FCAForwardPricer
    ) -> None:
        """MTM > 0 when current forward > strike (buyer benefits)."""
        spot = jnp.array(1.5, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.97, dtype=jnp.float64)

        strike = jnp.array(1.4, dtype=jnp.float64)
        mtm = pricer.mark_to_market(spot, p_d, p_f, strike)

        fwd = pricer.forward_rate(spot, p_d, p_f)
        expected = p_d * (fwd - strike)
        assert jnp.isclose(mtm, expected, atol=1e-12)
        assert mtm > 0

    def test_mark_to_market_out_of_the_money(
        self, pricer: FCAForwardPricer
    ) -> None:
        """MTM < 0 when current forward < strike."""
        spot = jnp.array(1.0, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.90, dtype=jnp.float64)

        strike = jnp.array(1.1, dtype=jnp.float64)
        mtm = pricer.mark_to_market(spot, p_d, p_f, strike)
        assert mtm < 0

    def test_forward_rate_is_jit_compatible(
        self, pricer: FCAForwardPricer
    ) -> None:
        """Forward pricing must work under jax.jit."""

        @jax.jit
        def compute(s, pd, pf):
            return pricer.forward_rate(s, pd, pf)

        spot = jnp.array(1.5, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.97, dtype=jnp.float64)

        result = compute(spot, p_d, p_f)
        expected = 1.5 * 0.97 / 0.95
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_mark_to_market_at_expiry(
        self, pricer: FCAForwardPricer
    ) -> None:
        """At expiry P_d(T,T) = P_f(T,T) = 1, so MTM = S(T) − K."""
        spot = jnp.array(1.6, dtype=jnp.float64)
        p_d = jnp.array(1.0, dtype=jnp.float64)
        p_f = jnp.array(1.0, dtype=jnp.float64)
        strike = jnp.array(1.5, dtype=jnp.float64)

        mtm = pricer.mark_to_market(spot, p_d, p_f, strike)
        assert jnp.isclose(mtm, 0.1, atol=1e-12)


# ═══════════════════════════════════════════════════
# Transaction Cost Tests
# ═══════════════════════════════════════════════════


class TestConstantBidOfferSpread:
    """Tests for the bid-offer spread transaction cost model."""

    def test_zero_spread(self) -> None:
        """Zero spread → zero cost."""
        tc = ConstantBidOfferSpread(spread_bps=0.0)
        notional = jnp.array(1_000_000.0, dtype=jnp.float64)
        assert jnp.isclose(tc.cost(notional), 0.0, atol=1e-15)

    def test_5bps_spread(self) -> None:
        """5 bps half-spread on 1M notional → 50.0 cost."""
        tc = ConstantBidOfferSpread(spread_bps=5.0)
        notional = jnp.array(1_000_000.0, dtype=jnp.float64)
        expected = 1_000_000.0 * 5.0 / 10_000.0
        assert jnp.isclose(tc.cost(notional), expected, atol=1e-10)

    def test_negative_notional(self) -> None:
        """Cost is based on |notional|."""
        tc = ConstantBidOfferSpread(spread_bps=10.0)
        notional = jnp.array(-500_000.0, dtype=jnp.float64)
        expected = 500_000.0 * 10.0 / 10_000.0
        assert jnp.isclose(tc.cost(notional), expected, atol=1e-10)

    def test_half_spread_property(self) -> None:
        """half_spread returns decimal fraction."""
        tc = ConstantBidOfferSpread(spread_bps=5.0)
        assert tc.half_spread == pytest.approx(0.0005)


# ═══════════════════════════════════════════════════
# FX Forward Model Tests
# ═══════════════════════════════════════════════════


class TestFXForward:
    """Tests for the FX forward simulation model."""

    @pytest.fixture
    def model(self) -> FXForward:
        return FXForward(
            name="gbp_usd_fwd",
            spot_fx_model="fx_usd",
            domestic_rate_model="gbp_nominal",
            foreign_rate_model="usd_nominal",
            tenors=(0.25, 0.5, 1.0),
        )

    def test_name(self, model: FXForward) -> None:
        assert model.name == "gbp_usd_fwd"

    def test_tenors(self, model: FXForward) -> None:
        assert model.tenors == (0.25, 0.5, 1.0)

    def test_step_with_deps(self, model: FXForward) -> None:
        """Step produces forward rates from dep outputs."""
        deps = {
            "fx_usd": {"level": jnp.array(1.5, dtype=jnp.float64)},
            "gbp_nominal": {
                "zcb_0.25": jnp.array(0.99, dtype=jnp.float64),
                "zcb_0.5": jnp.array(0.98, dtype=jnp.float64),
                "zcb_1.0": jnp.array(0.95, dtype=jnp.float64),
            },
            "usd_nominal": {
                "zcb_0.25": jnp.array(0.985, dtype=jnp.float64),
                "zcb_0.5": jnp.array(0.97, dtype=jnp.float64),
                "zcb_1.0": jnp.array(0.94, dtype=jnp.float64),
            },
        }

        outputs = model.step(t=0.0, dt=1 / 12, deps=deps)

        assert "spot" in outputs
        assert "forward_0.25" in outputs
        assert "forward_0.5" in outputs
        assert "forward_1.0" in outputs

        # Check CIP: F = S × P_f / P_d
        expected_1y = 1.5 * 0.94 / 0.95
        assert jnp.isclose(outputs["forward_1.0"], expected_1y, atol=1e-12)

    def test_step_no_deps(self) -> None:
        """With no deps, forward = spot (both ZCBs default to 1)."""
        model = FXForward(name="test")
        outputs = model.step(t=0.0, dt=1 / 12, deps={})
        for tenor in (0.25, 0.5, 1.0):
            assert jnp.isclose(
                outputs[f"forward_{tenor}"], outputs["spot"], atol=1e-12
            )

    def test_forward_rates_method(self, model: FXForward) -> None:
        """Direct forward_rates method with explicit ZCB dicts."""
        spot = jnp.array(1.3, dtype=jnp.float64)
        d_zcbs = {
            0.25: jnp.array(0.99, dtype=jnp.float64),
            0.5: jnp.array(0.98, dtype=jnp.float64),
            1.0: jnp.array(0.95, dtype=jnp.float64),
        }
        f_zcbs = {
            0.25: jnp.array(0.985, dtype=jnp.float64),
            0.5: jnp.array(0.97, dtype=jnp.float64),
            1.0: jnp.array(0.93, dtype=jnp.float64),
        }

        fwds = model.forward_rates(spot, d_zcbs, f_zcbs)

        for tenor in (0.25, 0.5, 1.0):
            expected = 1.3 * f_zcbs[tenor] / d_zcbs[tenor]
            assert jnp.isclose(fwds[tenor], expected, atol=1e-12)


# ═══════════════════════════════════════════════════
# Currency Hedging Tests
# ═══════════════════════════════════════════════════


class TestCurrencyHedgedEquityRebalancer:
    """Tests for the rolling FX hedge rebalancer."""

    @pytest.fixture
    def hedger(self) -> CurrencyHedgedEquityRebalancer:
        return CurrencyHedgedEquityRebalancer(
            hedge_ratio=0.5,
            roll_frequency_months=12,
            transaction_cost=ConstantBidOfferSpread(spread_bps=5.0),
        )

    @pytest.fixture
    def full_hedger(self) -> CurrencyHedgedEquityRebalancer:
        return CurrencyHedgedEquityRebalancer(
            hedge_ratio=1.0,
            roll_frequency_months=12,
            transaction_cost=ConstantBidOfferSpread(spread_bps=0.0),
        )

    def test_invalid_hedge_ratio(self) -> None:
        """hedge_ratio must be in [0, 1]."""
        with pytest.raises(ValueError, match="hedge_ratio"):
            CurrencyHedgedEquityRebalancer(hedge_ratio=1.5)

    def test_invalid_roll_frequency(self) -> None:
        """roll_frequency_months must be 1, 3, 6, or 12."""
        with pytest.raises(ValueError, match="roll_frequency_months"):
            CurrencyHedgedEquityRebalancer(roll_frequency_months=2)

    def test_hedge_ratio_property(self, hedger: CurrencyHedgedEquityRebalancer) -> None:
        assert hedger.hedge_ratio == 0.5

    def test_roll_tenor_property(self, hedger: CurrencyHedgedEquityRebalancer) -> None:
        assert hedger.roll_tenor == pytest.approx(1.0)

    def test_unhedged_return(self) -> None:
        """h=0 → hedged return = fully unhedged return."""
        h = CurrencyHedgedEquityRebalancer(hedge_ratio=0.0)
        eq_ret = jnp.array(0.05, dtype=jnp.float64)
        fx_ret = jnp.array(0.02, dtype=jnp.float64)
        fwd_prem = jnp.array(0.01, dtype=jnp.float64)

        result = h.hedge_return(eq_ret, fx_ret, fwd_prem)
        expected = (1.05) * (1.02) - 1.0
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_fully_hedged_return(
        self, full_hedger: CurrencyHedgedEquityRebalancer
    ) -> None:
        """h=1.0 → FX return replaced by forward premium."""
        eq_ret = jnp.array(0.05, dtype=jnp.float64)
        fx_ret = jnp.array(0.02, dtype=jnp.float64)
        fwd_prem = jnp.array(0.01, dtype=jnp.float64)

        result = full_hedger.hedge_return(eq_ret, fx_ret, fwd_prem)
        unhedged = 1.05 * 1.02 - 1.0
        hedge_gain = 1.0 * (0.01 - 0.02)
        expected = unhedged + hedge_gain
        assert jnp.isclose(result, expected, atol=1e-12)

    def test_zero_fx_return_no_hedge_impact(
        self, full_hedger: CurrencyHedgedEquityRebalancer
    ) -> None:
        """When FX is flat and forward = 0, hedged = unhedged."""
        eq_ret = jnp.array(0.05, dtype=jnp.float64)
        fx_ret = jnp.array(0.0, dtype=jnp.float64)
        fwd_prem = jnp.array(0.0, dtype=jnp.float64)

        result = full_hedger.hedge_return(eq_ret, fx_ret, fwd_prem)
        assert jnp.isclose(result, 0.05, atol=1e-12)

    def test_pnl_decomposition(
        self, hedger: CurrencyHedgedEquityRebalancer
    ) -> None:
        """P&L decomposition adds up correctly."""
        eq_ret = jnp.array(0.05, dtype=jnp.float64)
        fx_ret = jnp.array(0.03, dtype=jnp.float64)
        fwd_prem = jnp.array(0.01, dtype=jnp.float64)
        notional = jnp.array(1_000_000.0, dtype=jnp.float64)

        pnl = hedger.compute_pnl_decomposition(
            eq_ret, fx_ret, fwd_prem, notional
        )

        assert "unhedged_return" in pnl
        assert "hedge_gain" in pnl
        assert "transaction_cost" in pnl
        assert "hedged_return" in pnl

        # Verify decomposition sums correctly
        expected_hedged = (
            pnl["unhedged_return"] + pnl["hedge_gain"] - pnl["transaction_cost"]
        )
        assert jnp.isclose(pnl["hedged_return"], expected_hedged, atol=1e-12)

        # Transaction cost: 0.5 × 1M × 5bps = 250
        expected_tc = 0.5 * 1_000_000.0 * 5.0 / 10_000.0
        assert jnp.isclose(pnl["transaction_cost"], expected_tc, atol=1e-10)

    def test_should_roll_monthly(self) -> None:
        """Monthly roll: every step for monthly simulation."""
        h = CurrencyHedgedEquityRebalancer(roll_frequency_months=1)
        for i in range(12):
            assert h.should_roll(i, steps_per_year=12)

    def test_should_roll_quarterly(self) -> None:
        """Quarterly roll: every 3rd step for monthly simulation."""
        h = CurrencyHedgedEquityRebalancer(roll_frequency_months=3)
        for i in range(12):
            expected = i % 3 == 0
            assert h.should_roll(i, steps_per_year=12) == expected

    def test_should_roll_annually(self) -> None:
        """Annual roll: every 12th step for monthly simulation."""
        h = CurrencyHedgedEquityRebalancer(roll_frequency_months=12)
        for i in range(24):
            expected = i % 12 == 0
            assert h.should_roll(i, steps_per_year=12) == expected

    def test_forward_premium_computation(
        self, hedger: CurrencyHedgedEquityRebalancer
    ) -> None:
        """Forward premium = P_f/P_d − 1."""
        spot = jnp.array(1.5, dtype=jnp.float64)
        p_d = jnp.array(0.95, dtype=jnp.float64)
        p_f = jnp.array(0.97, dtype=jnp.float64)

        premium = hedger.compute_forward_premium(spot, p_d, p_f)
        # F/S = P_f/P_d so premium = P_f/P_d - 1
        expected = (1.5 * 0.97 / 0.95) / 1.5 - 1.0
        assert jnp.isclose(premium, expected, atol=1e-12)

    def test_hedging_reduces_fx_volatility(self) -> None:
        """Hedge dampens the impact of FX moves on total return."""
        full = CurrencyHedgedEquityRebalancer(hedge_ratio=1.0)
        none = CurrencyHedgedEquityRebalancer(hedge_ratio=0.0)

        eq_ret = jnp.array(0.0, dtype=jnp.float64)
        fwd_prem = jnp.array(0.0, dtype=jnp.float64)

        # Large positive FX move
        fx_up = jnp.array(0.10, dtype=jnp.float64)
        # Large negative FX move
        fx_down = jnp.array(-0.10, dtype=jnp.float64)

        # Unhedged: full FX exposure
        unhedged_range = float(
            none.hedge_return(eq_ret, fx_up, fwd_prem)
            - none.hedge_return(eq_ret, fx_down, fwd_prem)
        )

        # Fully hedged: no FX exposure
        hedged_range = float(
            full.hedge_return(eq_ret, fx_up, fwd_prem)
            - full.hedge_return(eq_ret, fx_down, fwd_prem)
        )

        assert hedged_range < unhedged_range


# ═══════════════════════════════════════════════════
# Hedge State Tests
# ═══════════════════════════════════════════════════


class TestHedgeState:
    """Tests for the hedge state container."""

    def test_default_state(self) -> None:
        state = HedgeState()
        assert state.forward_strike == 0.0
        assert state.periods_to_roll == 0
        assert state.cum_hedge_pnl == 0.0
        assert state.cum_transaction_costs == 0.0

    def test_custom_state(self) -> None:
        state = HedgeState(
            forward_strike=1.5,
            periods_to_roll=3,
            cum_hedge_pnl=100.0,
            cum_transaction_costs=5.0,
        )
        assert state.forward_strike == 1.5
        assert state.periods_to_roll == 3
        assert state.cum_hedge_pnl == 100.0
        assert state.cum_transaction_costs == 5.0


# ═══════════════════════════════════════════════════
# Protocol Conformance Tests
# ═══════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify protocol implementation at runtime."""

    def test_pricer_protocol(self) -> None:
        from hyesg.models.exchange_rates.forward import FXForwardPricer

        assert isinstance(FCAForwardPricer(), FXForwardPricer)

    def test_transaction_cost_protocol(self) -> None:
        from hyesg.models.exchange_rates.forward import TransactionCostModel

        assert isinstance(ConstantBidOfferSpread(), TransactionCostModel)

    def test_hedger_protocol(self) -> None:
        from hyesg.models.exchange_rates.hedging import CurrencyHedger

        assert isinstance(CurrencyHedgedEquityRebalancer(), CurrencyHedger)


# ═══════════════════════════════════════════════════
# Config Tests
# ═══════════════════════════════════════════════════


class TestFXForwardConfig:
    """Tests for the FX forward and hedge configuration schemas."""

    def test_fx_forward_config(self) -> None:
        from hyesg.config.params import FXForwardConfig

        cfg = FXForwardConfig(
            spot_fx_model="fx_usd",
            domestic_rate_model="gbp_nom",
            foreign_rate_model="usd_nom",
            tenors=(0.25, 1.0),
        )
        assert cfg.spot_fx_model == "fx_usd"
        assert cfg.tenors == (0.25, 1.0)

    def test_currency_hedge_config_defaults(self) -> None:
        from hyesg.config.params import CurrencyHedgeConfig

        cfg = CurrencyHedgeConfig()
        assert cfg.hedge_ratio == 1.0
        assert cfg.roll_frequency_months == 12
        assert cfg.spread_bps == 5.0

    def test_currency_hedge_config_validation(self) -> None:
        from pydantic import ValidationError

        from hyesg.config.params import CurrencyHedgeConfig

        with pytest.raises(ValidationError):
            CurrencyHedgeConfig(hedge_ratio=2.0)

        with pytest.raises(ValidationError):
            CurrencyHedgeConfig(roll_frequency_months=2)

    def test_config_is_frozen(self) -> None:
        from pydantic import ValidationError

        from hyesg.config.params import FXForwardConfig

        cfg = FXForwardConfig()
        with pytest.raises(ValidationError):
            cfg.spot_fx_model = "changed"  # type: ignore[misc]
