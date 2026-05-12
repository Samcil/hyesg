"""Tests for the SVJD (Stochastic Volatility Jump Diffusion) composition.

Tests cover:
- CIRVolAdapter: volatility process protocol compliance
- ConstantJumpAdapter / ZeroJumpAdapter: jump process protocol compliance
- svjd_equity_step: pure composition function with known inputs
- SVJDEquity: full model step matching Model protocol
- Zero-jump SVJD reduces to stochastic vol GBM
- Jump-adjusted sigma is applied correctly at initialization
- Backward compatibility: refactored Equity with vol/jump deps
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.config.params import GBMParams
from hyesg.core.types import FXState, JumpState, VolState
from hyesg.math.jump_utils import jump_adjusted_sigma
from hyesg.models.equity.svjd import (
    CIRVolAdapter,
    ConstantJumpAdapter,
    SVJDEquity,
    ZeroJumpAdapter,
    svjd_equity_step,
)

jax.config.update("jax_enable_x64", True)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def vol_adapter() -> CIRVolAdapter:
    """CIR vol adapter with typical calibration params."""
    return CIRVolAdapter(alpha=1.5, sigma=0.3, mu=0.04)


@pytest.fixture
def jump_adapter() -> ConstantJumpAdapter:
    """Constant-intensity jump adapter with Merton-style params."""
    return ConstantJumpAdapter(
        intensity=0.5, jump_mean=-0.02, jump_vol=0.08, max_k=20
    )


@pytest.fixture
def zero_jump() -> ZeroJumpAdapter:
    """Null-object jump adapter."""
    return ZeroJumpAdapter()


@pytest.fixture
def vol_state() -> VolState:
    """Initial vol state with variance = 0.04 (vol = 0.2)."""
    return VolState(
        variance=jnp.array(0.04, dtype=jnp.float64),
        volatility=jnp.array(0.2, dtype=jnp.float64),
    )


@pytest.fixture
def jump_state() -> JumpState:
    """Fresh jump state (no cumulative jumps)."""
    return JumpState(
        cum_jumps=jnp.array(0.0, dtype=jnp.float64),
        n_jumps=jnp.array(0.0, dtype=jnp.float64),
        last_jump_size=jnp.array(0.0, dtype=jnp.float64),
    )


# ── CIRVolAdapter Tests ─────────────────────────────────────────────────


class TestCIRVolAdapter:
    def test_step_returns_vol_state(
        self, vol_adapter: CIRVolAdapter, vol_state: VolState
    ) -> None:
        """step_vol returns (VolState, current_vol)."""
        dt = 1 / 12
        dW = jnp.array(0.5, dtype=jnp.float64)
        new_state, sigma = vol_adapter.step_vol(vol_state, 0.0, dt, dW)
        assert isinstance(new_state, VolState)
        assert jnp.isfinite(sigma)

    def test_uses_start_of_step_vol(
        self, vol_adapter: CIRVolAdapter, vol_state: VolState
    ) -> None:
        """Current vol returned is from START of step (pre-evolution)."""
        dt = 1 / 12
        dW = jnp.array(0.0, dtype=jnp.float64)
        _, sigma = vol_adapter.step_vol(vol_state, 0.0, dt, dW)
        assert jnp.isclose(sigma, 0.2, atol=1e-12)

    def test_mean_reversion(
        self, vol_adapter: CIRVolAdapter
    ) -> None:
        """With zero noise, variance mean-reverts toward mu."""
        v_high = VolState(
            variance=jnp.array(0.10, dtype=jnp.float64),
            volatility=jnp.array(jnp.sqrt(0.10), dtype=jnp.float64),
        )
        dt = 1 / 12
        dW = jnp.array(0.0, dtype=jnp.float64)
        new_state, _ = vol_adapter.step_vol(v_high, 0.0, dt, dW)
        # Variance should move toward mu=0.04 (decrease)
        assert new_state.variance < v_high.variance

    def test_current_vol(
        self, vol_adapter: CIRVolAdapter, vol_state: VolState
    ) -> None:
        """current_vol extracts volatility from state."""
        assert jnp.isclose(vol_adapter.current_vol(vol_state), 0.2, atol=1e-12)

    def test_euler_step_formula(
        self, vol_adapter: CIRVolAdapter, vol_state: VolState
    ) -> None:
        """Verify exact CIR Euler step with known inputs."""
        dt = 0.25
        dW = jnp.array(1.0, dtype=jnp.float64)
        new_state, _ = vol_adapter.step_vol(vol_state, 0.0, dt, dW)

        # Manual: V_new = 0.04 + 1.5*(0.04 - 0.04)*0.25 + 0.3*sqrt(0.04*0.25)*1.0
        v = 0.04
        expected = v + 1.5 * (0.04 - v) * dt + 0.3 * jnp.sqrt(v * dt) * 1.0
        assert jnp.isclose(new_state.variance, expected, atol=1e-12)


# ── ConstantJumpAdapter Tests ────────────────────────────────────────────


class TestConstantJumpAdapter:
    def test_sample_jumps_no_event(
        self, jump_adapter: ConstantJumpAdapter, jump_state: JumpState
    ) -> None:
        """Very low uniform → no jump (N=0)."""
        dt = 1 / 12
        # Very low uniform → P(N=0) = exp(-0.5/12) ≈ 0.959, so u=0.01 → N=0
        uniform = jnp.array(0.01, dtype=jnp.float64)
        normal = jnp.array(0.5, dtype=jnp.float64)
        new_state, jump, drift_adj = jump_adapter.sample_jumps(
            jump_state, dt, uniform, normal
        )
        assert jnp.isclose(jump, 0.0, atol=1e-12)
        # Drift adjustment is still nonzero (compensator applies every step)
        assert jnp.isfinite(drift_adj)

    def test_sample_jumps_with_event(
        self, jump_adapter: ConstantJumpAdapter, jump_state: JumpState
    ) -> None:
        """High uniform → at least one jump."""
        dt = 1.0  # large dt increases P(N>0)
        uniform = jnp.array(0.99, dtype=jnp.float64)
        normal = jnp.array(0.0, dtype=jnp.float64)
        _, jump, drift_adj = jump_adapter.sample_jumps(
            jump_state, dt, uniform, normal
        )
        # Should have nonzero jump
        assert not jnp.isclose(jump, 0.0, atol=1e-15)
        assert jnp.isfinite(drift_adj)

    def test_drift_adjustment_formula(
        self, jump_adapter: ConstantJumpAdapter, jump_state: JumpState
    ) -> None:
        """Verify drift adjustment = -λ·κ·dt."""
        dt = 0.5
        uniform = jnp.array(0.01, dtype=jnp.float64)
        normal = jnp.array(0.0, dtype=jnp.float64)
        _, _, drift_adj = jump_adapter.sample_jumps(
            jump_state, dt, uniform, normal
        )
        kappa = jnp.exp(-0.02 + 0.5 * 0.08**2) - 1.0
        expected = -0.5 * kappa * dt
        assert jnp.isclose(drift_adj, expected, atol=1e-12)

    def test_properties(self, jump_adapter: ConstantJumpAdapter) -> None:
        assert jump_adapter.intensity == 0.5
        assert jump_adapter.jump_mean == -0.02
        assert jump_adapter.jump_vol == 0.08

    def test_cumulative_state_update(
        self, jump_adapter: ConstantJumpAdapter, jump_state: JumpState
    ) -> None:
        """Jump state accumulates over steps."""
        dt = 1.0
        uniform = jnp.array(0.99, dtype=jnp.float64)
        normal = jnp.array(0.5, dtype=jnp.float64)
        new_state, _, _ = jump_adapter.sample_jumps(
            jump_state, dt, uniform, normal
        )
        # n_jumps should increase
        assert new_state.n_jumps >= jump_state.n_jumps


# ── ZeroJumpAdapter Tests ────────────────────────────────────────────────


class TestZeroJumpAdapter:
    def test_sample_always_zero(
        self, zero_jump: ZeroJumpAdapter, jump_state: JumpState
    ) -> None:
        """Zero adapter always returns zero jump and drift."""
        dt = 1.0
        uniform = jnp.array(0.99, dtype=jnp.float64)
        normal = jnp.array(2.0, dtype=jnp.float64)
        _, jump, drift_adj = zero_jump.sample_jumps(
            jump_state, dt, uniform, normal
        )
        assert jnp.isclose(jump, 0.0, atol=1e-15)
        assert jnp.isclose(drift_adj, 0.0, atol=1e-15)

    def test_properties(self, zero_jump: ZeroJumpAdapter) -> None:
        assert zero_jump.intensity == 0.0
        assert zero_jump.jump_mean == 0.0
        assert zero_jump.jump_vol == 0.0


# ── svjd_equity_step Tests ───────────────────────────────────────────────


class TestSVJDEquityStep:
    def test_deterministic_step_no_vol_no_jump(self) -> None:
        """Zero shocks + zero jump → deterministic drift step."""
        vol_adapter = CIRVolAdapter(alpha=1.5, sigma=0.0, mu=0.04)
        zero_jump = ZeroJumpAdapter()
        log_price = jnp.log(jnp.array(100.0, dtype=jnp.float64))
        vol_state = VolState(
            variance=jnp.array(0.04, dtype=jnp.float64),
            volatility=jnp.array(0.2, dtype=jnp.float64),
        )
        jump_state = JumpState(
            cum_jumps=jnp.array(0.0, dtype=jnp.float64),
            n_jumps=jnp.array(0.0, dtype=jnp.float64),
            last_jump_size=jnp.array(0.0, dtype=jnp.float64),
        )
        drift = 0.05  # r - q
        dt = 1 / 12

        new_log, new_vol, new_jump, outputs = svjd_equity_step(
            log_price=log_price,
            vol_state=vol_state,
            jump_state=jump_state,
            vol_process=vol_adapter,
            jump_process=zero_jump,
            drift=drift,
            dt=dt,
            dW_price=jnp.array(0.0, dtype=jnp.float64),
            dW_vol=jnp.array(0.0, dtype=jnp.float64),
            uniform=jnp.array(0.5, dtype=jnp.float64),
            normal_jump=jnp.array(0.0, dtype=jnp.float64),
        )

        # Expected: log(100) + (0.05 - 0.5*0.04) * (1/12)
        expected = jnp.log(100.0) + (0.05 - 0.5 * 0.04) * (1 / 12)
        assert jnp.isclose(new_log, expected, atol=1e-12)
        assert "Sigma" in outputs
        assert "Jump" in outputs
        assert jnp.isclose(outputs["Jump"], 0.0, atol=1e-15)

    def test_known_shock_step(self) -> None:
        """Known shocks → verify full equation."""
        vol_adapter = CIRVolAdapter(alpha=1.0, sigma=0.0, mu=0.04)
        zero_jump = ZeroJumpAdapter()
        log_p = jnp.array(0.0, dtype=jnp.float64)  # S=1
        vol_state = VolState(
            variance=jnp.array(0.04, dtype=jnp.float64),
            volatility=jnp.array(0.2, dtype=jnp.float64),
        )
        jump_state = JumpState(
            cum_jumps=jnp.array(0.0, dtype=jnp.float64),
            n_jumps=jnp.array(0.0, dtype=jnp.float64),
            last_jump_size=jnp.array(0.0, dtype=jnp.float64),
        )
        dt = 0.25
        dW_price = jnp.array(1.0, dtype=jnp.float64)

        new_log, _, _, _ = svjd_equity_step(
            log_price=log_p,
            vol_state=vol_state,
            jump_state=jump_state,
            vol_process=vol_adapter,
            jump_process=zero_jump,
            drift=0.03,
            dt=dt,
            dW_price=dW_price,
            dW_vol=jnp.array(0.0, dtype=jnp.float64),
            uniform=jnp.array(0.5, dtype=jnp.float64),
            normal_jump=jnp.array(0.0, dtype=jnp.float64),
        )

        # σ(t)=0.2 (from vol_state), drift=0.03
        # log_new = 0 + (0.03 - 0.5*0.04)*0.25 + 0.2*sqrt(0.25)*1.0
        expected = (0.03 - 0.02) * 0.25 + 0.2 * 0.5 * 1.0
        assert jnp.isclose(new_log, expected, atol=1e-12)

    def test_outputs_keys(self) -> None:
        """Step outputs contain expected keys."""
        vol_adapter = CIRVolAdapter(alpha=1.0, sigma=0.1, mu=0.04)
        zero_jump = ZeroJumpAdapter()
        log_p = jnp.array(0.0, dtype=jnp.float64)
        vol_s = VolState(
            variance=jnp.array(0.04, dtype=jnp.float64),
            volatility=jnp.array(0.2, dtype=jnp.float64),
        )
        jump_s = JumpState(
            cum_jumps=jnp.array(0.0, dtype=jnp.float64),
            n_jumps=jnp.array(0.0, dtype=jnp.float64),
            last_jump_size=jnp.array(0.0, dtype=jnp.float64),
        )
        _, _, _, outputs = svjd_equity_step(
            log_p, vol_s, jump_s, vol_adapter, zero_jump,
            0.05, 1 / 12,
            jnp.array(0.0), jnp.array(0.0),
            jnp.array(0.5), jnp.array(0.0),
        )
        assert set(outputs.keys()) == {
            "LogReturn", "Sigma", "Jump", "DriftAdjustment"
        }


# ── SVJDEquity Model Tests ───────────────────────────────────────────────


class TestSVJDEquity:
    def test_init_state(self) -> None:
        """init_state returns composite state dict."""
        model = SVJDEquity(
            initial_value=100.0, sigma=0.2,
            vol_alpha=1.5, vol_sigma=0.3, vol_mu=0.04,
        )
        state = model.init_state()
        assert "price_state" in state
        assert "vol_state" in state
        assert "jump_state" in state
        assert isinstance(state["price_state"], FXState)
        assert isinstance(state["vol_state"], VolState)
        assert isinstance(state["jump_state"], JumpState)

    def test_n_shocks(self) -> None:
        model = SVJDEquity(initial_value=100.0, sigma=0.2)
        assert model.n_shocks == 4

    def test_shock_config(self) -> None:
        model = SVJDEquity(initial_value=100.0, sigma=0.2, name="test_svjd")
        cfg = model.shock_config
        assert cfg.n_shocks == 4
        assert len(cfg.names) == 4
        assert "test_svjd_price_z" in cfg.names

    def test_step_deterministic(self) -> None:
        """Zero shocks, no jumps → deterministic drift."""
        model = SVJDEquity(
            initial_value=100.0, sigma=0.2,
            vol_alpha=1.5, vol_sigma=0.0, vol_mu=0.04,
        )
        state = model.init_state()
        dt = 1 / 12
        shocks = jnp.zeros(4, dtype=jnp.float64)
        deps: dict = {"rates": {"ShortRate": jnp.array(0.05, dtype=jnp.float64)}}

        new_state, outputs = model.step(state, 0.0, dt, shocks, deps)

        assert "TotalReturnIndex" in outputs
        assert "Sigma" in outputs
        assert jnp.isfinite(outputs["TotalReturnIndex"])

    def test_jump_adjusted_sigma(self) -> None:
        """Initial sigma is reduced by jump variance component."""
        raw_sigma = 0.3
        intensity = 0.5
        j_mean = -0.02
        j_vol = 0.08
        model = SVJDEquity(
            initial_value=100.0, sigma=raw_sigma,
            jump_intensity=intensity, jump_mean=j_mean, jump_vol=j_vol,
        )
        expected = float(jump_adjusted_sigma(raw_sigma, intensity, j_mean, j_vol))
        assert jnp.isclose(model.adjusted_sigma, expected, atol=1e-12)

    def test_zero_jump_matches_stochvol_gbm(self) -> None:
        """With zero jumps, SVJD should match stochastic vol GBM."""
        model = SVJDEquity(
            initial_value=100.0, sigma=0.2,
            vol_alpha=1.5, vol_sigma=0.0, vol_mu=0.04,
            jump_intensity=0.0,
        )
        state = model.init_state()
        dt = 1 / 12
        r = 0.05

        # Step with known shock
        shocks = jnp.array([0.5, 0.0, 0.5, 0.0], dtype=jnp.float64)
        deps: dict = {"rates": {"ShortRate": jnp.array(r, dtype=jnp.float64)}}
        new_state, outputs = model.step(state, 0.0, dt, shocks, deps)

        # Manual: log(100) + (0.05 - 0.5*σ²)*dt + σ*√dt*0.5
        # σ = adjusted_sigma (with zero jumps, no adjustment)
        sigma = model.adjusted_sigma
        expected_log = (
            jnp.log(100.0)
            + (r - 0.5 * sigma**2) * dt
            + sigma * jnp.sqrt(dt) * 0.5
        )
        assert jnp.isclose(new_state["price_state"].log_level, expected_log, atol=1e-10)
        assert jnp.isclose(outputs["Jump"], 0.0, atol=1e-15)

    def test_multi_step_produces_positive_prices(self) -> None:
        """Multiple SVJD steps keep prices positive."""
        model = SVJDEquity(
            initial_value=100.0, sigma=0.25,
            vol_alpha=1.5, vol_sigma=0.3, vol_mu=0.06,
            jump_intensity=0.3, jump_mean=-0.01, jump_vol=0.05,
        )
        state = model.init_state()
        dt = 1 / 12
        key = jax.random.PRNGKey(123)
        deps: dict = {"rates": {"ShortRate": jnp.array(0.04, dtype=jnp.float64)}}

        for i in range(24):  # 2 years of monthly steps
            key, subkey = jax.random.split(key)
            raw = jax.random.normal(subkey, shape=(4,))
            # Convert 3rd shock to uniform
            shocks = raw.at[2].set(jax.scipy.stats.norm.cdf(raw[2]))
            state, outputs = model.step(state, i * dt, dt, shocks, deps)
            assert outputs["TotalReturnIndex"] > 0

    def test_name_property(self) -> None:
        model = SVJDEquity(initial_value=100.0, sigma=0.2, name="my_equity")
        assert model.name == "my_equity"

    def test_no_deps_uses_zero_rate(self) -> None:
        """Empty deps → r=0."""
        model = SVJDEquity(
            initial_value=100.0, sigma=0.2,
            vol_alpha=1.0, vol_sigma=0.0, vol_mu=0.04,
        )
        state = model.init_state()
        shocks = jnp.zeros(4, dtype=jnp.float64)
        new_state, _ = model.step(state, 0.0, 1 / 12, shocks, {})
        # Should still work (drift = 0 - 0 = 0)
        assert jnp.isfinite(new_state["price_state"].level)


# ── Backward Compatibility: Equity with vol/jump deps ────────────────────


class TestEquityWithVolJumpDeps:
    """Test the refactored Equity model reads vol/jump from deps."""

    def test_equity_with_vol_model_dep(self) -> None:
        """Equity reads sigma from vol model dependency."""
        from hyesg.models.equity.equity import Equity

        params = GBMParams(sigma=0.2, initial_value=100.0)
        model = Equity(params=params, vol_model="cir_vol")
        state = model.init_state()
        dt = 1 / 12
        shocks = jnp.array([0.5], dtype=jnp.float64)

        # Provide stochastic vol = 0.3 via deps
        deps: dict = {
            "rates": {"ShortRate": jnp.array(0.05, dtype=jnp.float64)},
            "cir_vol": {"Sigma": jnp.array(0.3, dtype=jnp.float64)},
        }
        new_state, _ = model.step(state, 0.0, dt, shocks, deps)

        # Should use sigma=0.3 from deps, not 0.2 from params
        expected_log = (
            jnp.log(100.0)
            + (0.05 - 0.5 * 0.3**2) * dt
            + 0.3 * 0.5 * jnp.sqrt(dt)
        )
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)

    def test_equity_with_jump_model_dep(self) -> None:
        """Equity reads jump and drift_adjustment from jump model dependency."""
        from hyesg.models.equity.equity import Equity

        params = GBMParams(sigma=0.2, initial_value=100.0)
        model = Equity(params=params, jump_model="jumps")
        state = model.init_state()
        dt = 1 / 12
        shocks = jnp.array([0.5], dtype=jnp.float64)

        jump_val = jnp.array(0.01, dtype=jnp.float64)
        drift_adj = jnp.array(-0.005, dtype=jnp.float64)
        deps: dict = {
            "rates": {"ShortRate": jnp.array(0.05, dtype=jnp.float64)},
            "jumps": {"Jump": jump_val, "DriftAdjustment": drift_adj},
        }
        new_state, _ = model.step(state, 0.0, dt, shocks, deps)

        expected_log = (
            jnp.log(100.0)
            + (0.05 - 0.5 * 0.2**2) * dt
            + drift_adj
            + 0.2 * 0.5 * jnp.sqrt(dt)
            + jump_val
        )
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)

    def test_equity_backward_compatible_no_vol_jump(self) -> None:
        """Equity without vol/jump params behaves identically to old version."""
        from hyesg.models.equity.equity import Equity

        params = GBMParams(sigma=0.2, initial_value=100.0)
        model = Equity(params=params)
        state = model.init_state()
        dt = 1 / 12
        shocks = jnp.array([0.5], dtype=jnp.float64)
        deps: dict = {"rates": {"ShortRate": jnp.array(0.05, dtype=jnp.float64)}}

        new_state, outputs = model.step(state, 0.0, dt, shocks, deps)

        # Same as original GBM
        sigma = 0.2
        expected_log = (
            jnp.log(100.0) + (0.05 - 0.5 * sigma**2) * dt + sigma * 0.5 * jnp.sqrt(dt)
        )
        assert jnp.isclose(new_state.log_level, expected_log, atol=1e-12)
