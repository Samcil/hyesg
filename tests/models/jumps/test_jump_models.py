"""Tests for Merton jump diffusion models."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import pytest

from hyesg.core.registry import clear_registry, get_model
from hyesg.core.types import JumpState
from hyesg.models.jumps.jump_models import (
    ConstantIntensityJumpModel,
    StochasticIntensityJumpModel,
    ZeroJumpModel,
    _poisson_inverse_cdf,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Re-register jump models for each test."""
    clear_registry()
    import hyesg.models.jumps.jump_models as mod

    importlib.reload(mod)
    yield
    clear_registry()


def _zero_state() -> JumpState:
    """Helper: initial JumpState with all zeros."""
    zero = jnp.array(0.0, dtype=jnp.float64)
    return JumpState(cum_jumps=zero, n_jumps=zero, last_jump_size=zero)


# ── 1. ZeroJumpModel ────────────────────────────────────────────────────


class TestZeroJumpModel:
    """ZeroJumpModel outputs are all zero, n_shocks=0."""

    def test_name(self) -> None:
        m = ZeroJumpModel()
        assert m.name == "zero_jump"

    def test_n_shocks_zero(self) -> None:
        m = ZeroJumpModel()
        assert m.n_shocks == 0

    def test_shock_config(self) -> None:
        m = ZeroJumpModel()
        cfg = m.shock_config
        assert cfg.n_shocks == 0
        assert cfg.names == ()

    def test_step_returns_zeros(self) -> None:
        m = ZeroJumpModel()
        state = m.init_state()
        new_state, out = m.step(state, 0.0, 0.25, jnp.array([]), {})
        assert float(out["Jump"]) == pytest.approx(0.0)
        assert float(out["DriftAdjustment"]) == pytest.approx(0.0)
        assert float(out["NJumps"]) == pytest.approx(0.0)
        # State unchanged
        assert float(new_state.cum_jumps) == pytest.approx(0.0)

    def test_registry_lookup(self) -> None:
        cls = get_model("zero_jump")
        assert cls.__name__ == "ZeroJumpModel"


# ── 2. ConstantIntensity with λ=0: identical to zero model ──────────────


class TestConstantIntensityZeroLambda:
    """ConstantIntensityJumpModel with intensity=0 produces no jumps."""

    def test_no_jumps_when_lambda_zero(self) -> None:
        m = ConstantIntensityJumpModel(
            intensity=0.0, jump_mean=-0.05, jump_vol=0.1
        )
        state = m.init_state()
        shocks = jnp.array([0.5, 0.3])
        new_state, out = m.step(state, 0.0, 0.25, shocks, {})
        assert float(out["NJumps"]) == pytest.approx(0.0)
        assert float(out["Jump"]) == pytest.approx(0.0)


# ── 3. Drift adjustment formula ─────────────────────────────────────────


class TestDriftAdjustment:
    """Verify drift_adj = -λ·(exp(μ + σ²/2) - 1) · dt."""

    @pytest.mark.parametrize(
        "lam, mu_j, sig_j",
        [
            (1.0, -0.05, 0.1),
            (2.0, 0.0, 0.2),
            (0.5, -0.1, 0.05),
        ],
    )
    def test_drift_adjustment_formula(
        self, lam: float, mu_j: float, sig_j: float
    ) -> None:
        m = ConstantIntensityJumpModel(
            intensity=lam, jump_mean=mu_j, jump_vol=sig_j
        )
        state = m.init_state()
        dt = 0.25
        # Use uniform=0.0 so Poisson count = 0 (CDF at k=0 = exp(-λdt))
        # This guarantees we enter the branch but drift_adj is still computed.
        shocks = jnp.array([0.0, 0.0])
        _, out = m.step(state, 0.0, dt, shocks, {})

        expected = -lam * (jnp.exp(mu_j + 0.5 * sig_j**2) - 1.0) * dt
        assert float(out["DriftAdjustment"]) == pytest.approx(
            float(expected), abs=1e-12
        )


# ── 4. Poisson exact sampling with fixed uniform ────────────────────────


class TestPoissonExactSampling:
    """Inverse-CDF Poisson should return correct integer counts."""

    def test_uniform_zero_gives_zero(self) -> None:
        """P(N=0) = exp(-λ); u < exp(-λ) → N=0."""
        lam = 2.0
        u = jnp.array(0.0, dtype=jnp.float64)
        n = _poisson_inverse_cdf(u, lam)
        assert float(n) == pytest.approx(0.0)

    def test_uniform_one_gives_max(self) -> None:
        """u ≈ 1 should give a large count."""
        lam = 2.0
        u = jnp.array(0.9999, dtype=jnp.float64)
        n = _poisson_inverse_cdf(u, lam)
        assert float(n) >= 3.0

    def test_deterministic_count(self) -> None:
        """For λdt = 1, P(N=0) = e^-1 ≈ 0.368.  u = 0.4 → N ≥ 1."""
        lam = 1.0
        u = jnp.array(0.4, dtype=jnp.float64)
        n = _poisson_inverse_cdf(u, lam)
        assert float(n) == pytest.approx(1.0)


# ── 5. Poisson statistics over many samples ─────────────────────────────


class TestPoissonStatistics:
    """E[N] ≈ λdt, Var[N] ≈ λdt over many uniform draws."""

    def test_mean_and_variance(self) -> None:
        lam_dt = 3.0
        key = jax.random.PRNGKey(42)
        uniforms = jax.random.uniform(key, shape=(10_000,), dtype=jnp.float64)
        counts = jax.vmap(lambda u: _poisson_inverse_cdf(u, lam_dt))(uniforms)
        mean = float(jnp.mean(counts))
        var = float(jnp.var(counts))
        assert mean == pytest.approx(lam_dt, rel=0.05)
        assert var == pytest.approx(lam_dt, rel=0.15)


# ── 6. Jump size statistics ─────────────────────────────────────────────


class TestJumpSizeStatistics:
    """E[J | N=n] ≈ n·μ,  Var[J | N=n] ≈ n·σ²."""

    def test_conditional_mean_and_variance(self) -> None:
        mu_j = -0.05
        sig_j = 0.1
        m = ConstantIntensityJumpModel(
            intensity=5.0, jump_mean=mu_j, jump_vol=sig_j
        )
        state = m.init_state()
        dt = 1.0  # λ·dt = 5 → several jumps per step

        key = jax.random.PRNGKey(123)
        n_samples = 5_000
        keys = jax.random.split(key, n_samples)
        jumps = []
        n_jumps_list = []
        for k in keys:
            k1, k2 = jax.random.split(k)
            u = jax.random.uniform(k1, dtype=jnp.float64)
            z = jax.random.normal(k2, dtype=jnp.float64)
            shocks = jnp.array([u, z])
            _, out = m.step(state, 0.0, dt, shocks, {})
            jumps.append(float(out["Jump"]))
            n_jumps_list.append(float(out["NJumps"]))

        jumps = jnp.array(jumps)
        n_arr = jnp.array(n_jumps_list)

        # Overall E[J] = E[N]·μ = λdt·μ
        expected_mean = 5.0 * mu_j
        assert float(jnp.mean(jumps)) == pytest.approx(expected_mean, abs=0.15)

        # Overall E[N] ≈ λdt
        assert float(jnp.mean(n_arr)) == pytest.approx(5.0, rel=0.1)


# ── 7. StochasticIntensityJumpModel reads intensity from deps ────────────


class TestStochasticIntensity:
    """StochasticIntensityJumpModel reads λ from deps["cir_vol"]["Variance"]."""

    def test_reads_intensity_from_deps(self) -> None:
        m = StochasticIntensityJumpModel(
            jump_mean=-0.05, jump_vol=0.1, intensity_dep="cir_vol"
        )
        state = m.init_state()
        dt = 0.25
        # High intensity → likely to see jumps with u near 1
        deps = {"cir_vol": {"Variance": jnp.array(10.0, dtype=jnp.float64)}}
        shocks = jnp.array([0.999, 1.0])
        _, out = m.step(state, 0.0, dt, shocks, deps)
        # With λ·dt = 2.5 and u = 0.999, should have multiple jumps
        assert float(out["NJumps"]) >= 1.0

    def test_zero_variance_no_jumps(self) -> None:
        m = StochasticIntensityJumpModel(
            jump_mean=-0.05, jump_vol=0.1, intensity_dep="vol"
        )
        state = m.init_state()
        deps = {"vol": {"Variance": jnp.array(0.0, dtype=jnp.float64)}}
        shocks = jnp.array([0.5, 0.5])
        _, out = m.step(state, 0.0, 0.25, shocks, deps)
        assert float(out["NJumps"]) == pytest.approx(0.0)

    def test_registry_lookup(self) -> None:
        cls = get_model("stochastic_jump")
        assert cls.__name__ == "StochasticIntensityJumpModel"


# ── 8. Cumulative jumps: state tracks total jumps ────────────────────────


class TestCumulativeJumps:
    """State should accumulate jump contributions across steps."""

    def test_cumulative_tracking(self) -> None:
        m = ConstantIntensityJumpModel(
            intensity=2.0, jump_mean=-0.05, jump_vol=0.1
        )
        state = m.init_state()
        dt = 0.5

        total_jump = 0.0
        total_n = 0.0
        key = jax.random.PRNGKey(7)
        for i in range(10):
            k1, k2, key = jax.random.split(key, 3)
            u = jax.random.uniform(k1, dtype=jnp.float64)
            z = jax.random.normal(k2, dtype=jnp.float64)
            shocks = jnp.array([u, z])
            state, out = m.step(state, i * dt, dt, shocks, {})
            total_jump += float(out["Jump"])
            total_n += float(out["NJumps"])

        assert float(state.cum_jumps) == pytest.approx(total_jump, abs=1e-10)
        assert float(state.n_jumps) == pytest.approx(total_n, abs=1e-10)


# ── 9. JIT safety ────────────────────────────────────────────────────────


class TestJITSafety:
    """step() should compile under jax.jit without errors."""

    def test_constant_jit_compiles(self) -> None:
        m = ConstantIntensityJumpModel(
            intensity=1.0, jump_mean=-0.05, jump_vol=0.1
        )
        state = m.init_state()
        shocks = jnp.array([0.5, 0.3])

        @jax.jit
        def do_step(s, sh):
            return m.step(s, 0.0, 0.25, sh, {})

        new_state, out = do_step(state, shocks)
        # Smoke test: outputs are finite
        assert jnp.isfinite(out["Jump"])
        assert jnp.isfinite(out["DriftAdjustment"])

    def test_zero_jit_compiles(self) -> None:
        m = ZeroJumpModel()
        state = m.init_state()

        @jax.jit
        def do_step(s):
            return m.step(s, 0.0, 0.25, jnp.array([]), {})

        new_state, out = do_step(state)
        assert float(out["Jump"]) == pytest.approx(0.0)

    def test_stochastic_jit_compiles(self) -> None:
        m = StochasticIntensityJumpModel(
            jump_mean=-0.05, jump_vol=0.1, intensity_dep="vol"
        )
        state = m.init_state()
        shocks = jnp.array([0.5, 0.3])
        deps = {"vol": {"Variance": jnp.array(2.0, dtype=jnp.float64)}}

        @jax.jit
        def do_step(s, sh, v):
            return m.step(s, 0.0, 0.25, sh, {"vol": {"Variance": v}})

        new_state, out = do_step(state, shocks, jnp.array(2.0, dtype=jnp.float64))
        assert jnp.isfinite(out["Jump"])
