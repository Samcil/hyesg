"""Tests for the salary wedge G2++ model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from hyesg.math.curves.protocol import ParametricCurve
from hyesg.models.salary.salary_wedge import (
    SalaryWedgeModel,
    SalaryWedgeParams,
    SalaryWedgeState,
)

jax.config.update("jax_enable_x64", True)

# ── Test constants ──
ALPHA1 = 0.5
ALPHA2 = 0.8
SIGMA1 = 0.01
SIGMA2 = 0.015
RHO = 0.3
FLAT_RATE = 0.035
DT = 1.0


class FlatCurve(ParametricCurve):
    """Flat target salary growth curve for testing."""

    def __init__(self, rate: float) -> None:
        self._rate = rate

    def evaluate(self, x: float) -> float:
        return self._rate


class LinearCurve(ParametricCurve):
    """Linearly increasing target curve for testing."""

    def __init__(self, base: float, slope: float) -> None:
        self._base = base
        self._slope = slope

    def evaluate(self, x: float) -> float:
        return self._base + self._slope * x


@pytest.fixture
def params() -> SalaryWedgeParams:
    """Standard salary wedge parameters."""
    return SalaryWedgeParams(
        alpha1=ALPHA1,
        alpha2=ALPHA2,
        sigma1=SIGMA1,
        sigma2=SIGMA2,
        rho=RHO,
    )


@pytest.fixture
def target_curve() -> FlatCurve:
    """Flat target curve at 3.5%."""
    return FlatCurve(FLAT_RATE)


@pytest.fixture
def model(params: SalaryWedgeParams, target_curve: FlatCurve) -> SalaryWedgeModel:
    """Standard salary wedge model."""
    return SalaryWedgeModel(params=params, target_curve=target_curve)


# ── 1. State NamedTuple ──


class TestSalaryWedgeState:
    """Tests for SalaryWedgeState."""

    def test_is_named_tuple(self) -> None:
        """SalaryWedgeState is a NamedTuple."""
        assert issubclass(SalaryWedgeState, tuple)
        assert hasattr(SalaryWedgeState, "_fields")

    def test_fields(self) -> None:
        """SalaryWedgeState has expected fields."""
        assert SalaryWedgeState._fields == (
            "x1",
            "x2",
            "salary_rate",
            "salary_index",
        )


# ── 2. Params ──


class TestSalaryWedgeParams:
    """Tests for SalaryWedgeParams."""

    def test_params_defaults(self) -> None:
        """Default initial_x1 and initial_x2 are zero."""
        p = SalaryWedgeParams(
            alpha1=0.5, alpha2=0.8, sigma1=0.01, sigma2=0.015, rho=0.3
        )
        assert p.initial_x1 == 0.0
        assert p.initial_x2 == 0.0

    def test_params_custom_initial(self) -> None:
        """Custom initial factor values are stored."""
        p = SalaryWedgeParams(
            alpha1=0.5,
            alpha2=0.8,
            sigma1=0.01,
            sigma2=0.015,
            rho=0.3,
            initial_x1=0.01,
            initial_x2=-0.005,
        )
        assert p.initial_x1 == 0.01
        assert p.initial_x2 == -0.005

    def test_params_is_named_tuple(self) -> None:
        """SalaryWedgeParams is a NamedTuple."""
        assert issubclass(SalaryWedgeParams, tuple)


# ── 3. Init state ──


class TestInitState:
    """Tests for init_state."""

    def test_init_state_salary_rate(
        self, model: SalaryWedgeModel, target_curve: FlatCurve
    ) -> None:
        """Init salary_rate = phi(0) + x1_0 + x2_0."""
        state = model.init_state()
        expected = target_curve.evaluate(0.0) + 0.0 + 0.0
        assert jnp.isclose(state.salary_rate, expected, atol=1e-12)

    def test_init_state_salary_index(self, model: SalaryWedgeModel) -> None:
        """Init salary_index = 1.0."""
        state = model.init_state()
        assert jnp.isclose(state.salary_index, 1.0, atol=1e-15)

    def test_init_state_factors_zero(self, model: SalaryWedgeModel) -> None:
        """Default initial x1 and x2 are zero."""
        state = model.init_state()
        assert jnp.isclose(state.x1, 0.0, atol=1e-15)
        assert jnp.isclose(state.x2, 0.0, atol=1e-15)

    def test_init_state_custom_x0(self, target_curve: FlatCurve) -> None:
        """Custom initial factors are reflected in salary_rate."""
        p = SalaryWedgeParams(
            alpha1=ALPHA1,
            alpha2=ALPHA2,
            sigma1=SIGMA1,
            sigma2=SIGMA2,
            rho=RHO,
            initial_x1=0.02,
            initial_x2=-0.01,
        )
        model = SalaryWedgeModel(params=p, target_curve=target_curve)
        state = model.init_state()
        expected = FLAT_RATE + 0.02 + (-0.01)
        assert jnp.isclose(state.salary_rate, expected, atol=1e-12)
        assert jnp.isclose(state.x1, 0.02, atol=1e-15)
        assert jnp.isclose(state.x2, -0.01, atol=1e-15)


# ── 4. Phi function ──


class TestPhi:
    """Tests for the phi calibration function."""

    def test_phi_equals_target_at_zero(self, model: SalaryWedgeModel) -> None:
        """phi(0) = target(0) when x1_0=x2_0=0."""
        assert jnp.isclose(model.phi(0.0), FLAT_RATE, atol=1e-12)

    def test_phi_equals_target_at_future(self, model: SalaryWedgeModel) -> None:
        """phi(t) = target(t) for flat curve."""
        for t in [1.0, 5.0, 10.0]:
            assert jnp.isclose(model.phi(t), FLAT_RATE, atol=1e-12)

    def test_phi_follows_linear_curve(self, params: SalaryWedgeParams) -> None:
        """phi(t) follows a linear target curve."""
        curve = LinearCurve(base=0.03, slope=0.001)
        model = SalaryWedgeModel(params=params, target_curve=curve)
        for t in [0.0, 5.0, 10.0]:
            expected = 0.03 + 0.001 * t
            assert jnp.isclose(model.phi(t), expected, atol=1e-12)


# ── 5. Step dynamics ──


class TestStep:
    """Tests for the Euler step."""

    def test_step_zero_noise_mean_reversion(
        self, model: SalaryWedgeModel
    ) -> None:
        """Step with zero noise: OU factors mean-revert toward zero."""
        p = SalaryWedgeParams(
            alpha1=ALPHA1,
            alpha2=ALPHA2,
            sigma1=SIGMA1,
            sigma2=SIGMA2,
            rho=RHO,
            initial_x1=0.05,
            initial_x2=-0.03,
        )
        m = SalaryWedgeModel(params=p, target_curve=FlatCurve(FLAT_RATE))
        state = m.init_state()
        new_state = m.step(state, t=0.0, dt=DT, dw1=jnp.float64(0.0), dw2=jnp.float64(0.0))

        # x1 should move toward 0 (shrink in magnitude)
        assert jnp.abs(new_state.x1) < jnp.abs(state.x1)
        # x2 should move toward 0 (shrink in magnitude)
        assert jnp.abs(new_state.x2) < jnp.abs(state.x2)

    def test_step_noise_changes_factors(self, model: SalaryWedgeModel) -> None:
        """Step with non-zero noise produces non-zero x1, x2."""
        state = model.init_state()
        new_state = model.step(
            state, t=0.0, dt=DT, dw1=jnp.float64(1.0), dw2=jnp.float64(-1.0)
        )
        assert not jnp.isclose(new_state.x1, 0.0, atol=1e-15)
        assert not jnp.isclose(new_state.x2, 0.0, atol=1e-15)

    def test_salary_rate_equals_phi_plus_factors(
        self, model: SalaryWedgeModel
    ) -> None:
        """salary_rate = phi(t+dt) + x1 + x2 after step."""
        state = model.init_state()
        new_state = model.step(
            state, t=0.0, dt=DT, dw1=jnp.float64(0.5), dw2=jnp.float64(-0.3)
        )
        expected_rate = model.phi(DT) + new_state.x1 + new_state.x2
        assert jnp.isclose(new_state.salary_rate, expected_rate, atol=1e-12)

    def test_salary_index_accumulates(self, model: SalaryWedgeModel) -> None:
        """Salary index accumulates correctly: S_new = S_old * exp(r * dt)."""
        state = model.init_state()
        new_state = model.step(
            state, t=0.0, dt=DT, dw1=jnp.float64(0.0), dw2=jnp.float64(0.0)
        )
        expected_index = 1.0 * jnp.exp(state.salary_rate * DT)
        assert jnp.isclose(new_state.salary_index, expected_index, atol=1e-12)

    def test_multiple_steps_accumulate_index(
        self, model: SalaryWedgeModel
    ) -> None:
        """Multiple steps with positive rate produce monotonically increasing index."""
        state = model.init_state()
        indices = [state.salary_index]
        for i in range(10):
            state = model.step(
                state,
                t=i * DT,
                dt=DT,
                dw1=jnp.float64(0.0),
                dw2=jnp.float64(0.0),
            )
            indices.append(state.salary_index)
        # With positive target rate and zero noise, index should increase
        for i in range(1, len(indices)):
            assert indices[i] > indices[i - 1]


# ── 6. Volatility and mean reversion ──


class TestDynamicsProperties:
    """Tests for model dynamic properties."""

    def test_large_alpha_faster_decay(self, target_curve: FlatCurve) -> None:
        """Larger alpha → faster mean reversion (faster decay to phi)."""
        small_dt = 0.1  # small enough for Euler stability with alpha=2
        p_slow = SalaryWedgeParams(
            alpha1=0.1, alpha2=0.1, sigma1=SIGMA1, sigma2=SIGMA2, rho=RHO,
            initial_x1=0.05, initial_x2=0.05,
        )
        p_fast = SalaryWedgeParams(
            alpha1=2.0, alpha2=2.0, sigma1=SIGMA1, sigma2=SIGMA2, rho=RHO,
            initial_x1=0.05, initial_x2=0.05,
        )
        m_slow = SalaryWedgeModel(params=p_slow, target_curve=target_curve)
        m_fast = SalaryWedgeModel(params=p_fast, target_curve=target_curve)
        s_slow = m_slow.init_state()
        s_fast = m_fast.init_state()

        zero = jnp.float64(0.0)
        s_slow = m_slow.step(s_slow, t=0.0, dt=small_dt, dw1=zero, dw2=zero)
        s_fast = m_fast.step(s_fast, t=0.0, dt=small_dt, dw1=zero, dw2=zero)

        # Fast model x1 should be closer to 0 than slow model x1
        assert jnp.abs(s_fast.x1) < jnp.abs(s_slow.x1)

    def test_zero_volatility_deterministic(
        self, target_curve: FlatCurve
    ) -> None:
        """Zero volatility → deterministic path following target."""
        p = SalaryWedgeParams(
            alpha1=ALPHA1, alpha2=ALPHA2, sigma1=0.0, sigma2=0.0, rho=0.0
        )
        model = SalaryWedgeModel(params=p, target_curve=target_curve)
        state = model.init_state()

        # Even with non-zero noise, sigma=0 means factors stay at 0
        state = model.step(
            state, t=0.0, dt=DT, dw1=jnp.float64(2.0), dw2=jnp.float64(-3.0)
        )
        assert jnp.isclose(state.x1, 0.0, atol=1e-15)
        assert jnp.isclose(state.x2, 0.0, atol=1e-15)
        assert jnp.isclose(state.salary_rate, FLAT_RATE, atol=1e-12)


# ── 7. Correlation ──


class TestCorrelation:
    """Tests for correlated factors."""

    def test_correlated_brownian_increments(
        self, target_curve: FlatCurve
    ) -> None:
        """Correlated dw1, dw2 produce correlated factors."""
        p = SalaryWedgeParams(
            alpha1=ALPHA1, alpha2=ALPHA2, sigma1=0.1, sigma2=0.1, rho=0.9
        )
        model = SalaryWedgeModel(params=p, target_curve=target_curve)

        key = jax.random.PRNGKey(42)
        n_samples = 5000
        keys = jax.random.split(key, n_samples)

        x1_vals = []
        x2_vals = []
        for k in keys:
            k1, k2 = jax.random.split(k)
            z1 = jax.random.normal(k1, dtype=jnp.float64)
            z2 = jax.random.normal(k2, dtype=jnp.float64)
            # Apply Cholesky to correlate: dw2 = rho*z1 + sqrt(1-rho^2)*z2
            dw1 = z1
            dw2 = p.rho * z1 + jnp.sqrt(1.0 - p.rho**2) * z2

            state = model.init_state()
            new_state = model.step(state, t=0.0, dt=DT, dw1=dw1, dw2=dw2)
            x1_vals.append(new_state.x1)
            x2_vals.append(new_state.x2)

        x1_arr = jnp.array(x1_vals)
        x2_arr = jnp.array(x2_vals)

        # Compute sample correlation
        corr_matrix = jnp.corrcoef(x1_arr, x2_arr)
        sample_corr = corr_matrix[0, 1]

        # Correlation should be close to rho (from Cholesky)
        assert jnp.abs(sample_corr - p.rho) < 0.1


# ── 8. Output dict ──


class TestOutput:
    """Tests for the output method."""

    def test_output_keys(self, model: SalaryWedgeModel) -> None:
        """Output dict has salary_rate and salary_index keys."""
        state = model.init_state()
        out = model.output(state)
        assert "salary_rate" in out
        assert "salary_index" in out

    def test_output_values_match_state(self, model: SalaryWedgeModel) -> None:
        """Output values match state fields."""
        state = model.init_state()
        out = model.output(state)
        assert jnp.isclose(out["salary_rate"], state.salary_rate, atol=1e-15)
        assert jnp.isclose(out["salary_index"], state.salary_index, atol=1e-15)

    def test_output_after_step(self, model: SalaryWedgeModel) -> None:
        """Output after step reflects updated state."""
        state = model.init_state()
        state = model.step(
            state, t=0.0, dt=DT, dw1=jnp.float64(0.5), dw2=jnp.float64(-0.2)
        )
        out = model.output(state)
        assert jnp.isclose(out["salary_rate"], state.salary_rate, atol=1e-15)
        assert jnp.isclose(out["salary_index"], state.salary_index, atol=1e-15)


# ── 9. Import from package ──


class TestImports:
    """Tests for package-level imports."""

    def test_import_from_salary_package(self) -> None:
        """All public symbols importable from salary package."""
        from hyesg.models.salary import (
            SalaryWedgeModel,
            SalaryWedgeParams,
            SalaryWedgeState,
        )

        assert SalaryWedgeModel is not None
        assert SalaryWedgeParams is not None
        assert SalaryWedgeState is not None
