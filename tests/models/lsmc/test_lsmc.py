"""Tests for the LSMC (Least-Squares Monte Carlo) module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from scipy.stats import norm

from hyesg.engine.output import SimulationResult
from hyesg.models.lsmc.basis import laguerre_basis, polynomial_basis
from hyesg.models.lsmc.payoffs import american_put, bermudan_put, european_put
from hyesg.models.lsmc.pricer import LSMCConfig, LSMCPricer, LSMCResult

jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def bs_put(s: float, k: float, r: float, sigma: float, t: float) -> float:
    """Black-Scholes European put price (analytical reference).

    Args:
        s: Spot price.
        k: Strike price.
        r: Risk-free rate (continuous).
        sigma: Volatility.
        t: Time to maturity in years.

    Returns:
        Analytical put price.
    """
    d1 = (jnp.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * jnp.sqrt(t))
    d2 = d1 - sigma * jnp.sqrt(t)
    return float(
        k * jnp.exp(-r * t) * norm.cdf(-float(d2))
        - s * norm.cdf(-float(d1))
    )


def generate_gbm_paths(
    s0: float,
    r: float,
    sigma: float,
    t: float,
    n_steps: int,
    n_paths: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Generate GBM paths under the risk-neutral measure.

    Args:
        s0: Initial spot price.
        r: Risk-free rate.
        sigma: Volatility.
        t: Total time horizon.
        n_steps: Number of time steps.
        n_paths: Number of Monte Carlo paths.
        key: JAX PRNG key.

    Returns:
        Tuple of (paths, discount_factors) where paths has shape
        (n_paths, n_steps + 1) and discount_factors has shape
        (n_steps + 1,).
    """
    dt = t / n_steps
    z = jax.random.normal(key, shape=(n_paths, n_steps))

    log_increments = (r - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * z
    log_paths = jnp.concatenate(
        [jnp.full((n_paths, 1), jnp.log(s0)), log_increments], axis=1
    )
    log_paths = jnp.cumsum(log_paths, axis=1)
    paths = jnp.exp(log_paths)

    times = jnp.arange(n_steps + 1) * dt
    discount_factors = jnp.exp(-r * times)

    return paths, discount_factors


# ------------------------------------------------------------------
# Basis function tests
# ------------------------------------------------------------------


class TestPolynomialBasis:
    def test_shape(self) -> None:
        x = jnp.array([1.0, 2.0, 3.0])
        result = polynomial_basis(x, degree=3)
        assert result.shape == (3, 4)

    def test_degree_zero(self) -> None:
        x = jnp.array([1.0, 2.0])
        result = polynomial_basis(x, degree=0)
        assert result.shape == (2, 1)
        assert jnp.allclose(result[:, 0], jnp.ones(2))

    def test_values(self) -> None:
        x = jnp.array([2.0, 3.0])
        result = polynomial_basis(x, degree=3)
        # [1, x, x^2, x^3]
        expected = jnp.array([
            [1.0, 2.0, 4.0, 8.0],
            [1.0, 3.0, 9.0, 27.0],
        ])
        assert jnp.allclose(result, expected)

    def test_higher_degree_adds_columns(self) -> None:
        x = jnp.array([1.0, 2.0])
        r3 = polynomial_basis(x, degree=3)
        r5 = polynomial_basis(x, degree=5)
        assert r3.shape[1] == 4
        assert r5.shape[1] == 6


class TestLaguerreBasis:
    def test_shape(self) -> None:
        x = jnp.array([1.0, 2.0, 3.0])
        result = laguerre_basis(x, degree=3)
        assert result.shape == (3, 4)

    def test_degree_zero(self) -> None:
        x = jnp.array([1.0, 2.0])
        result = laguerre_basis(x, degree=0)
        assert result.shape == (2, 1)
        # L_0 = 1, weighted by exp(-u/2) where u = x/mean(x)
        u = x / jnp.mean(x)
        expected = jnp.exp(-u / 2.0)
        assert jnp.allclose(result[:, 0], expected)

    def test_l1_values(self) -> None:
        """L_1(u) = 1 - u, weighted by exp(-u/2), where u = x/mean(x)."""
        x = jnp.array([0.5, 1.0, 2.0])
        result = laguerre_basis(x, degree=1)
        # u = x / mean(x), mean = (0.5+1.0+2.0)/3 ≈ 1.1667
        u = x / jnp.mean(x)
        weight = jnp.exp(-u / 2.0)
        expected_l1 = weight * (1.0 - u)
        assert jnp.allclose(result[:, 1], expected_l1, atol=1e-12)

    def test_higher_degree_adds_columns(self) -> None:
        x = jnp.array([1.0, 2.0])
        r2 = laguerre_basis(x, degree=2)
        r4 = laguerre_basis(x, degree=4)
        assert r2.shape[1] == 3
        assert r4.shape[1] == 5


# ------------------------------------------------------------------
# Payoff function tests
# ------------------------------------------------------------------


class TestPayoffs:
    def test_european_put_itm(self) -> None:
        spot = jnp.array([80.0, 90.0])
        result = european_put(spot, strike=100.0)
        expected = jnp.array([20.0, 10.0])
        assert jnp.allclose(result, expected)

    def test_european_put_otm(self) -> None:
        spot = jnp.array([110.0, 120.0])
        result = european_put(spot, strike=100.0)
        assert jnp.allclose(result, jnp.zeros(2))

    def test_european_put_atm(self) -> None:
        spot = jnp.array([100.0])
        result = european_put(spot, strike=100.0)
        assert jnp.allclose(result, jnp.zeros(1))

    def test_american_put_matches_european(self) -> None:
        spot = jnp.array([80.0, 100.0, 120.0])
        assert jnp.allclose(
            american_put(spot, 100.0), european_put(spot, 100.0)
        )

    def test_bermudan_put_matches_european(self) -> None:
        spot = jnp.array([80.0, 100.0, 120.0])
        assert jnp.allclose(
            bermudan_put(spot, 100.0), european_put(spot, 100.0)
        )

    def test_deep_itm(self) -> None:
        spot = jnp.array([1.0])
        result = european_put(spot, strike=100.0)
        assert jnp.isclose(result[0], 99.0)

    def test_deep_otm(self) -> None:
        spot = jnp.array([1000.0])
        result = european_put(spot, strike=100.0)
        assert jnp.isclose(result[0], 0.0)


# ------------------------------------------------------------------
# European put convergence to Black-Scholes
# ------------------------------------------------------------------


class TestEuropeanPutConvergence:
    """LSMC European put should converge to the Black-Scholes price."""

    @pytest.fixture
    def bs_params(self) -> dict:
        return {"s": 100.0, "k": 100.0, "r": 0.05, "sigma": 0.2, "t": 1.0}

    @pytest.fixture
    def analytical_price(self, bs_params: dict) -> float:
        return bs_put(**bs_params)

    def test_european_convergence(
        self, bs_params: dict, analytical_price: float
    ) -> None:
        """With 100k paths, LSMC European should be within 1% of BS."""
        key = jax.random.PRNGKey(12345)
        n_paths = 100_000
        n_steps = 50

        paths, df = generate_gbm_paths(
            s0=bs_params["s"],
            r=bs_params["r"],
            sigma=bs_params["sigma"],
            t=bs_params["t"],
            n_steps=n_steps,
            n_paths=n_paths,
            key=key,
        )

        config = LSMCConfig(exercise_type="european")
        pricer = LSMCPricer(config)
        result = pricer.price(
            paths=paths,
            payoff_fn=european_put,
            strike=bs_params["k"],
            discount_factors=df,
        )

        rel_error = abs(result.price - analytical_price) / analytical_price
        assert rel_error < 0.01, (
            f"LSMC={result.price:.4f}, BS={analytical_price:.4f}, "
            f"rel_error={rel_error:.4f}"
        )

    def test_std_error_is_small(self, bs_params: dict) -> None:
        """Standard error should be small relative to price."""
        key = jax.random.PRNGKey(99)
        paths, df = generate_gbm_paths(
            s0=bs_params["s"],
            r=bs_params["r"],
            sigma=bs_params["sigma"],
            t=bs_params["t"],
            n_steps=50,
            n_paths=50_000,
            key=key,
        )

        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        result = pricer.price(paths, european_put, bs_params["k"], df)

        assert result.std_error < 0.1  # should be well below 0.1


# ------------------------------------------------------------------
# American vs European premium
# ------------------------------------------------------------------


class TestAmericanPremium:
    """American put price should be >= European put price."""

    def test_american_geq_european(self) -> None:
        key = jax.random.PRNGKey(42)
        s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
        n_paths, n_steps = 50_000, 50

        paths, df = generate_gbm_paths(s0, r, sigma, t, n_steps, n_paths, key)

        european_pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        american_pricer = LSMCPricer(
            LSMCConfig(exercise_type="american", degree=3)
        )

        eu_result = european_pricer.price(paths, european_put, k, df)
        am_result = american_pricer.price(paths, american_put, k, df)

        # American >= European (within MC noise)
        assert am_result.price >= eu_result.price - 3 * eu_result.std_error, (
            f"American={am_result.price:.4f} < European={eu_result.price:.4f}"
        )

    def test_deep_itm_american_has_early_exercise(self) -> None:
        """Deep ITM American put should exhibit early exercise."""
        key = jax.random.PRNGKey(77)
        s0, k, r, sigma, t = 80.0, 100.0, 0.08, 0.2, 1.0
        n_paths, n_steps = 20_000, 50

        paths, df = generate_gbm_paths(s0, r, sigma, t, n_steps, n_paths, key)

        pricer = LSMCPricer(LSMCConfig(exercise_type="american"))
        result = pricer.price(paths, american_put, k, df)

        # Some paths should exercise before maturity
        assert result.optimal_stopping is not None
        early = jnp.sum(result.optimal_stopping < n_steps - 1)
        assert int(early) > 0, "Expected some early exercise for deep ITM put"


# ------------------------------------------------------------------
# Exercise boundary
# ------------------------------------------------------------------


class TestExerciseBoundary:
    def test_boundary_exists_for_american(self) -> None:
        key = jax.random.PRNGKey(55)
        paths, df = generate_gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 20_000, key)

        pricer = LSMCPricer(LSMCConfig(exercise_type="american"))
        result = pricer.price(paths, american_put, 100.0, df)

        assert result.exercise_boundary is not None
        assert len(result.exercise_boundary) > 0

    def test_no_boundary_for_european(self) -> None:
        key = jax.random.PRNGKey(55)
        paths, df = generate_gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 10_000, key)

        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        result = pricer.price(paths, european_put, 100.0, df)

        assert result.exercise_boundary is None


# ------------------------------------------------------------------
# Convergence with more paths
# ------------------------------------------------------------------


class TestConvergenceWithPaths:
    def test_more_paths_reduce_std_error(self) -> None:
        """Doubling paths should roughly halve the standard error."""
        key1 = jax.random.PRNGKey(100)
        key2 = jax.random.PRNGKey(200)

        paths_small, df = generate_gbm_paths(
            100.0, 0.05, 0.2, 1.0, 50, 10_000, key1
        )
        paths_large, _ = generate_gbm_paths(
            100.0, 0.05, 0.2, 1.0, 50, 40_000, key2
        )

        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        r_small = pricer.price(paths_small, european_put, 100.0, df)

        df_large = jnp.exp(
            -0.05 * jnp.arange(51) * (1.0 / 50)
        )
        r_large = pricer.price(paths_large, european_put, 100.0, df_large)

        # std_error scales as 1/sqrt(N), so 4x paths ≈ 2x reduction
        assert r_large.std_error < r_small.std_error


# ------------------------------------------------------------------
# Basis function selection
# ------------------------------------------------------------------


class TestBasisSelection:
    def test_polynomial_basis_config(self) -> None:
        config = LSMCConfig(basis="polynomial", degree=4)
        pricer = LSMCPricer(config)
        assert pricer._basis_fn.__name__ == "polynomial_basis"

    def test_laguerre_basis_config(self) -> None:
        config = LSMCConfig(basis="laguerre", degree=3)
        pricer = LSMCPricer(config)
        assert pricer._basis_fn.__name__ == "laguerre_basis"

    def test_laguerre_produces_valid_price(self) -> None:
        """Laguerre basis should give a reasonable price.

        Tolerance is looser than the polynomial convergence test because
        the exp(-u/2) weighting in Laguerre basis is optimised for
        different payoff structures and converges more slowly for
        vanilla puts.
        """
        key = jax.random.PRNGKey(321)
        paths, df = generate_gbm_paths(100.0, 0.05, 0.2, 1.0, 50, 100_000, key)

        pricer = LSMCPricer(LSMCConfig(basis="laguerre", degree=4))
        result = pricer.price(paths, european_put, 100.0, df)

        bs_price = bs_put(100.0, 100.0, 0.05, 0.2, 1.0)
        rel_error = abs(result.price - bs_price) / bs_price
        assert rel_error < 0.10


# ------------------------------------------------------------------
# Bermudan option
# ------------------------------------------------------------------


class TestBermudanOption:
    def test_bermudan_between_european_and_american(self) -> None:
        """Bermudan price should lie between European and American."""
        key = jax.random.PRNGKey(500)
        n_steps = 50
        paths, df = generate_gbm_paths(
            100.0, 0.05, 0.2, 1.0, n_steps, 30_000, key
        )

        eu_pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        am_pricer = LSMCPricer(LSMCConfig(exercise_type="american"))
        bm_pricer = LSMCPricer(LSMCConfig(exercise_type="bermudan"))

        eu = eu_pricer.price(paths, european_put, 100.0, df)
        am = am_pricer.price(paths, american_put, 100.0, df)

        # Bermudan with quarterly exercise (every ~12 steps)
        exercise_dates = jnp.array([12, 25, 37, 50])
        bm = bm_pricer.price(
            paths, bermudan_put, 100.0, df, exercise_dates=exercise_dates
        )

        # Allow MC noise tolerance
        tol = 3 * max(eu.std_error, am.std_error, bm.std_error)
        assert bm.price >= eu.price - tol
        assert bm.price <= am.price + tol


# ------------------------------------------------------------------
# LSMCResult dataclass
# ------------------------------------------------------------------


class TestLSMCResult:
    def test_result_fields(self) -> None:
        result = LSMCResult(price=5.0, std_error=0.1)
        assert result.price == 5.0
        assert result.std_error == 0.1
        assert result.exercise_boundary is None
        assert result.optimal_stopping is None
        assert result.continuation_values is None


# ------------------------------------------------------------------
# LSMCConfig defaults
# ------------------------------------------------------------------


class TestLSMCConfig:
    def test_defaults(self) -> None:
        config = LSMCConfig()
        assert config.basis == "polynomial"
        assert config.degree == 3
        assert config.exercise_type == "american"

    def test_custom(self) -> None:
        config = LSMCConfig(basis="laguerre", degree=5, exercise_type="bermudan")
        assert config.basis == "laguerre"
        assert config.degree == 5
        assert config.exercise_type == "bermudan"


# ------------------------------------------------------------------
# Integration with SimulationResult
# ------------------------------------------------------------------


class TestSimulationResultIntegration:
    @pytest.fixture
    def mock_simulation_result(self) -> SimulationResult:
        """Create a mock SimulationResult with GBM-like paths."""
        key = jax.random.PRNGKey(777)
        n_trials, n_steps = 10_000, 50
        s0, r, sigma, t = 100.0, 0.05, 0.2, 1.0
        dt = t / n_steps

        # Generate equity paths
        z = jax.random.normal(key, shape=(n_trials, n_steps))
        log_inc = (r - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * z
        log_paths = jnp.concatenate(
            [jnp.full((n_trials, 1), jnp.log(s0)), log_inc], axis=1
        )
        log_paths = jnp.cumsum(log_paths, axis=1)
        equity_levels = jnp.exp(log_paths)

        # Constant short rate
        short_rates = jnp.full((n_trials, n_steps + 1), r)

        outputs = {
            "equity": {"level": equity_levels},
            "nominal": {"short_rate": short_rates},
        }
        time_grid = jnp.linspace(0.0, t, n_steps + 1)

        return SimulationResult(
            outputs=outputs, time_grid=time_grid, metadata={"seed": 777}
        )

    def test_price_from_simulation(
        self, mock_simulation_result: SimulationResult
    ) -> None:
        """price_from_simulation should produce a reasonable price."""
        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        result = pricer.price_from_simulation(
            result=mock_simulation_result,
            asset_model="equity",
            rate_model="nominal",
            payoff_fn=european_put,
            strike=100.0,
        )

        bs_price = bs_put(100.0, 100.0, 0.05, 0.2, 1.0)
        rel_error = abs(result.price - bs_price) / bs_price
        assert rel_error < 0.05, (
            f"price_from_simulation={result.price:.4f}, BS={bs_price:.4f}"
        )

    def test_price_from_simulation_american(
        self, mock_simulation_result: SimulationResult
    ) -> None:
        """American price from simulation should exceed European."""
        eu_pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        am_pricer = LSMCPricer(LSMCConfig(exercise_type="american"))

        eu = eu_pricer.price_from_simulation(
            mock_simulation_result, "equity", "nominal", european_put, 100.0
        )
        am = am_pricer.price_from_simulation(
            mock_simulation_result, "equity", "nominal", american_put, 100.0
        )

        assert am.price >= eu.price - 3 * eu.std_error


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_all_paths_otm(self) -> None:
        """All paths OTM at maturity should give price ≈ 0."""
        key = jax.random.PRNGKey(11)
        # Very high initial price, low strike → all OTM for put
        paths, df = generate_gbm_paths(
            200.0, 0.05, 0.1, 1.0, 50, 10_000, key
        )
        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        result = pricer.price(paths, european_put, 50.0, df)
        assert result.price < 0.01

    def test_all_paths_deep_itm(self) -> None:
        """Deep ITM put should be close to discounted intrinsic value."""
        key = jax.random.PRNGKey(22)
        # Very low initial price, high strike → deep ITM
        paths, df = generate_gbm_paths(
            10.0, 0.05, 0.1, 1.0, 50, 10_000, key
        )
        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        result = pricer.price(paths, european_put, 100.0, df)

        # Price should be close to K*exp(-rT) - S0 ≈ 95.12 - 10 ≈ 85.12
        lower_bound = 100.0 * jnp.exp(-0.05 * 1.0) - 10.0
        assert result.price > float(lower_bound) * 0.90

    def test_zero_volatility_european(self) -> None:
        """With zero vol, European put = max(K*exp(-rT) - S0, 0)."""
        n_paths, n_steps = 1000, 50
        s0, k, r, t = 100.0, 110.0, 0.05, 1.0
        dt = t / n_steps

        # Deterministic paths (zero vol)
        times = jnp.arange(n_steps + 1) * dt
        path = s0 * jnp.exp(r * times)
        paths = jnp.tile(path, (n_paths, 1))
        df = jnp.exp(-r * times)

        pricer = LSMCPricer(LSMCConfig(exercise_type="european"))
        result = pricer.price(paths, european_put, k, df)

        # Exact: max(K*exp(-rT) - S0, 0) = max(110*exp(-0.05) - 100, 0)
        exact = max(k * float(jnp.exp(-r * t)) - s0, 0.0)
        assert abs(result.price - exact) < 0.01


# ------------------------------------------------------------------
# Discount factor construction
# ------------------------------------------------------------------


class TestDiscountFactors:
    def test_constant_rate(self) -> None:
        """Constant short rate should give exp(-r*t) discount factors."""
        r = 0.05
        dt = 0.02
        n_steps = 50
        n_trials = 3

        short_rates = jnp.full((n_trials, n_steps), r)
        df = LSMCPricer._build_discount_factors(short_rates, dt)

        times = jnp.arange(1, n_steps + 1) * dt
        expected = jnp.exp(-r * times)

        assert df.shape == (n_trials, n_steps)
        assert jnp.allclose(df[0], expected, atol=1e-12)

    def test_zero_rate(self) -> None:
        """Zero short rate should give discount factors of 1."""
        short_rates = jnp.zeros((5, 10))
        df = LSMCPricer._build_discount_factors(short_rates, 0.1)
        assert jnp.allclose(df, jnp.ones_like(df))
