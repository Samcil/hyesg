"""Tests for hyesg.engine.output — SimulationResult and output extraction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from hyesg.core.types import OutputSpec
from hyesg.engine.output import SimulationResult, combine_regime_results, extract_outputs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_outputs() -> dict[str, dict[str, jnp.ndarray]]:
    """Create sample outputs shaped (n_trials, n_steps)."""
    n_trials, n_steps = 5, 10
    return {
        "cir": {
            "ShortRate": jnp.ones((n_trials, n_steps)) * 0.03,
        },
        "equity": {
            "TotalReturnIndex": jnp.ones((n_trials, n_steps)) * 100.0,
            "LogReturn": jnp.zeros((n_trials, n_steps)),
        },
    }


@pytest.fixture
def sample_time_grid() -> jnp.ndarray:
    """11 time points (10 steps)."""
    return jnp.linspace(0.0, 10.0, 11)


@pytest.fixture
def sample_result(sample_outputs, sample_time_grid) -> SimulationResult:
    """A SimulationResult for testing."""
    return SimulationResult(
        outputs=sample_outputs,
        time_grid=sample_time_grid,
        metadata={"seed": 42, "n_trials": 5},
    )


# ---------------------------------------------------------------------------
# SimulationResult creation
# ---------------------------------------------------------------------------


class TestSimulationResultCreation:
    """Tests for SimulationResult construction."""

    def test_create_empty(self):
        """Create result with no outputs."""
        result = SimulationResult(
            outputs={},
            time_grid=jnp.array([0.0, 1.0]),
        )
        assert result.model_names == []
        assert result.n_trials == 0
        assert result.n_steps == 0

    def test_create_with_outputs(self, sample_result):
        """Create result with sample outputs."""
        assert isinstance(sample_result, SimulationResult)
        assert sample_result.time_grid.shape == (11,)

    def test_metadata_default(self):
        """Default metadata is empty dict."""
        result = SimulationResult(
            outputs={},
            time_grid=jnp.array([0.0]),
        )
        assert result.metadata == {}

    def test_metadata_preserved(self, sample_result):
        """Metadata is preserved correctly."""
        assert sample_result.metadata["seed"] == 42
        assert sample_result.metadata["n_trials"] == 5


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestSimulationResultProperties:
    """Tests for SimulationResult properties."""

    def test_n_trials(self, sample_result):
        """n_trials matches first output dimension."""
        assert sample_result.n_trials == 5

    def test_n_steps(self, sample_result):
        """n_steps matches second output dimension."""
        assert sample_result.n_steps == 10

    def test_model_names(self, sample_result):
        """model_names returns sorted list."""
        assert sample_result.model_names == ["cir", "equity"]

    def test_model_names_sorted(self):
        """model_names are always sorted alphabetically."""
        result = SimulationResult(
            outputs={
                "zmodel": {"x": jnp.ones((2, 3))},
                "amodel": {"y": jnp.ones((2, 3))},
            },
            time_grid=jnp.array([0.0, 1.0, 2.0, 3.0]),
        )
        assert result.model_names == ["amodel", "zmodel"]


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------


class TestSelect:
    """Tests for SimulationResult.select()."""

    def test_select_existing(self, sample_result):
        """Select an existing field."""
        arr = sample_result.select("cir", "ShortRate")
        assert arr.shape == (5, 10)
        assert jnp.allclose(arr, 0.03)

    def test_select_equity_level(self, sample_result):
        """Select equity level field."""
        arr = sample_result.select("equity", "TotalReturnIndex")
        assert arr.shape == (5, 10)

    def test_select_missing_model(self, sample_result):
        """KeyError on missing model."""
        with pytest.raises(KeyError, match="Model 'missing' not found"):
            sample_result.select("missing", "field")

    def test_select_missing_field(self, sample_result):
        """KeyError on missing field."""
        with pytest.raises(KeyError, match="Field 'missing' not found"):
            sample_result.select("cir", "missing")


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------


class TestToDict:
    """Tests for SimulationResult.to_dict()."""

    def test_to_dict_keys(self, sample_result):
        """Flattened dict has model.field keys."""
        flat = sample_result.to_dict()
        assert "cir.ShortRate" in flat
        assert "equity.TotalReturnIndex" in flat
        assert "equity.LogReturn" in flat

    def test_to_dict_count(self, sample_result):
        """Correct number of entries in flattened dict."""
        flat = sample_result.to_dict()
        assert len(flat) == 3  # 1 CIR field + 2 equity fields

    def test_to_dict_values(self, sample_result):
        """Flattened values match originals."""
        flat = sample_result.to_dict()
        assert jnp.allclose(flat["cir.ShortRate"], 0.03)

    def test_to_dict_empty(self):
        """Empty result produces empty dict."""
        result = SimulationResult(
            outputs={},
            time_grid=jnp.array([0.0]),
        )
        assert result.to_dict() == {}


# ---------------------------------------------------------------------------
# extract_outputs
# ---------------------------------------------------------------------------


class TestExtractOutputs:
    """Tests for extract_outputs function."""

    def test_extract_all(self, sample_outputs):
        """None specs extracts everything."""
        result = extract_outputs(sample_outputs, None)
        assert "cir" in result
        assert "equity" in result
        assert "ShortRate" in result["cir"]

    def test_extract_filtered(self, sample_outputs):
        """Filter with OutputSpec."""
        specs = [
            OutputSpec(
                model_name="cir",
                member_name="ShortRate",
                output_name="ShortRate",
            ),
        ]
        result = extract_outputs(sample_outputs, specs)
        assert "cir" in result
        assert "equity" not in result

    def test_extract_renames(self, sample_outputs):
        """OutputSpec can rename output."""
        specs = [
            OutputSpec(
                model_name="equity",
                member_name="TotalReturnIndex",
                output_name="equity_price",
            ),
        ]
        result = extract_outputs(sample_outputs, specs)
        assert "equity_price" in result["equity"]
        assert "TotalReturnIndex" not in result["equity"]

    def test_extract_missing_model(self, sample_outputs):
        """Missing model in specs is silently skipped."""
        specs = [
            OutputSpec(
                model_name="nonexistent",
                member_name="x",
                output_name="x",
            ),
        ]
        result = extract_outputs(sample_outputs, specs)
        assert len(result) == 0

    def test_extract_missing_field(self, sample_outputs):
        """Missing field in specs is silently skipped."""
        specs = [
            OutputSpec(
                model_name="cir",
                member_name="nonexistent",
                output_name="x",
            ),
        ]
        result = extract_outputs(sample_outputs, specs)
        assert len(result) == 0

    def test_extract_empty_specs(self, sample_outputs):
        """Empty specs list produces empty result."""
        result = extract_outputs(sample_outputs, [])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# combine_regime_results
# ---------------------------------------------------------------------------


class TestCombineRegimeResults:
    """Tests for combine_regime_results."""

    def test_combine_single(self, sample_result):
        """Single result returns unchanged."""
        combined = combine_regime_results([sample_result])
        assert combined is sample_result

    def test_combine_multiple(self, sample_time_grid):
        """Combine two regimes concatenates trials."""
        r1 = SimulationResult(
            outputs={"m": {"x": jnp.ones((3, 5))}},
            time_grid=sample_time_grid,
            metadata={"regime": 0},
        )
        r2 = SimulationResult(
            outputs={"m": {"x": jnp.ones((4, 5)) * 2.0}},
            time_grid=sample_time_grid,
            metadata={"regime": 1},
        )
        combined = combine_regime_results([r1, r2])
        assert combined.n_trials == 7
        assert combined.n_steps == 5
        assert combined.outputs["m"]["x"].shape == (7, 5)

    def test_combine_empty(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot combine empty"):
            combine_regime_results([])

    def test_combine_metadata(self, sample_time_grid):
        """Combined metadata contains regime info."""
        r1 = SimulationResult(
            outputs={"m": {"x": jnp.ones((2, 3))}},
            time_grid=sample_time_grid,
            metadata={"seed": 1},
        )
        r2 = SimulationResult(
            outputs={"m": {"x": jnp.ones((2, 3))}},
            time_grid=sample_time_grid,
            metadata={"seed": 2},
        )
        combined = combine_regime_results([r1, r2])
        assert combined.metadata["n_regimes"] == 2
