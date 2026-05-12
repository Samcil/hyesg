"""Tests for CSV round-trip IO."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from hyesg.io.csv_io import from_csv, to_csv

jax.config.update("jax_enable_x64", True)

ATOL = 1e-10


# ---------------------------------------------------------------
# Tidy format
# ---------------------------------------------------------------


class TestTidyRoundTrip:
    """Round-trip tests for the tidy CSV layout."""

    def test_round_trip_values(self, sample_result, tmp_path):
        """Written then read values match within tolerance."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")

        for model in sample_result.model_names:
            for field in sorted(sample_result.outputs[model]):
                expected = np.asarray(sample_result.select(model, field))
                actual = np.asarray(loaded.select(model, field))
                np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_round_trip_time_grid(self, sample_result, tmp_path):
        """Time grid survives round-trip."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        np.testing.assert_allclose(
            np.asarray(loaded.time_grid[:sample_result.n_steps]),
            np.asarray(sample_result.time_grid[:sample_result.n_steps]),
            atol=ATOL,
        )

    def test_round_trip_metadata(self, sample_result, tmp_path):
        """Metadata round-trips via JSON sidecar."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        assert loaded.metadata["seed"] == 42
        assert loaded.metadata["description"] == "test simulation"

    def test_round_trip_shape(self, sample_result, tmp_path):
        """n_trials and n_steps are preserved."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        assert loaded.n_trials == sample_result.n_trials
        assert loaded.n_steps == sample_result.n_steps

    def test_model_names_preserved(self, sample_result, tmp_path):
        """Model names survive round-trip."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        assert loaded.model_names == sample_result.model_names

    def test_large_dataset(self, large_result, tmp_path):
        """100 trials × 12 steps round-trips correctly."""
        to_csv(large_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        assert loaded.n_trials == 100
        assert loaded.n_steps == 12
        for model in large_result.model_names:
            for field in sorted(large_result.outputs[model]):
                expected = np.asarray(large_result.select(model, field))
                actual = np.asarray(loaded.select(model, field))
                np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_single_field(self, single_field_result, tmp_path):
        """Single model / single field."""
        to_csv(single_field_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        expected = np.asarray(single_field_result.select("cir", "ShortRate"))
        actual = np.asarray(loaded.select("cir", "ShortRate"))
        np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_creates_results_csv(self, sample_result, tmp_path):
        """Tidy format creates results.csv."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        assert (tmp_path / "out" / "results.csv").exists()

    def test_creates_metadata_json(self, sample_result, tmp_path):
        """metadata.json is always created."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        assert (tmp_path / "out" / "metadata.json").exists()


# ---------------------------------------------------------------
# Wide format
# ---------------------------------------------------------------


class TestWideRoundTrip:
    """Round-trip tests for the wide CSV layout."""

    def test_round_trip_values(self, sample_result, tmp_path):
        """Written then read values match within tolerance."""
        to_csv(sample_result, tmp_path / "out", fmt="wide")
        loaded = from_csv(tmp_path / "out", fmt="wide")

        for model in sample_result.model_names:
            for field in sorted(sample_result.outputs[model]):
                expected = np.asarray(sample_result.select(model, field))
                actual = np.asarray(loaded.select(model, field))
                np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_round_trip_shape(self, sample_result, tmp_path):
        """Shape is preserved."""
        to_csv(sample_result, tmp_path / "out", fmt="wide")
        loaded = from_csv(tmp_path / "out", fmt="wide")
        assert loaded.n_trials == sample_result.n_trials
        assert loaded.n_steps == sample_result.n_steps

    def test_one_csv_per_model(self, sample_result, tmp_path):
        """Wide format creates one CSV per model."""
        to_csv(sample_result, tmp_path / "out", fmt="wide")
        csv_files = sorted(
            p.name for p in (tmp_path / "out").glob("*.csv")
        )
        assert "equity.csv" in csv_files
        assert "rates.csv" in csv_files

    def test_large_dataset_wide(self, large_result, tmp_path):
        """100 trials × 12 steps in wide format."""
        to_csv(large_result, tmp_path / "out", fmt="wide")
        loaded = from_csv(tmp_path / "out", fmt="wide")
        assert loaded.n_trials == 100
        assert loaded.n_steps == 12

    def test_metadata_wide(self, sample_result, tmp_path):
        """Metadata round-trips in wide format."""
        to_csv(sample_result, tmp_path / "out", fmt="wide")
        loaded = from_csv(tmp_path / "out", fmt="wide")
        assert loaded.metadata["seed"] == 42


# ---------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------


class TestAutoDetect:
    """Format auto-detection."""

    def test_detects_tidy(self, sample_result, tmp_path):
        """Auto-detects tidy from results.csv."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out")  # no fmt
        assert loaded.n_trials == sample_result.n_trials

    def test_detects_wide(self, sample_result, tmp_path):
        """Auto-detects wide from model CSVs."""
        to_csv(sample_result, tmp_path / "out", fmt="wide")
        loaded = from_csv(tmp_path / "out")  # no fmt
        assert loaded.n_trials == sample_result.n_trials


# ---------------------------------------------------------------
# Edge cases and errors
# ---------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and error handling."""

    def test_empty_result_tidy(self, empty_result, tmp_path):
        """Empty result writes without error."""
        to_csv(empty_result, tmp_path / "out", fmt="tidy")
        assert (tmp_path / "out" / "results.csv").exists()

    def test_invalid_fmt_raises(self, sample_result, tmp_path):
        """Invalid format string raises ValueError."""
        with pytest.raises(ValueError, match="fmt must be"):
            to_csv(sample_result, tmp_path / "out", fmt="parquet")

    def test_missing_dir_raises(self, tmp_path):
        """Reading from non-existent dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            from_csv(tmp_path / "nonexistent")

    def test_metadata_json_content(self, sample_result, tmp_path):
        """metadata.json is valid JSON with expected keys."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        with open(tmp_path / "out" / "metadata.json") as fh:
            meta = json.load(fh)
        assert meta["seed"] == 42
        assert meta["dt"] == 0.25

    def test_no_csv_files_raises(self, tmp_path):
        """Empty directory raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No CSV files"):
            from_csv(empty_dir)

    def test_output_is_jax_array(self, sample_result, tmp_path):
        """Loaded arrays are JAX arrays."""
        to_csv(sample_result, tmp_path / "out", fmt="tidy")
        loaded = from_csv(tmp_path / "out", fmt="tidy")
        arr = loaded.select("rates", "ShortRate")
        assert hasattr(arr, "__jax_array__") or "jax" in type(arr).__module__