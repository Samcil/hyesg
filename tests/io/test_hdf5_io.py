"""Tests for HDF5 round-trip IO."""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

h5py = pytest.importorskip("h5py")

from hyesg.io.hdf5_io import from_hdf5, to_hdf5  # noqa: E402

ATOL = 1e-12


class TestHdf5RoundTrip:
    """Round-trip tests for the HDF5 format."""

    def test_round_trip_values(self, sample_result, tmp_path):
        """Values survive round-trip."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)
        loaded = from_hdf5(hdf_path)

        for model in sample_result.model_names:
            for field in sorted(sample_result.outputs[model]):
                expected = np.asarray(sample_result.select(model, field))
                actual = np.asarray(loaded.select(model, field))
                np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_round_trip_time_grid(self, sample_result, tmp_path):
        """Time grid is preserved."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        np.testing.assert_allclose(
            np.asarray(loaded.time_grid),
            np.asarray(sample_result.time_grid),
            atol=ATOL,
        )

    def test_round_trip_shape(self, sample_result, tmp_path):
        """Shape is preserved."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        assert loaded.n_trials == sample_result.n_trials
        assert loaded.n_steps == sample_result.n_steps

    def test_model_names(self, sample_result, tmp_path):
        """Model names match."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        assert loaded.model_names == sample_result.model_names

    def test_metadata_round_trip(self, sample_result, tmp_path):
        """Metadata survives round-trip."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        assert loaded.metadata["seed"] == 42
        assert loaded.metadata["description"] == "test simulation"

    def test_hdf5_group_structure(self, sample_result, tmp_path):
        """HDF5 file has expected group hierarchy."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)

        with h5py.File(hdf_path, "r") as f:
            assert "time_grid" in f
            assert "models" in f
            assert "metadata" in f
            assert "rates" in f["models"]
            assert "equity" in f["models"]
            assert "ShortRate" in f["models"]["rates"]
            assert "forward_rate" in f["models"]["rates"]

    def test_hdf5_dataset_shapes(self, sample_result, tmp_path):
        """Datasets have correct shapes."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)

        with h5py.File(hdf_path, "r") as f:
            assert f["time_grid"].shape == (sample_result.n_steps + 1,)
            assert f["models"]["rates"]["ShortRate"].shape == (4, 3)

    def test_large_dataset(self, large_result, tmp_path):
        """100 trials × 12 steps."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(large_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        assert loaded.n_trials == 100
        assert loaded.n_steps == 12

    def test_single_field(self, single_field_result, tmp_path):
        """Single model / single field."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(single_field_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        expected = np.asarray(single_field_result.select("cir", "ShortRate"))
        actual = np.asarray(loaded.select("cir", "ShortRate"))
        np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_missing_file_raises(self, tmp_path):
        """Reading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            from_hdf5(tmp_path / "nonexistent.h5")

    def test_output_is_jax_array(self, sample_result, tmp_path):
        """Loaded arrays are JAX arrays."""
        hdf_path = tmp_path / "results.h5"
        to_hdf5(sample_result, hdf_path)
        loaded = from_hdf5(hdf_path)
        arr = loaded.select("rates", "ShortRate")
        assert hasattr(arr, "__jax_array__") or "jax" in type(arr).__module__