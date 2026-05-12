"""Tests for Parquet round-trip IO."""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from hyesg.io.parquet_io import from_parquet, to_parquet  # noqa: E402

ATOL = 1e-12


class TestParquetRoundTrip:
    """Round-trip tests for the Parquet format."""

    def test_round_trip_values(self, sample_result, tmp_path):
        """Values survive round-trip."""
        to_parquet(sample_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")

        for model in sample_result.model_names:
            for field in sorted(sample_result.outputs[model]):
                expected = np.asarray(sample_result.select(model, field))
                actual = np.asarray(loaded.select(model, field))
                np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_round_trip_time_grid(self, sample_result, tmp_path):
        """Time grid is preserved."""
        to_parquet(sample_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        np.testing.assert_allclose(
            np.asarray(loaded.time_grid),
            np.asarray(sample_result.time_grid),
            atol=ATOL,
        )

    def test_round_trip_shape(self, sample_result, tmp_path):
        """Shape is preserved."""
        to_parquet(sample_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        assert loaded.n_trials == sample_result.n_trials
        assert loaded.n_steps == sample_result.n_steps

    def test_model_names(self, sample_result, tmp_path):
        """Model names match."""
        to_parquet(sample_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        assert loaded.model_names == sample_result.model_names

    def test_metadata_round_trip(self, sample_result, tmp_path):
        """Metadata survives via Parquet file metadata."""
        to_parquet(sample_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        assert loaded.metadata["seed"] == 42
        assert loaded.metadata["description"] == "test simulation"

    def test_metadata_in_parquet_schema(self, sample_result, tmp_path):
        """Metadata is stored in Parquet schema metadata."""
        to_parquet(sample_result, tmp_path / "pq")
        pq_files = list((tmp_path / "pq").glob("*.parquet"))
        model_files = [f for f in pq_files if not f.name.startswith("_")]
        assert len(model_files) > 0

        table = pq.read_table(model_files[0])
        schema_meta = table.schema.metadata
        assert b"hyesg_metadata" in schema_meta

    def test_large_dataset(self, large_result, tmp_path):
        """100 trials × 12 steps."""
        to_parquet(large_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        assert loaded.n_trials == 100
        assert loaded.n_steps == 12

        for model in large_result.model_names:
            for field in sorted(large_result.outputs[model]):
                expected = np.asarray(large_result.select(model, field))
                actual = np.asarray(loaded.select(model, field))
                np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_single_field(self, single_field_result, tmp_path):
        """Single model / single field."""
        to_parquet(single_field_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        expected = np.asarray(single_field_result.select("cir", "ShortRate"))
        actual = np.asarray(loaded.select("cir", "ShortRate"))
        np.testing.assert_allclose(actual, expected, atol=ATOL)

    def test_creates_parquet_files(self, sample_result, tmp_path):
        """One .parquet per model plus _time_grid.parquet."""
        to_parquet(sample_result, tmp_path / "pq")
        pq_files = sorted(p.name for p in (tmp_path / "pq").glob("*.parquet"))
        assert "_time_grid.parquet" in pq_files
        assert "equity.parquet" in pq_files
        assert "rates.parquet" in pq_files

    def test_missing_dir_raises(self, tmp_path):
        """Reading from non-existent dir raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            from_parquet(tmp_path / "nonexistent")

    def test_output_is_jax_array(self, sample_result, tmp_path):
        """Loaded arrays are JAX arrays."""
        to_parquet(sample_result, tmp_path / "pq")
        loaded = from_parquet(tmp_path / "pq")
        arr = loaded.select("rates", "ShortRate")
        assert hasattr(arr, "__jax_array__") or "jax" in type(arr).__module__