"""Tests for the streaming writer."""

from __future__ import annotations

import csv
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hyesg.io.csv_io import from_csv
from hyesg.io.streaming import StreamingWriter

jax.config.update("jax_enable_x64", True)


class TestStreamingWriter:
    """StreamingWriter tests."""

    def test_basic_write(self, tmp_path):
        """StreamingWriter produces a valid CSV."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)

        n_trials = 3
        for step in range(4):
            t = float(step) * 0.25
            step_outputs = {
                "model_a": {
                    "ShortRate": jnp.ones(n_trials) * (step + 1) * 0.01,
                },
            }
            writer.write_timestep(t, step_outputs)

        writer.finalize(metadata={"seed": 7})
        assert (out_dir / "results.csv").exists()
        assert (out_dir / "metadata.json").exists()

    def test_timesteps_written_counter(self, tmp_path):
        """timesteps_written property tracks writes."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)
        assert writer.timesteps_written == 0

        writer.write_timestep(0.0, {"m": {"f": jnp.array([1.0])}})
        assert writer.timesteps_written == 1

        writer.write_timestep(0.5, {"m": {"f": jnp.array([2.0])}})
        assert writer.timesteps_written == 2
        writer.finalize()

    def test_csv_header(self, tmp_path):
        """CSV header is written correctly."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)
        writer.write_timestep(0.0, {"m": {"f": jnp.array([1.0])}})
        writer.finalize()

        with open(out_dir / "results.csv") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == ["time", "trial", "model", "variable", "value"]

    def test_csv_row_count(self, tmp_path):
        """CSV has correct number of data rows."""
        out_dir = tmp_path / "stream"
        n_trials, n_steps = 3, 5
        writer = StreamingWriter(out_dir)

        for step in range(n_steps):
            writer.write_timestep(
                float(step),
                {"m": {"f": jnp.ones(n_trials)}},
            )
        writer.finalize()

        with open(out_dir / "results.csv") as fh:
            rows = list(csv.reader(fh))
        # 1 header + n_trials * n_steps data rows
        assert len(rows) == 1 + n_trials * n_steps

    def test_output_readable_by_from_csv(self, tmp_path):
        """StreamingWriter output is compatible with from_csv."""
        out_dir = tmp_path / "stream"
        n_trials, n_steps = 2, 3
        key = jax.random.PRNGKey(42)

        writer = StreamingWriter(out_dir)
        expected_values = []
        for step in range(n_steps):
            k1, key = jax.random.split(key)
            vals = jax.random.normal(k1, (n_trials,))
            expected_values.append(vals)
            writer.write_timestep(
                float(step),
                {"mymodel": {"output": vals}},
            )
        writer.finalize(metadata={"seed": 42})

        loaded = from_csv(out_dir, fmt="tidy")
        assert loaded.n_trials == n_trials
        assert loaded.n_steps == n_steps
        arr = np.asarray(loaded.select("mymodel", "output"))
        for step in range(n_steps):
            np.testing.assert_allclose(
                arr[:, step],
                np.asarray(expected_values[step]),
                atol=1e-10,
            )

    def test_multiple_models_fields(self, tmp_path):
        """StreamingWriter handles multiple models and fields."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)

        for step in range(2):
            writer.write_timestep(
                float(step),
                {
                    "model_a": {
                        "x": jnp.array([1.0, 2.0]),
                        "y": jnp.array([3.0, 4.0]),
                    },
                    "model_b": {
                        "z": jnp.array([5.0, 6.0]),
                    },
                },
            )
        writer.finalize()

        loaded = from_csv(out_dir, fmt="tidy")
        assert sorted(loaded.model_names) == ["model_a", "model_b"]
        assert loaded.n_trials == 2
        assert loaded.n_steps == 2

    def test_context_manager(self, tmp_path):
        """StreamingWriter works as a context manager."""
        out_dir = tmp_path / "stream"
        with StreamingWriter(out_dir) as writer:
            writer.write_timestep(0.0, {"m": {"f": jnp.array([1.0])}})
        # File should be closed after exiting context
        assert (out_dir / "results.csv").exists()

    def test_finalize_writes_metadata(self, tmp_path):
        """finalize() writes metadata.json."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)
        writer.write_timestep(0.0, {"m": {"f": jnp.array([1.0])}})
        writer.finalize(metadata={"key": "value"})

        with open(out_dir / "metadata.json") as fh:
            meta = json.load(fh)
        assert meta["key"] == "value"

    def test_finalize_no_metadata(self, tmp_path):
        """finalize() without metadata writes empty dict."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)
        writer.write_timestep(0.0, {"m": {"f": jnp.array([1.0])}})
        writer.finalize()

        with open(out_dir / "metadata.json") as fh:
            meta = json.load(fh)
        assert meta == {}

    def test_invalid_format_raises(self, tmp_path):
        """Non-csv format raises ValueError."""
        with pytest.raises(ValueError, match="only supports fmt='csv'"):
            StreamingWriter(tmp_path / "stream", fmt="parquet")

    def test_scalar_array_per_trial(self, tmp_path):
        """Scalar (0-d) arrays are handled gracefully."""
        out_dir = tmp_path / "stream"
        writer = StreamingWriter(out_dir)
        writer.write_timestep(0.0, {"m": {"f": jnp.float64(3.14)}})
        writer.finalize()

        with open(out_dir / "results.csv") as fh:
            rows = list(csv.reader(fh))
        # header + 1 data row
        assert len(rows) == 2
