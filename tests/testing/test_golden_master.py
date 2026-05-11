"""Tests for golden master save/load and creation."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from hyesg.testing.golden_master import GoldenMaster

jax.config.update("jax_enable_x64", True)


class TestGoldenMasterSaveLoad:
    """Round-trip save/load tests."""

    def test_save_load_roundtrip(self, golden_master, tmp_path):
        """Saving and loading a golden master preserves all data."""
        path = tmp_path / "test_golden.npz"
        golden_master.save(path)

        loaded = GoldenMaster.load(path)

        assert loaded.name == golden_master.name
        assert set(loaded.outputs.keys()) == set(golden_master.outputs.keys())

        for model in golden_master.outputs:
            for field_name in golden_master.outputs[model]:
                np.testing.assert_array_equal(
                    np.asarray(loaded.outputs[model][field_name]),
                    np.asarray(golden_master.outputs[model][field_name]),
                )

    def test_save_load_preserves_time_grid(self, golden_master, tmp_path):
        """Time grid is preserved through save/load."""
        path = tmp_path / "test_time.npz"
        golden_master.save(path)
        loaded = GoldenMaster.load(path)

        np.testing.assert_array_equal(
            np.asarray(loaded.time_grid),
            np.asarray(golden_master.time_grid),
        )

    def test_save_load_preserves_metadata(self, golden_master, tmp_path):
        """Metadata dict survives serialisation."""
        path = tmp_path / "test_meta.npz"
        golden_master.save(path)
        loaded = GoldenMaster.load(path)

        assert loaded.metadata["csharp_version"] == "3.2.1"
        assert loaded.metadata["created"] == "2024-01-15"
        assert loaded.metadata["config"] == "test_config"

    def test_load_nonexistent_raises(self, tmp_path):
        """Loading a missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GoldenMaster.load(tmp_path / "nonexistent.npz")

    def test_save_creates_file(self, golden_master, tmp_path):
        """Save creates the .npz file on disk."""
        path = tmp_path / "created.npz"
        assert not path.exists()
        golden_master.save(path)
        assert path.exists()

    def test_roundtrip_preserves_dtypes(self, golden_master, tmp_path):
        """Array dtypes are preserved through save/load."""
        path = tmp_path / "dtypes.npz"
        golden_master.save(path)
        loaded = GoldenMaster.load(path)

        for model in golden_master.outputs:
            for field_name in golden_master.outputs[model]:
                orig = golden_master.outputs[model][field_name]
                reloaded = loaded.outputs[model][field_name]
                assert orig.dtype == reloaded.dtype

    def test_roundtrip_preserves_shapes(self, golden_master, tmp_path):
        """Array shapes are preserved through save/load."""
        path = tmp_path / "shapes.npz"
        golden_master.save(path)
        loaded = GoldenMaster.load(path)

        for model in golden_master.outputs:
            for field_name in golden_master.outputs[model]:
                orig = golden_master.outputs[model][field_name]
                reloaded = loaded.outputs[model][field_name]
                assert orig.shape == reloaded.shape

    def test_multiple_save_load_cycles(self, golden_master, tmp_path):
        """Multiple save/load cycles don't degrade data."""
        gm = golden_master
        for i in range(3):
            path = tmp_path / f"cycle_{i}.npz"
            gm.save(path)
            gm = GoldenMaster.load(path)

        for model in golden_master.outputs:
            for field_name in golden_master.outputs[model]:
                np.testing.assert_array_equal(
                    np.asarray(gm.outputs[model][field_name]),
                    np.asarray(golden_master.outputs[model][field_name]),
                )


class TestGoldenMasterFromSimulationResult:
    """Tests for GoldenMaster.from_simulation_result."""

    def test_create_from_result(self, sample_result):
        """Creates a valid golden master from a SimulationResult."""
        gm = GoldenMaster.from_simulation_result(
            sample_result, name="from_result"
        )
        assert gm.name == "from_result"
        assert "model_a" in gm.outputs

    def test_preserves_outputs(self, sample_result):
        """Output arrays reference the same data."""
        gm = GoldenMaster.from_simulation_result(
            sample_result, name="preserve"
        )
        for model in sample_result.outputs:
            for field_name in sample_result.outputs[model]:
                np.testing.assert_array_equal(
                    np.asarray(gm.outputs[model][field_name]),
                    np.asarray(sample_result.outputs[model][field_name]),
                )

    def test_preserves_time_grid(self, sample_result):
        """Time grid is preserved."""
        gm = GoldenMaster.from_simulation_result(
            sample_result, name="tg"
        )
        np.testing.assert_array_equal(
            np.asarray(gm.time_grid),
            np.asarray(sample_result.time_grid),
        )

    def test_extra_metadata_merged(self, sample_result):
        """Extra keyword metadata is merged with result metadata."""
        gm = GoldenMaster.from_simulation_result(
            sample_result,
            name="meta",
            csharp_version="3.0",
            notes="test run",
        )
        assert gm.metadata["csharp_version"] == "3.0"
        assert gm.metadata["notes"] == "test run"
        assert gm.metadata["test"] is True  # from sample_result

    def test_roundtrip_via_file(self, sample_result, tmp_path):
        """Created golden master can be saved and loaded."""
        gm = GoldenMaster.from_simulation_result(
            sample_result, name="roundtrip"
        )
        path = tmp_path / "rt.npz"
        gm.save(path)
        loaded = GoldenMaster.load(path)
        assert loaded.name == "roundtrip"
        assert set(loaded.outputs.keys()) == set(gm.outputs.keys())
