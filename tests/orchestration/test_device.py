"""Tests for hyesg.orchestration.device."""

from __future__ import annotations

import pytest

from hyesg.orchestration.device import DeviceInfo, detect_devices, select_device


class TestDetectDevices:
    """Tests for detect_devices()."""

    def test_returns_device_info(self) -> None:
        """detect_devices should return a DeviceInfo instance."""
        info = detect_devices()
        assert isinstance(info, DeviceInfo)

    def test_platform_is_string(self) -> None:
        """Platform should be a non-empty string."""
        info = detect_devices()
        assert isinstance(info.platform, str)
        assert len(info.platform) > 0

    def test_cpu_always_available(self) -> None:
        """CPU should always be available as a platform."""
        info = detect_devices()
        # In CI and most environments, platform is "cpu"
        assert info.platform in {"cpu", "gpu", "tpu"}

    def test_device_count_positive(self) -> None:
        """Device count should be at least 1."""
        info = detect_devices()
        assert info.device_count >= 1

    def test_device_names_match_count(self) -> None:
        """Number of device names should match device_count."""
        info = detect_devices()
        assert len(info.device_names) == info.device_count

    def test_default_device_is_string(self) -> None:
        """default_device should be a non-empty string."""
        info = detect_devices()
        assert isinstance(info.default_device, str)
        assert len(info.default_device) > 0

    def test_device_info_is_frozen(self) -> None:
        """DeviceInfo should be immutable (frozen dataclass)."""
        info = detect_devices()
        with pytest.raises(AttributeError):
            info.platform = "tpu"  # type: ignore[misc]

    def test_repeated_calls_consistent(self) -> None:
        """Repeated calls should return consistent results."""
        info1 = detect_devices()
        info2 = detect_devices()
        assert info1.platform == info2.platform
        assert info1.device_count == info2.device_count


class TestSelectDevice:
    """Tests for select_device()."""

    def test_select_cpu(self) -> None:
        """Selecting CPU should succeed without error."""
        select_device("cpu")

    def test_select_auto(self) -> None:
        """Selecting auto should succeed without error."""
        select_device("auto")

    def test_select_invalid_raises(self) -> None:
        """Selecting an unavailable device should raise ValueError."""
        with pytest.raises(ValueError, match="not available"):
            select_device("nonexistent_device")

    def test_select_default_is_auto(self) -> None:
        """Default argument should be 'auto'."""
        # Should not raise
        select_device()
