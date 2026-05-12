"""Device detection and selection for JAX compute backends.

Provides ``DeviceInfo`` and utilities to auto-detect or manually
select the JAX compute device (CPU, GPU, TPU).

.. todo::
    **F40 Integration Path** — Device selection should be invoked
    as a ``PipelineStep`` early in the orchestration pipeline,
    before ``SimulateStep``.  The chosen device should be stored in
    ``PipelineContext.metadata["device"]`` so that downstream steps
    can query the active backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Information about available compute devices.

    Attributes:
        platform: Backend platform (``"cpu"``, ``"gpu"``, ``"tpu"``).
        device_count: Number of devices of the selected platform.
        device_names: Human-readable names of each device.
        default_device: Name of the default device.
    """

    platform: str
    device_count: int
    device_names: list[str]
    default_device: str


def detect_devices() -> DeviceInfo:
    """Auto-detect available JAX devices.

    Returns:
        DeviceInfo describing the current JAX backend.
    """
    devices = jax.devices()
    platform = devices[0].platform if devices else "cpu"
    device_names = [str(d) for d in devices]
    default_device = str(jax.devices()[0]) if devices else "cpu:0"

    return DeviceInfo(
        platform=platform,
        device_count=len(devices),
        device_names=device_names,
        default_device=default_device,
    )


def select_device(preference: str = "auto") -> None:
    """Select the compute device for JAX.

    Args:
        preference: One of ``"cpu"``, ``"gpu"``, ``"tpu"``, or
            ``"auto"`` (best available).

    Raises:
        ValueError: If the requested device is not available.
    """
    if preference == "auto":
        # JAX default is already "best available"
        info = detect_devices()
        logger.info("Auto-selected device: %s", info.default_device)
        return

    available_platforms = {d.platform for d in jax.devices()}

    if preference not in available_platforms:
        raise ValueError(
            f"Requested device '{preference}' is not available. "
            f"Available platforms: {sorted(available_platforms)}"
        )

    jax.config.update("jax_default_device", jax.devices(preference)[0])
    logger.info("Selected device: %s", preference)
