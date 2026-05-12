"""Workload distribution for multi-device parallel execution.

Distributes Monte Carlo trials across compute devices (GPUs/TPUs)
with optional even-number rounding for antithetic variance reduction.

All functions are pure.
"""

from __future__ import annotations


def distribute_trials(
    total_trials: int,
    n_devices: int,
    require_even: bool = True,
) -> tuple[int, ...]:
    """Distribute trials across devices for parallel execution.

    Splits ``total_trials`` as evenly as possible across ``n_devices``.
    When ``require_even`` is True, each device's share is rounded up
    to an even number so that antithetic pairing (original + negated)
    works without leftover trials.

    Args:
        total_trials: Total number of trials to distribute.
        n_devices: Number of compute devices.
        require_even: If ``True``, round each device's share up to
            an even number for antithetic pairing.

    Returns:
        Tuple of trial counts per device.  The sum may be ≥
        ``total_trials`` when rounding is applied.

    Raises:
        ValueError: If *total_trials* < 1 or *n_devices* < 1.
    """
    if total_trials < 1:
        msg = f"total_trials must be >= 1, got {total_trials}"
        raise ValueError(msg)
    if n_devices < 1:
        msg = f"n_devices must be >= 1, got {n_devices}"
        raise ValueError(msg)

    base = total_trials // n_devices
    if require_even and base % 2 != 0:
        base += 1

    remainder = total_trials - base * (n_devices - 1)
    if remainder < 1:
        remainder = base
    if require_even and remainder % 2 != 0:
        remainder += 1

    return tuple([base] * (n_devices - 1) + [remainder])
