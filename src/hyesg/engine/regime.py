"""Regime system — proportional trial ordering and regime specifications.

Provides the regime trial map that interleaves trials from multiple
economic regimes proportionally (largest-remainder greedy), matching
the C# ESG engine's interleaving behaviour.

Example with 3 regimes (Strong=2500, Moderate=1500, Weak=1000)::

    Total = 5000
    Proportions: [0.50, 0.30, 0.20]
    Ordering: [S, M, S, W, S, M, S, S, W, M, ...] (interleaved)

All functions are pure and compatible with ``jax.jit``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence


class RegimeSpec(NamedTuple):
    """Specification for a single economic regime.

    Attributes:
        name: Human-readable regime label (e.g. "Strong").
        trials: Number of trials assigned to this regime.
        weight: Regime probability weight (informational; not used
            for trial ordering — ``trials`` is authoritative).
        seed_offset: Optional offset added to the master seed for
            this regime's PRNG derivation.
    """

    name: str
    trials: int
    weight: float
    seed_offset: int = 0


class RegimeTrialMap(NamedTuple):
    """Mapping from global trial index to regime index.

    Attributes:
        trial_to_regime: Tuple of length ``total_trials`` where
            entry *i* gives the regime index for global trial *i*.
        regime_trial_counts: Tuple of cumulative trial counts per
            regime (useful for slicing results).
        total_trials: Sum of all regime trial counts.
    """

    trial_to_regime: tuple[int, ...]
    regime_trial_counts: tuple[int, ...]
    total_trials: int


def build_proportional_trial_map(
    regimes: Sequence[RegimeSpec],
) -> RegimeTrialMap:
    """Build interleaved trial ordering using largest-remainder greedy.

    Given regimes with trial counts, interleave proportionally so that
    each regime's trials are spread evenly across the global trial
    sequence rather than being grouped in contiguous blocks.

    Algorithm:
        1. Compute proportion for each regime: ``p_i = trials_i / total``.
        2. For each trial position, assign to the regime with the
           largest fractional remainder (greedy).  After assignment,
           increment that regime's counter and recompute remainders.

    This produces a maximally interleaved ordering that matches the
    C# engine's ``BuildProportionalTrialMap`` behaviour.

    Args:
        regimes: Sequence of :class:`RegimeSpec` objects.

    Returns:
        :class:`RegimeTrialMap` with the trial-to-regime mapping.

    Raises:
        ValueError: If *regimes* is empty or any regime has
            non-positive trial count.
    """
    if not regimes:
        msg = "At least one regime must be provided"
        raise ValueError(msg)

    n_regimes = len(regimes)
    trial_counts = [r.trials for r in regimes]

    if any(t <= 0 for t in trial_counts):
        msg = f"All regime trial counts must be positive, got {trial_counts}"
        raise ValueError(msg)

    total = sum(trial_counts)
    proportions = [t / total for t in trial_counts]

    # Greedy largest-remainder assignment
    assigned = [0] * n_regimes
    ordering: list[int] = []

    for _ in range(total):
        # Fractional remainder = proportion * total_so_far_plus_one - assigned
        # We want the regime whose "ideal cumulative" is furthest ahead
        # of its actual assignment count.
        best_regime = -1
        best_remainder = -1.0

        for r_idx in range(n_regimes):
            if assigned[r_idx] >= trial_counts[r_idx]:
                continue
            # How many trials regime r_idx *should* have by now
            ideal = proportions[r_idx] * (len(ordering) + 1)
            remainder = ideal - assigned[r_idx]
            if remainder > best_remainder:
                best_remainder = remainder
                best_regime = r_idx

        ordering.append(best_regime)
        assigned[best_regime] += 1

    # Build cumulative counts
    cumulative: list[int] = []
    running = 0
    for count in trial_counts:
        running += count
        cumulative.append(running)

    return RegimeTrialMap(
        trial_to_regime=tuple(ordering),
        regime_trial_counts=tuple(cumulative),
        total_trials=total,
    )
