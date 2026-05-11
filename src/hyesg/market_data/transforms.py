"""Transform market data into simulation curve inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hyesg.math.curves.protocol import ParametricCurve

if TYPE_CHECKING:
    from hyesg.config.models import SimulationConfig
    from hyesg.market_data.snapshot import MarketDataSnapshot


def to_simulation_curves(
    snapshot: MarketDataSnapshot,
    config: SimulationConfig,
) -> dict[str, ParametricCurve]:
    """Map market data to the curves required by each model.

    Inspects ``config.models`` and matches model types to the
    appropriate curves from the snapshot:

    * ``"cir2pp"`` / ``"g1pp"`` → nominal zero curve
    * ``"g2pp"`` → real (inflation) zero curve
    * ``"credit"`` → credit spread curve (requires ``rating``
      and ``currency`` in model params)

    Models that do not match any known type are silently skipped.

    Args:
        snapshot: A frozen :class:`MarketDataSnapshot` with
            pre-loaded curves.
        config: Simulation configuration describing the models.

    Returns:
        Mapping of ``model_name → ParametricCurve`` for every
        model whose curve requirement could be satisfied from the
        snapshot.
    """
    result: dict[str, ParametricCurve] = {}

    nominal_model_types = {"cir2pp", "g1pp", "cirpp", "vasicek", "cir"}
    real_model_types = {"g2pp"}
    credit_model_types = {"credit"}

    for model in config.models:
        model_type = model.type.lower()

        if model_type in nominal_model_types:
            currency = model.params.get("currency", "GBP")
            curve = snapshot.zero_curves.get(currency)
            if curve is not None:
                result[model.name] = curve

        elif model_type in real_model_types:
            currency = model.params.get("currency", "GBP")
            curve = snapshot.inflation_curves.get(currency)
            if curve is not None:
                result[model.name] = curve

        elif model_type in credit_model_types:
            rating = model.params.get("rating", "AAA")
            currency = model.params.get("currency", "GBP")
            rating_map = snapshot.credit_curves.get(rating, {})
            curve = rating_map.get(currency)
            if curve is not None:
                result[model.name] = curve

    return result
