"""Economy-level configuration for multi-economy ESG simulations.

Defines the ``Economy`` dataclass that groups all model configurations
for a single currency zone (e.g. GBP, USD, EUR).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EconomyModelConfig(BaseModel):
    """Configuration reference for a model instance within an economy.

    Unlike the simulation-level ``ModelConfig``, this captures the
    *structural* role of each model in the economy without wiring
    dependencies or outputs — those are resolved later by the
    dependency graph and correlation assembler.

    Attributes:
        model_type: Model type key (e.g. ``"cir2pp"``, ``"g2pp"``,
            ``"gbm"``, ``"fx_gbm"``).
        label: Unique label across all economies (e.g. ``"gbp_nominal"``).
        params: Model-specific parameters as a dict.
    """

    model_config = ConfigDict(frozen=True)

    model_type: str
    label: str
    params: dict = Field(default_factory=dict)


class Economy(BaseModel):
    """Economy specification — one per currency zone.

    Groups all model configurations that belong to a single economy,
    including rates, inflation, equities, credit, and salary models.
    The domestic economy is distinguished by ``is_domestic=True``.

    Attributes:
        name: Economy identifier (e.g. ``"GBP"``, ``"USD"``).
        is_domestic: Whether this is the domestic (base) economy.
        nominal_rate_model: Nominal short-rate model (required).
        real_rate_model: Real short-rate model (optional).
        inflation_model: Inflation model (optional).
        fx_model: FX exchange rate model (optional, absent for domestic).
        equity_models: List of equity/growth-asset models.
        credit_pool: Credit default-intensity model (optional).
        salary_model: Salary/wage model (optional).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    is_domestic: bool = False
    nominal_rate_model: EconomyModelConfig
    real_rate_model: Optional[EconomyModelConfig] = None
    inflation_model: Optional[EconomyModelConfig] = None
    fx_model: Optional[EconomyModelConfig] = None
    equity_models: list[EconomyModelConfig] = Field(default_factory=list)
    credit_pool: Optional[EconomyModelConfig] = None
    salary_model: Optional[EconomyModelConfig] = None

    @property
    def all_models(self) -> list[EconomyModelConfig]:
        """Return all model configs in dependency order.

        Order follows the natural dependency chain:
        nominal → fx → real → inflation → equities → credit → salary.

        Returns:
            List of ``EconomyModelConfig`` in execution-safe order.
        """
        models: list[EconomyModelConfig] = [self.nominal_rate_model]
        if self.fx_model:
            models.append(self.fx_model)
        if self.real_rate_model:
            models.append(self.real_rate_model)
        if self.inflation_model:
            models.append(self.inflation_model)
        models.extend(self.equity_models)
        if self.credit_pool:
            models.append(self.credit_pool)
        if self.salary_model:
            models.append(self.salary_model)
        return models
