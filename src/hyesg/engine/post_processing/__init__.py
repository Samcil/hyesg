"""Post-processing pipeline for simulation results.

Provides 14 processor types, a recipe-based chaining mechanism,
output path specifications, and inner Monte Carlo utilities.
"""

from __future__ import annotations

from hyesg.engine.post_processing.inner_mc import (
    MemoizingSkipForward,
    double_antithetic_paths,
)
from hyesg.engine.post_processing.output_paths import OutputPathSpec
from hyesg.engine.post_processing.processors import (
    AnnualisationProcessor,
    ConditionalExpectationProcessor,
    CurrencyConversionProcessor,
    CustomProcessor,
    EquilibriumSwapRateProcessor,
    EstimateFilterProcessor,
    FeeDeductionProcessor,
    InflationAdjustmentProcessor,
    LSMCRegressionProcessor,
    OutputFormattingProcessor,
    PathStatisticsProcessor,
    PercentileExtractionProcessor,
    PortfolioAggregationProcessor,
    SABRCalibrationProcessor,
)
from hyesg.engine.post_processing.protocol import (
    PostProcessor,
    ProcessedResults,
    SimulationResults,
)
from hyesg.engine.post_processing.recipes import (
    CompositeProcessor,
    PostProcessingRecipe,
)

__all__ = [
    "AnnualisationProcessor",
    "CompositeProcessor",
    "ConditionalExpectationProcessor",
    "CurrencyConversionProcessor",
    "CustomProcessor",
    "EquilibriumSwapRateProcessor",
    "EstimateFilterProcessor",
    "FeeDeductionProcessor",
    "InflationAdjustmentProcessor",
    "LSMCRegressionProcessor",
    "MemoizingSkipForward",
    "OutputFormattingProcessor",
    "OutputPathSpec",
    "PathStatisticsProcessor",
    "PercentileExtractionProcessor",
    "PortfolioAggregationProcessor",
    "PostProcessingRecipe",
    "PostProcessor",
    "ProcessedResults",
    "SABRCalibrationProcessor",
    "SimulationResults",
    "double_antithetic_paths",
]
