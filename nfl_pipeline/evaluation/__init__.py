"""
Evaluation module for model assessment and selection.
"""

from nfl_pipeline.evaluation.evaluator import ModelEvaluator
from nfl_pipeline.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_euclidean_distance,
    calculate_position_error,
    calculate_combined_rmse,
    calculate_combined_mae
)
from nfl_pipeline.evaluation.selector import ModelSelector

__all__ = [
    'ModelEvaluator',
    'ModelSelector',
    'calculate_regression_metrics',
    'calculate_euclidean_distance',
    'calculate_position_error',
    'calculate_combined_rmse',
    'calculate_combined_mae'
]
