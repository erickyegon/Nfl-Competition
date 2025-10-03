"""
Metrics Module for NFL ML Pipeline
Contains all metric calculation functions for model evaluation.
"""

import numpy as np
from typing import Dict, Union
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)


def calculate_regression_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of regression metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'medae': np.median(np.abs(y_true - y_pred)),  # Median absolute error
    }

    # Additional custom metrics
    metrics['rmse_percentage'] = (metrics['rmse'] / np.mean(y_true)) * 100
    metrics['mae_percentage'] = (metrics['mae'] / np.mean(y_true)) * 100

    return metrics


def calculate_euclidean_distance(pred_x: np.ndarray, pred_y: np.ndarray,
                                 true_x: np.ndarray, true_y: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between predicted and true positions.

    Args:
        pred_x: Predicted X coordinates
        pred_y: Predicted Y coordinates
        true_x: True X coordinates
        true_y: True Y coordinates

    Returns:
        Array of Euclidean distances
    """
    return np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)


def calculate_position_error(y_pred_x: np.ndarray, y_pred_y: np.ndarray,
                             y_true_x: np.ndarray, y_true_y: np.ndarray) -> Dict[str, float]:
    """
    Calculate position-specific error metrics.

    Args:
        y_pred_x: Predicted X coordinates
        y_pred_y: Predicted Y coordinates
        y_true_x: True X coordinates
        y_true_y: True Y coordinates

    Returns:
        Dictionary of position error metrics
    """
    distances = calculate_euclidean_distance(y_pred_x, y_pred_y, y_true_x, y_true_y)

    return {
        'mean_position_error': np.mean(distances),
        'median_position_error': np.median(distances),
        'max_position_error': np.max(distances),
        'position_error_std': np.std(distances),
        'position_error_90th_percentile': np.percentile(distances, 90),
        'position_error_95th_percentile': np.percentile(distances, 95),
    }


def calculate_combined_rmse(x_rmse: float, y_rmse: float) -> float:
    """
    Calculate combined RMSE from X and Y coordinate RMSEs.

    Args:
        x_rmse: RMSE for X coordinate
        y_rmse: RMSE for Y coordinate

    Returns:
        Combined RMSE
    """
    return np.sqrt(x_rmse**2 + y_rmse**2)


def calculate_combined_mae(x_mae: float, y_mae: float) -> float:
    """
    Calculate combined MAE from X and Y coordinate MAEs.

    Args:
        x_mae: MAE for X coordinate
        y_mae: MAE for Y coordinate

    Returns:
        Combined MAE
    """
    return np.sqrt(x_mae**2 + y_mae**2)


def calculate_percentage_improvement(baseline: float, improved: float) -> float:
    """
    Calculate percentage improvement over baseline.

    Args:
        baseline: Baseline metric value
        improved: Improved metric value

    Returns:
        Percentage improvement (positive means improvement)
    """
    return ((baseline - improved) / baseline) * 100


if __name__ == "__main__":
    # Test metrics
    print("Testing metric functions...")

    # Create sample data
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1

    # Test regression metrics
    metrics = calculate_regression_metrics(y_true, y_pred)
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")

    # Test position error
    x_true = np.random.randn(100) * 10
    y_true = np.random.randn(100) * 10
    x_pred = x_true + np.random.randn(100) * 0.5
    y_pred = y_true + np.random.randn(100) * 0.5

    pos_metrics = calculate_position_error(x_pred, y_pred, x_true, y_true)
    print(f"Mean Position Error: {pos_metrics['mean_position_error']:.4f}")

    print("All metric functions working correctly")
