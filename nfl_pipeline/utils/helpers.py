"""
Utility functions for the NFL ML Pipeline
Common functions used across different pipeline components.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from datetime import datetime
import json
import warnings
import gc
import psutil
import os
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# PipelineLogger has been moved to utils/logging.py

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage"""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    yield

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = final_memory - initial_memory

    logger.info(f"Memory usage: {final_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")


def force_garbage_collection():
    """Force garbage collection and log memory usage"""
    initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    gc.collect()

    final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    freed_memory = initial_memory - final_memory

    logger.info(f"Garbage collection freed {freed_memory:.1f} MB")


# ============================================================================
# TIMING UTILITIES
# ============================================================================

@contextmanager
def timer(description: str = "Operation"):
    """Context manager for timing operations"""
    start_time = time.time()
    logger.info(f"Starting {description}...")

    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.info(f"{description} completed in {duration:.2f} seconds")


def time_function(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"Function {func.__name__} executed in {duration:.2f} seconds")
        return result
    return wrapper


# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_memory_usage(df: pd.DataFrame) -> float:
    """Get memory usage of DataFrame in MB"""
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def optimize_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame data types for memory efficiency"""
    df_optimized = df.copy()

    # Convert object columns to category if they have few unique values
    for col in df_optimized.select_dtypes(include=['object']):
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:  # Less than 50% unique
            df_optimized[col] = df_optimized[col].astype('category')

    # Downcast numeric types
    for col in df_optimized.select_dtypes(include=['int64']):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

    for col in df_optimized.select_dtypes(include=['float64']):
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

    return df_optimized


def safe_divide(a: Union[float, np.ndarray], b: Union[float, np.ndarray],
                default: float = 0.0) -> Union[float, np.ndarray]:
    """Safe division that handles division by zero"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(a, b, out=np.full_like(a, default, dtype=float),
                          where=(b != 0))
    return result


def angular_difference(angle1: Union[float, np.ndarray],
                      angle2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the smallest angular difference between two angles"""
    diff = angle1 - angle2
    # Normalize to [-180, 180]
    diff = diff % 360
    diff = np.where(diff > 180, diff - 360, diff)
    return np.abs(diff)


def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize angle to [0, 360) range"""
    return angle % 360


def height_to_inches(height_str: str) -> Optional[float]:
    """Convert height string (e.g., '6-2') to inches"""
    if pd.isna(height_str):
        return np.nan  # Return NaN instead of None for proper numeric handling

    try:
        feet, inches = str(height_str).split('-')
        return float(int(feet) * 12 + int(inches))  # Return as float
    except (ValueError, AttributeError):
        return np.nan  # Return NaN instead of None


# ============================================================================
# MODEL EVALUATION UTILITIES
# ============================================================================

# Metric calculation functions have been moved to evaluation/metrics.py


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def plot_feature_importance(feature_names: List[str],
                          importance_values: np.ndarray,
                          title: str = "Feature Importance",
                          top_n: Optional[int] = None,
                          save_path: Optional[Path] = None):
    """Plot feature importance"""
    if top_n:
        # Sort and take top N
        indices = np.argsort(importance_values)[-top_n:]
        feature_names = [feature_names[i] for i in indices]
        importance_values = importance_values[indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_names)), importance_values)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")

    plt.show()


def plot_model_comparison(results: Dict[str, Union[Dict, float]],
                         metric: str = 'rmse',
                         save_path: Optional[Path] = None):
    """Plot model comparison - handles both dict and float values"""
    model_names = list(results.keys())

    # FIXED: Handle both dict values and float values
    metric_values = []
    for name in model_names:
        val = results[name]
        if isinstance(val, dict):
            metric_values.append(val.get(metric, val.get('rmse', 0)))
        else:
            # If it's a float/number, use it directly
            metric_values.append(float(val))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, metric_values)
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45, ha='right')

    # Highlight best model
    best_idx = np.argmin(metric_values)
    bars[best_idx].set_color('green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # logger.info(f"Model comparison plot saved to {save_path}")

    plt.close()  # Close instead of show to avoid blocking


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str = "Residuals Plot",
                  save_path: Optional[Path] = None):
    """Plot residuals analysis"""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')

    # Residuals distribution
    axes[1].hist(residuals, bins=50, alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residuals Distribution')

    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residuals plot saved to {save_path}")

    plt.show()


# ============================================================================
# FILE I/O UTILITIES
# ============================================================================

def save_json(data: Dict, filepath: Path, **kwargs):
    """Save dictionary to JSON file with error handling"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str, **kwargs)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")
        raise


def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file with error handling"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        raise


def ensure_directory(path: Path):
    """Ensure directory exists, create if necessary"""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_dataframe(df: pd.DataFrame,
                      required_columns: Optional[List[str]] = None,
                      dtypes: Optional[Dict[str, str]] = None) -> List[str]:
    """Validate DataFrame structure and return list of issues"""
    issues = []

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")

    # Check data types
    if dtypes:
        for col, expected_dtype in dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if expected_dtype not in actual_dtype:
                    issues.append(f"Column {col}: expected {expected_dtype}, got {actual_dtype}")

    # Check for completely missing columns
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    completely_missing = missing_percentages[missing_percentages == 100].index.tolist()
    if completely_missing:
        issues.append(f"Completely missing columns: {completely_missing}")

    return issues


def validate_config(config) -> List[str]:
    """Validate pipeline configuration"""
    issues = []

    # Check paths exist
    if not config.data_dir.exists():
        issues.append(f"Data directory does not exist: {config.data_dir}")

    # Check model list is not empty
    if not config.models_to_evaluate:
        issues.append("No models specified for evaluation")

    # Check CV parameters
    if config.n_splits < 2:
        issues.append("n_splits must be at least 2")

    return issues


# ExperimentTracker has been moved to utils/tracking.py


# ============================================================================
# CONSTANTS
# ============================================================================

# Football field dimensions (yards)
FIELD_LENGTH = 120  # yards
FIELD_WIDTH = 53.3  # yards

# Common angle conversions
DEG_TO_RAD = np.pi / 180
RAD_TO_DEG = 180 / np.pi

# Common time conversions
FRAMES_PER_SECOND = 10  # NFL tracking data frame rate

# Position group mappings
POSITION_GROUPS = {
    'QB': 'QB',
    'RB': 'RB', 'FB': 'RB',
    'WR': 'WR', 'TE': 'TE',
    'OL': 'OL', 'T': 'OL', 'G': 'OL', 'C': 'OL',
    'DL': 'DL', 'DE': 'DL', 'DT': 'DL', 'NT': 'DL',
    'LB': 'LB', 'ILB': 'LB', 'OLB': 'LB', 'MLB': 'LB',
    'DB': 'DB', 'CB': 'DB', 'S': 'DB', 'SS': 'DB', 'FS': 'DB'
}


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test angular difference
    angle1, angle2 = 10, 350
    diff = angular_difference(angle1, angle2)
    print(f"Angular difference between {angle1}° and {angle2}°: {diff}°")

    # Test height conversion
    height = height_to_inches("6-2")
    print(f"Height 6-2 in inches: {height}")

    # Test safe divide
    result = safe_divide(10, 0, default=999)
    print(f"Safe divide 10/0: {result}")

    print("All utility functions working correctly")