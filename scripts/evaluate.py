"""
Evaluation script for NFL models.

Evaluates model performance on validation or test data.

Usage:
    python scripts/evaluate.py --model path/to/model.pkl
    python scripts/evaluate.py --experiment experiment_name
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_pipeline.core.config import get_quick_config
from nfl_pipeline.data.loader import DataLoader
from nfl_pipeline.data.preprocessor import DataPreprocessor
from nfl_pipeline.features.nfl_domain import FeatureEngineer
from nfl_pipeline.evaluation.metrics import (
    calculate_regression_metrics,
    calculate_position_error,
    calculate_combined_rmse
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate NFL model performance')
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='Experiment name to load model from'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Directory containing saved model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Dataset to evaluate on'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='quick',
        choices=['quick', 'lstm', 'full'],
        help='Configuration preset to use'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config == 'quick':
        from nfl_pipeline import get_quick_config
        config = get_quick_config()
    elif args.config == 'lstm':
        from nfl_pipeline import get_lstm_config
        config = get_lstm_config()
    elif args.config == 'full':
        from nfl_pipeline import get_full_config
        config = get_full_config()

    print("=" * 80)
    print("NFL MODEL EVALUATION")
    print("=" * 80)

    # Determine model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif args.experiment:
        model_dir = config.models_dir / args.experiment
    else:
        model_dir = config.models_dir / config.experiment_name

    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return

    print(f"Loading model from: {model_dir}")
    print(f"Evaluating on: {args.dataset} dataset")

    # Load data
    print("\nLoading and preprocessing data...")
    loader = DataLoader(config)
    data = loader.load_data()

    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess(data)

    # Feature engineering
    print("Generating features...")
    feature_engineer = FeatureEngineer(config)

    if args.dataset == 'train':
        X, y = feature_engineer.create_features(processed_data['train'])
    elif args.dataset == 'val':
        X, y = feature_engineer.create_features(processed_data['val'])
    else:  # test
        X, y = feature_engineer.create_features(processed_data['test'])

    # Load model and make predictions
    print("\nMaking predictions...")
    try:
        # Try to load model (implementation depends on how models are saved)
        # This is a placeholder - actual implementation depends on model format
        from nfl_pipeline.prediction.predictor import NFLPredictor

        predictor = NFLPredictor(config)
        predictions = predictor.predict_from_saved_model(X, model_dir)

        pred_x = predictions[:, 0]
        pred_y = predictions[:, 1]
        true_x = y[:, 0]
        true_y = y[:, 1]

        # Calculate metrics
        print("\n" + "=" * 80)
        print(f"EVALUATION RESULTS - {args.dataset.upper()} SET")
        print("=" * 80)

        # X coordinate metrics
        metrics_x = calculate_regression_metrics(true_x, pred_x)
        print("\nX Coordinate Metrics:")
        print(f"  RMSE: {metrics_x['rmse']:.4f}")
        print(f"  MAE:  {metrics_x['mae']:.4f}")
        print(f"  R2:   {metrics_x['r2']:.4f}")

        # Y coordinate metrics
        metrics_y = calculate_regression_metrics(true_y, pred_y)
        print("\nY Coordinate Metrics:")
        print(f"  RMSE: {metrics_y['rmse']:.4f}")
        print(f"  MAE:  {metrics_y['mae']:.4f}")
        print(f"  R2:   {metrics_y['r2']:.4f}")

        # Combined metrics
        combined_rmse = calculate_combined_rmse(metrics_x['rmse'], metrics_y['rmse'])
        print(f"\nCombined RMSE: {combined_rmse:.4f}")

        # Position error metrics
        pos_metrics = calculate_position_error(pred_x, pred_y, true_x, true_y)
        print("\nPosition Error Metrics:")
        print(f"  Mean:   {pos_metrics['mean_position_error']:.4f}")
        print(f"  Median: {pos_metrics['median_position_error']:.4f}")
        print(f"  90th %: {pos_metrics['position_error_90th_percentile']:.4f}")
        print(f"  95th %: {pos_metrics['position_error_95th_percentile']:.4f}")
        print(f"  Max:    {pos_metrics['max_position_error']:.4f}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
