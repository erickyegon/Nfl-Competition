"""
Prediction script for NFL player movement prediction.

Makes predictions on test data using trained models.

Usage:
    python scripts/predict.py --model path/to/model.pkl
    python scripts/predict.py --experiment experiment_name
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
from nfl_pipeline.prediction.predictor import NFLPredictor


def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained NFL models')
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
        '--output',
        type=str,
        default='predictions.csv',
        help='Output file for predictions'
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
    print("NFL PLAYER MOVEMENT PREDICTION")
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

    # Load data
    print("\nLoading and preprocessing data...")
    loader = DataLoader(config)
    data = loader.load_data()

    preprocessor = DataPreprocessor(config)
    processed_data = preprocessor.preprocess(data)

    # Feature engineering
    print("Generating features...")
    feature_engineer = FeatureEngineer(config)
    X_test, _ = feature_engineer.create_features(processed_data['test'])

    # Make predictions
    print("\nMaking predictions...")
    predictor = NFLPredictor(config)

    # Load best model (you may need to specify which model to load)
    # For now, try to load from the experiment directory
    try:
        predictions = predictor.predict_from_saved_model(X_test, model_dir)

        # Create submission dataframe
        submission = pd.DataFrame({
            'pred_x': predictions[:, 0],
            'pred_y': predictions[:, 1]
        })

        # Save predictions
        output_path = Path(args.output)
        submission.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        print(f"Prediction shape: {submission.shape}")

    except Exception as e:
        print(f"Error making predictions: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 80)
    print("PREDICTION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
