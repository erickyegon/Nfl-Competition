"""
Main entry point for training NFL player movement prediction models.

Usage:
    python train.py                    # Use default config
    python train.py --config quick     # Use quick config
    python train.py --config lstm      # Use LSTM config
"""

import argparse
from nfl_pipeline import NFLPipeline, get_quick_config, get_lstm_config, get_full_config

def main():
    parser = argparse.ArgumentParser(description='NFL Player Movement Prediction Pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='quick',
        choices=['quick', 'lstm', 'full'],
        help='Configuration preset to use'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to train (e.g., ridge xgboost lstm)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs for neural network models'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config == 'quick':
        config = get_quick_config()
    elif args.config == 'lstm':
        config = get_lstm_config()
    elif args.config == 'full':
        config = get_full_config()

    # Override with command line arguments
    if args.models:
        config.models_to_evaluate = args.models
    if args.epochs:
        config.nn_epochs = args.epochs

    # Initialize and run pipeline
    print("=" * 80)
    print("NFL PLAYER MOVEMENT PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Models: {config.models_to_evaluate}")
    print("=" * 80)

    pipeline = NFLPipeline(config)
    results = pipeline.run_pipeline()

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Model: {results['selected_model']['name']}")
    print(f"Validation RMSE: {results['selected_model']['rmse']:.4f}")
    print("=" * 80)

    return results

if __name__ == '__main__':
    main()
