"""
Evaluation Module for NFL ML Pipeline
Handles model evaluation, comparison, and selection.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns

from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.helpers import timer, plot_model_comparison, plot_residuals
from nfl_pipeline.utils.tracking import ExperimentTracker
from nfl_pipeline.evaluation.metrics import (
    calculate_regression_metrics, calculate_euclidean_distance
)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison.

    Provides detailed metrics, visualizations, and model selection
    based on various evaluation criteria.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger
        self.experiment_tracker = ExperimentTracker(config)

    def evaluate_all_models(self, model_results: Dict[str, Dict],
                           ensemble_results: Dict[str, Any],
                           data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate all models and create comprehensive comparison.

        Args:
            model_results: Results from individual models
            ensemble_results: Results from ensemble models
            data: Training and validation data

        Returns:
            Comprehensive evaluation results
        """
        with timer("Model Evaluation"):
            self.logger.info("=" * 80)
            self.logger.info("MODEL EVALUATION")
            self.logger.info("=" * 80)

            # Combine all results
            all_results = self._combine_results(model_results, ensemble_results)

            # Calculate comprehensive metrics
            evaluation_results = self._calculate_comprehensive_metrics(
                all_results, data
            )

            # Generate evaluation plots
            if self.config.save_evaluation_plots:
                self._generate_evaluation_plots(evaluation_results, data)

            # Log results to experiment tracker
            self._log_evaluation_results(evaluation_results)

            return evaluation_results

    def _combine_results(self, model_results: Dict[str, Dict],
                        ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine base model and ensemble results"""
        combined = {}

        # Add base models
        for name, result in model_results.items():
            combined[f'base_{name}'] = {
                'type': 'base',
                'name': name,
                'x_rmse': result['x_results']['metrics']['val_rmse'],
                'y_rmse': result['y_results']['metrics']['val_rmse'],
                'combined_rmse': result['combined_val_rmse'],
                'x_mae': result['x_results']['metrics']['val_mae'],
                'y_mae': result['y_results']['metrics']['val_mae'],
                'combined_mae': np.sqrt(
                    result['x_results']['metrics']['val_mae']**2 +
                    result['y_results']['metrics']['val_mae']**2
                ),
                'result': result
            }

        # Add ensemble models
        for name, result in ensemble_results.items():
            combined[f'ensemble_{name}'] = {
                'type': 'ensemble',
                'name': name,
                'x_rmse': result['x_rmse'],
                'y_rmse': result['y_rmse'],
                'combined_rmse': result['combined_rmse'],
                'x_mae': None,  # Ensembles may not have MAE
                'y_mae': None,
                'combined_mae': None,
                'result': result
            }

        return combined

    def _calculate_comprehensive_metrics(self, all_results: Dict[str, Any],
                                       data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        evaluation_results = {
            'model_comparison': {},
            'best_models': {},
            'summary_stats': {}
        }

        # Extract metrics for comparison
        model_names = []
        rmse_values = []
        mae_values = []

        for model_key, model_info in all_results.items():
            model_names.append(model_key)
            rmse_values.append(model_info['combined_rmse'])

            # Use RMSE as MAE for ensembles that don't have MAE
            mae_val = model_info.get('combined_mae')
            if mae_val is None:
                mae_val = model_info['combined_rmse']  # Approximation
            mae_values.append(mae_val)

        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'model': model_names,
            'rmse': rmse_values,
            'mae': mae_values,
            'type': [all_results[m]['type'] for m in model_names]
        })

        # Sort by RMSE
        comparison_df = comparison_df.sort_values('rmse').reset_index(drop=True)

        evaluation_results['model_comparison'] = comparison_df

        # Find best models
        best_overall = comparison_df.iloc[0]
        best_base = comparison_df[comparison_df['type'] == 'base'].iloc[0]
        best_ensemble = comparison_df[comparison_df['type'] == 'ensemble'].iloc[0] if len(comparison_df[comparison_df['type'] == 'ensemble']) > 0 else None

        evaluation_results['best_models'] = {
            'overall': {
                'name': best_overall['model'],
                'rmse': best_overall['rmse'],
                'mae': best_overall['mae']
            },
            'base': {
                'name': best_base['model'],
                'rmse': best_base['rmse'],
                'mae': best_base['mae']
            },
            'ensemble': {
                'name': best_ensemble['model'] if best_ensemble is not None else None,
                'rmse': best_ensemble['rmse'] if best_ensemble is not None else None,
                'mae': best_ensemble['mae'] if best_ensemble is not None else None
            }
        }

        # Calculate summary statistics
        evaluation_results['summary_stats'] = {
            'n_models_evaluated': len(comparison_df),
            'rmse_range': {
                'min': comparison_df['rmse'].min(),
                'max': comparison_df['rmse'].max(),
                'mean': comparison_df['rmse'].mean(),
                'std': comparison_df['rmse'].std()
            },
            'best_improvement_over_baseline': self._calculate_improvement(
                comparison_df, baseline_type='base'
            )
        }

        # Display results
        self._display_evaluation_summary(evaluation_results)

        return evaluation_results

    def _calculate_improvement(self, comparison_df: pd.DataFrame,
                             baseline_type: str = 'base') -> Dict[str, float]:
        """Calculate improvement over baseline"""
        if baseline_type not in ['base', 'ensemble']:
            return {}

        baseline_df = comparison_df[comparison_df['type'] == baseline_type]
        if len(baseline_df) == 0:
            return {}

        baseline_rmse = baseline_df['rmse'].min()
        best_rmse = comparison_df['rmse'].min()

        improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

        return {
            'baseline_rmse': baseline_rmse,
            'best_rmse': best_rmse,
            'improvement_percent': improvement
        }

    def _display_evaluation_summary(self, evaluation_results: Dict[str, Any]):
        """Display comprehensive evaluation summary"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 80)

        comparison_df = evaluation_results['model_comparison']

        self.logger.info("Model Performance Ranking:")
        self.logger.info("-" * 80)

        for idx, row in comparison_df.iterrows():
            rank = idx + 1
            model_name = row['model']
            rmse = row['rmse']
            mae = row['mae']
            model_type = row['type']

            self.logger.info(f"{rank:2d}. {model_name:25s} RMSE: {rmse:.4f}, MAE: {mae:.4f} ({model_type})")

        # Best models summary
        best_models = evaluation_results['best_models']

        self.logger.info("")
        self.logger.info("Best Models:")
        self.logger.info(f"  Overall: {best_models['overall']['name']} (RMSE: {best_models['overall']['rmse']:.4f})")
        self.logger.info(f"  Base:    {best_models['base']['name']} (RMSE: {best_models['base']['rmse']:.4f})")

        if best_models['ensemble']['name']:
            self.logger.info(f"  Ensemble: {best_models['ensemble']['name']} (RMSE: {best_models['ensemble']['rmse']:.4f})")

        # Summary statistics
        stats = evaluation_results['summary_stats']
        rmse_stats = stats['rmse_range']

        self.logger.info("")
        self.logger.info("Summary Statistics:")
        self.logger.info(f"  Models evaluated: {stats['n_models_evaluated']}")
        self.logger.info(f"  RMSE range: [{rmse_stats['min']:.4f}, {rmse_stats['max']:.4f}]")
        self.logger.info(f"  RMSE mean ± std: {rmse_stats['mean']:.4f} ± {rmse_stats['std']:.4f}")

        improvement = stats.get('best_improvement_over_baseline', {})
        if improvement:
            self.logger.info(f"  Best improvement over base: {improvement['improvement_percent']:+.2f}%")

    def _generate_evaluation_plots(self, evaluation_results: Dict[str, Any],
                                 data: Dict[str, Any]):
        """Generate evaluation plots and save them"""
        try:
            comparison_df = evaluation_results['model_comparison']

            # Model comparison plot
            model_names = comparison_df['model'].tolist()
            rmse_values = comparison_df['rmse'].tolist()

            plot_path = self.config.output_dir / f"{self.config.experiment_name}_model_comparison.png"
            plot_model_comparison(
                dict(zip(model_names, rmse_values)),
                metric='rmse',
                save_path=plot_path
            )

            # Residual plots for best model
            best_model_name = evaluation_results['best_models']['overall']['name']

            if best_model_name.startswith('base_'):
                model_key = best_model_name.replace('base_', '')
                # This would require access to model results - simplified for now
                pass

            self.logger.info(f"Evaluation plots saved to {self.config.output_dir}")

        except Exception as e:
            self.logger.warning(f"Could not generate evaluation plots: {e}")

    def _log_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Log evaluation results to experiment tracker"""
        # Log best model
        best_model = evaluation_results['best_models']['overall']
        self.experiment_tracker.log_metrics({
            'best_model_name': best_model['name'],
            'best_model_rmse': best_model['rmse'],
            'best_model_mae': best_model['mae']
        }, 'evaluation')

        # Log summary stats
        stats = evaluation_results['summary_stats']
        self.experiment_tracker.log_metrics({
            'n_models_evaluated': stats['n_models_evaluated'],
            'rmse_min': stats['rmse_range']['min'],
            'rmse_max': stats['rmse_range']['max'],
            'rmse_mean': stats['rmse_range']['mean'],
            'rmse_std': stats['rmse_range']['std']
        }, 'evaluation_summary')


# ModelSelector has been moved to evaluation/selector.py


class PredictionEvaluator:
    """
    Evaluate predictions on test data and generate submission.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                           dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth.

        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of the dataset for logging

        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating predictions on {dataset_name} set...")

        # Calculate metrics
        metrics = calculate_regression_metrics(y_true, y_pred)

        # Calculate position-specific metrics
        if y_true.shape[1] == 2 and y_pred.shape[1] == 2:  # X and Y coordinates
            position_metrics = calculate_euclidean_distance(
                y_pred[:, 0], y_pred[:, 1], y_true[:, 0], y_true[:, 1]
            )
            metrics.update({
                'mean_position_error': np.mean(position_metrics),
                'median_position_error': np.median(position_metrics),
                'position_error_90th': np.percentile(position_metrics, 90),
                'position_error_95th': np.percentile(position_metrics, 95)
            })

        # Log metrics
        self.logger.info(f"{dataset_name.upper()} METRICS:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric_name}: {value:.4f}")

        return metrics


if __name__ == "__main__":
    # Test evaluation
    from nfl_pipeline.core.config import get_quick_config

    config = get_quick_config()
    evaluator = ModelEvaluator(config)

    # Create mock results
    mock_model_results = {
        'ridge': {
            'x_results': {'metrics': {'val_rmse': 2.1, 'val_mae': 1.8}},
            'y_results': {'metrics': {'val_rmse': 2.3, 'val_mae': 1.9}},
            'combined_val_rmse': 3.1
        },
        'random_forest': {
            'x_results': {'metrics': {'val_rmse': 1.8, 'val_mae': 1.5}},
            'y_results': {'metrics': {'val_rmse': 2.0, 'val_mae': 1.7}},
            'combined_val_rmse': 2.7
        }
    }

    mock_ensemble_results = {
        'averaging': {'combined_rmse': 2.5, 'x_rmse': 1.7, 'y_rmse': 1.9}
    }

    mock_data = {'dummy': 'data'}

    try:
        results = evaluator.evaluate_all_models(mock_model_results, mock_ensemble_results, mock_data)
        print("Model evaluation test successful")
        print(f"Best model: {results['best_models']['overall']['name']}")
        print(f"Best RMSE: {results['best_models']['overall']['rmse']:.4f}")
    except Exception as e:
        print(f"✗ Model evaluation test failed: {e}")
        import traceback
        traceback.print_exc()