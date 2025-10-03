"""
Model Selection Module for NFL ML Pipeline
Handles model selection based on evaluation criteria.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.helpers import timer


class ModelSelector:
    """
    Select the best model based on evaluation criteria.

    Provides various selection strategies and handles model persistence.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def select_best_model(self, evaluation_results: Dict[str, Any],
                         model_results: Dict[str, Dict],
                         ensemble_results: Dict[str, Any],
                         selection_metric: str = 'rmse') -> Dict[str, Any]:
        """
        Select the best model based on evaluation results.

        Args:
            evaluation_results: Results from model evaluation
            model_results: Individual model results
            ensemble_results: Ensemble model results
            selection_metric: Metric to use for selection (default: 'rmse')

        Returns:
            Selected model information
        """
        with timer("Model Selection"):
            self.logger.info("=" * 80)
            self.logger.info("FINAL MODEL SELECTION")
            self.logger.info("=" * 80)

            # Get best model from evaluation
            best_model_info = evaluation_results['best_models']['overall']
            best_model_name = best_model_info['name']

            # Extract the actual model
            if best_model_name.startswith('base_'):
                model_key = best_model_name.replace('base_', '')
                selected_result = model_results[model_key]
                model_type = 'base'
            elif best_model_name.startswith('ensemble_'):
                ensemble_key = best_model_name.replace('ensemble_', '')
                selected_result = ensemble_results[ensemble_key]
                model_type = 'ensemble'
            else:
                raise ValueError(f"Unknown model type for {best_model_name}")

            # Create selection result
            selection_result = {
                'name': best_model_name,
                'type': model_type,
                'rmse': best_model_info['rmse'],
                'mae': best_model_info['mae'],
                'result': selected_result,
                'evaluation_results': evaluation_results,
                'selection_metric': selection_metric
            }

            # Log selection
            self.logger.info(f"SELECTED MODEL: {best_model_name.upper()}")
            self.logger.info(f"   Type: {model_type.upper()}")
            self.logger.info(f"   Validation RMSE: {best_model_info['rmse']:.4f}")
            if best_model_info['mae']:
                self.logger.info(f"   Validation MAE: {best_model_info['mae']:.4f}")

            # Display final ranking
            self._display_final_ranking(evaluation_results)

            return selection_result

    def select_model_by_criteria(self, evaluation_results: Dict[str, Any],
                                 criteria: str = 'best_rmse') -> str:
        """
        Select model based on specific criteria.

        Args:
            evaluation_results: Results from model evaluation
            criteria: Selection criteria ('best_rmse', 'best_mae', 'best_base', 'best_ensemble')

        Returns:
            Selected model name
        """
        comparison_df = evaluation_results['model_comparison']

        if criteria == 'best_rmse':
            return comparison_df.iloc[0]['model']
        elif criteria == 'best_mae':
            return comparison_df.sort_values('mae').iloc[0]['model']
        elif criteria == 'best_base':
            base_models = comparison_df[comparison_df['type'] == 'base']
            return base_models.iloc[0]['model'] if len(base_models) > 0 else None
        elif criteria == 'best_ensemble':
            ensemble_models = comparison_df[comparison_df['type'] == 'ensemble']
            return ensemble_models.iloc[0]['model'] if len(ensemble_models) > 0 else None
        else:
            raise ValueError(f"Unknown criteria: {criteria}")

    def get_top_n_models(self, evaluation_results: Dict[str, Any],
                        n: int = 3) -> List[Dict[str, Any]]:
        """
        Get top N models by RMSE.

        Args:
            evaluation_results: Results from model evaluation
            n: Number of top models to return

        Returns:
            List of top model information
        """
        comparison_df = evaluation_results['model_comparison']
        top_models = []

        for idx, row in comparison_df.head(n).iterrows():
            top_models.append({
                'rank': idx + 1,
                'name': row['model'],
                'type': row['type'],
                'rmse': row['rmse'],
                'mae': row['mae']
            })

        return top_models

    def _display_final_ranking(self, evaluation_results: Dict[str, Any]):
        """Display final model ranking"""
        comparison_df = evaluation_results['model_comparison']

        self.logger.info("")
        self.logger.info("Final Model Ranking:")
        self.logger.info("-" * 80)

        for idx, row in comparison_df.head(10).iterrows():  # Top 10
            rank = idx + 1
            model_name = row['model']
            rmse = row['rmse']
            model_type = row['type']

            marker = "[BEST]" if idx == 0 else "      "
            self.logger.info(f"{marker} {rank:2d}. {model_name:25s} RMSE: {rmse:.4f} ({model_type})")

    def save_selection_results(self, selection_result: Dict[str, Any]):
        """
        Save model selection results to disk.

        Args:
            selection_result: Model selection result dictionary
        """
        import json

        # Create simplified version for JSON serialization
        save_data = {
            'selected_model': selection_result['name'],
            'model_type': selection_result['type'],
            'rmse': float(selection_result['rmse']),
            'mae': float(selection_result['mae']) if selection_result['mae'] else None,
            'selection_metric': selection_result.get('selection_metric', 'rmse'),
            'experiment_name': self.config.experiment_name
        }

        save_path = self.config.output_dir / f"{self.config.experiment_name}_selection.json"
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        self.logger.info(f"Selection results saved to {save_path}")


class EnsembleSelector:
    """
    Select optimal ensemble configuration.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def select_ensemble_members(self, model_results: Dict[str, Dict],
                               n_members: int = 3,
                               diversity_weight: float = 0.2) -> List[str]:
        """
        Select ensemble members balancing performance and diversity.

        Args:
            model_results: Individual model results
            n_members: Number of members to select
            diversity_weight: Weight for diversity vs. performance (0-1)

        Returns:
            List of selected model names
        """
        # Extract performance metrics
        model_scores = {}
        for name, result in model_results.items():
            rmse = result['combined_val_rmse']
            model_scores[name] = rmse

        # Sort by performance
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1])

        # For now, select top N by performance
        # TODO: Add diversity calculation based on prediction correlations
        selected = [name for name, _ in sorted_models[:n_members]]

        self.logger.info(f"Selected {n_members} ensemble members: {selected}")
        return selected


if __name__ == "__main__":
    # Test selector
    from nfl_pipeline.core.config import get_quick_config

    config = get_quick_config()
    selector = ModelSelector(config)

    # Create mock evaluation results
    mock_comparison = pd.DataFrame({
        'model': ['base_random_forest', 'base_ridge', 'ensemble_averaging'],
        'rmse': [2.5, 2.8, 2.4],
        'mae': [2.0, 2.3, 1.9],
        'type': ['base', 'base', 'ensemble']
    })

    mock_evaluation_results = {
        'model_comparison': mock_comparison,
        'best_models': {
            'overall': {'name': 'ensemble_averaging', 'rmse': 2.4, 'mae': 1.9},
            'base': {'name': 'base_random_forest', 'rmse': 2.5, 'mae': 2.0},
            'ensemble': {'name': 'ensemble_averaging', 'rmse': 2.4, 'mae': 1.9}
        }
    }

    try:
        top_models = selector.get_top_n_models(mock_evaluation_results, n=3)
        print("Model selector test successful")
        print(f"Top model: {top_models[0]['name']} (RMSE: {top_models[0]['rmse']:.4f})")
    except Exception as e:
        print(f"Model selector test failed: {e}")
        import traceback
        traceback.print_exc()
