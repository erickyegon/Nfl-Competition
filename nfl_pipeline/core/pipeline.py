"""
Main Orchestrator for NFL ML Pipeline
Coordinates all pipeline components for end-to-end execution.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import pandas as pd
import numpy as np

from nfl_pipeline.core.config import PipelineConfig, get_quick_config, get_lstm_config, get_tuned_config, get_full_config, get_production_config
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.tracking import ExperimentTracker
from nfl_pipeline.utils.helpers import timer, force_garbage_collection
from nfl_pipeline.data.loader import DataLoader
from nfl_pipeline.features.engineer import FeatureEngineer
from nfl_pipeline.data.preprocessor import DataPreparation
from nfl_pipeline.models.traditional import ModelTrainer, HyperparameterTuner
from nfl_pipeline.models.ensemble import EnsembleBuilder
from nfl_pipeline.evaluation.evaluator import ModelEvaluator
from nfl_pipeline.evaluation.selector import ModelSelector
from nfl_pipeline.prediction.predictor import PredictionGenerator

# Import LSTM model if available
try:
    from nfl_pipeline.models.sequence import LSTMTrainer
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("Warning: LSTM models not available. Install PyTorch to use LSTM: pip install torch")


class NFLPipeline:
    """
    Main orchestrator for the NFL Big Data Bowl ML Pipeline.

    This class coordinates all pipeline components to provide
    end-to-end machine learning pipeline execution.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = PipelineLogger(config).logger
        self.experiment_tracker = ExperimentTracker(config)

        # Initialize components
        self.components = {}

        # Log initialization
        self.logger.info("=" * 80)
        self.logger.info("NFL BIG DATA BOWL 2026 - ML PIPELINE INITIALIZED")
        self.logger.info(f"Experiment: {config.experiment_name}")
        self.logger.info("=" * 80)

    def run_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete ML pipeline.

        Returns:
            Dictionary containing pipeline results and artifacts
        """
        start_time = time.time()
        self.logger.info("Starting pipeline execution...")

        try:
            # Step 1: Data Loading
            input_df, output_df = self._run_data_loading()

            # Step 2: Feature Engineering (NON-TEMPORAL features only)
            # This avoids data leakage - temporal features will be added AFTER split
            self.logger.info("⚠️  Creating NON-TEMPORAL features first to avoid data leakage")
            input_engineered = self._run_feature_engineering(input_df, include_temporal=False)

            # Clear memory
            del input_df
            force_garbage_collection()

            # Step 3: Data Preparation (includes train/test split)
            # Temporal features will be added separately per split inside prepare_modeling_data
            prepared_data = self._run_data_preparation(input_engineered, output_df)

            # Clear memory
            del input_engineered, output_df
            force_garbage_collection()

            # Step 4: Model Training
            model_results = self._run_model_training(prepared_data)

            # Step 5: Ensemble Building
            ensemble_results = self._run_ensemble_building(model_results, prepared_data)

            # Step 6: Model Evaluation
            evaluation_results = self._run_model_evaluation(
                model_results, ensemble_results, prepared_data
            )

            # Step 7: Model Selection
            selected_model = self._run_model_selection(
                evaluation_results, model_results, ensemble_results
            )

            # Step 8: Prediction Generation
            submission = self._run_prediction_generation(
                selected_model, model_results, ensemble_results
            )

            # Calculate total execution time
            total_time = time.time() - start_time

            # Log completion
            self._log_pipeline_completion(selected_model, total_time)

            # Prepare final results
            results = {
                'selected_model': selected_model,
                'model_results': model_results,
                'ensemble_results': ensemble_results,
                'evaluation_results': evaluation_results,
                'prepared_data': prepared_data,
                'submission': submission,
                'execution_time': total_time,
                'config': self.config
            }

            # Save final experiment results
            self.experiment_tracker.save_results()

            return results

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _run_data_loading(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run data loading step"""
        with timer("Data Loading"):
            self.logger.info("Step 1: Loading training data...")

            data_loader = DataLoader(self.config)
            input_df, output_df = data_loader.load_training_data()

            self.components['data_loader'] = data_loader
            return input_df, output_df

    def _run_feature_engineering(self, input_df: pd.DataFrame, include_temporal: bool = True) -> pd.DataFrame:
        """Run feature engineering step with optional temporal features"""
        with timer("Feature Engineering"):
            if include_temporal:
                self.logger.info("Step 2: Engineering features (including temporal)...")
            else:
                self.logger.info("Step 2: Engineering features (NON-temporal only, avoiding leakage)...")

            feature_engineer = FeatureEngineer(self.config)
            input_engineered = feature_engineer.engineer_features(input_df, include_temporal=include_temporal)

            self.components['feature_engineer'] = feature_engineer
            return input_engineered

    def _run_data_preparation(self, input_engineered: pd.DataFrame,
                            output_df: pd.DataFrame) -> Dict[str, Any]:
        """Run data preparation step"""
        with timer("Data Preparation"):
            self.logger.info("Step 3: Preparing data for modeling...")

            data_prep = DataPreparation(self.config)
            prepared_data = data_prep.prepare_modeling_data(input_engineered, output_df)

            self.components['data_prep'] = data_prep
            return prepared_data

    def _run_model_training(self, prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run model training step"""
        with timer("Model Training"):
            self.logger.info("Step 4: Training models...")

            trainer = ModelTrainer(self.config)
            model_results = {}

            # Separate traditional models from LSTM (which predicts x,y jointly)
            traditional_models = [m for m in self.config.models_to_evaluate if m != 'lstm']
            use_lstm = 'lstm' in self.config.models_to_evaluate

            # Train traditional models separately for X and Y coordinates
            if traditional_models:
                self.logger.info("Training X-coordinate models...")
                x_results = self._train_coordinate_models(
                    trainer, prepared_data['X_train'], prepared_data['y_train_x'],
                    prepared_data['X_val'], prepared_data['y_val_x'], 'x', traditional_models
                )

                self.logger.info("Training Y-coordinate models...")
                y_results = self._train_coordinate_models(
                    trainer, prepared_data['X_train'], prepared_data['y_train_y'],
                    prepared_data['X_val'], prepared_data['y_val_y'], 'y', traditional_models
                )

                # Combine results for traditional models
                for model_name in traditional_models:
                    if model_name in x_results and model_name in y_results:
                        # Calculate combined RMSE - FIXED: correct key names
                        x_val_rmse = x_results[model_name]['val_metrics']['rmse']
                        y_val_rmse = y_results[model_name]['val_metrics']['rmse']
                        combined_rmse = (x_val_rmse**2 + y_val_rmse**2)**0.5

                        # Create result structure with proper metrics format
                        model_results[model_name] = {
                            'x_results': {
                                'model': x_results[model_name]['model'],
                                'predictions': {
                                    'train': x_results[model_name]['train_predictions'],
                                    'val': x_results[model_name]['val_predictions']
                                },
                                'metrics': {
                                    'train_rmse': x_results[model_name]['train_metrics']['rmse'],
                                    'val_rmse': x_val_rmse,
                                    'train_mae': x_results[model_name]['train_metrics']['mae'],
                                    'val_mae': x_results[model_name]['val_metrics']['mae']
                                }
                            },
                            'y_results': {
                                'model': y_results[model_name]['model'],
                                'predictions': {
                                    'train': y_results[model_name]['train_predictions'],
                                    'val': y_results[model_name]['val_predictions']
                                },
                                'metrics': {
                                    'train_rmse': y_results[model_name]['train_metrics']['rmse'],
                                    'val_rmse': y_val_rmse,
                                    'train_mae': y_results[model_name]['train_metrics']['mae'],
                                    'val_mae': y_results[model_name]['val_metrics']['mae']
                                }
                            },
                            'combined_val_rmse': combined_rmse
                        }

            # Train LSTM model (predicts x,y jointly)
            if use_lstm:
                try:
                    self.logger.info("Training LSTM trajectory model (joint x,y prediction)...")
                    lstm_result = self._train_lstm_model(prepared_data)
                    if lstm_result is not None:
                        model_results['lstm'] = lstm_result
                except Exception as e:
                    self.logger.error(f"LSTM training failed: {e}")
                    import traceback
                    traceback.print_exc()

            self.components['trainer'] = trainer
            return model_results

    def _train_coordinate_models(self, trainer: ModelTrainer, X_train: np.ndarray,
                               y_train: np.ndarray, X_val: np.ndarray,
                               y_val: np.ndarray, coordinate: str,
                               model_names: List[str] = None) -> Dict[str, Any]:
        """Train models for a specific coordinate - OPTIMIZED with parallel training"""
        results = {}

        # Use provided model names or fall back to config
        if model_names is None:
            model_names = self.config.models_to_evaluate

        # Use parallel training for multiple models
        if len(model_names) > 1:
            self.logger.info(f"Training {len(model_names)} models in parallel for {coordinate}-coordinate...")
            results = trainer.train_models_parallel(
                model_names,
                X_train, y_train, X_val, y_val
            )
        else:
            # Single model - no need for parallelization
            for model_name in model_names:
                try:
                    result = trainer.train_model(
                        model_name, X_train, y_train, X_val, y_val
                    )
                    results[model_name] = result
                except Exception as e:
                    self.logger.warning(f"Failed to train {model_name} for {coordinate}: {e}")
                    continue

        # EFFICIENT HYPERPARAMETER TUNING STRATEGY
        # Only tune top-performing models to save computational resources
        if self.config.hyperparameter_tuning and len(results) > 0:
            # Step 1: Rank models by validation RMSE
            model_rankings = []
            for model_name, result in results.items():
                val_rmse = result['val_metrics']['rmse']
                model_rankings.append((model_name, val_rmse))

            # Sort by RMSE (lower is better)
            model_rankings.sort(key=lambda x: x[1])

            # Step 2: SMART SELECTION - Skip tuning if:
            # a) Best model is already Ridge (linear models rarely benefit from tuning)
            # b) Performance gap is very small (< 2% improvement potential)
            best_model = model_rankings[0][0]
            best_rmse = model_rankings[0][1]

            # Don't tune linear models (Ridge, Lasso) - they have minimal hyperparameters
            skip_tuning_models = {'ridge', 'lasso', 'linear_regression', 'elastic_net'}

            if best_model in skip_tuning_models:
                self.logger.info(f"⚡ Smart Skip: {best_model} is linear model, tuning unlikely to help")
                self.logger.info(f"   Best RMSE: {best_rmse:.4f} - proceeding with default parameters")
                models_to_tune = []
            elif len(model_rankings) > 1:
                # Check if second-best is very close (within 2%)
                second_best_rmse = model_rankings[1][1]
                gap_pct = ((second_best_rmse - best_rmse) / best_rmse) * 100

                if gap_pct < 2.0:
                    self.logger.info(f"⚡ Smart Skip: Top models very close ({gap_pct:.1f}% gap)")
                    self.logger.info(f"   {model_rankings[0][0]}: {best_rmse:.4f} vs {model_rankings[1][0]}: {second_best_rmse:.4f}")
                    self.logger.info(f"   Tuning unlikely to justify time cost - using defaults")
                    models_to_tune = []
                else:
                    # Tune only the best tree-based model
                    n_models_to_tune = 1
                    models_to_tune = [name for name, _ in model_rankings[:n_models_to_tune]]
                    self.logger.info(f"🎯 Efficient Tuning Strategy: Tuning top {n_models_to_tune} model for {coordinate}-coordinate")
                    self.logger.info(f"   Models selected: {models_to_tune}")
                    self.logger.info(f"   Skipping tuning for: {[name for name, _ in model_rankings[n_models_to_tune:]]}")
            else:
                models_to_tune = [best_model]
                self.logger.info(f"🎯 Tuning only model: {best_model}")

            # Step 3: Tune only selected models
            for model_name in models_to_tune:
                try:
                    self.logger.info(f"   Tuning {model_name} (baseline RMSE: {results[model_name]['val_metrics']['rmse']:.4f})...")
                    tuner = HyperparameterTuner(self.config)
                    tuned_model = tuner.tune_model(model_name, X_train, y_train)

                    if tuned_model is not None:
                        # Re-evaluate tuned model
                        tuned_result = trainer.train_model(
                            model_name, X_train, y_train, X_val, y_val
                        )

                        # Compare improvement
                        old_rmse = results[model_name]['val_metrics']['rmse']
                        new_rmse = tuned_result['val_metrics']['rmse']
                        improvement = ((old_rmse - new_rmse) / old_rmse) * 100

                        if new_rmse < old_rmse:
                            self.logger.info(f"   ✓ {model_name} improved: {old_rmse:.4f} → {new_rmse:.4f} ({improvement:.1f}% better)")
                            results[model_name] = tuned_result
                        else:
                            self.logger.info(f"   → {model_name} no improvement: {old_rmse:.4f} → {new_rmse:.4f} (keeping baseline)")

                except Exception as e:
                    self.logger.warning(f"Failed to tune {model_name} for {coordinate}: {e}")

        return results

    def _run_ensemble_building(self, model_results: Dict[str, Any],
                             prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run ensemble building step"""
        with timer("Ensemble Building"):
            self.logger.info("Step 5: Building ensembles...")

            ensemble_builder = EnsembleBuilder(self.config)
            ensemble_results = ensemble_builder.build_ensembles(model_results, prepared_data)

            self.components['ensemble_builder'] = ensemble_builder
            return ensemble_results

    def _run_model_evaluation(self, model_results: Dict[str, Any],
                            ensemble_results: Dict[str, Any],
                            prepared_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run model evaluation step"""
        with timer("Model Evaluation"):
            self.logger.info("Step 6: Evaluating models...")

            evaluator = ModelEvaluator(self.config)
            evaluation_results = evaluator.evaluate_all_models(
                model_results, ensemble_results, prepared_data
            )

            self.components['evaluator'] = evaluator
            return evaluation_results

    def _run_model_selection(self, evaluation_results: Dict[str, Any],
                           model_results: Dict[str, Any],
                           ensemble_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run model selection step"""
        with timer("Model Selection"):
            self.logger.info("Step 7: Selecting best model...")

            selector = ModelSelector(self.config)
            selected_model = selector.select_best_model(
                evaluation_results, model_results, ensemble_results
            )

            self.components['selector'] = selector
            return selected_model

    def _run_prediction_generation(self, selected_model: Dict[str, Any],
                                 model_results: Dict[str, Any],
                                 ensemble_results: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Run prediction generation step"""
        with timer("Prediction Generation"):
            self.logger.info("Step 8: Generating predictions...")

            predictor = PredictionGenerator(self.config)
            submission = predictor.generate_predictions(
                selected_model, model_results, ensemble_results,
                self.components.get('feature_engineer'),
                self.components.get('data_prep'),
                self.components.get('ensemble_builder')
            )

            self.components['predictor'] = predictor
            return submission

    def _train_lstm_model(self, prepared_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Train LSTM model for joint (x,y) trajectory prediction.

        Args:
            prepared_data: Dictionary with training and validation data

        Returns:
            Dictionary with LSTM results in standard format, or None if failed
        """
        if not LSTM_AVAILABLE:
            self.logger.warning("LSTM not available. Skipping LSTM training.")
            return None

        try:
            # Prepare joint targets (x,y combined)
            y_train_joint = np.column_stack([
                prepared_data['y_train_x'],
                prepared_data['y_train_y']
            ])

            y_val_joint = np.column_stack([
                prepared_data['y_val_x'],
                prepared_data['y_val_y']
            ])

            # Initialize LSTM trainer
            lstm_trainer = LSTMTrainer(self.config)

            # Train LSTM
            lstm_results = lstm_trainer.train(
                X_train=prepared_data['X_train'],
                y_train=y_train_joint,
                train_metadata=prepared_data['train_metadata'],
                X_val=prepared_data['X_val'],
                y_val=y_val_joint,
                val_metadata=prepared_data['val_metadata'],
                sequence_length=5,
                epochs=self.config.nn_epochs,
                batch_size=self.config.nn_batch_size,
                learning_rate=self.config.nn_learning_rate,
                patience=10
            )

            # Extract predictions
            val_predictions = lstm_results['val_predictions']
            val_pred_x = val_predictions[:, 0]
            val_pred_y = val_predictions[:, 1]

            # Calculate individual RMSEs
            x_rmse = np.sqrt(np.mean((prepared_data['y_val_x'][:len(val_pred_x)] - val_pred_x)**2))
            y_rmse = np.sqrt(np.mean((prepared_data['y_val_y'][:len(val_pred_y)] - val_pred_y)**2))
            combined_rmse = lstm_results['val_rmse']

            # Format results to match traditional model structure
            result = {
                'x_results': {
                    'model': lstm_results['model'],  # Store the full model
                    'predictions': {
                        'train': None,  # Not computed to save memory
                        'val': val_pred_x
                    },
                    'metrics': {
                        'train_rmse': lstm_results['train_rmse'],
                        'val_rmse': x_rmse,
                        'train_mae': 0.0,  # Not computed
                        'val_mae': 0.0     # Not computed
                    }
                },
                'y_results': {
                    'model': lstm_results['model'],  # Same model for both
                    'predictions': {
                        'train': None,
                        'val': val_pred_y
                    },
                    'metrics': {
                        'train_rmse': lstm_results['train_rmse'],
                        'val_rmse': y_rmse,
                        'train_mae': 0.0,
                        'val_mae': 0.0
                    }
                },
                'combined_val_rmse': combined_rmse,
                'lstm_info': {
                    'sequence_length': lstm_results.get('sequence_length', 5),
                    'device': lstm_results.get('device', 'cpu'),
                    'model_state': lstm_results.get('model_state')
                }
            }

            self.logger.info(f"LSTM training completed. Combined RMSE: {combined_rmse:.4f}")

            return result

        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _log_pipeline_completion(self, selected_model: Dict[str, Any], total_time: float):
        """Log pipeline completion"""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE EXECUTION COMPLETED")
        self.logger.info("=" * 80)
        self.logger.info(f"Total execution time: {total_time:.2f} seconds")
        self.logger.info(f"Selected model: {selected_model['name']}")
        self.logger.info(f"Best RMSE: {selected_model['rmse']:.4f}")
        self.logger.info("=" * 80)

    def save_pipeline_artifacts(self):
        """Save all pipeline artifacts"""
        self.logger.info("Saving pipeline artifacts...")

        # Save configuration
        self.config.save_config()

        # Save component artifacts
        if 'trainer' in self.components:
            # Model saving would be implemented here
            pass

        self.logger.info("Pipeline artifacts saved")

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution"""
        return {
            'experiment_name': self.config.experiment_name,
            'config': self.config.to_dict(),
            'components': list(self.components.keys()),
            'artifacts': {
                'models_dir': str(self.config.models_dir),
                'output_dir': str(self.config.output_dir),
                'logs_dir': str(self.config.logs_dir)
            }
        }


def run_quick_pipeline() -> Dict[str, Any]:
    """Run pipeline with quick configuration"""
    config = get_quick_config()
    pipeline = NFLPipeline(config)
    return pipeline.run_pipeline()


def run_full_pipeline() -> Dict[str, Any]:
    """Run pipeline with full configuration"""
    config = get_full_config()
    pipeline = NFLPipeline(config)
    return pipeline.run_pipeline()


def run_production_pipeline() -> Dict[str, Any]:
    """Run pipeline with production configuration"""
    config = get_production_config()
    pipeline = NFLPipeline(config)
    return pipeline.run_pipeline()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='NFL ML Pipeline')
    parser.add_argument('--config', choices=['quick', 'tuned', 'full', 'production'],
                       default='quick', help='Pipeline configuration')
    parser.add_argument('--experiment-name', type=str,
                       help='Custom experiment name')

    args = parser.parse_args()

    # Select configuration
    if args.config == 'quick':
        config = get_quick_config()
    elif args.config == 'tuned':
        config = get_tuned_config()
    elif args.config == 'full':
        config = get_full_config()
    elif args.config == 'production':
        config = get_production_config()

    # Override experiment name if provided
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    # Run pipeline
    try:
        pipeline = NFLPipeline(config)
        results = pipeline.run_pipeline()

        print("\n🎉 Pipeline completed successfully!")
        print(f"Best model: {results['selected_model']['name']}")
        print(f"Best RMSE: {results['selected_model']['rmse']:.4f}")

        if results.get('submission') is not None:
            print("Submission file generated")
        else:
            print("No test data available - no submission generated")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()