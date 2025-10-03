"""
Prediction Module for NFL ML Pipeline
Handles test data prediction and submission generation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.tracking import ExperimentTracker
from nfl_pipeline.utils.helpers import timer
from nfl_pipeline.data.loader import DataLoader
from nfl_pipeline.features.engineer import FeatureEngineer
from nfl_pipeline.data.preprocessor import DataPreparation
from nfl_pipeline.models.ensemble import EnsembleBuilder


class PredictionGenerator:
    """
    Generate predictions for test data and create submission files.

    Handles the complete prediction pipeline from test data loading
    to submission file creation.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger
        self.experiment_tracker = ExperimentTracker(config)

    def generate_predictions(self, selected_model: Dict[str, Any],
                           model_results: Dict[str, Dict],
                           ensemble_results: Dict[str, Any],
                           feature_engineer: FeatureEngineer,
                           data_prep: DataPreparation,
                           ensemble_builder: Optional[EnsembleBuilder] = None) -> Optional[pd.DataFrame]:
        """
        Generate predictions for test data.

        Args:
            selected_model: The selected best model
            model_results: Results from base models
            ensemble_results: Results from ensemble models
            feature_engineer: Trained feature engineer
            data_prep: Trained data preparation pipeline
            ensemble_builder: Trained ensemble builder (optional)

        Returns:
            Submission dataframe or None if test data not available
        """
        with timer("Prediction Generation"):
            self.logger.info("=" * 80)
            self.logger.info("GENERATING TEST PREDICTIONS")
            self.logger.info("=" * 80)

            # Load test data
            test_input, test_meta = self._load_test_data()
            if test_input is None or test_meta is None:
                self.logger.warning("Test data not available, skipping prediction generation")
                return None

            # Engineer features for test data
            test_features = self._engineer_test_features(feature_engineer, test_input)

            # Prepare test data for prediction
            X_test = self._prepare_test_features(test_features, test_meta, data_prep)

            # Generate predictions
            predictions = self._generate_model_predictions(
                selected_model, model_results, ensemble_results,
                X_test, ensemble_builder
            )

            # Create submission file
            submission = self._create_submission_file(test_meta, predictions)

            # Save submission
            submission_path = self._save_submission(submission)

            # Log results
            self._log_prediction_results(submission, predictions)

            return submission

    def _load_test_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load test data files"""
        self.logger.info("Loading test data...")

        data_loader = DataLoader(self.config)
        test_input, test_meta = data_loader.load_test_data()

        if test_input is not None and test_meta is not None:
            self.logger.info(f"Test input shape: {test_input.shape}")
            self.logger.info(f"Test metadata shape: {test_meta.shape}")
        else:
            self.logger.warning("Test data files not found")

        return test_input, test_meta

    def _engineer_test_features(self, feature_engineer: FeatureEngineer,
                              test_input: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for test data"""
        self.logger.info("Engineering features for test data...")
        test_features = feature_engineer.engineer_features(test_input)
        self.logger.info(f"Test features shape: {test_features.shape}")
        return test_features

    def _prepare_test_features(self, test_features: pd.DataFrame,
                             test_meta: pd.DataFrame,
                             data_prep: DataPreparation) -> np.ndarray:
        """Prepare test features for model prediction"""
        self.logger.info("Preparing test features for prediction...")

        # Parse test metadata to get required predictions
        test_meta = self._parse_test_metadata(test_meta)

        # Get last frame for each player
        last_frames = self._get_test_last_frames(test_features)

        # Merge with test metadata
        test_data = self._merge_test_data(last_frames, test_meta)

        # Select feature columns (same as training)
        feature_cols = [col for col in test_data.columns
                       if col not in ['id', 'game_id', 'play_id', 'nfl_id',
                                     'frame_id', 'x', 'y', 'player_birth_date',
                                     'player_height', 'player_position',
                                     'player_role', 'player_side', 'play_direction',
                                     'position_group']]

        # Handle missing values
        test_data[feature_cols] = test_data[feature_cols].fillna(
            test_data[feature_cols].median()
        )

        # Convert to numpy array
        X_test = test_data[feature_cols].values

        # Apply scaling (same as training)
        if 'feature_scaler' in data_prep.scalers:
            X_test = data_prep.scalers['feature_scaler'].transform(X_test)

        self.logger.info(f"Prepared test features shape: {X_test.shape}")
        return X_test

    def _parse_test_metadata(self, test_meta: pd.DataFrame) -> pd.DataFrame:
        """Parse test metadata to extract game/play/player/frame IDs"""
        test_meta = test_meta.copy()

        # Parse IDs from the 'id' column (format: gameId_playId_playerId_frameId)
        id_parts = test_meta['id'].str.split('_', expand=True)

        if len(id_parts.columns) >= 4:
            test_meta['game_id'] = id_parts[0].astype(int)
            test_meta['play_id'] = id_parts[1].astype(int)
            test_meta['nfl_id'] = id_parts[2].astype(int)
            test_meta['frame_id'] = id_parts[3].astype(int)
        else:
            raise ValueError("Test metadata 'id' column format is incorrect")

        return test_meta

    def _get_test_last_frames(self, test_features: pd.DataFrame) -> pd.DataFrame:
        """Get the last frame of input for each play-player combination in test data"""
        groupby_cols = ['game_id', 'play_id', 'nfl_id']

        # Get last frame for each player in each play
        last_frames = test_features.groupby(groupby_cols).tail(1).copy()

        self.logger.info(f"Test last frames: {last_frames.shape}")
        return last_frames

    def _merge_test_data(self, last_frames: pd.DataFrame,
                        test_meta: pd.DataFrame) -> pd.DataFrame:
        """Merge last frames with test metadata"""
        # Merge on game_id, play_id, nfl_id
        merge_cols = ['game_id', 'play_id', 'nfl_id']
        test_data = test_meta.merge(
            last_frames,
            on=merge_cols,
            how='left'
        )

        # Check for missing data
        missing_count = test_data.isnull().any(axis=1).sum()
        if missing_count > 0:
            self.logger.warning(f"{missing_count} test samples have missing feature data")

        return test_data

    def _generate_model_predictions(self, selected_model: Dict[str, Any],
                                  model_results: Dict[str, Dict],
                                  ensemble_results: Dict[str, Any],
                                  X_test: np.ndarray,
                                  ensemble_builder: Optional[EnsembleBuilder]) -> Dict[str, np.ndarray]:
        """Generate predictions using the selected model"""
        self.logger.info("Generating predictions...")

        model_type = selected_model['type']
        model_name = selected_model['name']

        if model_type == 'base':
            predictions = self._predict_with_base_model(selected_model, model_results, X_test)
        elif model_type == 'ensemble':
            predictions = self._predict_with_ensemble(
                selected_model, ensemble_results, model_results, X_test, ensemble_builder
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.logger.info(f"Predictions generated using {model_name} ({model_type})")
        return predictions

    def _predict_with_base_model(self, selected_model: Dict[str, Any],
                               model_results: Dict[str, Dict],
                               X_test: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions using a base model"""
        # Extract base model name
        model_key = selected_model['name'].replace('base_', '')
        result = model_results[model_key]

        # Get trained models
        x_model = result['x_results']['model']
        y_model = result['y_results']['model']

        # Predict
        x_pred = x_model.predict(X_test)
        y_pred = y_model.predict(X_test)

        return {'x': x_pred, 'y': y_pred}

    def _predict_with_ensemble(self, selected_model: Dict[str, Any],
                             ensemble_results: Dict[str, Any],
                             model_results: Dict[str, Dict],
                             X_test: np.ndarray,
                             ensemble_builder: Optional[EnsembleBuilder]) -> Dict[str, np.ndarray]:
        """Generate predictions using an ensemble model"""
        if ensemble_builder is None:
            raise ValueError("Ensemble builder required for ensemble predictions")

        ensemble_name = selected_model['name'].replace('ensemble_', '')

        x_pred, y_pred = ensemble_builder.predict_with_ensemble(
            ensemble_name, model_results, X_test
        )

        return {'x': x_pred, 'y': y_pred}

    def _create_submission_file(self, test_meta: pd.DataFrame,
                              predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Create submission dataframe"""
        self.logger.info("Creating submission file...")

        submission = pd.DataFrame({
            'id': test_meta['id'],
            'x': predictions['x'],
            'y': predictions['y']
        })

        # Validate submission format
        self._validate_submission(submission)

        self.logger.info(f"Submission file created with {len(submission)} predictions")
        return submission

    def _validate_submission(self, submission: pd.DataFrame):
        """Validate submission file format"""
        required_cols = ['id', 'x', 'y']

        # Check columns
        missing_cols = set(required_cols) - set(submission.columns)
        if missing_cols:
            raise ValueError(f"Submission missing columns: {missing_cols}")

        # Check for missing values
        missing_count = submission.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(f"Submission contains {missing_count} missing values")

        # Check data types
        if not pd.api.types.is_string_dtype(submission['id']):
            self.logger.warning("ID column should be string type")

        for col in ['x', 'y']:
            if not pd.api.types.is_numeric_dtype(submission[col]):
                raise ValueError(f"Column {col} should be numeric")

    def _save_submission(self, submission: pd.DataFrame) -> Path:
        """Save submission file"""
        output_path = self.config.output_dir / f"{self.config.experiment_name}_submission.csv"
        submission.to_csv(output_path, index=False)
        self.logger.info(f"Submission saved to: {output_path}")
        return output_path

    def _log_prediction_results(self, submission: pd.DataFrame,
                              predictions: Dict[str, np.ndarray]):
        """Log prediction results and statistics"""
        # Basic statistics
        x_stats = submission['x'].describe()
        y_stats = submission['y'].describe()

        self.logger.info("Prediction Statistics:")
        self.logger.info(f"  X predictions - Mean: {x_stats['mean']:.2f}, Std: {x_stats['std']:.2f}")
        self.logger.info(f"  Y predictions - Mean: {y_stats['mean']:.2f}, Std: {y_stats['std']:.2f}")
        self.logger.info(f"  Total predictions: {len(submission)}")

        # Log to experiment tracker
        self.experiment_tracker.log_metrics({
            'n_predictions': len(submission),
            'x_pred_mean': x_stats['mean'],
            'x_pred_std': x_stats['std'],
            'y_pred_mean': y_stats['mean'],
            'y_pred_std': y_stats['std']
        }, 'predictions')

        self.experiment_tracker.log_artifact('submission', self.config.output_dir / f"{self.config.experiment_name}_submission.csv")


class PredictionValidator:
    """Validate predictions and check for common issues"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def validate_predictions(self, predictions: Dict[str, np.ndarray],
                           test_meta: pd.DataFrame) -> List[str]:
        """Validate predictions for common issues"""
        issues = []

        x_pred = predictions['x']
        y_pred = predictions['y']

        # Check array shapes
        if len(x_pred) != len(y_pred):
            issues.append("X and Y prediction arrays have different lengths")

        if len(x_pred) != len(test_meta):
            issues.append(f"Predictions length ({len(x_pred)}) doesn't match test data ({len(test_meta)})")

        # Check for NaN values
        if np.isnan(x_pred).any():
            nan_count = np.isnan(x_pred).sum()
            issues.append(f"X predictions contain {nan_count} NaN values")

        if np.isnan(y_pred).any():
            nan_count = np.isnan(y_pred).sum()
            issues.append(f"Y predictions contain {nan_count} NaN values")

        # Check for infinite values
        if np.isinf(x_pred).any():
            inf_count = np.isinf(x_pred).sum()
            issues.append(f"X predictions contain {inf_count} infinite values")

        if np.isinf(y_pred).any():
            inf_count = np.isinf(y_pred).sum()
            issues.append(f"Y predictions contain {inf_count} infinite values")

        # Check prediction ranges (reasonable football field limits)
        if np.any((x_pred < 0) | (x_pred > 120)):
            out_of_bounds = np.sum((x_pred < 0) | (x_pred > 120))
            issues.append(f"{out_of_bounds} X predictions are outside field bounds [0, 120]")

        if np.any((y_pred < 0) | (y_pred > 53.3)):
            out_of_bounds = np.sum((y_pred < 0) | (y_pred > 53.3))
            issues.append(f"{out_of_bounds} Y predictions are outside field bounds [0, 53.3]")

        # Check for constant predictions (potential model issues)
        if np.std(x_pred) < 1e-6:
            issues.append("X predictions are nearly constant (possible model issue)")

        if np.std(y_pred) < 1e-6:
            issues.append("Y predictions are nearly constant (possible model issue)")

        return issues


if __name__ == "__main__":
    # Test prediction generator
    from nfl_pipeline.core.config import get_quick_config

    config = get_quick_config()
    predictor = PredictionGenerator(config)

    # Mock selected model
    selected_model = {
        'name': 'base_ridge',
        'type': 'base'
    }

    # Mock model results
    mock_results = {
        'ridge': {
            'x_results': {'model': None},  # Would be actual trained model
            'y_results': {'model': None}
        }
    }

    print("Prediction module created successfully")
    print("Note: Full testing requires trained models and test data")