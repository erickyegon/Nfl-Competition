"""
Data Preparation Module for NFL ML Pipeline
Handles data splitting, scaling, preprocessing, and modeling data preparation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    PowerTransformer, QuantileTransformer
)

from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.tracking import ExperimentTracker
from nfl_pipeline.utils.helpers import (
    timer, get_memory_usage, validate_dataframe
)


class DataPreparation:
    """
    Prepare data for modeling with proper train/validation splits and scaling.

    Handles:
    - Data merging and alignment
    - Train/validation splitting
    - Feature scaling
    - Outlier handling
    - Missing value imputation
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger
        self.experiment_tracker = ExperimentTracker(config)

        # Initialize scalers and transformers
        self.scalers = {}
        self.outlier_detectors = {}
        self.imputers = {}

    def prepare_modeling_data(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Prepare and merge input/output data for modeling.

        Args:
            input_df: Engineered input features
            output_df: Target output positions

        Returns:
            Dictionary containing prepared datasets and metadata
        """
        with timer("Data Preparation"):
            self.logger.info("=" * 80)
            self.logger.info("DATA PREPARATION")
            self.logger.info("=" * 80)

            # Step 1: Merge input and output data
            merged_data = self._merge_input_output(input_df, output_df)

            # Step 2: Handle missing values
            merged_data = self._handle_missing_values(merged_data)

            # Step 3: Handle outliers
            if self.config.handle_outliers:
                merged_data = self._handle_outliers(merged_data)

            # Step 4: Split data BEFORE adding temporal features (to prevent leakage)
            train_data, val_data = self._split_data(merged_data)

            # Step 5: Add temporal features AFTER split (prevents leakage)
            if self.config.use_temporal_features:
                self.logger.info("Adding temporal features AFTER split to prevent data leakage...")
                train_data = self._add_temporal_features_to_split(train_data)
                val_data = self._add_temporal_features_to_split(val_data)

            # Step 6: Separate features and targets
            X_train, y_train_x, y_train_y, train_metadata = self._separate_features_targets(train_data)
            X_val, y_val_x, y_val_y, val_metadata = self._separate_features_targets(val_data)

            # Step 7: Scale features
            X_train_scaled, X_val_scaled = self._scale_features(X_train, X_val)

            # Step 8: Log data statistics
            self._log_data_statistics(X_train_scaled, X_val_scaled, y_train_x, y_val_x)

            # Prepare result dictionary
            result = {
                'X_train': X_train_scaled,
                'y_train_x': y_train_x,
                'y_train_y': y_train_y,
                'X_val': X_val_scaled,
                'y_val_x': y_val_x,
                'y_val_y': y_val_y,
                'feature_names': X_train.columns.tolist(),
                'train_metadata': train_metadata,
                'val_metadata': val_metadata,
                'scalers': self.scalers.copy()
            }

            # Log to experiment tracker
            self.experiment_tracker.log_params({
                'n_features': len(result['feature_names']),
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'scaler_type': self.config.scaler_type,
                'outlier_handling': self.config.handle_outliers,
                'missing_value_strategy': self.config.imputation_strategy
            })

            return result

    def _merge_input_output(self, input_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
        """Merge input and output data for modeling"""
        self.logger.info("Merging input and output data...")

        # For each input row (last frame before pass), we need to predict multiple output frames
        # We'll create a dataset where each row represents a prediction task

        # Get the last frame of input for each play-player combination
        last_input_frame = self._get_last_input_frames(input_df)

        # Merge with output data - each input frame will have multiple output frames
        merged_data = self._create_prediction_tasks(last_input_frame, output_df)

        self.logger.info(f"Merged dataset shape: {merged_data.shape}")
        return merged_data

    def _get_last_input_frames(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Get the last frame of input for each play-player combination"""
        groupby_cols = ['game_id', 'play_id', 'nfl_id']

        # Get last frame for each player in each play
        last_frames = input_df.groupby(groupby_cols).tail(1).copy()

        self.logger.info(f"Last input frames: {last_frames.shape}")
        return last_frames

    def _create_prediction_tasks(self, input_frames: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
        """Create prediction tasks by merging input with all corresponding output frames - OPTIMIZED"""
        # Vectorized merge approach - much faster than row iteration
        self.logger.info("Creating prediction tasks using vectorized merge...")

        # Filter input frames if player_to_predict column exists
        if 'player_to_predict' in input_frames.columns:
            input_frames = input_frames[input_frames['player_to_predict']].copy()

        # Merge on game_id, play_id, nfl_id (one-to-many)
        # This creates cartesian product for each player's input with all their output frames
        merged_df = input_frames.merge(
            output_df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']],
            on=['game_id', 'play_id', 'nfl_id'],
            how='inner',
            suffixes=('', '_output')
        )

        # Rename output columns
        merged_df = merged_df.rename(columns={
            'frame_id_output': 'output_frame_id',
            'x_output': 'target_x',
            'y_output': 'target_y'
        })

        self.logger.info(f"Created {len(merged_df)} prediction tasks (vectorized)")
        return merged_df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        self.logger.info("Handling missing values...")

        # Check for missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if missing_cols.empty:
            self.logger.info("No missing values found")
            return df

        self.logger.info(f"Found missing values in {len(missing_cols)} columns")

        # Apply imputation strategy
        if self.config.imputation_strategy == 'median':
            df = self._impute_median(df, missing_cols)
        elif self.config.imputation_strategy == 'mean':
            df = self._impute_mean(df, missing_cols)
        elif self.config.imputation_strategy == 'most_frequent':
            df = self._impute_most_frequent(df, missing_cols)
        elif self.config.imputation_strategy == 'knn':
            df = self._impute_knn(df, missing_cols)
        else:
            raise ValueError(f"Unknown imputation strategy: {self.config.imputation_strategy}")

        # Verify no missing values remain
        remaining_missing = df.isnull().sum().sum()
        if remaining_missing > 0:
            self.logger.warning(f"Still have {remaining_missing} missing values after imputation")

        return df

    def _impute_median(self, df: pd.DataFrame, missing_cols: pd.Series) -> pd.DataFrame:
        """Impute missing values with median"""
        for col in missing_cols.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.logger.info(f"Imputed {missing_cols[col]} missing values in {col} with median: {median_val}")
        return df

    def _impute_mean(self, df: pd.DataFrame, missing_cols: pd.Series) -> pd.DataFrame:
        """Impute missing values with mean"""
        for col in missing_cols.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                self.logger.info(f"Imputed {missing_cols[col]} missing values in {col} with mean: {mean_val}")
        return df

    def _impute_most_frequent(self, df: pd.DataFrame, missing_cols: pd.Series) -> pd.DataFrame:
        """Impute missing values with most frequent value"""
        for col in missing_cols.index:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                self.logger.info(f"Imputed {missing_cols[col]} missing values in {col} with mode: {mode_val[0]}")
        return df

    def _impute_knn(self, df: pd.DataFrame, missing_cols: pd.Series) -> pd.DataFrame:
        """Impute missing values using KNN imputation"""
        try:
            from sklearn.impute import KNNImputer

            # Select numeric columns for imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            imputer = KNNImputer(n_neighbors=5)

            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['knn_imputer'] = imputer

            self.logger.info("Applied KNN imputation to numeric columns")

        except ImportError:
            self.logger.warning("KNNImputer not available, falling back to median imputation")
            return self._impute_median(df, missing_cols)

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the dataset"""
        self.logger.info(f"Handling outliers using {self.config.outlier_method} method...")

        if self.config.outlier_method == 'iqr':
            df = self._handle_outliers_iqr(df)
        elif self.config.outlier_method == 'zscore':
            df = self._handle_outliers_zscore(df)
        elif self.config.outlier_method == 'isolation_forest':
            df = self._handle_outliers_isolation_forest(df)
        else:
            raise ValueError(f"Unknown outlier method: {self.config.outlier_method}")

        return df

    def _handle_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method with reasonable threshold"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'output_frame_id',
                       'target_x', 'target_y']  # Don't modify IDs or targets

        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        outliers_removed = 0
        # Use 99th percentile method instead of IQR - more robust for skewed data
        for col in feature_cols:
            # Use percentiles instead of IQR for better handling of skewed distributions
            lower_bound = df[col].quantile(0.001)  # 0.1th percentile
            upper_bound = df[col].quantile(0.999)  # 99.9th percentile

            # Count outliers
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            # Only cap if more than 0.5% of data points are outliers
            if outliers / len(df) > 0.005:
                outliers_removed += outliers
                # Cap outliers
                df[col] = np.clip(df[col], lower_bound, upper_bound)

        if outliers_removed > 0:
            self.logger.info(f"Capped {outliers_removed} outlier values using percentile method ({outliers_removed/len(df)*100:.2f}% of data)")

        return df

    def _handle_outliers_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using Z-score method"""
        from scipy import stats

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'output_frame_id',
                       'target_x', 'target_y']

        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        outliers_removed = 0
        for col in feature_cols:
            z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
            outliers = (z_scores > 3).sum()  # 3 standard deviations
            outliers_removed += outliers

            # Cap outliers at 3 standard deviations
            mean_val = df[col].mean()
            std_val = df[col].std()
            df[col] = np.clip(df[col], mean_val - 3*std_val, mean_val + 3*std_val)

        if outliers_removed > 0:
            self.logger.info(f"Capped {outliers_removed} outlier values using Z-score method")

        return df

    def _handle_outliers_isolation_forest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using Isolation Forest"""
        try:
            from sklearn.ensemble import IsolationForest

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'output_frame_id',
                           'target_x', 'target_y']

            feature_cols = [col for col in numeric_cols if col not in exclude_cols]

            if len(feature_cols) > 1:
                iso_forest = IsolationForest(
                    contamination=0.1,  # Assume 10% outliers
                    random_state=self.config.random_state
                )

                outlier_preds = iso_forest.fit_predict(df[feature_cols])
                outlier_mask = outlier_preds == -1

                outliers_count = outlier_mask.sum()
                if outliers_count > 0:
                    # Remove outlier rows
                    df = df[~outlier_mask].copy()
                    self.logger.info(f"Removed {outliers_count} outlier rows using Isolation Forest")

                self.outlier_detectors['isolation_forest'] = iso_forest

        except ImportError:
            self.logger.warning("IsolationForest not available, skipping outlier removal")

        return df

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation sets"""
        self.logger.info("Splitting data into train/validation sets...")

        if self.config.cv_strategy == 'timeseries':
            train_data, val_data = self._split_timeseries(df)
        elif self.config.cv_strategy == 'kfold':
            train_data, val_data = self._split_random(df)
        elif self.config.cv_strategy == 'stratified':
            train_data, val_data = self._split_stratified(df)
        else:
            raise ValueError(f"Unknown CV strategy: {self.config.cv_strategy}")

        self.logger.info(f"Train set: {len(train_data)} samples")
        self.logger.info(f"Validation set: {len(val_data)} samples")

        return train_data, val_data

    def _split_timeseries(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data using time series approach - by week/game for proper temporal validation"""
        # Extract week number from game_id if available (e.g., input_2023_w01.csv -> week 1)
        # Otherwise sort by game_id to approximate temporal order

        if 'week' in df.columns:
            # If week column exists, use it directly
            df_sorted = df.sort_values(['week', 'game_id']).reset_index(drop=True)
            unique_weeks = sorted(df['week'].unique())
            # Train on first 80% of weeks, validate on last 20%
            split_week_idx = int(len(unique_weeks) * (1 - self.config.val_size))
            split_week = unique_weeks[split_week_idx]
            train_data = df_sorted[df_sorted['week'] < split_week]
            val_data = df_sorted[df_sorted['week'] >= split_week]
            self.logger.info(f"Temporal split: training on weeks {unique_weeks[:split_week_idx]}, validating on weeks {unique_weeks[split_week_idx:]}")
        else:
            # Fallback: sort by game_id and split sequentially
            # This ensures temporal ordering - earlier games in train, later in validation
            df_sorted = df.sort_values(['game_id', 'play_id']).reset_index(drop=True)
            unique_games = sorted(df['game_id'].unique())
            split_game_idx = int(len(unique_games) * (1 - self.config.val_size))
            split_game = unique_games[split_game_idx]
            train_data = df_sorted[df_sorted['game_id'] < split_game]
            val_data = df_sorted[df_sorted['game_id'] >= split_game]
            self.logger.info(f"Temporal split: training on {split_game_idx} games, validating on {len(unique_games) - split_game_idx} games")

        return train_data, val_data

    def _split_random(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data randomly"""
        # Shuffle the data
        df_shuffled = df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)

        split_idx = int(len(df_shuffled) * (1 - self.config.val_size))
        train_data = df_shuffled.iloc[:split_idx]
        val_data = df_shuffled.iloc[split_idx:]

        return train_data, val_data

    def _split_stratified(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data using stratified approach (if applicable)"""
        # For now, fall back to random split since we don't have clear stratification variable
        self.logger.warning("Stratified split not fully implemented, using random split")
        return self._split_random(df)

    def _add_temporal_features_to_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features to a data split (train or validation).
        This is done AFTER splitting to prevent data leakage.
        """
        self.logger.info(f"  Creating temporal features for {len(df)} samples...")

        # Group by game, play, and player for temporal calculations
        groupby_cols = ['game_id', 'play_id', 'nfl_id']

        # Sort within groups
        if 'frame_id' in df.columns:
            df = df.sort_values(groupby_cols + ['frame_id'])

        # Position changes (velocity approximation)
        for col in ['x', 'y']:
            if col in df.columns:
                df[f'{col}_change'] = df.groupby(groupby_cols)[col].diff()
                df[f'{col}_change'] = df[f'{col}_change'].fillna(0)

        # Speed changes
        if 's' in df.columns:
            df['s_change'] = df.groupby(groupby_cols)['s'].diff()
            df['s_change'] = df['s_change'].fillna(0)

        # Acceleration changes
        if 'a' in df.columns:
            df['a_change'] = df.groupby(groupby_cols)['a'].diff()
            df['a_change'] = df['a_change'].fillna(0)

        # Direction changes
        if 'dir' in df.columns:
            df['dir_change'] = df.groupby(groupby_cols)['dir'].diff()
            df['dir_change'] = df['dir_change'].fillna(0)
            # Handle wraparound (e.g., 359 -> 1 should be +2, not -358)
            df.loc[df['dir_change'] > 180, 'dir_change'] -= 360
            df.loc[df['dir_change'] < -180, 'dir_change'] += 360

        # Rolling statistics (3-frame window to avoid too much smoothing)
        window = 3
        for col in ['s', 'a', 'x', 'y']:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df.groupby(groupby_cols)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std'] = df.groupby(groupby_cols)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                df[f'{col}_rolling_std'] = df[f'{col}_rolling_std'].fillna(0)

        self.logger.info(f"  Added temporal features. Total columns: {len(df.columns)}")

        return df

    def _separate_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
        """Separate features and targets - FIXED to handle non-numeric columns"""
        # Identify feature columns (exclude metadata and targets)
        exclude_cols = [
            'game_id', 'play_id', 'nfl_id', 'frame_id', 'output_frame_id',
            'target_x', 'target_y', 'player_birth_date', 'player_height',
            'player_position', 'player_role', 'player_side', 'play_direction',
            'position_group', 'player_name', 'player_display_name',  # Player metadata
            'player_to_predict'  # Boolean flag, not a feature
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Separate features
        X = df[feature_cols].copy()

        # IMPORTANT FIX: Remove any remaining non-numeric columns
        # This handles categorical columns that weren't explicitly excluded
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns

        if len(non_numeric_cols) > 0:
            self.logger.warning(f"⚠️  Found {len(non_numeric_cols)} non-numeric columns after feature engineering: {list(non_numeric_cols)}")
            self.logger.warning("    These should have been encoded during feature engineering!")
            self.logger.warning(f"    Dropping: {list(non_numeric_cols)}")
            self.logger.info("    ℹ️  Check feature_engineering.py to ensure all categorical features are properly encoded")
            X = X[numeric_cols]

        y_x = df['target_x'].values
        y_y = df['target_y'].values

        # Keep metadata
        metadata_cols = ['game_id', 'play_id', 'nfl_id', 'output_frame_id']
        metadata = df[metadata_cols]

        return X, y_x, y_y, metadata

    def _scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using the configured scaler"""
        self.logger.info(f"Scaling features using {self.config.scaler_type} scaler...")

        if self.config.scaler_type == 'robust':
            scaler = RobustScaler()
        elif self.config.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.config.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.config.scaler_type == 'none':
            # No scaling
            self.scalers['feature_scaler'] = None
            return X_train.values, X_val.values
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")

        # Fit on training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers['feature_scaler'] = scaler

        self.logger.info("Feature scaling completed")
        return X_train_scaled, X_val_scaled

    def _log_data_statistics(self, X_train: np.ndarray, X_val: np.ndarray,
                           y_train_x: np.ndarray, y_val_x: np.ndarray):
        """Log statistics about the prepared data"""
        self.logger.info("")
        self.logger.info("DATA STATISTICS")
        self.logger.info("-" * 80)
        self.logger.info(f"Features: {X_train.shape[1]}")
        self.logger.info(f"Train samples: {X_train.shape[0]}")
        self.logger.info(f"Validation samples: {X_val.shape[0]}")
        self.logger.info(f"Train X target range: [{y_train_x.min():.2f}, {y_train_x.max():.2f}]")
        self.logger.info(f"Val X target range: [{y_val_x.min():.2f}, {y_val_x.max():.2f}]")

        # Memory usage
        train_memory = X_train.nbytes / 1024 / 1024  # MB
        val_memory = X_val.nbytes / 1024 / 1024  # MB
        self.logger.info(f"Train data memory: {train_memory:.2f} MB")
        self.logger.info(f"Val data memory: {val_memory:.2f} MB")

    def inverse_transform_targets(self, y_pred_x: np.ndarray, y_pred_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform target predictions if needed"""
        # For now, targets are not scaled, so return as-is
        return y_pred_x, y_pred_y

    def get_cross_validation_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits for hyperparameter tuning"""
        if self.config.cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds_tuning)
        elif self.config.cv_strategy == 'kfold':
            cv = KFold(n_splits=self.config.cv_folds_tuning, shuffle=True, random_state=self.config.random_state)
        else:
            cv = KFold(n_splits=self.config.cv_folds_tuning, shuffle=True, random_state=self.config.random_state)

        splits = []
        for train_idx, val_idx in cv.split(X):
            splits.append((train_idx, val_idx))

        return splits


class DataValidator:
    """Additional data validation utilities"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def validate_prepared_data(self, prepared_data: Dict[str, Any]) -> List[str]:
        """Validate prepared modeling data"""
        issues = []

        # Check shapes
        X_train = prepared_data['X_train']
        X_val = prepared_data['X_val']

        if X_train.shape[1] != X_val.shape[1]:
            issues.append(f"Feature count mismatch: train={X_train.shape[1]}, val={X_val.shape[1]}")

        # Check for NaN values
        if np.isnan(X_train).any():
            issues.append("NaN values found in training features")
        if np.isnan(X_val).any():
            issues.append("NaN values found in validation features")

        # Check target shapes
        y_train_x = prepared_data['y_train_x']
        y_val_x = prepared_data['y_val_x']

        if len(y_train_x) != X_train.shape[0]:
            issues.append("Training target X length mismatch")
        if len(y_val_x) != X_val.shape[0]:
            issues.append("Validation target X length mismatch")

        return issues


if __name__ == "__main__":
    # Test data preparation
    from nfl_pipeline.core.config import get_quick_config

    config = get_quick_config()
    prep = DataPreparation(config)

    # Create sample data
    input_data = pd.DataFrame({
        'game_id': [1, 1, 2, 2],
        'play_id': [1, 1, 1, 1],
        'nfl_id': [1, 2, 1, 2],
        'frame_id': [1, 1, 1, 1],
        'x': [10, 20, 30, 40],
        'y': [15, 25, 35, 45],
        's': [5, 6, 7, 8],
        'a': [1, 2, 3, 4],
        'dir': [90, 180, 270, 0]
    })

    output_data = pd.DataFrame({
        'game_id': [1, 1, 2, 2],
        'play_id': [1, 1, 1, 1],
        'nfl_id': [1, 2, 1, 2],
        'frame_id': [2, 2, 2, 2],
        'x': [12, 22, 32, 42],
        'y': [17, 27, 37, 47]
    })

    try:
        prepared = prep.prepare_modeling_data(input_data, output_data)
        print("Data preparation test successful")
        print(f"Train shape: {prepared['X_train'].shape}")
        print(f"Val shape: {prepared['X_val'].shape}")
        print(f"Features: {len(prepared['feature_names'])}")

    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        import traceback
        traceback.print_exc()