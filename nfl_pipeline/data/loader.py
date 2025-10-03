"""
Data Loading Module for NFL ML Pipeline
Handles loading, validation, and preprocessing of training and test data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import warnings
from datetime import datetime

from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.helpers import (
    timer, get_memory_usage, optimize_dataframe_types,
    validate_dataframe, force_garbage_collection
)


class DataLoader:
    """
    Handles all data loading operations with validation and error handling.

    This class is responsible for:
    - Loading training input and output files
    - Data validation and integrity checks
    - Memory optimization
    - Error handling and logging
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger
        self.input_dfs = []
        self.output_dfs = []

    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load all training input and output files from the train directory.

        Returns:
            Tuple of (input_df, output_df) concatenated across all weeks
        """
        with timer("Data Loading"):
            self.logger.info("=" * 80)
            self.logger.info("LOADING TRAINING DATA")
            self.logger.info("=" * 80)

            train_dir = self.config.train_dir

            # Find all input and output files
            input_files = self._find_data_files(train_dir, 'input')
            output_files = self._find_data_files(train_dir, 'output')

            self.logger.info(f"Found {len(input_files)} input files and {len(output_files)} output files")

            if len(input_files) == 0:
                raise FileNotFoundError(f"No input files found in {train_dir}")

            # Load input files
            self._load_input_files(input_files)

            # Load output files
            self._load_output_files(output_files)

            # Concatenate all data
            input_df = self._concatenate_dataframes(self.input_dfs, "input")
            output_df = self._concatenate_dataframes(self.output_dfs, "output")

            # Validate data integrity
            self._validate_data(input_df, output_df)

            # Optimize memory usage
            input_df = self._optimize_memory(input_df, "input")
            output_df = self._optimize_memory(output_df, "output")

            # Final summary
            self._log_data_summary(input_df, output_df)

            return input_df, output_df

    def load_test_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load test data if available.

        Returns:
            Tuple of (test_input_df, test_metadata_df) or (None, None) if not found
        """
        try:
            test_input_path = self.config.data_dir / 'test_input.csv'
            test_meta_path = self.config.data_dir / 'test.csv'

            if not test_input_path.exists() or not test_meta_path.exists():
                self.logger.warning("Test data files not found")
                return None, None

            with timer("Test Data Loading"):
                self.logger.info("Loading test data...")

                test_input = pd.read_csv(test_input_path)
                test_meta = pd.read_csv(test_meta_path)

                # Validate test data
                self._validate_test_data(test_input, test_meta)

                # Optimize memory
                test_input = self._optimize_memory(test_input, "test_input")
                test_meta = self._optimize_memory(test_meta, "test_metadata")

                self.logger.info(f"Test input shape: {test_input.shape}")
                self.logger.info(f"Test metadata shape: {test_meta.shape}")

                return test_input, test_meta

        except Exception as e:
            self.logger.error(f"Error loading test data: {e}")
            return None, None

    def _find_data_files(self, directory: Path, file_type: str) -> List[Path]:
        """Find all data files of specified type"""
        if file_type == 'input':
            pattern = 'input_*.csv'
        elif file_type == 'output':
            pattern = 'output_*.csv'
        else:
            raise ValueError(f"Unknown file type: {file_type}")

        files = sorted(directory.glob(pattern))

        if not files:
            # Try alternative patterns
            if file_type == 'input':
                files = sorted(directory.glob('*input*.csv'))
            elif file_type == 'output':
                files = sorted(directory.glob('*output*.csv'))

        return files

    def _load_input_files(self, input_files: List[Path]):
        """Load all input CSV files with memory optimization"""
        for file_path in input_files:
            try:
                self.logger.info(f"Loading {file_path.name}...")

                # Load with optimized dtypes to reduce memory
                dtypes = {
                    'game_id': 'int32',
                    'play_id': 'int32',
                    'nfl_id': 'int32',
                    'frame_id': 'int16',
                    'x': 'float32',
                    'y': 'float32',
                    's': 'float32',
                    'a': 'float32',
                    'o': 'float32',
                    'dir': 'float32'
                }

                df = pd.read_csv(file_path, dtype=dtypes)

                # Basic validation
                self._validate_input_file(df, file_path.name)

                self.input_dfs.append(df)

                memory_mb = get_memory_usage(df)
                self.logger.info(f"  Shape: {df.shape}, Memory: {memory_mb:.2f} MB")

            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                raise

    def _load_output_files(self, output_files: List[Path]):
        """Load all output CSV files with memory optimization"""
        for file_path in output_files:
            try:
                self.logger.info(f"Loading {file_path.name}...")

                # Load with optimized dtypes
                dtypes = {
                    'game_id': 'int32',
                    'play_id': 'int32',
                    'nfl_id': 'int32',
                    'frame_id': 'int16',
                    'x': 'float32',
                    'y': 'float32'
                }

                df = pd.read_csv(file_path, dtype=dtypes)

                # Basic validation
                self._validate_output_file(df, file_path.name)

                self.output_dfs.append(df)

                memory_mb = get_memory_usage(df)
                self.logger.info(f"  Shape: {df.shape}, Memory: {memory_mb:.2f} MB")

            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")
                raise

    def _concatenate_dataframes(self, dfs: List[pd.DataFrame], name: str) -> pd.DataFrame:
        """Concatenate list of DataFrames with memory management"""
        if not dfs:
            raise ValueError(f"No {name} dataframes to concatenate")

        self.logger.info(f"Concatenating {len(dfs)} {name} dataframes...")

        # Concatenate
        combined_df = pd.concat(dfs, ignore_index=True)

        # Force garbage collection
        force_garbage_collection()

        self.logger.info(f"Combined {name} data: {combined_df.shape}")
        return combined_df

    def _validate_data(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Perform comprehensive data validation checks"""
        with timer("Data Validation"):
            self.logger.info("-" * 80)
            self.logger.info("DATA VALIDATION")
            self.logger.info("-" * 80)

            # Check for missing values
            self._check_missing_values(input_df, "input")
            self._check_missing_values(output_df, "output")

            # Check for duplicate rows
            self._check_duplicates(input_df, "input")
            self._check_duplicates(output_df, "output")

            # Validate required columns
            self._validate_required_columns(input_df, output_df)

            # Validate data types
            self._validate_data_types(input_df, output_df)

            # Validate data ranges
            self._validate_data_ranges(input_df, output_df)

            # Validate data consistency
            self._validate_data_consistency(input_df, output_df)

            self.logger.info("Data validation passed")

    def _validate_test_data(self, test_input: pd.DataFrame, test_meta: pd.DataFrame):
        """Validate test data structure"""
        # Check required columns exist
        required_input_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        # FIXED: Test metadata has game_id, play_id, nfl_id, frame_id instead of 'id'
        # We'll create the 'id' column if it doesn't exist
        required_meta_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']

        missing_input = set(required_input_cols) - set(test_input.columns)
        missing_meta = set(required_meta_cols) - set(test_meta.columns)

        if missing_input:
            raise ValueError(f"Test input missing columns: {missing_input}")
        if missing_meta:
            raise ValueError(f"Test metadata missing columns: {missing_meta}")

        # Create composite 'id' column if it doesn't exist
        if 'id' not in test_meta.columns:
            test_meta['id'] = (
                test_meta['game_id'].astype(str) + '_' +
                test_meta['play_id'].astype(str) + '_' +
                test_meta['nfl_id'].astype(str) + '_' +
                test_meta['frame_id'].astype(str)
            )
            self.logger.info("Created composite 'id' column for test metadata")

    def _validate_input_file(self, df: pd.DataFrame, filename: str):
        """Validate individual input file"""
        required_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 's', 'a', 'o', 'dir']

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Input file {filename} missing required columns: {missing_cols}")

        # Check for empty dataframe
        if df.empty:
            raise ValueError(f"Input file {filename} is empty")

    def _validate_output_file(self, df: pd.DataFrame, filename: str):
        """Validate individual output file"""
        required_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']

        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Output file {filename} missing required columns: {missing_cols}")

        # Check for empty dataframe
        if df.empty:
            raise ValueError(f"Output file {filename} is empty")

    def _check_missing_values(self, df: pd.DataFrame, name: str):
        """Check for missing values in dataframe"""
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if not missing_cols.empty:
            self.logger.warning(f"Missing values in {name} data:")
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                self.logger.warning(f"  {col}: {count} ({percentage:.2f}%)")
        else:
            self.logger.info(f"No missing values in {name} data")

    def _check_duplicates(self, df: pd.DataFrame, name: str):
        """Check for duplicate rows"""
        # Define key columns for uniqueness check
        if name == 'input':
            key_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        else:  # output
            key_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']

        available_key_cols = [col for col in key_cols if col in df.columns]

        if available_key_cols:
            duplicates = df.duplicated(subset=available_key_cols).sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate rows in {name} data")
            else:
                self.logger.info(f"No duplicate rows in {name} data")
        else:
            self.logger.warning(f"Cannot check duplicates in {name} data - key columns missing")

    def _validate_required_columns(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Validate that all required columns are present"""
        required_input_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id',
                              'x', 'y', 's', 'a', 'o', 'dir']
        required_output_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']

        missing_input = set(required_input_cols) - set(input_df.columns)
        missing_output = set(required_output_cols) - set(output_df.columns)

        if missing_input:
            raise ValueError(f"Missing required input columns: {missing_input}")
        if missing_output:
            raise ValueError(f"Missing required output columns: {missing_output}")

    def _validate_data_types(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Validate data types are appropriate"""
        # Numeric columns that should be numeric
        numeric_cols = ['x', 'y', 's', 'a', 'o', 'dir', 'game_id', 'play_id', 'nfl_id', 'frame_id']

        for df, name in [(input_df, 'input'), (output_df, 'output')]:
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        self.logger.warning(f"{name} column {col} should be numeric but is {df[col].dtype}")

    def _validate_data_ranges(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Validate that data values are in reasonable ranges"""
        # Field position validation
        if 'x' in input_df.columns:
            x_range = input_df['x'].agg(['min', 'max'])
            if x_range['min'] < 0 or x_range['max'] > 120:
                self.logger.warning(f"Input X position range: {x_range['min']:.1f} to {x_range['max']:.1f} (expected 0-120)")

        if 'y' in input_df.columns:
            y_range = input_df['y'].agg(['min', 'max'])
            if y_range['min'] < 0 or y_range['max'] > 53.3:
                self.logger.warning(f"Input Y position range: {y_range['min']:.1f} to {y_range['max']:.1f} (expected 0-53.3)")

        # Speed validation (reasonable range for football players)
        if 's' in input_df.columns:
            speed_range = input_df['s'].agg(['min', 'max'])
            if speed_range['max'] > 30:  # Max speed in NFL is around 25-30 mph
                self.logger.warning(f"Input speed range: {speed_range['min']:.1f} to {speed_range['max']:.1f} mph")

        # Angle validation
        for col in ['o', 'dir']:
            if col in input_df.columns:
                angle_range = input_df[col].agg(['min', 'max'])
                if angle_range['min'] < 0 or angle_range['max'] > 360:
                    self.logger.warning(f"Input {col} angle range: {angle_range['min']:.1f} to {angle_range['max']:.1f} (expected 0-360)")

    def _validate_data_consistency(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Validate consistency between input and output data"""
        # Check if game/play/player combinations match
        input_keys = set(zip(input_df['game_id'], input_df['play_id'], input_df['nfl_id']))
        output_keys = set(zip(output_df['game_id'], output_df['play_id'], output_df['nfl_id']))

        # Output should be subset of input (players we need to predict)
        missing_in_input = output_keys - input_keys
        if missing_in_input:
            self.logger.warning(f"{len(missing_in_input)} player-game-play combinations in output not found in input")

    def _optimize_memory(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        initial_memory = get_memory_usage(df)

        # Optimize data types
        df_optimized = optimize_dataframe_types(df)

        final_memory = get_memory_usage(df_optimized)
        savings = initial_memory - final_memory

        self.logger.info(f"Memory optimization for {name}: {initial_memory:.2f} MB -> {final_memory:.2f} MB (saved {savings:.2f} MB)")

        return df_optimized

    def _log_data_summary(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Log comprehensive data summary"""
        self.logger.info("")
        self.logger.info("DATA SUMMARY")
        self.logger.info("-" * 80)

        # Dataset overview
        n_games = input_df['game_id'].nunique()
        n_plays = input_df.groupby('game_id')['play_id'].nunique().sum()
        n_players = input_df['nfl_id'].nunique()

        self.logger.info(f"Total games: {n_games}")
        self.logger.info(f"Total plays: {n_plays}")
        self.logger.info(f"Unique players: {n_players}")
        self.logger.info(f"Average players per play: {input_df.groupby(['game_id', 'play_id'])['nfl_id'].nunique().mean():.1f}")

        # Position statistics
        if 'x' in input_df.columns:
            self.logger.info(f"X position range: [{input_df['x'].min():.1f}, {input_df['x'].max():.1f}] yards")
        if 'y' in input_df.columns:
            self.logger.info(f"Y position range: [{input_df['y'].min():.1f}, {input_df['y'].max():.1f}] yards")

        # Speed statistics
        if 's' in input_df.columns:
            self.logger.info(f"Speed range: [{input_df['s'].min():.1f}, {input_df['s'].max():.1f}] mph")
            self.logger.info(f"Average speed: {input_df['s'].mean():.2f} mph")

        # Memory usage
        input_memory = get_memory_usage(input_df)
        output_memory = get_memory_usage(output_df)
        total_memory = input_memory + output_memory

        self.logger.info(f"Input data memory: {input_memory:.2f} MB")
        self.logger.info(f"Output data memory: {output_memory:.2f} MB")
        self.logger.info(f"Total memory usage: {total_memory:.2f} MB")


class DataInspector:
    """Utility class for inspecting and understanding the data"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def inspect_data(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Perform detailed data inspection"""
        self.logger.info("DATA INSPECTION")
        self.logger.info("=" * 80)

        # Basic info
        self._inspect_basic_info(input_df, output_df)

        # Column analysis
        self._inspect_columns(input_df, "input")
        self._inspect_columns(output_df, "output")

        # Statistical summary
        self._inspect_statistics(input_df, "input")
        self._inspect_statistics(output_df, "output")

    def _inspect_basic_info(self, input_df: pd.DataFrame, output_df: pd.DataFrame):
        """Inspect basic DataFrame information"""
        self.logger.info("Basic Information:")
        self.logger.info(f"Input shape: {input_df.shape}")
        self.logger.info(f"Output shape: {output_df.shape}")
        self.logger.info(f"Input memory: {get_memory_usage(input_df):.2f} MB")
        self.logger.info(f"Output memory: {get_memory_usage(output_df):.2f} MB")

    def _inspect_columns(self, df: pd.DataFrame, name: str):
        """Inspect DataFrame columns"""
        self.logger.info(f"\n{name.upper()} Columns:")
        for col in df.columns:
            dtype = df[col].dtype
            n_unique = df[col].nunique()
            n_missing = df[col].isnull().sum()
            self.logger.info(f"  {col}: {dtype}, {n_unique} unique, {n_missing} missing")

    def _inspect_statistics(self, df: pd.DataFrame, name: str):
        """Inspect statistical properties"""
        self.logger.info(f"\n{name.upper()} Statistics:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe()
            self.logger.info(f"\n{stats}")


if __name__ == "__main__":
    # Test data loader
    from nfl_pipeline.core.config import get_quick_config

    config = get_quick_config()
    loader = DataLoader(config)

    try:
        input_df, output_df = loader.load_training_data()
        print("Data loading test successful")

        # Test inspector
        inspector = DataInspector(config)
        inspector.inspect_data(input_df, output_df)

    except Exception as e:
        print(f"Data loading test failed: {e}")
        import traceback
        traceback.print_exc()