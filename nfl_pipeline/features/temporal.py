"""
Temporal Features - Rolling statistics, position changes, motion tracking
"""

import numpy as np
import pandas as pd
from typing import Dict

from nfl_pipeline.features.base import BaseFeatureEngineer
from nfl_pipeline.utils.helpers import timer


class TemporalFeatureEngineer(BaseFeatureEngineer):
    """
    Create temporal/time-series features for player movement.

    Features include:
    - Position changes (frame-to-frame deltas)
    - Rolling statistics (moving averages)
    - Motion changes (acceleration, deceleration)

    IMPORTANT: These features should be created AFTER train/test split
    to prevent data leakage!
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all temporal features"""
        with timer("Temporal Features"):
            self.logger.info("⚠️  Creating temporal features - ensure this is AFTER split!")

            # Sort data for temporal operations
            df = self._sort_for_temporal_features(df)

            df = self._position_change_features(df)
            df = self._rolling_statistics_features(df)
            df = self._motion_change_features(df)

            return df

    def _sort_for_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe for temporal feature creation"""
        sort_cols = []
        for col in ['game_id', 'play_id', 'nfl_id', 'frame_id']:
            if col in df.columns:
                sort_cols.append(col)

        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
            self.logger.debug(f"Sorted by {sort_cols} for temporal features")

        return df

    def _position_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Frame-to-frame position changes"""
        groupby_cols = ['game_id', 'play_id', 'nfl_id']

        # Position deltas (change from previous frame)
        for col in ['x', 'y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_change'] = df.groupby(groupby_cols)[col].diff()
                df[f'{col}_change'] = df[f'{col}_change'].fillna(0).astype(np.float32)
                self.feature_names.append(f'{col}_change')

        # Distance traveled (Euclidean distance from previous frame)
        if 'x_change' in df.columns and 'y_change' in df.columns:
            df['distance_traveled'] = np.sqrt(
                df['x_change']**2 + df['y_change']**2
            ).astype(np.float32)
            self.feature_names.append('distance_traveled')

        return df

    def _rolling_statistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window statistics (moving averages, std)"""
        groupby_cols = ['game_id', 'play_id', 'nfl_id']
        windows = [3, 5, 10]  # Different time windows

        for window in windows:
            for col in ['s', 'a', 'x', 'y']:
                if col in df.columns:
                    # Rolling mean
                    col_name = f'{col}_rolling_mean_{window}'
                    df[col_name] = df.groupby(groupby_cols)[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    ).astype(np.float32)
                    self.feature_names.append(col_name)

                    # Rolling std (measure of variability)
                    if window >= 5:  # Only for larger windows
                        col_name = f'{col}_rolling_std_{window}'
                        df[col_name] = df.groupby(groupby_cols)[col].transform(
                            lambda x: x.rolling(window=window, min_periods=2).std()
                        ).fillna(0).astype(np.float32)
                        self.feature_names.append(col_name)

        # Exponential moving average (gives more weight to recent frames)
        for col in ['s', 'a']:
            if col in df.columns:
                col_name = f'{col}_ewm'
                df[col_name] = df.groupby(groupby_cols)[col].transform(
                    lambda x: x.ewm(span=5, min_periods=1).mean()
                ).astype(np.float32)
                self.feature_names.append(col_name)

        return df

    def _motion_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect acceleration, deceleration, and direction changes"""
        groupby_cols = ['game_id', 'play_id', 'nfl_id']

        # Speed change rate (jerk - derivative of acceleration)
        if 'a' in df.columns:
            df['jerk'] = df.groupby(groupby_cols)['a'].diff().fillna(0).astype(np.float32)
            self.feature_names.append('jerk')

        # Is player accelerating, decelerating, or steady?
        if 's_change' in df.columns:
            df['is_accelerating'] = (df['s_change'] > 0.5).astype(np.int8)
            df['is_decelerating'] = (df['s_change'] < -0.5).astype(np.int8)
            df['is_steady_speed'] = (df['s_change'].abs() <= 0.5).astype(np.int8)
            self.feature_names.extend(['is_accelerating', 'is_decelerating', 'is_steady_speed'])

        # Sharp direction changes (potential route breaks)
        if 'dir_change' in df.columns:
            df['sharp_dir_change'] = (df['dir_change'].abs() > 45).astype(np.int8)
            self.feature_names.append('sharp_dir_change')

        return df
