"""
Spatial Features - Field position, distances, interactions, coverage
"""

import numpy as np
import pandas as pd
from typing import Dict

from nfl_pipeline.features.base import BaseFeatureEngineer
from nfl_pipeline.utils.helpers import timer


class SpatialFeatureEngineer(BaseFeatureEngineer):
    """
    Create spatial features for player movement.

    Features include:
    - Field position (sideline distance, goal line distance)
    - Player interactions (distances, spacing)
    - Coverage features (defenders nearby)
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all spatial features"""
        with timer("Spatial Features"):
            self.logger.info("Creating spatial features...")

            df = self._field_position_features(df)
            df = self._sideline_features(df)
            df = self._goal_line_features(df)
            df = self._interaction_features(df)

            return df

    def _field_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create field position features"""
        if 'x' in df.columns and 'y' in df.columns:
            # Distance from center of field
            df['dist_from_center'] = np.abs(df['y'] - 26.65).astype(np.float32)

            # Absolute position (distance from own endzone)
            df['field_position'] = df['x'].abs().astype(np.float32)

            # Position quadrant (dividing field into regions)
            df['field_x_third'] = pd.cut(
                df['x'], bins=[-np.inf, -33.33, 33.33, np.inf], labels=[0, 1, 2]
            ).astype(np.int8)

            df['field_y_half'] = (df['y'] > 26.65).astype(np.int8)

            self.feature_names.extend([
                'dist_from_center', 'field_position', 'field_x_third', 'field_y_half'
            ])

        return df

    def _sideline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance from sidelines"""
        if 'y' in df.columns:
            # Field width is 53.3 yards (160 feet), y ranges from 0 to 53.3
            df['dist_to_left_sideline'] = df['y'].astype(np.float32)
            df['dist_to_right_sideline'] = (53.3 - df['y']).astype(np.float32)
            df['dist_to_nearest_sideline'] = np.minimum(
                df['dist_to_left_sideline'], df['dist_to_right_sideline']
            ).astype(np.float32)

            # Is player near sideline? (within 5 yards)
            df['near_sideline'] = (df['dist_to_nearest_sideline'] < 5).astype(np.int8)

            self.feature_names.extend([
                'dist_to_left_sideline', 'dist_to_right_sideline',
                'dist_to_nearest_sideline', 'near_sideline'
            ])

        return df

    def _goal_line_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance from goal lines"""
        if 'x' in df.columns:
            # Field length is 120 yards (100 + 2*10 for endzones)
            # x ranges from -10 to 110
            df['dist_to_own_goal'] = np.abs(df['x'] + 10).astype(np.float32)
            df['dist_to_opp_goal'] = np.abs(df['x'] - 110).astype(np.float32)

            # Is player in red zone? (within 20 yards of goal)
            df['in_red_zone'] = (df['dist_to_opp_goal'] < 20).astype(np.int8)

            # Is player in own territory?
            df['in_own_territory'] = (df['x'] < 50).astype(np.int8)

            self.feature_names.extend([
                'dist_to_own_goal', 'dist_to_opp_goal', 'in_red_zone', 'in_own_territory'
            ])

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Player interaction and spacing features"""
        if all(col in df.columns for col in ['x', 'y', 'game_id', 'play_id', 'frame_id']):
            # Group by game, play, frame to calculate inter-player features
            grouped = df.groupby(['game_id', 'play_id', 'frame_id'])

            # Number of players in frame (should be ~22 for offense + defense)
            df['num_players_in_frame'] = grouped['nfl_id'].transform('count').astype(np.int8)

            # Average distance to other players (proxy for spacing)
            # This is expensive, so we'll do a simplified version
            if len(df) < 100000:  # Only for smaller datasets
                df['avg_dist_to_others'] = df.apply(
                    lambda row: self._calc_avg_distance(row, df), axis=1
                ).astype(np.float32)
                self.feature_names.append('avg_dist_to_others')

            self.feature_names.append('num_players_in_frame')

        return df

    def _calc_avg_distance(self, row, df: pd.DataFrame) -> float:
        """Calculate average distance to other players in same frame"""
        # Get all players in same game/play/frame
        same_frame = df[
            (df['game_id'] == row['game_id']) &
            (df['play_id'] == row['play_id']) &
            (df['frame_id'] == row['frame_id']) &
            (df['nfl_id'] != row['nfl_id'])  # Exclude self
        ]

        if len(same_frame) == 0:
            return 0.0

        # Calculate Euclidean distances
        distances = np.sqrt(
            (same_frame['x'] - row['x'])**2 + (same_frame['y'] - row['y'])**2
        )

        return distances.mean()
