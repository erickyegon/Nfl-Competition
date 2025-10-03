"""
Physics-Based Features - Velocity, acceleration, momentum, kinetic energy
"""

import numpy as np
import pandas as pd
from typing import Dict

from nfl_pipeline.features.base import BaseFeatureEngineer
from nfl_pipeline.utils.helpers import safe_divide, timer


class PhysicsFeatureEngineer(BaseFeatureEngineer):
    """
    Create physics-based features for player movement.

    Features include:
    - Velocity components (x, y)
    - Acceleration components (x, y)
    - Momentum
    - Kinetic energy
    - Direction changes
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all physics-based features"""
        with timer("Physics Features"):
            self.logger.info("Creating physics features...")

            df = self._velocity_components(df)
            df = self._acceleration_components(df)
            df = self._momentum_features(df)
            df = self._kinetic_energy_features(df)
            df = self._direction_change_features(df)

            return df

    def _velocity_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose velocity into x and y components"""
        if 's' in df.columns and 'dir' in df.columns:
            df['v_x'] = (df['s'] * np.sin(np.radians(df['dir']))).astype(np.float32)
            df['v_y'] = (df['s'] * np.cos(np.radians(df['dir']))).astype(np.float32)
            self.feature_names.extend(['v_x', 'v_y'])
        return df

    def _acceleration_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose acceleration into x and y components"""
        if 'a' in df.columns and 'dir' in df.columns:
            df['a_x'] = (df['a'] * np.sin(np.radians(df['dir']))).astype(np.float32)
            df['a_y'] = (df['a'] * np.cos(np.radians(df['dir']))).astype(np.float32)
            self.feature_names.extend(['a_x', 'a_y'])
        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum (mass * velocity approximation)"""
        if 's' in df.columns and 'player_weight' in df.columns:
            # Momentum = mass * velocity (weight as proxy for mass)
            df['momentum'] = (df['player_weight'] * df['s']).astype(np.float32)
            self.feature_names.append('momentum')

            # Momentum components
            if 'v_x' in df.columns and 'v_y' in df.columns:
                df['momentum_x'] = (df['player_weight'] * df['v_x']).astype(np.float32)
                df['momentum_y'] = (df['player_weight'] * df['v_y']).astype(np.float32)
                self.feature_names.extend(['momentum_x', 'momentum_y'])

        return df

    def _kinetic_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate kinetic energy (0.5 * mass * velocity^2)"""
        if 's' in df.columns and 'player_weight' in df.columns:
            # KE = 0.5 * m * v^2
            df['kinetic_energy'] = (0.5 * df['player_weight'] * df['s'] ** 2).astype(np.float32)
            self.feature_names.append('kinetic_energy')

        return df

    def _direction_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate direction changes and angular velocity"""
        if 'dir' in df.columns:
            groupby_cols = ['game_id', 'play_id', 'nfl_id']

            # Direction change (angular difference between frames)
            df['dir_change'] = df.groupby(groupby_cols)['dir'].diff().fillna(0).astype(np.float32)

            # Normalize to [-180, 180]
            df['dir_change'] = df['dir_change'].apply(
                lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x)
            ).astype(np.float32)

            # Angular velocity (direction change per time unit)
            df['angular_velocity'] = df['dir_change'].abs().astype(np.float32)

            self.feature_names.extend(['dir_change', 'angular_velocity'])

        return df
