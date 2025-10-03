"""
NFL Domain Features - Route detection, coverage, pass context, formations
"""

import numpy as np
import pandas as pd
from typing import Dict

from nfl_pipeline.features.base import BaseFeatureEngineer
from nfl_pipeline.utils.helpers import timer, POSITION_GROUPS


class NFLDomainFeatureEngineer(BaseFeatureEngineer):
    """
    Create NFL-specific domain features.

    Features include:
    - Route detection (depth, breaks, patterns)
    - Coverage analysis (defenders nearby)
    - Pass context (QB distance, pressure)
    - Player roles and formations
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all NFL domain features"""
        with timer("NFL Domain Features"):
            self.logger.info("Creating NFL-specific features...")

            df = self._route_detection_features(df)
            df = self._coverage_analysis_features(df)
            df = self._pass_context_features(df)
            df = self._position_group_features(df)

            return df

    def _route_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect route patterns for receivers"""
        if 'x' in df.columns and 'y' in df.columns:
            # Route depth (how far downfield)
            df['route_depth'] = df['x'].abs().astype(np.float32)

            # Lateral movement (distance from sideline)
            df['lateral_movement'] = (df['y'] - 26.65).abs().astype(np.float32)

            # Route angle category (8 directions)
            if 'dir' in df.columns:
                df['route_angle_category'] = pd.cut(
                    df['dir'] % 360,
                    bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                    labels=[0, 1, 2, 3, 4, 5, 6, 7]
                ).astype(np.int8)
                self.feature_names.append('route_angle_category')

            # Route break detection (slow speed + depth > 5 yards)
            if 's' in df.columns:
                df['is_route_break'] = (
                    (df['s'] < 2.0) & (df['route_depth'] > 5)
                ).astype(np.int8)
                self.feature_names.append('is_route_break')

            # Deep route indicator
            df['is_deep_route'] = (df['route_depth'] > 20).astype(np.int8)

            # Short route indicator (quick slant, screen)
            df['is_short_route'] = (df['route_depth'] < 5).astype(np.int8)

            self.feature_names.extend([
                'route_depth', 'lateral_movement', 'is_deep_route', 'is_short_route'
            ])

        return df

    def _coverage_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze defensive coverage"""
        if 'player_position' in df.columns:
            # Identify defensive players
            df['is_defender'] = df['player_position'].isin([
                'CB', 'S', 'FS', 'SS', 'DB', 'LB', 'ILB', 'OLB', 'MLB', 'DL', 'DE', 'DT', 'NT'
            ]).astype(np.int8)

            # Identify offensive skill players
            df['is_receiver'] = df['player_position'].isin([
                'WR', 'TE', 'RB', 'FB'
            ]).astype(np.int8)

            # Count defenders in proximity (requires spatial join - simplified)
            # For full implementation, you'd calculate actual distances
            self.feature_names.extend(['is_defender', 'is_receiver'])

        return df

    def _pass_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features related to passing situations"""
        if all(col in df.columns for col in ['player_position', 'x', 'y']):
            # Identify QB
            df['is_qb'] = (df['player_position'] == 'QB').astype(np.int8)

            # Distance from line of scrimmage (assuming x=0 is LOS)
            df['dist_from_los'] = df['x'].abs().astype(np.float32)

            # In passing zone (typically 0-40 yards from LOS)
            df['in_passing_zone'] = (df['dist_from_los'] <= 40).astype(np.int8)

            self.feature_names.extend(['is_qb', 'dist_from_los', 'in_passing_zone'])

        return df

    def _position_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode player positions into groups"""
        if 'player_position' in df.columns:
            # Map positions to groups
            df['position_group'] = df['player_position'].map(POSITION_GROUPS)

            # One-hot encode position groups
            df = self._encode_categorical(df, 'position_group', 'pos')

            # Also encode specific positions (if not too many)
            unique_positions = df['player_position'].nunique()
            if unique_positions <= 20:
                df = self._encode_categorical(df, 'player_position', 'position')

        return df


class FeatureEngineer(BaseFeatureEngineer):
    """
    Main feature engineering orchestrator.

    Combines all feature engineering modules into a single interface.
    """

    def __init__(self, config):
        super().__init__(config)
        self.physics = PhysicsFeatureEngineer(config) if config.use_physics_features else None
        self.spatial = SpatialFeatureEngineer(config) if config.use_spatial_features else None
        self.temporal = TemporalFeatureEngineer(config) if config.use_temporal_features else None
        self.nfl_domain = NFLDomainFeatureEngineer(config) if config.use_nfl_features else None

    def engineer_features(self, input_df: pd.DataFrame, include_temporal: bool = True) -> pd.DataFrame:
        """
        Main feature engineering pipeline.

        Args:
            input_df: Raw input dataframe
            include_temporal: Whether to include temporal features (should be False before split)

        Returns:
            DataFrame with engineered features
        """
        with timer("Feature Engineering"):
            self.logger.info("=" * 80)
            self.logger.info("FEATURE ENGINEERING")
            self.logger.info("=" * 80)

            df = input_df.copy()

            # Basic transformations
            df = self._basic_transformations(df)

            # Physics features
            if self.physics:
                df = self.physics.transform(df)
                self.feature_names.extend(self.physics.feature_names)

            # Spatial features
            if self.spatial:
                df = self.spatial.transform(df)
                self.feature_names.extend(self.spatial.feature_names)

            # NFL domain features
            if self.nfl_domain:
                df = self.nfl_domain.transform(df)
                self.feature_names.extend(self.nfl_domain.feature_names)

            # Temporal features (ONLY if requested)
            if include_temporal and self.temporal:
                self.logger.info("⚠️  Including temporal features")
                df = self.temporal.transform(df)
                self.feature_names.extend(self.temporal.feature_names)
            elif not include_temporal:
                self.logger.info("✓ Skipping temporal features (will add after split)")

            self.logger.info(f"✓ Feature engineering complete. Total features: {len(df.columns)}")

            return df


# Import for backward compatibility
from nfl_pipeline.features.physics import PhysicsFeatureEngineer
from nfl_pipeline.features.spatial import SpatialFeatureEngineer
from nfl_pipeline.features.temporal import TemporalFeatureEngineer
