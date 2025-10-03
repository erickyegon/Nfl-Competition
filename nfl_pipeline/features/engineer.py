"""
Feature Engineering Module for NFL ML Pipeline
Creates physics-based, temporal, spatial, and role-based features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import PolynomialFeatures
import pickle
import hashlib

from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import PipelineLogger
from nfl_pipeline.utils.helpers import (
    timer, angular_difference, height_to_inches,
    POSITION_GROUPS, safe_divide, normalize_angle
)


class FeatureEngineer:
    """
    Comprehensive feature engineering for player movement prediction.

    Creates physics-based, temporal, spatial, and role-based features
    from raw player tracking data.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger
        self.feature_names = []
        self.polynomial_features = None
        self.cache_dir = config.output_dir / 'feature_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = True  # Enable caching by default

    def _get_cache_key(self, input_df: pd.DataFrame) -> str:
        """Generate cache key based on data hash and config"""
        # Create hash from data shape, columns, and config
        data_info = f"{input_df.shape}_{list(input_df.columns)}_{self.config.use_physics_features}_{self.config.use_temporal_features}_{self.config.use_spatial_features}_{self.config.use_role_features}"
        return hashlib.md5(data_info.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load features from cache if available"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                self.logger.info(f"Loading features from cache: {cache_key[:8]}...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.feature_names = cached_data['feature_names']
                    return cached_data['features']
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, features: pd.DataFrame):
        """Save features to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'features': features,
                    'feature_names': self.feature_names
                }, f)
            self.logger.info(f"Features saved to cache: {cache_key[:8]}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def engineer_features(self, input_df: pd.DataFrame, include_temporal: bool = True) -> pd.DataFrame:
        """
        Main feature engineering pipeline with caching support.

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

            # Disable cache if temporal features are excluded
            use_cache_for_this_run = self.use_cache and include_temporal

            # Check cache first
            if use_cache_for_this_run:
                cache_key = self._get_cache_key(input_df)
                cached_features = self._load_from_cache(cache_key)
                if cached_features is not None:
                    self.logger.info("✓ Using cached features")
                    return cached_features

            df = input_df.copy()

            # Sort by game, play, player, and frame for temporal features
            df = self._sort_for_temporal_features(df)

            # 1. Basic transformations
            if self.config.verbose > 0:
                self.logger.info("Creating basic transformations...")
            df = self._basic_transformations(df)

            # 2. Physics-based features (no leakage - uses current frame only)
            if self.config.use_physics_features:
                df = self._physics_features(df)

            # 3. Spatial features (no leakage - uses current frame only)
            if self.config.use_spatial_features:
                df = self._spatial_features(df)

            # 4. Temporal features (ONLY if include_temporal=True to avoid leakage)
            if self.config.use_temporal_features and include_temporal:
                self.logger.info("⚠️  Including temporal features - ensure this is AFTER train/test split!")
                try:
                    df = self._temporal_features(df)
                except KeyboardInterrupt:
                    self.logger.warning("Temporal features computation interrupted by user - skipping")
                    self.config.use_temporal_features = False
                except Exception as e:
                    self.logger.warning(f"Temporal features computation failed: {e} - skipping")
                    self.config.use_temporal_features = False

            # 5. Role-based features
            if self.config.use_role_features:
                try:
                    df = self._role_features(df)
                except MemoryError:
                    self.logger.warning("Role features computation failed due to memory constraints - skipping")
                    self.config.use_role_features = False
                except Exception as e:
                    self.logger.warning(f"Role features computation failed: {e} - skipping")
                    self.config.use_role_features = False

            # 6. NFL-specific domain features (routes, coverage, pass context)
            if self.config.use_nfl_features:
                try:
                    df = self._nfl_domain_features(df)
                except Exception as e:
                    self.logger.warning(f"NFL domain features computation failed: {e} - skipping")
                    self.config.use_nfl_features = False

            # 7. Interaction features
            df = self._interaction_features(df)

            # 8. Polynomial features (optional)
            if self.config.use_polynomial_features:
                df = self._polynomial_features(df)

            # Store feature names
            self.feature_names = df.columns.tolist()

            memory_mb = self._get_memory_usage(df)
            self.logger.info(f"Feature engineering complete. Total features: {len(df.columns)}, Memory: {memory_mb:.2f} MB")

            # Save to cache
            if self.use_cache:
                self._save_to_cache(cache_key, df)

            return df

    def _sort_for_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe for temporal feature calculations"""
        sort_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']
        available_sort_cols = [col for col in sort_cols if col in df.columns]

        if available_sort_cols:
            df = df.sort_values(available_sort_cols).reset_index(drop=True)

        return df

    def _basic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic feature transformations and encodings"""
        # Encode categorical variables
        df = self._encode_play_direction(df)
        df = self._encode_player_side(df)

        # Parse player attributes
        df = self._parse_player_height(df)
        df = self._calculate_player_age(df)

        # One-hot encode player roles
        df = self._encode_player_roles(df)

        return df

    def _encode_play_direction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode play direction"""
        if 'play_direction' in df.columns:
            df['play_direction_encoded'] = (df['play_direction'] == 'right').astype(int)
        return df

    def _encode_player_side(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode player side (offense/defense)"""
        if 'player_side' in df.columns:
            df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
        return df

    def _parse_player_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse player height to inches and convert to numeric"""
        if 'player_height' in df.columns:
            df['player_height_inches'] = df['player_height'].apply(height_to_inches).astype(np.float32)
        return df

    def _calculate_player_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate player age from birth date"""
        if 'player_birth_date' in df.columns:
            df['player_birth_date'] = pd.to_datetime(df['player_birth_date'].astype(str), errors='coerce')
            reference_date = pd.to_datetime('2023-09-01')  # Start of 2023 season
            df['player_age_days'] = (reference_date - df['player_birth_date']).dt.days
            df['player_age_years'] = df['player_age_days'] / 365.25
        return df

    def _encode_player_roles(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode player roles"""
        if 'player_role' in df.columns:
            # Targeted receiver
            df['is_targeted_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)

            # Passer (QB)
            df['is_passer'] = (df['player_role'] == 'Passer').astype(int)

            # Defensive coverage
            df['is_defensive_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)

            # Route runner
            df['is_route_runner'] = (df['player_role'] == 'Other Route Runner').astype(int)

        return df

    def _physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create physics-based features (velocity, acceleration, momentum)"""
        if self.config.verbose > 0:
            self.logger.info("Creating physics-based features...")

        # Velocity components
        df = self._velocity_components(df)

        # Acceleration components
        df = self._acceleration_components(df)

        # Momentum
        df = self._momentum_features(df)

        # Kinetic energy
        df = self._kinetic_energy_features(df)

        # Direction differences
        df = self._direction_features(df)

        return df

    def _velocity_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate velocity components from speed and direction"""
        if all(col in df.columns for col in ['s', 'dir']):
            # Convert direction to radians
            dir_rad = np.radians(df['dir'])

            # Velocity components
            df['velocity_x'] = df['s'] * np.cos(dir_rad)
            df['velocity_y'] = df['s'] * np.sin(dir_rad)

            # Normalized velocity components (direction only)
            df['velocity_x_norm'] = np.cos(dir_rad)
            df['velocity_y_norm'] = np.sin(dir_rad)

        return df

    def _acceleration_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate acceleration components"""
        if all(col in df.columns for col in ['a', 'dir']):
            # Convert direction to radians
            dir_rad = np.radians(df['dir'])

            # Acceleration components
            df['acceleration_x'] = df['a'] * np.cos(dir_rad)
            df['acceleration_y'] = df['a'] * np.sin(dir_rad)

        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum features"""
        if all(col in df.columns for col in ['player_weight', 's']):
            # Linear momentum (mass * velocity)
            df['momentum'] = df['player_weight'] * df['s']

            # Momentum components
            if all(col in df.columns for col in ['velocity_x', 'velocity_y']):
                df['momentum_x'] = df['player_weight'] * df['velocity_x']
                df['momentum_y'] = df['player_weight'] * df['velocity_y']

        return df

    def _kinetic_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate kinetic energy features"""
        if all(col in df.columns for col in ['player_weight', 's']):
            # KE = 0.5 * m * v^2
            df['kinetic_energy'] = 0.5 * df['player_weight'] * (df['s'] ** 2)

            # Translational kinetic energy (simplified)
            df['kinetic_energy_trans'] = df['kinetic_energy']  # Alias for clarity

        return df

    def _direction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate direction-related features"""
        if all(col in df.columns for col in ['o', 'dir']):
            # Direction difference (orientation vs direction of motion)
            df['dir_diff'] = angular_difference(df['o'], df['dir'])

            # Normalized direction difference
            df['dir_diff_norm'] = df['dir_diff'] / 180.0  # Scale to [-1, 1]

        return df

    def _spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create spatial relationship features"""
        if self.config.verbose > 0:
            self.logger.info("Creating spatial features...")

        # Distance to ball landing location
        df = self._ball_landing_features(df)

        # Field position features
        df = self._field_position_features(df)

        # Sideline distance features
        df = self._sideline_features(df)

        # Goal line distance features
        df = self._goal_line_features(df)

        return df

    def _ball_landing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features related to ball landing location"""
        if all(col in df.columns for col in ['x', 'y', 'ball_land_x', 'ball_land_y']):
            # Distance to ball landing location
            df['dist_to_ball_land'] = np.sqrt(
                (df['x'] - df['ball_land_x'])**2 +
                (df['y'] - df['ball_land_y'])**2
            )

            # Horizontal and vertical distance components
            df['dx_to_ball'] = df['ball_land_x'] - df['x']
            df['dy_to_ball'] = df['ball_land_y'] - df['y']

            # Angle to ball landing location
            df['angle_to_ball'] = np.degrees(
                np.arctan2(df['dy_to_ball'], df['dx_to_ball'])
            )

            # Normalize angle
            df['angle_to_ball_norm'] = normalize_angle(df['angle_to_ball']) / 360.0

            # Alignment with ball trajectory
            if 'dir' in df.columns:
                df['alignment_with_ball'] = angular_difference(
                    df['dir'], df['angle_to_ball']
                )

        return df

    def _field_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate field position features"""
        if 'x' in df.columns:
            # Field position normalized (0-1)
            df['field_position_norm'] = df['x'] / 120.0

            # Field zones
            df['in_red_zone'] = (df['x'] <= 20).astype(int)
            df['in_opponent_territory'] = (df['x'] > 60).astype(int)
            df['at_midfield'] = ((df['x'] >= 55) & (df['x'] <= 65)).astype(int)

        return df

    def _sideline_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance to sidelines"""
        if 'y' in df.columns:
            # Distance from sidelines
            df['dist_from_left_sideline'] = df['y']
            df['dist_from_right_sideline'] = 53.3 - df['y']
            df['dist_to_nearest_sideline'] = np.minimum(
                df['dist_from_left_sideline'],
                df['dist_from_right_sideline']
            )

            # Normalized sideline distance
            df['sideline_dist_norm'] = df['dist_to_nearest_sideline'] / 26.65

        return df

    def _goal_line_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance to goal lines"""
        if 'x' in df.columns:
            # Distance from goal lines
            df['dist_from_own_goal'] = df['x']
            df['dist_from_opp_goal'] = 120 - df['x']

            # Normalized goal line distance
            df['goal_dist_norm'] = df['dist_from_own_goal'] / 120.0

        return df

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features (changes over time)"""
        if self.config.verbose > 0:
            self.logger.info("Creating temporal features...")

        # Group by game, play, and player for temporal calculations
        groupby_cols = ['game_id', 'play_id', 'nfl_id']

        # Position changes (velocity approximation)
        df = self._position_change_features(df, groupby_cols)

        # Speed and acceleration changes
        df = self._motion_change_features(df, groupby_cols)

        # Direction changes
        df = self._direction_change_features(df, groupby_cols)

        # Rolling statistics - disabled for performance
        # df = self._rolling_statistics_features(df, groupby_cols)

        # Time-to-ball features
        df = self._time_to_ball_features(df)

        return df

    def _position_change_features(self, df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
        """Calculate position change features"""
        for col in ['x', 'y']:
            if col in df.columns:
                df[f'{col}_change'] = df.groupby(groupby_cols)[col].diff()
                df[f'{col}_change_abs'] = df[f'{col}_change'].abs()
                df[f'{col}_change_rate'] = df[f'{col}_change'] * 10  # per second (10 fps)

        return df

    def _motion_change_features(self, df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
        """Calculate speed and acceleration change features"""
        for col in ['s', 'a']:
            if col in df.columns:
                df[f'{col}_change'] = df.groupby(groupby_cols)[col].diff()
                df[f'{col}_change_rate'] = df[f'{col}_change'] * 10  # per second

        return df

    def _direction_change_features(self, df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
        """Calculate direction change features"""
        if 'dir' in df.columns:
            df['dir_change'] = df.groupby(groupby_cols)['dir'].diff()

            # Handle wraparound for angles
            df['dir_change'] = df['dir_change'].apply(
                lambda x: x - 360 if x > 180 else (x + 360 if x < -180 else x) if pd.notna(x) else x
            )

            # Direction change rate
            df['dir_change_rate'] = df['dir_change'] * 10  # per second

        return df

    def _rolling_statistics_features(self, df: pd.DataFrame, groupby_cols: List[str]) -> pd.DataFrame:
        """Calculate rolling statistics - OPTIMIZED"""
        window = 2  # Reduced window for performance: 2 frames = 0.2 seconds

        roll_cols = ['s', 'a', 'x_change', 'y_change', 'dir_change']
        stats = ['mean', 'std']  # Reduced stats for performance

        if self.config.verbose > 0:
            available_cols = [c for c in roll_cols if c in df.columns]
            self.logger.info(f"Computing rolling statistics for {len(available_cols)} columns (optimized)...")

        # Vectorized approach - compute all rolling stats at once per group
        for col in roll_cols:
            if col in df.columns:
                # Use transform with rolling - this is more efficient than looping
                grouped = df.groupby(groupby_cols, group_keys=False)[col]

                # Compute rolling window
                rolling = grouped.rolling(window=window, min_periods=1)

                # Calculate multiple stats at once
                df[f'{col}_rolling_mean'] = rolling.mean().values
                df[f'{col}_rolling_std'] = rolling.std().values

        return df

    def _time_to_ball_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-to-ball-landing features"""
        if 'num_frames_output' in df.columns:
            # Time to ball land in seconds
            df['time_to_ball_land'] = df['num_frames_output'] / 10.0

            # Urgency features
            df['time_pressure'] = 1.0 / (df['time_to_ball_land'] + 0.1)  # Avoid division by zero

        return df

    def _role_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create role-specific features"""
        if self.config.verbose > 0:
            self.logger.info("Creating role-based features...")

        # Position group encoding
        df = self._position_group_features(df)

        # Role count features
        df = self._role_count_features(df)

        # Role-specific behavior features
        df = self._role_behavior_features(df)

        return df

    def _position_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode position groups as numeric features"""
        if 'player_position' in df.columns:
            df['position_group'] = df['player_position'].map(
                lambda x: POSITION_GROUPS.get(x, 'OTHER') if pd.notna(x) else 'UNKNOWN'
            )

            # One-hot encode position groups and convert to int
            position_dummies = pd.get_dummies(df['position_group'], prefix='pos', dtype=np.int8)
            df = pd.concat([df, position_dummies], axis=1)

            # Drop the string column
            df = df.drop(columns=['position_group'])

        return df

    def _role_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate role count features - OPTIMIZED for large datasets"""
        if 'player_role' in df.columns:
            # OPTIMIZATION: Use transform instead of merge for much better performance
            # This avoids the expensive merge operation on millions of rows
            df['role_count'] = df.groupby(
                ['game_id', 'play_id', 'frame_id', 'player_role'],
                observed=True  # Suppress FutureWarning
            )['player_role'].transform('size')

        return df

    def _role_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create role-specific behavior features"""
        # Targeted receiver specific features
        if 'is_targeted_receiver' in df.columns:
            # Targeted receivers tend to be faster and more aligned with ball
            pass  # Features calculated in interaction section

        # Defensive coverage specific features
        if 'is_defensive_coverage' in df.columns:
            # Defensive players may have different movement patterns
            pass

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between existing features"""
        if self.config.verbose > 0:
            self.logger.info("Creating interaction features...")

        # Speed × Distance to ball (urgency metric)
        if all(col in df.columns for col in ['s', 'dist_to_ball_land']):
            df['speed_x_dist_to_ball'] = df['s'] * df['dist_to_ball_land']

        # Speed × Time to ball land (expected displacement)
        if all(col in df.columns for col in ['s', 'time_to_ball_land']):
            df['expected_displacement'] = df['s'] * df['time_to_ball_land']

        # Alignment × Speed (effective speed towards ball)
        if all(col in df.columns for col in ['alignment_with_ball', 's']):
            df['effective_speed_to_ball'] = df['s'] * np.cos(np.radians(df['alignment_with_ball']))

        # Role × Distance interactions
        if all(col in df.columns for col in ['is_targeted_receiver', 'dist_to_ball_land']):
            df['targeted_x_dist'] = df['is_targeted_receiver'] * df['dist_to_ball_land']

        # Momentum × Direction alignment
        if all(col in df.columns for col in ['momentum', 'dir_diff_norm']):
            df['momentum_alignment'] = df['momentum'] * (1 - df['dir_diff_norm'])

        # Field position × Speed
        if all(col in df.columns for col in ['field_position_norm', 's']):
            df['field_speed_interaction'] = df['field_position_norm'] * df['s']

        return df

    def _polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features"""
        if self.config.verbose > 0:
            self.logger.info(f"Creating polynomial features (degree={self.config.polynomial_degree})...")

        # Select numeric features for polynomial expansion
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id']  # ID columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(feature_cols) > 0 and self.config.polynomial_degree > 1:
            # Limit to most important features to avoid explosion
            important_features = feature_cols[:min(10, len(feature_cols))]

            poly = PolynomialFeatures(
                degree=self.config.polynomial_degree,
                interaction_only=False,
                include_bias=False
            )

            poly_features = poly.fit_transform(df[important_features])
            poly_feature_names = poly.get_feature_names_out(important_features)

            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(
                poly_features,
                columns=[f'poly_{name}' for name in poly_feature_names],
                index=df.index
            )

            # Concatenate with original dataframe
            df = pd.concat([df, poly_df], axis=1)

            self.polynomial_features = poly

        return df

    def _nfl_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create NFL-specific domain features (routes, coverage, pass context)"""
        if self.config.verbose > 0:
            self.logger.info("Creating NFL-specific domain features...")

        # Route-based features
        df = self._route_detection_features(df)

        # Coverage features
        df = self._coverage_analysis_features(df)

        # Pass context features
        df = self._pass_context_features(df)

        # Player interaction features
        df = self._player_interaction_features(df)

        return df

    def _route_detection_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect route patterns from player movement"""
        if all(col in df.columns for col in ['x', 'y', 's', 'dir']):
            # Route depth (how far downfield from line of scrimmage)
            # Assuming x=0 is line of scrimmage
            df['route_depth'] = df['x'].abs()

            # Lateral movement (horizontal route component)
            df['lateral_movement'] = df['y'] - 26.65  # Distance from field center

            # Route angle (direction of movement)
            if 'dir' in df.columns:
                # Classify route type by direction
                # 0-45 deg = go route, 45-90 = out, 90-135 = comeback, etc.
                df['route_angle_category'] = pd.cut(
                    df['dir'] % 360,
                    bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                    labels=[0, 1, 2, 3, 4, 5, 6, 7]
                ).astype(np.int8)

            # Speed-based route indicators
            if 's' in df.columns:
                df['is_route_break'] = ((df['s'] < 2.0) & (df['route_depth'] > 5)).astype(np.int8)
                df['is_deep_route'] = (df['route_depth'] > 20).astype(np.int8)
                df['is_short_route'] = (df['route_depth'] < 10).astype(np.int8)

        return df

    def _coverage_analysis_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze defensive coverage from player positions"""
        if all(col in df.columns for col in ['x', 'y', 'game_id', 'play_id', 'frame_id']):
            # Group by play and frame to analyze player relationships
            groupby_cols = ['game_id', 'play_id', 'frame_id']

            # Count defenders in vicinity (within 5 yards)
            if 'is_offense' in df.columns:
                # For each offensive player, count nearby defenders
                def count_nearby_defenders(group):
                    offense = group[group['is_offense'] == 1]
                    defense = group[group['is_offense'] == 0]

                    counts = []
                    for _, off_player in offense.iterrows():
                        # Calculate distance to all defenders
                        distances = np.sqrt(
                            (defense['x'] - off_player['x'])**2 +
                            (defense['y'] - off_player['y'])**2
                        )
                        nearby = (distances < 5).sum()
                        counts.append(nearby)

                    # Add 0 for defensive players
                    counts.extend([0] * len(defense))
                    return counts

                # This is computationally expensive, so we'll use a simpler approximation
                # Count total defenders on the same side of the field
                df['defenders_same_side'] = df.groupby(groupby_cols + ['is_offense'])['nfl_id'].transform('count')
                df.loc[df['is_offense'] == 0, 'defenders_same_side'] = 0

            # Coverage tightness indicator
            if 'y' in df.columns:
                df['sideline_pressure'] = (53.3 - df['y'].abs()).clip(0, 10) / 10

        return df

    def _pass_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pass-specific context features"""
        # Time-related features (if frame_id available)
        if 'frame_id' in df.columns:
            # Frames since snap (assuming earlier frames = pre-snap)
            df['frames_elapsed'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].transform(
                lambda x: x - x.min()
            )
            df['time_since_snap'] = df['frames_elapsed'] * 0.1  # 10 fps = 0.1s per frame

        # Ball location features (if available)
        # Note: In real data, ball location might be in a separate column
        # For now, we'll approximate using passer position
        if 'is_passer' in df.columns:
            # Get passer position for each play/frame
            passer_pos = df[df['is_passer'] == 1].groupby(['game_id', 'play_id', 'frame_id'])[['x', 'y']].first()
            passer_pos.columns = ['ball_x', 'ball_y']

            # Merge back
            df = df.merge(passer_pos, on=['game_id', 'play_id', 'frame_id'], how='left')

            # Distance to ball
            if 'ball_x' in df.columns:
                df['distance_to_ball'] = np.sqrt(
                    (df['x'] - df['ball_x'])**2 +
                    (df['y'] - df['ball_y'])**2
                )
                df['angle_to_ball_deg'] = np.degrees(
                    np.arctan2(df['y'] - df['ball_y'], df['x'] - df['ball_x'])
                )

        # Formation indicators
        if all(col in df.columns for col in ['x', 'y', 'is_offense']):
            # Receiver spread (width of offensive formation)
            receiver_spread = df[df['is_offense'] == 1].groupby(['game_id', 'play_id', 'frame_id'])['y'].agg(['min', 'max'])
            receiver_spread['formation_width'] = receiver_spread['max'] - receiver_spread['min']
            df = df.merge(receiver_spread[['formation_width']], on=['game_id', 'play_id', 'frame_id'], how='left')

        return df

    def _player_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features based on player-to-player interactions"""
        if all(col in df.columns for col in ['x', 'y', 'game_id', 'play_id', 'frame_id']):
            # Nearest teammate distance
            groupby_cols = ['game_id', 'play_id', 'frame_id']

            if 'is_offense' in df.columns:
                # For each player, find distance to nearest teammate
                def nearest_teammate_dist(group):
                    distances = []
                    for idx, player in group.iterrows():
                        # Get teammates (same side)
                        teammates = group[(group['is_offense'] == player['is_offense']) & (group.index != idx)]
                        if len(teammates) > 0:
                            team_dist = np.sqrt(
                                (teammates['x'] - player['x'])**2 +
                                (teammates['y'] - player['y'])**2
                            )
                            distances.append(team_dist.min())
                        else:
                            distances.append(np.nan)
                    return distances

                # Simplified version for performance
                df['isolation_score'] = df.groupby(groupby_cols + ['is_offense'])['nfl_id'].transform('count')
                df['isolation_score'] = 1.0 / df['isolation_score']  # Inverse = more isolated

        return df

    def _get_memory_usage(self, df: pd.DataFrame) -> float:
        """Get memory usage in MB"""
        return df.memory_usage(deep=True).sum() / 1024 / 1024

    def get_feature_importance_template(self) -> Dict[str, List[str]]:
        """Get template for organizing features by category"""
        return {
            'basic': ['play_direction_encoded', 'is_offense', 'player_height_inches',
                      'player_age_years'],
            'physics': ['velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
                        'momentum', 'kinetic_energy', 'dir_diff'],
            'spatial': ['dist_to_ball_land', 'angle_to_ball', 'alignment_with_ball',
                        'field_position_norm', 'dist_to_nearest_sideline'],
            'temporal': ['x_change', 'y_change', 's_change', 'dir_change'],
            'role': ['is_targeted_receiver', 'is_passer', 'position_group'],
            'interaction': ['speed_x_dist_to_ball', 'expected_displacement',
                            'effective_speed_to_ball']
        }


class FeatureSelector:
    """Feature selection utilities"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger(config).logger

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'mutual_info') -> List[str]:
        """
        Select most important features

        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('mutual_info', 'f_regression', 'rf_importance')

        Returns:
            List of selected feature names
        """
        if not self.config.feature_selection:
            return X.columns.tolist()

        self.logger.info(f"Selecting features using {method}...")

        if method == 'mutual_info':
            selected = self._select_mutual_info(X, y)
        elif method == 'f_regression':
            selected = self._select_f_regression(X, y)
        elif method == 'rf_importance':
            selected = self._select_rf_importance(X, y)
        else:
            self.logger.warning(f"Unknown selection method: {method}")
            return X.columns.tolist()

        self.logger.info(f"Selected {len(selected)} out of {len(X.columns)} features")
        return selected

    def _select_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using mutual information"""
        from sklearn.feature_selection import SelectKBest, mutual_info_regression

        if self.config.max_features:
            k = min(self.config.max_features, len(X.columns))
        else:
            k = min(50, len(X.columns))  # Default to 50 features

        selector = SelectKBest(mutual_info_regression, k=k)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        return selected_features

    def _select_f_regression(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using F-regression"""
        from sklearn.feature_selection import SelectKBest, f_regression

        if self.config.max_features:
            k = min(self.config.max_features, len(X.columns))
        else:
            k = min(50, len(X.columns))

        selector = SelectKBest(f_regression, k=k)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()

        return selected_features

    def _select_rf_importance(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using random forest importance"""
        from sklearn.ensemble import RandomForestRegressor

        # Train a quick RF model
        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )

        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_

        # Select features above threshold
        if self.config.feature_selection_threshold:
            threshold = self.config.feature_selection_threshold
        else:
            threshold = np.mean(importances)  # Default to mean importance

        selected_mask = importances >= threshold
        selected_features = X.columns[selected_mask].tolist()

        return selected_features


if __name__ == "__main__":
    # Test feature engineering
    from nfl_pipeline.core.config import get_quick_config

    config = get_quick_config()
    engineer = FeatureEngineer(config)

    # Create sample data
    sample_data = pd.DataFrame({
        'game_id': [1, 1, 1],
        'play_id': [1, 1, 1],
        'nfl_id': [1, 1, 1],
        'frame_id': [1, 2, 3],
        'x': [10, 11, 12],
        'y': [20, 21, 22],
        's': [5, 6, 7],
        'a': [1, 2, 3],
        'dir': [90, 95, 100],
        'o': [85, 90, 95],
        'player_weight': [200, 200, 200],
        'ball_land_x': [30, 30, 30],
        'ball_land_y': [25, 25, 25]
    })

    try:
        engineered = engineer.engineer_features(sample_data)
        print("Feature engineering test successful")
        print(f"Original features: {len(sample_data.columns)}")
        print(f"Engineered features: {len(engineered.columns)}")
        print(f"New features: {list(set(engineered.columns) - set(sample_data.columns))[:5]}...")

    except Exception as e:
        print(f"Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()