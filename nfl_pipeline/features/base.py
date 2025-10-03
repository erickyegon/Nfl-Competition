"""
Base Feature Engineer - Foundation for all feature engineering modules
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import pickle
import hashlib

from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.utils.logging import get_logger
from nfl_pipeline.utils.helpers import timer


class BaseFeatureEngineer:
    """
    Base class for all feature engineering modules.

    Provides common functionality for caching, logging, and feature tracking.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_logger()
        self.feature_names = []
        self.cache_dir = config.output_dir / 'feature_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = True

    def _get_cache_key(self, input_df: pd.DataFrame, feature_type: str) -> str:
        """Generate cache key based on data hash and feature type"""
        data_info = f"{input_df.shape}_{list(input_df.columns)}_{feature_type}_{self.config.__dict__}"
        return hashlib.md5(data_info.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load features from cache if available"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                self.logger.info(f"Loading features from cache: {cache_key[:8]}...")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    return cached_data['features']
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, features: pd.DataFrame):
        """Save features to cache"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({'features': features}, f)
            self.logger.info(f"Features saved to cache: {cache_key[:8]}")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def _basic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic transformations to common features"""
        # Player height parsing
        if 'player_height' in df.columns:
            from nfl_pipeline.utils.helpers import height_to_inches
            df['player_height_inches'] = df['player_height'].apply(height_to_inches).astype(np.float32)
            df = df.drop(columns=['player_height'])

        # Normalize angles to [0, 360)
        for col in ['dir', 'o']:
            if col in df.columns:
                from nfl_pipeline.utils.helpers import normalize_angle
                df[col] = df[col].apply(normalize_angle).astype(np.float32)

        return df

    def _encode_categorical(self, df: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
        """Encode categorical column with proper dtype"""
        if column in df.columns:
            dummies = pd.get_dummies(df[column], prefix=prefix, dtype=np.int8)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[column])
        return df

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (to be implemented by subclasses)"""
        return {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement transform()")
