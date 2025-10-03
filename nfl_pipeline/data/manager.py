"""
Data Management - Handles data flow through raw → processed → features pipeline
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import pickle
import hashlib
from datetime import datetime

from nfl_pipeline.utils.logging import get_logger
from nfl_pipeline.utils.helpers import timer, get_memory_usage


class DataManager:
    """
    Manages data flow through the pipeline stages:
    1. Raw data (original CSV files)
    2. Processed data (cleaned, merged, validated)
    3. Feature data (engineered features ready for modeling)
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.logger = get_logger()

        # Create directory structure
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.features_dir = self.data_dir / 'features'

        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Data Manager initialized:")
        self.logger.info(f"  Raw:       {self.raw_dir}")
        self.logger.info(f"  Processed: {self.processed_dir}")
        self.logger.info(f"  Features:  {self.features_dir}")

    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate hash of dataframe for versioning"""
        info = f"{df.shape}_{list(df.columns)}_{df.memory_usage(deep=True).sum()}"
        return hashlib.md5(info.encode()).hexdigest()[:12]

    def _get_timestamp(self) -> str:
        """Get current timestamp for versioning"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    # ========================================================================
    # RAW DATA MANAGEMENT
    # ========================================================================

    def move_to_raw(self):
        """
        Move existing train/test data to raw/ subfolder.
        Called once during initial setup.
        """
        # Check if train/test folders exist at data root
        train_dir = self.data_dir / 'train'
        test_dir = self.data_dir / 'test'

        if train_dir.exists():
            import shutil
            target = self.raw_dir / 'train'
            if not target.exists():
                shutil.move(str(train_dir), str(target))
                self.logger.info(f"✓ Moved train/ → raw/train/")

        if test_dir.exists():
            import shutil
            target = self.raw_dir / 'test'
            if not target.exists():
                shutil.move(str(test_dir), str(target))
                self.logger.info(f"✓ Moved test/ → raw/test/")

        # Move CSV files
        for csv_file in self.data_dir.glob('*.csv'):
            target = self.raw_dir / csv_file.name
            if not target.exists():
                import shutil
                shutil.move(str(csv_file), str(target))
                self.logger.info(f"✓ Moved {csv_file.name} → raw/")

    def get_raw_train_dir(self) -> Path:
        """Get path to raw training data"""
        return self.raw_dir / 'train'

    def get_raw_test_dir(self) -> Path:
        """Get path to raw test data"""
        return self.raw_dir / 'test'

    # ========================================================================
    # PROCESSED DATA MANAGEMENT
    # ========================================================================

    def save_processed_data(self,
                           input_df: pd.DataFrame,
                           output_df: pd.DataFrame,
                           metadata: Optional[Dict] = None) -> Path:
        """
        Save processed (cleaned, merged) data.

        Args:
            input_df: Processed input dataframe
            output_df: Processed output dataframe
            metadata: Optional metadata about processing

        Returns:
            Path to saved file
        """
        with timer("Saving processed data"):
            timestamp = self._get_timestamp()
            data_hash = self._get_data_hash(input_df)

            filename = f"processed_{timestamp}_{data_hash}.pkl"
            filepath = self.processed_dir / filename

            data = {
                'input_df': input_df,
                'output_df': output_df,
                'metadata': metadata or {},
                'timestamp': timestamp,
                'shape': {'input': input_df.shape, 'output': output_df.shape}
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            # Create symlink to latest
            latest_link = self.processed_dir / 'latest.pkl'
            if latest_link.exists():
                latest_link.unlink()
            try:
                latest_link.symlink_to(filepath.name)
            except (OSError, NotImplementedError):
                pass

            mem_mb = (input_df.memory_usage(deep=True).sum() +
                     output_df.memory_usage(deep=True).sum()) / 1024**2
            self.logger.info(f"✓ Processed data saved: {filepath.name}")
            self.logger.info(f"  Size: {mem_mb:.1f} MB")

            return filepath

    def load_processed_data(self, version: str = 'latest') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load processed data.

        Args:
            version: 'latest' or specific filename

        Returns:
            Tuple of (input_df, output_df, metadata)
        """
        if version == 'latest':
            filepath = self.processed_dir / 'latest.pkl'
        else:
            filepath = self.processed_dir / version

        if not filepath.exists():
            raise FileNotFoundError(f"Processed data not found: {filepath}")

        with timer("Loading processed data"):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.logger.info(f"✓ Loaded processed data: {filepath.name}")
            self.logger.info(f"  Input shape: {data['input_df'].shape}")
            self.logger.info(f"  Output shape: {data['output_df'].shape}")

            return data['input_df'], data['output_df'], data['metadata']

    # ========================================================================
    # FEATURE DATA MANAGEMENT
    # ========================================================================

    def save_features(self,
                     features_df: pd.DataFrame,
                     feature_names: list,
                     split: str = 'full',
                     metadata: Optional[Dict] = None) -> Path:
        """
        Save engineered features.

        Args:
            features_df: DataFrame with engineered features
            feature_names: List of feature column names
            split: 'full', 'train', 'val', or 'test'
            metadata: Optional metadata about features

        Returns:
            Path to saved file
        """
        with timer(f"Saving {split} features"):
            timestamp = self._get_timestamp()
            data_hash = self._get_data_hash(features_df)

            filename = f"features_{split}_{timestamp}_{data_hash}.pkl"
            filepath = self.features_dir / filename

            data = {
                'features': features_df,
                'feature_names': feature_names,
                'split': split,
                'metadata': metadata or {},
                'timestamp': timestamp,
                'shape': features_df.shape
            }

            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

            # Create symlink to latest for this split
            latest_link = self.features_dir / f'latest_{split}.pkl'
            if latest_link.exists():
                latest_link.unlink()
            try:
                latest_link.symlink_to(filepath.name)
            except (OSError, NotImplementedError):
                pass

            mem_mb = features_df.memory_usage(deep=True).sum() / 1024**2
            self.logger.info(f"✓ Features saved: {filepath.name}")
            self.logger.info(f"  Shape: {features_df.shape}")
            self.logger.info(f"  Size: {mem_mb:.1f} MB")

            return filepath

    def load_features(self, split: str = 'full', version: str = 'latest') -> Tuple[pd.DataFrame, list, Dict]:
        """
        Load engineered features.

        Args:
            split: 'full', 'train', 'val', or 'test'
            version: 'latest' or specific filename

        Returns:
            Tuple of (features_df, feature_names, metadata)
        """
        if version == 'latest':
            filepath = self.features_dir / f'latest_{split}.pkl'
        else:
            filepath = self.features_dir / version

        if not filepath.exists():
            raise FileNotFoundError(f"Features not found: {filepath}")

        with timer(f"Loading {split} features"):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.logger.info(f"✓ Loaded features: {filepath.name}")
            self.logger.info(f"  Shape: {data['features'].shape}")
            self.logger.info(f"  # Features: {len(data['feature_names'])}")

            return data['features'], data['feature_names'], data['metadata']

    def save_train_val_features(self,
                                train_data: Dict,
                                val_data: Dict,
                                feature_names: list,
                                metadata: Optional[Dict] = None):
        """
        Save train and validation features together.

        Args:
            train_data: Dict with X_train, y_train_x, y_train_y, metadata
            val_data: Dict with X_val, y_val_x, y_val_y, metadata
            feature_names: List of feature names
            metadata: Optional metadata
        """
        timestamp = self._get_timestamp()

        # Save train features
        train_features = pd.DataFrame(train_data['X_train'], columns=feature_names)
        train_features['target_x'] = train_data['y_train_x']
        train_features['target_y'] = train_data['y_train_y']

        self.save_features(
            train_features,
            feature_names,
            split='train',
            metadata={**metadata, 'timestamp': timestamp} if metadata else {'timestamp': timestamp}
        )

        # Save validation features
        val_features = pd.DataFrame(val_data['X_val'], columns=feature_names)
        val_features['target_x'] = val_data['y_val_x']
        val_features['target_y'] = val_data['y_val_y']

        self.save_features(
            val_features,
            feature_names,
            split='val',
            metadata={**metadata, 'timestamp': timestamp} if metadata else {'timestamp': timestamp}
        )

        self.logger.info(f"✓ Train/Val features saved with timestamp: {timestamp}")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def list_processed_versions(self) -> list:
        """List all processed data versions"""
        files = sorted(self.processed_dir.glob('processed_*.pkl'), reverse=True)
        return [f.name for f in files]

    def list_feature_versions(self, split: str = None) -> list:
        """List all feature versions"""
        if split:
            pattern = f'features_{split}_*.pkl'
        else:
            pattern = 'features_*.pkl'

        files = sorted(self.features_dir.glob(pattern), reverse=True)
        return [f.name for f in files]

    def cleanup_old_versions(self, keep_n: int = 5):
        """
        Clean up old versions, keeping only the N most recent.

        Args:
            keep_n: Number of versions to keep
        """
        # Cleanup processed data
        processed_files = sorted(self.processed_dir.glob('processed_*.pkl'), reverse=True)
        for file in processed_files[keep_n:]:
            file.unlink()
            self.logger.info(f"Deleted old processed data: {file.name}")

        # Cleanup features
        for split in ['full', 'train', 'val', 'test']:
            feature_files = sorted(
                self.features_dir.glob(f'features_{split}_*.pkl'),
                reverse=True
            )
            for file in feature_files[keep_n:]:
                file.unlink()
                self.logger.info(f"Deleted old features: {file.name}")

    def get_data_info(self) -> Dict:
        """Get information about all data stages"""
        info = {
            'raw': {
                'train_files': len(list((self.raw_dir / 'train').glob('*.csv'))) if (self.raw_dir / 'train').exists() else 0,
                'test_files': len(list((self.raw_dir / 'test').glob('*.csv'))) if (self.raw_dir / 'test').exists() else 0,
            },
            'processed': {
                'versions': len(self.list_processed_versions()),
                'latest': 'latest.pkl' if (self.processed_dir / 'latest.pkl').exists() else None
            },
            'features': {
                'full': len(self.list_feature_versions('full')),
                'train': len(self.list_feature_versions('train')),
                'val': len(self.list_feature_versions('val')),
                'test': len(self.list_feature_versions('test')),
            }
        }
        return info

    def print_data_info(self):
        """Print data information"""
        info = self.get_data_info()

        print("\n" + "="*60)
        print("DATA PIPELINE STATUS")
        print("="*60)

        print(f"\n📁 RAW DATA:")
        print(f"  Train files: {info['raw']['train_files']}")
        print(f"  Test files:  {info['raw']['test_files']}")

        print(f"\n🔧 PROCESSED DATA:")
        print(f"  Versions: {info['processed']['versions']}")
        print(f"  Latest:   {info['processed']['latest']}")

        print(f"\n⚡ FEATURE DATA:")
        print(f"  Full:  {info['features']['full']} versions")
        print(f"  Train: {info['features']['train']} versions")
        print(f"  Val:   {info['features']['val']} versions")
        print(f"  Test:  {info['features']['test']} versions")

        print("="*60 + "\n")
