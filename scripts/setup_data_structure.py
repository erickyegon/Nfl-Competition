"""
One-time script to organize existing data into raw/processed/features structure
"""

from pathlib import Path
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_pipeline.data.manager import DataManager
from nfl_pipeline.utils.logging import get_logger


def setup_data_structure(data_dir: Path):
    """
    Organize existing data into proper structure:
    - Move train/test folders to raw/
    - Move CSV files to raw/
    - Create processed/ and features/ directories
    """
    print("\n" + "="*60)
    print("SETTING UP DATA STRUCTURE")
    print("="*60 + "\n")

    # Initialize DataManager (creates directories)
    dm = DataManager(data_dir)

    # Move existing data to raw/
    print("\n📦 Moving existing data to raw/...")
    dm.move_to_raw()

    # Verify structure
    print("\n✓ Data structure created!")
    dm.print_data_info()

    # Create README in each folder
    create_readme_files(dm)

    print("="*60)
    print("SETUP COMPLETE!")
    print("="*60 + "\n")

    print("Your data is now organized as:")
    print(f"  📁 {data_dir}/")
    print(f"     ├── raw/              # Original CSV files")
    print(f"     │   ├── train/")
    print(f"     │   ├── test/")
    print(f"     │   └── *.csv")
    print(f"     ├── processed/        # Cleaned, merged data")
    print(f"     └── features/         # Engineered features")
    print()


def create_readme_files(dm: DataManager):
    """Create README files in each data folder"""

    # Raw data README
    raw_readme = dm.raw_dir / 'README.md'
    raw_readme.write_text("""# Raw Data

This folder contains the original, unmodified data files.

## Contents:
- `train/` - Training data CSVs (input_*.csv, output_*.csv)
- `test/` - Test data CSVs
- `*.csv` - Additional raw data files

## Important:
- **NEVER modify files in this folder**
- This is the source of truth for all data
- All processing starts from here
""")

    # Processed data README
    processed_readme = dm.processed_dir / 'README.md'
    processed_readme.write_text("""# Processed Data

This folder contains cleaned and merged data, ready for feature engineering.

## File Format:
- Files: `processed_{timestamp}_{hash}.pkl`
- Contains: `input_df`, `output_df`, `metadata`

## Usage:
```python
from nfl_pipeline.data.manager import DataManager

dm = DataManager('data/')
input_df, output_df, metadata = dm.load_processed_data('latest')
```

## Versioning:
- Each processing run creates a new version
- `latest.pkl` symlinks to most recent version
- Old versions are auto-cleaned (keeps 5 most recent)
""")

    # Features README
    features_readme = dm.features_dir / 'README.md'
    features_readme.write_text("""# Feature Data

This folder contains engineered features, ready for model training.

## File Format:
- Files: `features_{split}_{timestamp}_{hash}.pkl`
- Splits: `full`, `train`, `val`, `test`
- Contains: `features_df`, `feature_names`, `metadata`

## Usage:
```python
from nfl_pipeline.data.manager import DataManager

dm = DataManager('data/')

# Load training features
features_df, feature_names, metadata = dm.load_features('train', 'latest')

# Load validation features
features_df, feature_names, metadata = dm.load_features('val', 'latest')
```

## Versioning:
- Each feature engineering run creates new versions
- `latest_{split}.pkl` symlinks to most recent version
- Old versions are auto-cleaned (keeps 5 most recent)

## Feature Pipeline:
1. Raw data → Processed data (cleaned, merged)
2. Processed data → Features (engineered, split)
3. Features → Models (training, prediction)
""")

    print("\n📝 README files created in each data folder")


if __name__ == "__main__":
    # Default data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    setup_data_structure(data_dir)
