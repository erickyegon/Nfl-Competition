# Data Pipeline Architecture

## 📁 Data Folder Structure

```
data/
├── raw/                    # 📦 Original, unmodified data
│   ├── train/              # Training CSV files
│   │   ├── input_2023_w01.csv
│   │   ├── output_2023_w01.csv
│   │   └── ...
│   ├── test/               # Test CSV files
│   │   ├── input_2024_w01.csv
│   │   └── ...
│   └── *.csv               # Additional raw files
│
├── processed/              # 🔧 Cleaned & merged data
│   ├── processed_{timestamp}_{hash}.pkl
│   ├── latest.pkl → processed_20251003_142030_a1b2c3.pkl
│   └── README.md
│
└── features/               # ⚡ Engineered features
    ├── features_full_{timestamp}_{hash}.pkl
    ├── features_train_{timestamp}_{hash}.pkl
    ├── features_val_{timestamp}_{hash}.pkl
    ├── latest_train.pkl → features_train_20251003_142045_d4e5f6.pkl
    ├── latest_val.pkl → features_val_20251003_142045_d4e5f6.pkl
    └── README.md
```

---

## 🔄 Data Flow Pipeline

```
┌─────────────┐
│  RAW DATA   │  Original CSV files (never modified)
└──────┬──────┘
       │
       │ 1. Load & Merge
       ↓
┌─────────────────┐
│ PROCESSED DATA  │  Cleaned, merged, validated
└────────┬────────┘
         │
         │ 2. Feature Engineering
         ↓
┌──────────────────┐
│  FEATURE DATA    │  Engineered features (train/val/test splits)
└────────┬─────────┘
         │
         │ 3. Model Training
         ↓
┌─────────────┐
│   MODELS    │  Trained models
└─────────────┘
```

---

## 🚀 One-Time Setup

**First time only** - Organize existing data:

```bash
# Run setup script to move data to raw/
python scripts/setup_data_structure.py
```

This will:
✅ Create `raw/`, `processed/`, `features/` directories
✅ Move existing `train/` and `test/` folders to `raw/`
✅ Move CSV files to `raw/`
✅ Create README files in each folder

---

## 📖 Usage Examples

### 1. Data Manager - Core Interface

```python
from nfl_pipeline.data.manager import DataManager

# Initialize
dm = DataManager('data/')

# Check status
dm.print_data_info()
```

### 2. Save/Load Processed Data

```python
# After loading and cleaning raw data
dm.save_processed_data(
    input_df=cleaned_input,
    output_df=cleaned_output,
    metadata={'version': '1.0', 'cleaning_steps': ['removed_nulls', 'fixed_dtypes']}
)

# Load processed data
input_df, output_df, metadata = dm.load_processed_data('latest')
```

### 3. Save/Load Features

```python
# After feature engineering
dm.save_features(
    features_df=engineered_features,
    feature_names=['x', 'y', 's', 'a', 'v_x', 'v_y', ...],
    split='train',
    metadata={'num_features': 120, 'feature_groups': ['physics', 'spatial', 'nfl_domain']}
)

# Load features
features_df, feature_names, metadata = dm.load_features('train', 'latest')
```

### 4. Complete Pipeline Example

```python
from nfl_pipeline.data.manager import DataManager
from nfl_pipeline.data.loader import DataLoader
from nfl_pipeline.data.preprocessor import DataPreprocessor
from nfl_pipeline.features.nfl_domain import FeatureEngineer

# 1. Load raw data
dm = DataManager('data/')
loader = DataLoader(config)
input_df, output_df = loader.load_data(dm.get_raw_train_dir())

# 2. Process and save
preprocessor = DataPreprocessor(config)
clean_input, clean_output = preprocessor.clean_data(input_df, output_df)
dm.save_processed_data(clean_input, clean_output)

# 3. Engineer features and save
engineer = FeatureEngineer(config)
features = engineer.engineer_features(clean_input, include_temporal=False)
dm.save_features(features, engineer.feature_names, split='full')

# 4. Later: Load features for training
features_df, feature_names, _ = dm.load_features('train', 'latest')
```

---

## 🔑 Key Benefits

### ✅ Separation of Concerns
- **Raw**: Never modified, source of truth
- **Processed**: Reusable cleaned data
- **Features**: Versioned feature sets

### ✅ Reproducibility
- Each stage saves metadata
- Timestamp + hash versioning
- Easy to track data lineage

### ✅ Efficiency
- Skip data loading if processed exists
- Skip feature engineering if features exist
- Reuse processed data for different feature sets

### ✅ Experimentation
- Compare different feature sets
- A/B test preprocessing steps
- Keep multiple versions

### ✅ Production Ready
- Clear data flow
- Proper versioning
- Easy rollback

---

## 📊 Data Versioning

### File Naming Convention

```
processed_{timestamp}_{hash}.pkl
         ↓           ↓
    20251003_142030_a1b2c3.pkl
```

- **Timestamp**: When created (YYYYMMDD_HHMMSS)
- **Hash**: Data fingerprint (first 12 chars of MD5)

### Symlinks

```
latest.pkl → processed_20251003_142030_a1b2c3.pkl
```

Always points to most recent version.

### Auto-Cleanup

```python
# Keep only 5 most recent versions
dm.cleanup_old_versions(keep_n=5)
```

Automatically removes old versions to save space.

---

## 🛠️ Data Manager API

### Initialization
```python
dm = DataManager('data/')
```

### Raw Data
```python
dm.get_raw_train_dir()  # Path to raw/train/
dm.get_raw_test_dir()   # Path to raw/test/
dm.move_to_raw()        # One-time: organize existing data
```

### Processed Data
```python
dm.save_processed_data(input_df, output_df, metadata)
dm.load_processed_data('latest')  # or specific version
dm.list_processed_versions()
```

### Feature Data
```python
dm.save_features(features_df, feature_names, split='train', metadata)
dm.load_features('train', 'latest')  # or specific version
dm.list_feature_versions('train')
dm.save_train_val_features(train_data, val_data, feature_names)
```

### Utilities
```python
dm.get_data_info()           # Dict with status
dm.print_data_info()         # Pretty print status
dm.cleanup_old_versions(5)   # Keep 5 most recent
```

---

## 📝 Metadata Examples

### Processed Data Metadata
```python
{
    'version': '1.0',
    'cleaning_steps': ['removed_nulls', 'fixed_dtypes', 'merged_files'],
    'num_samples': 563829,
    'date_range': '2023-W01 to 2023-W18',
    'outliers_removed': 2847
}
```

### Feature Data Metadata
```python
{
    'num_features': 120,
    'feature_groups': {
        'physics': 15,
        'spatial': 18,
        'temporal': 35,
        'nfl_domain': 22,
        'basic': 30
    },
    'include_temporal': True,
    'engineered_from': 'processed_20251003_142030_a1b2c3.pkl'
}
```

---

## 🔍 Troubleshooting

### Problem: Data not found
```python
# Check data status
dm.print_data_info()

# List available versions
print(dm.list_processed_versions())
print(dm.list_feature_versions('train'))
```

### Problem: Out of disk space
```python
# Clean up old versions
dm.cleanup_old_versions(keep_n=3)  # Keep only 3 most recent
```

### Problem: Need to start fresh
```bash
# Delete processed and features (keeps raw data safe)
rm -rf data/processed/*
rm -rf data/features/*
```

---

## 🎯 Best Practices

1. **Never modify raw data** - It's the source of truth
2. **Always save metadata** - Document what you did
3. **Use versions** - Don't overwrite, create new versions
4. **Clean up regularly** - Remove old versions to save space
5. **Document changes** - Add notes in metadata
6. **Test on small data** - Use `max_files=1` in config for quick tests

---

## 📚 Integration with Pipeline

The pipeline automatically uses DataManager:

```python
# In pipeline.py
self.data_manager = DataManager(self.config.data_dir)

# Stage 1: Load or use cached processed data
if processed_exists:
    input_df, output_df, _ = self.data_manager.load_processed_data('latest')
else:
    input_df, output_df = self._load_raw_data()
    self.data_manager.save_processed_data(input_df, output_df)

# Stage 2: Load or create features
if features_exist:
    features, names, _ = self.data_manager.load_features('train', 'latest')
else:
    features = self._engineer_features(input_df)
    self.data_manager.save_features(features, names, 'train')
```

---

**Ready to use!** Run `python scripts/setup_data_structure.py` to get started. 🚀
