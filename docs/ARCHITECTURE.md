## NFL Pipeline - Professional Architecture Guide

**Version**: 2.1.0
**Last Updated**: 2025-10-03

---

## Directory Structure

```
NFL Competition/
├── nfl_pipeline/              # Main Python package
│   ├── __init__.py            # Package exports
│   ├── core/                  # Core pipeline components
│   │   ├── pipeline.py        # Main orchestrator (NFLPipeline class)
│   │   └── config.py          # Configuration management
│   ├── data/                  # Data handling
│   │   ├── loader.py          # DataLoader class
│   │   └── preprocessor.py    # DataPreparation class
│   ├── features/              # Feature engineering
│   │   └── engineer.py        # FeatureEngineer class
│   ├── models/                # All model implementations
│   │   ├── traditional.py     # ModelTrainer (Ridge, XGBoost, etc.)
│   │   ├── sequence.py        # LSTMTrainer (sequence models)
│   │   └── ensemble.py        # EnsembleBuilder
│   ├── evaluation/            # Model evaluation
│   │   └── evaluator.py       # ModelEvaluator, ModelSelector
│   ├── utils/                 # Utilities
│   │   └── helpers.py         # Logging, timing, helpers
│   └── prediction/            # Prediction generation
│       └── predictor.py       # PredictionGenerator
├── data/                      # Data files (gitignored)
│   ├── train/                 # Training data
│   ├── test/                  # Test data
│   └── external/              # External data sources
├── outputs/                   # All outputs (gitignored)
│   ├── models/                # Saved models by experiment
│   │   ├── 2025_10_03_exp1/  # Timestamped experiments
│   │   └── best/              # Best performing models
│   ├── predictions/           # Prediction files
│   ├── feature_cache/         # Cached features
│   └── figures/               # Plots and visualizations
├── logs/                      # Log files (gitignored)
├── notebooks/                 # Jupyter notebooks
│   ├── experiments.ipynb
│   └── archive/               # Old notebooks
├── tests/                     # Unit tests
│   ├── test_features.py
│   ├── test_models.py
│   └── test_pipeline.py
├── configs/                   # YAML configuration files
│   ├── default.yaml
│   ├── quick.yaml
│   └── lstm.yaml
├── scripts/                   # Utility scripts
│   └── update_imports.py      # Import updater
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # This file
│   ├── PROJECT_STATUS.md      # Current status
│   ├── LSTM_INTEGRATION.md    # LSTM guide
│   └── API.md                 # API documentation
├── train.py                   # Main training script
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
└── README.md                  # Main README
```

---

## Installation

### Option 1: Development Mode (Recommended)
```bash
cd "c:\projects\NFL Competition"
pip install -e .
```

This installs the package in editable mode - changes to code are immediately available.

### Option 2: Regular Installation
```bash
pip install -e ".[lstm]"  # With LSTM support
pip install -e ".[dev]"   # With development tools
```

---

## Usage

### Quick Start
```bash
# Using default configuration
python train.py

# Using LSTM
python train.py --config lstm

# Custom models
python train.py --models ridge xgboost lstm --epochs 30
```

### Python API
```python
from nfl_pipeline import NFLPipeline, get_quick_config

# Load configuration
config = get_quick_config()

# Customize
config.models_to_evaluate = ['xgboost', 'lstm']
config.use_nfl_features = True

# Run pipeline
pipeline = NFLPipeline(config)
results = pipeline.run_pipeline()
```

---

## Module Responsibilities

### Core (`nfl_pipeline/core/`)
- **pipeline.py**: Main orchestrator, coordinates all components
- **config.py**: Configuration management, presets

### Data (`nfl_pipeline/data/`)
- **loader.py**: Load training/test data from disk
- **preprocessor.py**: Data preparation, splitting, scaling

### Features (`nfl_pipeline/features/`)
- **engineer.py**: All feature engineering logic
  - Physics features (velocity, acceleration)
  - Spatial features (field position, distances)
  - Temporal features (changes over time)
  - NFL-specific features (routes, coverage)

### Models (`nfl_pipeline/models/`)
- **traditional.py**: Traditional ML models (Ridge, XGBoost, etc.)
- **sequence.py**: LSTM and sequence models
- **ensemble.py**: Ensemble methods (voting, stacking)

### Evaluation (`nfl_pipeline/evaluation/`)
- **evaluator.py**: Model evaluation, metrics, selection

### Utils (`nfl_pipeline/utils/`)
- **helpers.py**: Logging, timing, memory management

### Prediction (`nfl_pipeline/prediction/`)
- **predictor.py**: Generate predictions for test data

---

## Import Convention

All imports use absolute paths from package root:

```python
# Good ✅
from nfl_pipeline.core.config import PipelineConfig
from nfl_pipeline.models.sequence import LSTMTrainer

# Bad ❌
from config import PipelineConfig
from sequence_models import LSTMTrainer
```

---

## Output Organization

### Models
```
outputs/models/
├── 2025_10_03_142530_experiment1/
│   ├── xgboost_x.joblib
│   ├── xgboost_y.joblib
│   ├── lstm_model.pt
│   └── metadata.json
└── best/
    ├── best_model_x.joblib
    ├── best_model_y.joblib
    └── config.json
```

### Predictions
```
outputs/predictions/
├── 2025_10_03_142530_experiment1/
│   └── submission.csv
└── latest_submission.csv
```

### Feature Cache
```
outputs/feature_cache/
└── a2fbafbc_features.pkl
```

---

## Configuration Management

### Presets
- **get_quick_config()**: Fast testing (3 models, no tuning)
- **get_lstm_config()**: LSTM comparison (XGBoost vs LSTM)
- **get_full_config()**: Comprehensive (all models, full tuning)
- **get_production_config()**: Production ready

### Custom Config
```python
config = PipelineConfig(
    models_to_evaluate=['xgboost', 'lstm'],
    use_nfl_features=True,
    nn_epochs=50,
    hyperparameter_tuning=False
)
```

---

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_features.py

# With coverage
pytest --cov=nfl_pipeline tests/
```

---

## Logging

Logs are organized by experiment:
```
logs/
├── nfl_pipeline_20251003_142530.log
└── latest.log -> nfl_pipeline_20251003_142530.log
```

---

## Best Practices

### 1. Always Use Package Imports
```python
from nfl_pipeline.core.config import PipelineConfig  # ✅
from config import PipelineConfig  # ❌
```

### 2. Use Entry Point Scripts
```bash
python train.py  # ✅
python nfl_pipeline/core/pipeline.py  # ❌
```

### 3. Version Your Experiments
The pipeline automatically timestamps outputs - use descriptive experiment names in config.

### 4. Cache Features
Enable feature caching to speed up iterations:
```python
config.use_cache = True  # Default
```

---

## Migration from Old Structure

If you have old code importing from root:
```bash
python scripts/update_imports.py
```

This automatically updates all imports to new package structure.

---

## Performance Tips

1. **Use Feature Caching**: Saves 30-40% of pipeline time
2. **Disable Tuning for Testing**: `hyperparameter_tuning=False`
3. **Reduce Models**: Start with `['ridge', 'xgboost']`
4. **LSTM Epochs**: Use 20-30 for testing, 50+ for production

---

## Troubleshooting

### ImportError after restructuring
```bash
# Reinstall package
pip install -e .
```

### Can't find modules
```python
# Make sure you're importing from nfl_pipeline
from nfl_pipeline import NFLPipeline  # ✅
from main import NFLPipeline  # ❌
```

### LSTM not found
```bash
# Install LSTM extras
pip install -e ".[lstm]"
```

---

## Contributing

1. Follow package import conventions
2. Add tests for new features
3. Update documentation
4. Use type hints
5. Follow PEP 8 style guide

---

**The architecture is now production-ready and follows Python best practices! 🚀**
