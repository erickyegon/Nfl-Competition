# NFL Big Data Bowl 2026 - Modular ML Pipeline

A comprehensive, modular machine learning pipeline for predicting NFL player movement during pass plays. Built with best practices in mind, featuring extensive configurability, robust error handling, and production-ready code.

## 🏗️ Architecture

This pipeline is built with a modular architecture where each component has a single responsibility:

```
nfl-ml-pipeline/
├── config.py              # Configuration management
├── utils.py               # Utility functions and logging
├── data_loader.py         # Data loading and validation
├── feature_engineering.py # Feature creation and engineering
├── data_preparation.py    # Data preprocessing and splitting
├── models.py              # Model definitions and training
├── ensemble.py            # Ensemble methods
├── evaluation.py          # Model evaluation and selection
├── prediction.py          # Test prediction and submission
├── main.py                # Pipeline orchestrator
├── experiments.ipynb      # Jupyter notebook for experimentation
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install core dependencies manually
pip install numpy pandas scikit-learn matplotlib seaborn

# Optional: advanced models
pip install xgboost lightgbm catboost

# Optional: neural networks
pip install torch torchvision
```

### Basic Usage

```python
from config import get_quick_config
from main import NFLPipeline

# Get quick configuration
config = get_quick_config()

# Run complete pipeline
pipeline = NFLPipeline(config)
results = pipeline.run_pipeline()

print(f"Best model: {results['selected_model']['name']}")
print(f"Best RMSE: {results['selected_model']['rmse']:.4f}")
```

### Command Line

```bash
# Quick configuration
python main.py --config quick

# Full configuration
python main.py --config full

# Production configuration
python main.py --config production

# Custom experiment name
python main.py --config quick --experiment-name my_experiment
```

## 📊 Pipeline Components

### 1. Configuration (`config.py`)
- Centralized configuration management
- Preset configurations (quick, full, production)
- Validation and type safety
- JSON serialization support

### 2. Data Loading (`data_loader.py`)
- CSV file loading with validation
- Memory optimization
- Data integrity checks
- Support for multiple weeks/files

### 3. Feature Engineering (`feature_engineering.py`)
- Physics-based features (velocity, acceleration, momentum)
- Spatial features (distances, angles, field positions)
- Temporal features (changes over time, rolling statistics)
- Role-based features (player positions, team roles)

### 4. Data Preparation (`data_preparation.py`)
- Train/validation splitting (time-series aware)
- Feature scaling (Robust, Standard, MinMax)
- Outlier handling (IQR, Z-score, Isolation Forest)
- Missing value imputation

### 5. Models (`models.py`)
- Multiple algorithm support (Linear, Tree-based, SVM, KNN)
- Advanced models (XGBoost, LightGBM, CatBoost)
- Neural networks (PyTorch-based)
- Hyperparameter tuning (Grid, Random, Bayesian)

### 6. Ensembles (`ensemble.py`)
- Simple averaging
- Weighted averaging (performance-based)
- Stacking with meta-learners
- Blending with holdout sets

### 7. Evaluation (`evaluation.py`)
- Comprehensive metrics (RMSE, MAE, R², MAPE)
- Model comparison and ranking
- Statistical significance testing
- Visualization and reporting

### 8. Prediction (`prediction.py`)
- Test data processing
- Batch prediction generation
- Submission file creation
- Prediction validation

## 🎯 Key Features

### Configurable Pipeline
- **Feature Engineering**: Enable/disable physics, spatial, temporal, role features
- **Models**: Choose from 10+ algorithms including advanced models
- **Ensembles**: Multiple ensemble strategies
- **Cross-Validation**: Time-series, K-fold, stratified splitting
- **Scaling**: Robust, Standard, MinMax scalers
- **Tuning**: Grid search, random search, hyperparameter optimization

### Production Ready
- **Error Handling**: Comprehensive exception handling and logging
- **Memory Management**: Automatic garbage collection and memory monitoring
- **Validation**: Data integrity checks and schema validation
- **Logging**: Structured logging with file and console output
- **Artifacts**: Automatic saving of models, configs, and results

### Experimentation Friendly
- **Jupyter Notebook**: Interactive experimentation environment
- **Modular Design**: Test individual components
- **Configuration Presets**: Quick start with optimized settings
- **Visualization**: Built-in plotting and analysis tools

## 📈 Performance & Results

The pipeline has been designed to achieve competitive performance on NFL player movement prediction:

- **Baseline Models**: Linear models provide solid baselines
- **Tree Models**: Random Forest, XGBoost typically perform best
- **Ensembles**: Often improve by 5-15% over best base model
- **Feature Engineering**: Physics + spatial + temporal features crucial

## 🔧 Configuration Options

### Quick Config (Default)
```python
config = get_quick_config()
# - Ridge, Random Forest, XGBoost
# - Basic feature engineering
# - 3-fold CV, 20 tuning iterations
# - Fast execution for experimentation
```

### Full Config
```python
config = get_full_config()
# - All available models
# - Complete feature engineering
# - 5-fold CV, 50 tuning iterations
# - All ensembles enabled
# - Comprehensive evaluation
```

### Production Config
```python
config = get_production_config()
# - Best performing models
# - Optimized for production use
# - GPU acceleration
# - Maximum tuning iterations
# - Minimal logging
```

## 📁 Directory Structure

After running the pipeline:

```
outputs/
├── experiment_name/
│   ├── models/           # Saved model artifacts
│   ├── logs/            # Log files
│   ├── plots/           # Evaluation plots
│   ├── submission.csv   # Test predictions
│   └── results.json     # Performance summary
└── pipeline_config.json # Configuration backup
```

## 🧪 Experimentation

Use the provided Jupyter notebook (`experiments.ipynb`) for:

1. **Interactive Development**: Test individual components
2. **Parameter Tuning**: Experiment with different configurations
3. **Visualization**: Analyze model performance and features
4. **Debugging**: Step-through pipeline execution

### Example Experimentation Workflow

```python
# Load and explore data
data_loader = DataLoader(config)
input_df, output_df = data_loader.load_training_data()

# Test feature engineering
feature_engineer = FeatureEngineer(config)
features = feature_engineer.engineer_features(input_df)

# Train a single model
trainer = ModelTrainer(config)
result = trainer.train_model('xgboost', X_train, y_train, X_val, y_val)

# Compare multiple models
evaluator = ModelEvaluator(config)
comparison = evaluator.evaluate_all_models(model_results, ensemble_results, data)
```

## 🤝 Contributing

This is a modular, well-documented codebase designed for:

- **ML Researchers**: Easy experimentation with different approaches
- **Data Scientists**: Production-ready pipeline components
- **Engineers**: Scalable and maintainable code structure

### Adding New Components

1. **Models**: Add to `ModelFactory.get_model()` and `get_hyperparameter_grid()`
2. **Features**: Extend `FeatureEngineer` with new feature creation methods
3. **Ensembles**: Implement new ensemble strategies in `EnsembleBuilder`
4. **Metrics**: Add custom metrics to `evaluation.py`

## 📄 License

This project is provided as-is for educational and research purposes in the NFL Big Data Bowl competition.

## 🙏 Acknowledgments

- NFL for providing the tracking data
- Scikit-learn, XGBoost, LightGBM, CatBoost communities
- PyTorch for neural network support

---

**Happy modeling! 🏈🤖**