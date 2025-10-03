# NFL Pipeline - Quick Start Guide

## Installation

```bash
cd "c:\projects\NFL Competition"
pip install -e .
```

## Run Your First Model

```bash
# Basic training (Ridge + XGBoost)
python train.py

# With LSTM
python train.py --config lstm
```

## That's It! ✅

The pipeline will:
1. Load data from `./data/train/`
2. Engineer 90+ features
3. Train models
4. Evaluate and select best model
5. Save results to `./outputs/`

## Results

Check:
- **Models**: `outputs/models/latest/`
- **Logs**: `logs/latest.log`
- **Predictions**: `outputs/predictions/`

## Customize

```python
from nfl_pipeline import NFLPipeline, get_quick_config

config = get_quick_config()
config.models_to_evaluate = ['xgboost', 'lstm']
config.nn_epochs = 30

pipeline = NFLPipeline(config)
results = pipeline.run_pipeline()
```

## Expected Performance

| Model | RMSE | Time |
|-------|------|------|
| Ridge | 2.5-2.6 | 2 min |
| XGBoost | 2.3-2.4 | 5 min |
| LSTM | 1.5-1.8 | 15 min |

## Need Help?

- **Architecture**: See `docs/ARCHITECTURE.md`
- **Full Status**: See `docs/PROJECT_STATUS.md`
- **LSTM Guide**: See `docs/LSTM_INTEGRATION_COMPLETE.md`

---

**For most users, just run `python train.py` and you're done!** 🎉
