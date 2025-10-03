# NFL Big Data Bowl 2026 - Submission Guide

## Competition Overview

**Goal:** Predict NFL player movement during video frames after the ball is thrown.

**Key Details:**
- **Frame Rate:** 10 frames per second
- **Data:** Input = player tracking BEFORE pass, Target = positions DURING pass
- **Evaluation Metric:** Root Mean Squared Error (RMSE)
- **Format:** CSV file with id, x, y columns

---

## Submission File Format

### Required Format
```csv
id,x,y
1_1_1_1,45.2,26.3
1_1_1_2,46.1,25.9
1_1_1_3,47.0,25.5
```

### ID Format
The `id` column must follow this exact format:
```
{game_id}_{play_id}_{nfl_id}_{frame_id}
```

**Example:** `2022090800_97_44539_5`
- `game_id`: 2022090800 (Game identifier)
- `play_id`: 97 (Play within the game)
- `nfl_id`: 44539 (Player identifier)
- `frame_id`: 5 (Frame number during the pass)

### Columns
1. **id** (string): Unique identifier as described above
2. **x** (float): Predicted X-coordinate (yards, 0-120)
3. **y** (float): Predicted Y-coordinate (yards, 0-53.3)

---

## Generating Submission

### Method 1: Using the Complete Pipeline (Recommended)

```python
from config import get_quick_config
from main import NFLPipeline

# Configure pipeline
config = get_quick_config()
config.experiment_name = "my_submission"

# Run complete pipeline
pipeline = NFLPipeline(config)
results = pipeline.run_pipeline()

# Submission file automatically generated if test data exists
# Location: ./outputs/my_submission_submission.csv
```

### Method 2: Using Jupyter Notebook

Open `experiments_improved.ipynb` and run all cells. The notebook will:
1. Load and explore data
2. Train models
3. Generate predictions
4. Create submission file

### Method 3: Manual Submission Generation

```python
from prediction import PredictionGenerator
from config import get_quick_config

config = get_quick_config()
predictor = PredictionGenerator(config)

# Assuming you have trained models
submission = predictor.generate_predictions(
    selected_model=best_model,
    model_results=model_results,
    ensemble_results=ensemble_results,
    feature_engineer=feature_engineer,
    data_prep=data_prep
)

# Save
submission.to_csv('submission.csv', index=False)
```

---

## Validating Your Submission

### Automated Validation

The pipeline includes automatic validation:

```python
from prediction import PredictionValidator

validator = PredictionValidator(config)
issues = validator.validate_predictions(predictions, test_meta)

if issues:
    print("⚠️ Validation Issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ Submission validated successfully!")
```

### Manual Validation Checklist

✅ **File Format**
- [ ] File is CSV format
- [ ] Has header row: `id,x,y`
- [ ] No index column

✅ **ID Column**
- [ ] Format: `{game_id}_{play_id}_{nfl_id}_{frame_id}`
- [ ] All IDs from test.csv are present
- [ ] No duplicate IDs
- [ ] String type (not numeric)

✅ **X Column**
- [ ] Numeric (float) values
- [ ] No NaN or infinite values
- [ ] Values in reasonable range (0-120 yards)
- [ ] Not all identical (indicates model issue)

✅ **Y Column**
- [ ] Numeric (float) values
- [ ] No NaN or infinite values
- [ ] Values in reasonable range (0-53.3 yards)
- [ ] Not all identical (indicates model issue)

✅ **Row Count**
- [ ] Matches number of rows in test.csv
- [ ] Each test ID has exactly one prediction

---

## Common Issues & Solutions

### Issue 1: Missing Test Data
**Problem:** No test_input.csv or test.csv files
**Solution:**
```bash
# Ensure files are in correct location
./nfl/data/test_input.csv
./nfl/data/test.csv
```

### Issue 2: ID Format Incorrect
**Problem:** IDs don't match required format
**Solution:**
```python
# The pipeline automatically creates correct IDs from test.csv
# Ensure test.csv has 'id' column in correct format
test_df = pd.read_csv('nfl/data/test.csv')
print(test_df['id'].head())  # Should be: game_play_player_frame
```

### Issue 3: NaN Predictions
**Problem:** Model produces NaN values
**Solution:**
```python
# Check for missing features
print(f"Missing values in features: {X_test.isna().sum().sum()}")

# Fill NaN predictions with sensible defaults
submission['x'] = submission['x'].fillna(submission['x'].median())
submission['y'] = submission['y'].fillna(submission['y'].median())
```

### Issue 4: Out of Bounds Predictions
**Problem:** X or Y coordinates outside field dimensions
**Solution:**
```python
# Clip predictions to field bounds
submission['x'] = submission['x'].clip(0, 120)
submission['y'] = submission['y'].clip(0, 53.3)
```

### Issue 5: Wrong Number of Predictions
**Problem:** Submission has different row count than test.csv
**Solution:**
```python
# Ensure all test IDs are included
test_df = pd.read_csv('nfl/data/test.csv')
submission = submission.merge(test_df[['id']], on='id', how='right')
```

---

## Example: Complete Submission Workflow

```python
# 1. Import libraries
from config import get_quick_config
from main import NFLPipeline
import pandas as pd

# 2. Configure
config = get_quick_config()
config.experiment_name = "final_submission"
config.models_to_evaluate = ['ridge', 'random_forest', 'xgboost']

# 3. Run pipeline
print("Running pipeline...")
pipeline = NFLPipeline(config)
results = pipeline.run_pipeline()

# 4. Check submission
submission_path = config.output_dir / f"{config.experiment_name}_submission.csv"
submission = pd.read_csv(submission_path)

print(f"\n✓ Submission created!")
print(f"  Location: {submission_path}")
print(f"  Predictions: {len(submission)}")
print(f"  Best Model: {results['selected_model']['name']}")
print(f"  RMSE: {results['selected_model']['rmse']:.6f}")

# 5. Validate
print(f"\nFirst 5 predictions:")
print(submission.head())

print(f"\nSample IDs:")
for id_val in submission['id'].head(3):
    parts = str(id_val).split('_')
    print(f"  {id_val}")
    print(f"    Game: {parts[0]}, Play: {parts[1]}, Player: {parts[2]}, Frame: {parts[3]}")

# 6. Basic validation
assert len(submission) == len(pd.read_csv('nfl/data/test.csv')), "Row count mismatch!"
assert list(submission.columns) == ['id', 'x', 'y'], "Column mismatch!"
assert submission['x'].between(0, 120).all(), "X out of bounds!"
assert submission['y'].between(0, 53.3).all(), "Y out of bounds!"

print("\n✅ Validation passed! Ready to submit.")
```

---

## Field Dimensions Reference

**Football Field:**
- **Length (X):** 0 to 120 yards
  - 0 = Own end zone
  - 10-110 = Playing field
  - 120 = Opponent end zone

- **Width (Y):** 0 to 53.3 yards
  - 0 = Left sideline
  - 26.65 = Center
  - 53.3 = Right sideline

**Typical Player Movement:**
- Pass catchers: Move toward ball landing location
- Defenders: Move toward ball or receiver
- Frame rate: 10 fps (0.1 seconds per frame)

---

## Submission Best Practices

### 1. Model Validation
```python
# Always validate on local CV before submission
print(f"Local CV RMSE: {results['selected_model']['rmse']:.6f}")
```

### 2. Ensemble Methods
```python
# Ensemble often improves performance
config.use_ensemble = True
config.use_stacking = True
```

### 3. Feature Engineering
```python
# Enable all feature types for best results
config.use_physics_features = True
config.use_temporal_features = True
config.use_spatial_features = True
config.use_role_features = True
```

### 4. Hyperparameter Tuning
```python
# Tune for better performance
config.hyperparameter_tuning = True
config.n_iter = 50  # More iterations = better tuning
```

### 5. Caching for Speed
```python
# Use caching for faster iterations
# Features are cached automatically
# Delete cache if you change feature engineering:
import shutil
shutil.rmtree('./outputs/feature_cache')
```

---

## Evaluation Metric

**Root Mean Squared Error (RMSE):**

```
RMSE = sqrt(mean((predicted_x - actual_x)² + (predicted_y - actual_y)²))
```

**Lower is better!**

Typical RMSE values:
- **< 2.0:** Excellent
- **2.0 - 3.0:** Good
- **3.0 - 5.0:** Average
- **> 5.0:** Needs improvement

---

## Leaderboard Notes

- **Public Leaderboard:** Based on subset of test data
- **Private Leaderboard:** Based on remaining test data (final ranking)
- **Overfitting:** Optimize for validation set, not training set
- **Ensemble:** Often helps reduce overfitting

---

## Final Checklist

Before submitting:

✅ Run complete pipeline successfully
✅ Check submission file exists
✅ Validate file format (CSV with id,x,y)
✅ Validate ID format (game_play_player_frame)
✅ Check for NaN/infinite values
✅ Verify row count matches test.csv
✅ Confirm predictions in valid ranges
✅ Review local CV score
✅ Test submission file can be read
✅ Check file size is reasonable

---

## Getting Help

If you encounter issues:

1. **Check logs:** `./logs/{experiment_name}.log`
2. **Review notebook:** `experiments_improved.ipynb`
3. **Validate submission:** Use PredictionValidator
4. **Check documentation:**
   - [README.md](README.md)
   - [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)

---

## Quick Command Reference

```bash
# Run pipeline
python main.py --config quick --experiment-name my_submission

# Check submission
python -c "import pandas as pd; df = pd.read_csv('outputs/submission.csv'); print(df.head()); print(f'Rows: {len(df)}')"

# Validate submission
python -c "from prediction import PredictionValidator; from config import get_quick_config; validator = PredictionValidator(get_quick_config()); # Add validation logic"
```

---

**Good luck with your submission! 🏈🚀**
