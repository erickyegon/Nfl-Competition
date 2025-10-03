# 📓 NFL Player Movement Prediction - Jupyter Notebooks

**Self-contained, well-documented notebooks for the complete ML pipeline**

---

## 🎯 Quick Start

### 1. Run the Complete Pipeline (Recommended)
```bash
jupyter notebook 01_end_to_end_pipeline.ipynb
```
**Time:** ~5-10 minutes (with sample data)
**What it does:** Complete ML workflow from data loading to model evaluation

### 2. Explore the Data
```bash
jupyter notebook 02_data_exploration.ipynb
```
**Time:** ~3-5 minutes
**What it does:** Deep dive into data statistics, distributions, and patterns

---

## 📚 Available Notebooks

| # | Notebook | Purpose | Status | Time |
|---|----------|---------|--------|------|
| 1 | `01_end_to_end_pipeline.ipynb` | Complete ML pipeline | ✅ Ready | 5-10 min |
| 2 | `02_data_exploration.ipynb` | Data analysis & visualization | ✅ Ready | 3-5 min |
| 3 | `03_feature_engineering.ipynb` | Feature engineering deep dive | 📝 Template | 10-15 min |
| 4 | `04_model_comparison.ipynb` | Model comparison & selection | 📝 Template | 15-20 min |
| 5 | `05_lstm_sequence_modeling.ipynb` | LSTM sequence models | 📝 Template | 20-30 min |
| 6 | `06_prediction_and_evaluation.ipynb` | Final predictions & evaluation | 📝 Template | 10-15 min |

---

## ✅ Completed Notebooks

### 1️⃣ End-to-End Pipeline (MAIN)
**File:** `01_end_to_end_pipeline.ipynb`

**Complete ML workflow:**
1. Setup & Configuration
2. Data Loading (with sampling)
3. Data Exploration
4. Data Preparation (cleaning, outlier removal)
5. Feature Engineering (physics, spatial, NFL features)
6. Train/Val Split (temporal split)
7. Temporal Features (AFTER split)
8. Model Training (Ridge, RF, XGBoost)
9. Model Evaluation (metrics + visualizations)
10. Save Results

**Key Features:**
- Self-contained functions
- ~1000 lines of well-documented code
- Configurable sample size
- Saves models and visualizations
- Educational comments

**Output:**
```
outputs/end_to_end_pipeline/
├── data_exploration.png
├── model_comparison.png
├── predictions_visualization.png
├── best_model_x.pkl
├── best_model_y.pkl
├── feature_names.pkl
├── model_comparison.csv
└── summary.json
```

---

### 2️⃣ Data Exploration
**File:** `02_data_exploration.ipynb`

**Deep dive analysis:**
1. Data Dictionary (column descriptions)
2. Statistical Analysis
3. Distribution Analysis (9 plots with statistics)
4. Correlation Analysis (heatmap)
5. Player Position Analysis
6. Game/Play Analysis
7. Field Position Heatmaps (4 visualizations)
8. Key Insights

**Visualizations:**
- 9 distribution plots (speed, acceleration, positions, etc.)
- Speed boxplots by player role
- Correlation heatmap
- Position distribution charts
- Field heatmaps (position density, speed, ball landing, zones)

**Output:**
```
outputs/data_exploration/
├── distributions.png
├── speed_by_role.png
├── correlation_heatmap.png
├── position_distribution.png
├── field_heatmaps.png
└── insights.txt
```

---

## 📝 Notebook Templates (To Create)

### 3️⃣ Feature Engineering Deep Dive
**Template:** See `NOTEBOOKS_SUMMARY.md` section 3

**Sections:**
- Physics Features (velocity, momentum, kinetic energy)
- Spatial Features (distances, angles, field zones)
- Temporal Features (⚠️ with data leakage warnings)
- NFL Domain Features (routes, roles, formations)
- Feature Importance Analysis
- Feature Correlation Heatmap

**Key Functions:**
```python
create_physics_features(df)
create_spatial_features(df)
create_temporal_features(df)  # ⚠️ Use after split!
create_nfl_features(df)
```

---

### 4️⃣ Model Comparison
**Template:** See `NOTEBOOKS_SUMMARY.md` section 4

**Models to Compare:**
- Linear: Ridge, Lasso, ElasticNet
- Tree: Random Forest, XGBoost, LightGBM
- Neural: Simple MLP

**Analysis:**
- Cross-validation
- Hyperparameter tuning
- Performance comparison (RMSE, MAE, R², speed)
- Feature importance across models
- Ensemble methods

---

### 5️⃣ LSTM Sequence Modeling
**Template:** See `NOTEBOOKS_SUMMARY.md` section 5

**Sequence Modeling:**
- Create sequences from tracking data
- LSTM architecture (2-layer)
- Training with early stopping
- Trajectory prediction
- Comparison with traditional models

**PyTorch Example:**
```python
class PlayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)
```

---

### 6️⃣ Prediction & Evaluation
**Template:** See `NOTEBOOKS_SUMMARY.md` section 6

**Final Analysis:**
- Load trained models
- Generate predictions
- Detailed evaluation metrics
- Error analysis (by position, field region, speed)
- Prediction visualization
- Export submission.csv

---

## 🔧 Configuration

### Sample vs Full Data
All notebooks support sampling for faster execution:

```python
# In each notebook config section
USE_SAMPLE = True  # Set to False for full data
SAMPLE_SIZE = 50000  # Adjust as needed
MAX_FILES = 2  # Number of weekly files to load
```

### Output Directories
Each notebook saves to its own directory:
```
outputs/
├── end_to_end_pipeline/
├── data_exploration/
├── feature_engineering/
├── model_comparison/
├── lstm_modeling/
└── predictions/
```

---

## ⚠️ Important: Data Leakage Warning

**Temporal features** use information from past AND future frames!

### ✅ Correct Approach:
```python
# 1. Create non-temporal features
df = create_physics_features(df)
df = create_spatial_features(df)
df = create_nfl_features(df)

# 2. Split data
train_df, val_df = train_val_split(df)

# 3. Create temporal features AFTER split
train_df = create_temporal_features(train_df)
val_df = create_temporal_features(val_df)
```

### ❌ Wrong Approach:
```python
# DON'T DO THIS!
df = create_temporal_features(df)  # Creates leakage!
train_df, val_df = train_val_split(df)
```

All notebooks include warnings about this!

---

## 📊 Expected Performance

### Typical Results (with good features):

| Model | X RMSE | Y RMSE | Training Time |
|-------|--------|--------|---------------|
| Ridge | 3-5 yards | 2-3 yards | < 1 min |
| Random Forest | 2-4 yards | 1.5-2.5 yards | 2-5 min |
| XGBoost | 2-3.5 yards | 1.5-2 yards | 3-8 min |
| LSTM | 2-3 yards | 1.5-2.5 yards | 10-20 min |

### Top Important Features (Typical):
1. Current position (x, y)
2. Speed (s)
3. Distance to ball landing
4. Player role (targeted receiver)
5. Velocity components (velocity_x, velocity_y)
6. Momentum
7. Field position features
8. Temporal changes (if used correctly)

---

## 🚀 Running the Notebooks

### Prerequisites
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# Optional for LSTM
pip install torch
```

### Launch Jupyter
```bash
# From the notebooks directory
cd notebooks
jupyter notebook
```

### Run in Order:
1. **Start with 01** - Get familiar with the pipeline
2. **Explore with 02** - Understand the data
3. **Build features with 03** - Create advanced features
4. **Compare models with 04** - Find best model
5. **Try LSTM with 05** - Sequence modeling
6. **Generate predictions with 06** - Final submission

---

## 📖 Documentation

### Detailed Templates:
See `NOTEBOOKS_SUMMARY.md` for:
- Complete notebook outlines
- Key code snippets
- Function definitions
- Visualization examples

### Pipeline Documentation:
See project root for:
- `README.md` - Project overview
- `QUICK_START.md` - Getting started guide
- `DATA_PIPELINE.md` - Data pipeline details

---

## 🎓 Learning Path

### For Beginners:
1. Run `01_end_to_end_pipeline.ipynb`
2. Read through each cell
3. Modify configurations (sample size, models)
4. Experiment with different features

### For Advanced Users:
1. Study `02_data_exploration.ipynb` for insights
2. Create custom features in `03_feature_engineering.ipynb`
3. Tune hyperparameters in `04_model_comparison.ipynb`
4. Build ensemble models
5. Try LSTM in `05_lstm_sequence_modeling.ipynb`

---

## 💡 Tips & Tricks

### Speed Up Execution:
```python
USE_SAMPLE = True
SAMPLE_SIZE = 10000  # Start small
MAX_FILES = 1  # Use one week only
```

### Save Intermediate Results:
```python
# After feature engineering
df.to_pickle('features.pkl')

# Load later
df = pd.read_pickle('features.pkl')
```

### Debug Models:
```python
# Add verbose output
model = RandomForestRegressor(verbose=2)

# Check shapes
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
```

### Visualize More:
```python
# Add custom plots
plt.figure(figsize=(12, 6))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Custom Prediction Plot')
plt.show()
```

---

## 🐛 Troubleshooting

### Common Issues:

**1. Out of Memory**
```python
# Reduce sample size
SAMPLE_SIZE = 10000
```

**2. Missing XGBoost**
```bash
pip install xgboost
# Or set: MODELS = ['ridge', 'random_forest']
```

**3. Slow Execution**
```python
# Use fewer files
MAX_FILES = 1
# Use fewer estimators
n_estimators = 50
```

**4. Data Leakage**
- Always create temporal features AFTER split
- Check for look-ahead bias
- Validate with unseen data

---

## 📈 Next Steps

After running all notebooks:

1. **Optimize Features** - Select most important features
2. **Tune Hyperparameters** - Grid/random search
3. **Build Ensemble** - Combine best models
4. **Validate on Test** - Generate final predictions
5. **Submit to Competition** - Use submission.csv

---

## 🤝 Contributing

To add new notebooks:
1. Follow the structure of existing notebooks
2. Include table of contents
3. Add clear markdown explanations
4. Use self-contained functions
5. Save visualizations to outputs
6. Update this README

---

## 📞 Support

For issues or questions:
1. Check `NOTEBOOKS_SUMMARY.md` for detailed templates
2. Review existing notebooks for examples
3. Check project documentation in root folder

---

**Happy Modeling! 🏈🤖**

*Last Updated: October 2025*
