# 📚 NFL Competition Notebooks - Complete Suite

This directory contains a comprehensive set of Jupyter notebooks for the NFL Player Movement Prediction pipeline.

---

## ✅ Created Notebooks

### 1. **01_end_to_end_pipeline.ipynb** (MAIN) ✅
**Status:** Complete
**Purpose:** Complete ML pipeline from data loading to predictions

**Key Sections:**
- Setup & Configuration
- Data Loading (with sampling)
- Data Exploration (quick analysis)
- Data Preparation (cleaning, outlier removal)
- Feature Engineering (physics, spatial, NFL features)
- Train/Val Split (temporal split by game)
- Temporal Features (AFTER split to avoid leakage)
- Model Training (Ridge, Random Forest, XGBoost)
- Model Evaluation (RMSE, MAE, R²)
- Predictions & Visualization
- Save Results (models, metrics, plots)

**Features:**
- Self-contained functions
- Progress indicators
- Visualizations saved to outputs
- Modular and easy to modify
- ~1000 lines of well-documented code

---

### 2. **02_data_exploration.ipynb** ✅
**Status:** Complete
**Purpose:** Deep dive into data analysis

**Key Sections:**
- Setup & Data Loading
- Data Dictionary (column descriptions)
- Statistical Analysis (describe, missing values)
- Distribution Analysis (histograms with stats)
- Speed Analysis by Player Role
- Correlation Analysis (heatmap, top correlations)
- Player Position Analysis (counts, physical attributes)
- Game/Play Analysis (plays per game, frames per play)
- Field Position Heatmaps (position density, speed heatmap, ball landing, zones)
- Key Insights (summary of findings)

**Visualizations:**
- 9 distribution plots with mean/median lines
- Speed boxplots by player role
- Correlation heatmap
- Position distribution (bar + pie charts)
- 4 field heatmaps (position, speed, ball landing, zones)

---

## 📝 Remaining Notebooks (Templates)

### 3. **03_feature_engineering.ipynb**
**Purpose:** Feature engineering deep dive

**Sections:**
1. **Setup** - Import libraries, configure paths
2. **Data Loading** - Load sample data
3. **Physics Features Section**:
   - Velocity decomposition (v_x, v_y)
   - Acceleration components
   - Momentum calculation (mass × velocity)
   - Kinetic energy (0.5 × mass × v²)
   - Direction difference (orientation vs motion)
   - Visualizations (velocity scatter, momentum hist, KE hist, dir diff)

4. **Spatial Features Section**:
   - Distance to ball landing
   - Field position (normalized, zones)
   - Sideline proximity
   - Angle to ball
   - Visualizations (distance distributions, field zones)

5. **Temporal Features Section**:
   - ⚠️ WARNING about data leakage (prominent warning boxes)
   - Position changes (dx, dy)
   - Speed/acceleration changes
   - Direction changes with wraparound handling
   - Demonstration (not applied to avoid leakage in notebook)

6. **NFL Domain Features Section**:
   - Player role encoding (targeted receiver, passer, defensive)
   - Route detection (depth, lateral movement)
   - Speed-based indicators (sprinting, jogging, stationary)
   - Formation features

7. **Feature Importance Analysis**:
   - Train quick Random Forest
   - Top 20 feature importances
   - Bar plot visualization
   - Save to outputs

8. **Feature Correlation**:
   - Correlation with targets
   - Top positive/negative correlations
   - Heatmap of top 15 features
   - Multicollinearity detection

**Key Functions:**
```python
def create_physics_features(df):
    # Velocity, acceleration, momentum, KE

def create_spatial_features(df):
    # Distances, angles, field positions

def create_temporal_features(df):
    # ⚠️ WARNING: Use only after split!

def create_nfl_features(df):
    # Routes, roles, formations
```

---

### 4. **04_model_comparison.ipynb**
**Purpose:** Compare multiple models and select best

**Sections:**
1. **Setup & Data Loading**
2. **Data Preparation** - Load prepared features
3. **Train/Val Split** - Temporal split

4. **Model Training** - Train multiple models:
   - **Linear Models**: Ridge, Lasso, ElasticNet
   - **Tree Models**: Random Forest, XGBoost, LightGBM
   - **Neural Network**: Simple MLP (optional)

5. **Cross-Validation**:
   - K-fold CV
   - Time-series CV
   - Results comparison

6. **Hyperparameter Tuning**:
   - Grid search examples
   - Random search
   - Best parameters for each model

7. **Model Comparison**:
   - RMSE comparison (bar chart)
   - Training time comparison
   - Prediction speed comparison
   - Feature importance comparison
   - Learning curves

8. **Best Model Selection**:
   - Overall best performer
   - Best by metric
   - Recommendations

9. **Ensemble Methods**:
   - Simple averaging
   - Weighted averaging
   - Stacking (meta-learner)
   - Ensemble vs individual comparison

**Key Visualizations:**
- Model performance bar charts (RMSE, MAE, R²)
- Training time comparison
- Feature importance across models
- Learning curves
- Ensemble improvement plot

---

### 5. **05_lstm_sequence_modeling.ipynb**
**Purpose:** LSTM and sequence models for trajectory prediction

**Sections:**
1. **Setup & Imports**
   - PyTorch/TensorFlow imports
   - Sequence utilities

2. **Sequence Creation**:
   - Create sequences from tracking data
   - Window size selection (e.g., 10 frames)
   - Sequence padding
   - Train/val split for sequences

3. **LSTM Architecture**:
   - Model definition (2-layer LSTM)
   - Input: (batch, sequence_length, features)
   - Output: (batch, 2) for x,y prediction
   - Architecture diagram/explanation

4. **Training**:
   - DataLoader setup
   - Training loop with early stopping
   - Validation monitoring
   - Loss curves

5. **Sequence Predictions**:
   - Generate predictions
   - Compare with traditional models
   - Trajectory visualization

6. **Results Analysis**:
   - RMSE comparison (LSTM vs traditional)
   - Trajectory plots (predicted vs actual)
   - Error analysis by sequence length
   - When LSTM works best

**Sample Code:**
```python
class PlayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
```

---

### 6. **06_prediction_and_evaluation.ipynb**
**Purpose:** Final predictions and detailed evaluation

**Sections:**
1. **Setup & Load Models**
   - Load trained models
   - Load feature names
   - Load test data

2. **Generate Predictions**:
   - Apply feature engineering to test data
   - Generate predictions
   - Create submission file

3. **Detailed Evaluation**:
   - **Overall Metrics**: RMSE, MAE, R², MAPE
   - **Residual Analysis**:
     - Residual plots (scatter, histogram)
     - Q-Q plot for normality
     - Residuals vs predicted

4. **Error Analysis**:
   - **By Player Position**: Which positions harder to predict?
   - **By Field Region**: Errors in red zone vs midfield
   - **By Speed Range**: High speed vs low speed errors
   - **By Distance to Ball**: Near vs far predictions

5. **Prediction Visualization**:
   - Scatter plots (actual vs predicted)
   - Trajectory plots for sample plays
   - Heatmaps of errors on field
   - Animation examples (optional)

6. **Confidence Intervals**:
   - Prediction uncertainty
   - Confidence bounds
   - Reliability analysis

7. **Export Final Predictions**:
   - Save submission.csv
   - Save prediction analysis
   - Generate report

**Key Visualizations:**
- Predictions vs actual scatter (with diagonal)
- Residual distribution (histogram + KDE)
- Error heatmap on field
- Error by position (boxplots)
- Trajectory comparison (animated or multi-frame)

---

## 🚀 Quick Start

### Run Complete Pipeline:
```bash
jupyter notebook 01_end_to_end_pipeline.ipynb
```

### Explore Data:
```bash
jupyter notebook 02_data_exploration.ipynb
```

### Feature Engineering:
```bash
jupyter notebook 03_feature_engineering.ipynb
```

### Model Comparison:
```bash
jupyter notebook 04_model_comparison.ipynb
```

### Try LSTM:
```bash
jupyter notebook 05_lstm_sequence_modeling.ipynb
```

### Generate Predictions:
```bash
jupyter notebook 06_prediction_and_evaluation.ipynb
```

---

## 📊 Common Features Across All Notebooks

### ✅ Structure:
- Clear markdown headers with emojis
- Table of contents at top
- Well-commented code
- Output cells showing results
- Visualizations for key insights

### ✅ Code Style:
- Self-contained functions (not importing from nfl_pipeline)
- Functions mirror the modular pipeline functionality
- Clear docstrings
- Progress indicators (print statements)
- Error handling

### ✅ Educational:
- Explain WHY each step is done
- Show intermediate results
- Include tips and warnings (especially for temporal features!)
- Link concepts to NFL football context

### ✅ Practical:
- Can run independently
- Use sample data for speed (configurable)
- Toggle between sample/full data
- Save intermediate results
- Easy to modify and experiment

---

## 🔑 Key Functions (Self-Contained)

All notebooks include these self-contained functions:

```python
# Data Loading
def load_data(data_dir, max_files=None, sample_size=None):
    """Load input/output CSV files"""

# Data Preparation
def prepare_data(input_df, output_df):
    """Merge, clean, handle missing values"""

# Feature Engineering
def create_physics_features(df):
    """Velocity, acceleration, momentum, kinetic energy"""

def create_spatial_features(df):
    """Distance to ball, field position, sideline proximity"""

def create_temporal_features(df):
    """⚠️ WARNING: Use only AFTER split! Position/speed changes"""

def create_nfl_features(df):
    """Routes, roles, formations, coverage"""

# Splitting
def train_val_split(df, val_size=0.2):
    """Temporal split by game/week"""

# Modeling
def train_model(X_train, y_train, X_val, y_val, model_type='ridge'):
    """Train sklearn model, return model and metrics"""

# Evaluation
def evaluate_model(model, X_val, y_val):
    """Calculate RMSE, MAE, R²"""

# Visualization
def visualize_predictions(y_true, y_pred):
    """Plot predictions vs actual, residuals"""
```

---

## ⚠️ Important Notes

### Data Leakage Warning:
**Temporal features** (position changes, rolling stats) use information from **past and future frames**.

**MUST create these AFTER train/test split!**

All notebooks include prominent warnings about this.

### Sample vs Full Data:
Each notebook has a `USE_SAMPLE` flag:
```python
USE_SAMPLE = True  # Set to False for full data
SAMPLE_SIZE = 50000  # Adjust as needed
```

### Output Directories:
Each notebook saves outputs to:
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

## 📈 Expected Results

### Baseline Performance:
- **Ridge Regression**: RMSE ~3-5 yards (baseline)
- **Random Forest**: RMSE ~2-4 yards
- **XGBoost**: RMSE ~2-3.5 yards (typically best)
- **LSTM**: RMSE ~2-3 yards (good for trajectories)

### Feature Importance (Typical):
1. Current position (x, y)
2. Speed (s)
3. Distance to ball landing
4. Player role (targeted receiver)
5. Velocity components
6. Field position
7. Temporal changes (if used)

---

## 🎯 Next Steps After Running Notebooks

1. **Tune Hyperparameters** - Use notebook 04 for grid search
2. **Try Ensembles** - Combine best models
3. **Feature Selection** - Remove low-importance features
4. **Cross-Validation** - Validate on multiple splits
5. **Submit Predictions** - Use notebook 06 to generate submission

---

## 📚 Additional Resources

### Created:
- ✅ `01_end_to_end_pipeline.ipynb` - Complete pipeline
- ✅ `02_data_exploration.ipynb` - Data analysis

### To Create (use templates above):
- 📝 `03_feature_engineering.ipynb` - Feature deep dive
- 📝 `04_model_comparison.ipynb` - Model selection
- 📝 `05_lstm_sequence_modeling.ipynb` - Sequence models
- 📝 `06_prediction_and_evaluation.ipynb` - Final predictions

### Quick Create Commands:
You can create the remaining notebooks by copying the structure from notebooks 01-02 and following the templates above.

---

**Happy Modeling! 🏈🤖**
