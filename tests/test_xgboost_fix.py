"""Quick test for XGBoost early stopping fix"""
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from models import ModelFactory, ModelTrainer, XGBOOST_AVAILABLE
from config import get_quick_config

print("Testing XGBoost Early Stopping Fix...")

if not XGBOOST_AVAILABLE:
    print("✗ XGBoost not available")
    sys.exit(1)

# Test 1: Model has early_stopping_rounds parameter
model = ModelFactory.get_model('xgboost', random_state=42)
print(f"1. early_stopping_rounds: {getattr(model, 'early_stopping_rounds', 'NOT SET')}")

# Test 2: Train with validation set
config = get_quick_config()
trainer = ModelTrainer(config)

X_train = np.random.randn(100, 10)
y_train = np.random.randn(100)
X_val = np.random.randn(20, 10)
y_val = np.random.randn(20)

print("2. Training XGBoost with validation set...")
try:
    result = trainer.train_model('xgboost', X_train, y_train, X_val, y_val)
    print("✓ SUCCESS: XGBoost trained with validation set")
    print(f"   Train RMSE: {result['train_metrics']['rmse']:.4f}")
    print(f"   Val RMSE: {result['val_metrics']['rmse']:.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
