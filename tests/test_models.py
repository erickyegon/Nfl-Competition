"""
Unit tests for model modules.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_pipeline.models.base import BaseModel, SklearnModel
from nfl_pipeline.models.traditional import ModelFactory, ModelTrainer
from nfl_pipeline.core.config import get_quick_config


@pytest.fixture
def config():
    """Get test configuration"""
    return get_quick_config()


@pytest.fixture
def sample_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X_train = np.random.randn(n_samples, n_features)
    y_train_x = X_train.sum(axis=1) + np.random.randn(n_samples) * 0.1
    y_train_y = X_train.sum(axis=1) + np.random.randn(n_samples) * 0.1

    X_val = np.random.randn(50, n_features)
    y_val_x = X_val.sum(axis=1) + np.random.randn(50) * 0.1
    y_val_y = X_val.sum(axis=1) + np.random.randn(50) * 0.1

    return {
        'X_train': X_train,
        'y_train_x': y_train_x,
        'y_train_y': y_train_y,
        'X_val': X_val,
        'y_val_x': y_val_x,
        'y_val_y': y_val_y
    }


class TestBaseModel:
    """Test base model interface"""

    def test_sklearn_model_initialization(self, config):
        """Test SklearnModel initialization"""
        from sklearn.linear_model import Ridge

        model = SklearnModel(config, Ridge, name="test_ridge", alpha=1.0)

        assert model.name == "test_ridge"
        assert model.is_trained == False
        assert model.model_x is None
        assert model.model_y is None

    def test_sklearn_model_training(self, config, sample_data):
        """Test SklearnModel training"""
        from sklearn.linear_model import Ridge

        model = SklearnModel(config, Ridge, name="test_ridge", alpha=1.0)

        results = model.train(
            sample_data['X_train'],
            sample_data['y_train_x'],
            sample_data['y_train_y'],
            sample_data['X_val'],
            sample_data['y_val_x'],
            sample_data['y_val_y']
        )

        assert model.is_trained == True
        assert 'x_results' in results
        assert 'y_results' in results
        assert 'metrics' in results['x_results']

    def test_sklearn_model_prediction(self, config, sample_data):
        """Test SklearnModel prediction"""
        from sklearn.linear_model import Ridge

        model = SklearnModel(config, Ridge, name="test_ridge", alpha=1.0)

        # Train first
        model.train(
            sample_data['X_train'],
            sample_data['y_train_x'],
            sample_data['y_train_y']
        )

        # Predict
        pred_x, pred_y = model.predict(sample_data['X_val'])

        assert len(pred_x) == len(sample_data['X_val'])
        assert len(pred_y) == len(sample_data['X_val'])

    def test_model_save_load(self, config, sample_data):
        """Test model save and load"""
        from sklearn.linear_model import Ridge

        model = SklearnModel(config, Ridge, name="test_ridge", alpha=1.0)

        # Train
        model.train(
            sample_data['X_train'],
            sample_data['y_train_x'],
            sample_data['y_train_y']
        )

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)
            model.save(save_dir)

            # Create new model and load
            model2 = SklearnModel(config, Ridge, name="test_ridge", alpha=1.0)
            model2.load(save_dir)

            assert model2.is_trained == True

            # Predictions should be the same
            pred1_x, pred1_y = model.predict(sample_data['X_val'])
            pred2_x, pred2_y = model2.predict(sample_data['X_val'])

            np.testing.assert_array_almost_equal(pred1_x, pred2_x)
            np.testing.assert_array_almost_equal(pred1_y, pred2_y)


class TestModelFactory:
    """Test model factory"""

    def test_get_ridge_model(self):
        """Test getting Ridge model"""
        model = ModelFactory.get_model('ridge', random_state=42)

        from sklearn.linear_model import Ridge
        assert isinstance(model, Ridge)

    def test_get_random_forest_model(self):
        """Test getting Random Forest model"""
        model = ModelFactory.get_model('random_forest', random_state=42)

        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(model, RandomForestRegressor)

    def test_invalid_model_name(self):
        """Test that invalid model name raises error"""
        with pytest.raises(ValueError):
            ModelFactory.get_model('invalid_model')

    def test_get_hyperparameter_grid(self):
        """Test getting hyperparameter grid"""
        grid = ModelFactory.get_hyperparameter_grid('ridge')

        assert 'alpha' in grid
        assert isinstance(grid['alpha'], list)


class TestModelTrainer:
    """Test model trainer"""

    def test_trainer_initialization(self, config):
        """Test trainer initialization"""
        trainer = ModelTrainer(config)

        assert trainer.config == config
        assert isinstance(trainer.trained_models, dict)

    def test_train_single_model(self, config, sample_data):
        """Test training a single model"""
        trainer = ModelTrainer(config)

        result = trainer.train_model(
            'ridge',
            sample_data['X_train'],
            sample_data['y_train_x'],
            sample_data['X_val'],
            sample_data['y_val_x']
        )

        assert 'model' in result
        assert 'train_metrics' in result
        assert 'val_metrics' in result
        assert 'train_predictions' in result

    def test_model_stored_after_training(self, config, sample_data):
        """Test that model is stored after training"""
        trainer = ModelTrainer(config)

        trainer.train_model(
            'ridge',
            sample_data['X_train'],
            sample_data['y_train_x']
        )

        assert 'ridge' in trainer.trained_models

    def test_feature_importance(self, config, sample_data):
        """Test getting feature importance"""
        trainer = ModelTrainer(config)

        trainer.train_model(
            'random_forest',
            sample_data['X_train'],
            sample_data['y_train_x']
        )

        feature_names = [f'feature_{i}' for i in range(sample_data['X_train'].shape[1])]
        importance = trainer.get_feature_importance('random_forest', feature_names)

        assert importance is not None
        assert len(importance) == len(feature_names)


def test_model_reproducibility(config, sample_data):
    """Test that model training is reproducible with same seed"""
    trainer1 = ModelTrainer(config)
    trainer2 = ModelTrainer(config)

    result1 = trainer1.train_model(
        'ridge',
        sample_data['X_train'],
        sample_data['y_train_x']
    )

    result2 = trainer2.train_model(
        'ridge',
        sample_data['X_train'],
        sample_data['y_train_x']
    )

    # Predictions should be identical
    np.testing.assert_array_almost_equal(
        result1['train_predictions'],
        result2['train_predictions']
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
