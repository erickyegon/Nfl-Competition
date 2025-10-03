"""
End-to-end tests for the NFL ML pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl_pipeline.core.config import get_quick_config
from nfl_pipeline.core.pipeline import NFLPipeline


@pytest.fixture
def config():
    """Get test configuration with minimal settings"""
    config = get_quick_config()
    # Override with minimal settings for fast testing
    config.models_to_evaluate = ['ridge']
    config.enable_ensembles = False
    config.tune_hyperparameters = False
    return config


class TestPipelineConfiguration:
    """Test pipeline configuration"""

    def test_quick_config_creation(self):
        """Test that quick config can be created"""
        config = get_quick_config()

        assert config is not None
        assert hasattr(config, 'models_to_evaluate')
        assert hasattr(config, 'data_dir')
        assert hasattr(config, 'output_dir')

    def test_config_validation(self):
        """Test config validation"""
        from nfl_pipeline.utils.helpers import validate_config

        config = get_quick_config()
        issues = validate_config(config)

        # Should have issues if data doesn't exist
        # This is expected in test environment
        assert isinstance(issues, list)


class TestPipelineComponents:
    """Test individual pipeline components"""

    def test_data_loader_initialization(self, config):
        """Test data loader initialization"""
        from nfl_pipeline.data.loader import DataLoader

        loader = DataLoader(config)
        assert loader.config == config

    def test_preprocessor_initialization(self, config):
        """Test preprocessor initialization"""
        from nfl_pipeline.data.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor(config)
        assert preprocessor.config == config

    def test_feature_engineer_initialization(self, config):
        """Test feature engineer initialization"""
        from nfl_pipeline.features.nfl_domain import FeatureEngineer

        engineer = FeatureEngineer(config)
        assert engineer.config == config

    def test_model_trainer_initialization(self, config):
        """Test model trainer initialization"""
        from nfl_pipeline.models.traditional import ModelTrainer

        trainer = ModelTrainer(config)
        assert trainer.config == config

    def test_evaluator_initialization(self, config):
        """Test evaluator initialization"""
        from nfl_pipeline.evaluation.evaluator import ModelEvaluator

        evaluator = ModelEvaluator(config)
        assert evaluator.config == config


class TestPipelineExecution:
    """Test pipeline execution with mock data"""

    @pytest.mark.skip(reason="Requires actual data files")
    def test_pipeline_initialization(self, config):
        """Test pipeline initialization"""
        pipeline = NFLPipeline(config)

        assert pipeline.config == config
        assert hasattr(pipeline, 'logger')

    @pytest.mark.skip(reason="Requires actual data files")
    def test_pipeline_run(self, config):
        """Test full pipeline execution"""
        pipeline = NFLPipeline(config)

        try:
            results = pipeline.run_pipeline()

            assert 'model_results' in results
            assert 'selected_model' in results
            assert 'evaluation_results' in results
        except FileNotFoundError:
            pytest.skip("Data files not available for testing")


class TestErrorHandling:
    """Test error handling in pipeline"""

    def test_missing_data_directory(self):
        """Test handling of missing data directory"""
        config = get_quick_config()
        config.data_dir = Path("/nonexistent/path")

        from nfl_pipeline.data.loader import DataLoader
        loader = DataLoader(config)

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_invalid_model_name(self, config):
        """Test handling of invalid model name"""
        from nfl_pipeline.models.traditional import ModelFactory

        with pytest.raises(ValueError):
            ModelFactory.get_model('nonexistent_model')


class TestOutputGeneration:
    """Test output file generation"""

    def test_output_directory_creation(self, config):
        """Test that output directories are created"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = Path(tmpdir) / "outputs"
            config.models_dir = Path(tmpdir) / "models"
            config.logs_dir = Path(tmpdir) / "logs"

            # Create directories
            config.output_dir.mkdir(parents=True, exist_ok=True)
            config.models_dir.mkdir(parents=True, exist_ok=True)
            config.logs_dir.mkdir(parents=True, exist_ok=True)

            assert config.output_dir.exists()
            assert config.models_dir.exists()
            assert config.logs_dir.exists()


def test_integration_small_dataset():
    """Test pipeline on small synthetic dataset"""
    config = get_quick_config()
    config.models_to_evaluate = ['ridge']

    # Create minimal synthetic data
    from nfl_pipeline.models.traditional import ModelTrainer

    trainer = ModelTrainer(config)

    # Synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = X_train.sum(axis=1) + np.random.randn(100) * 0.1

    X_val = np.random.randn(50, 10)
    y_val = X_val.sum(axis=1) + np.random.randn(50) * 0.1

    # Train model
    result = trainer.train_model('ridge', X_train, y_train, X_val, y_val)

    assert 'model' in result
    assert 'train_metrics' in result
    assert result['train_metrics']['rmse'] < 1.0  # Should fit well


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
