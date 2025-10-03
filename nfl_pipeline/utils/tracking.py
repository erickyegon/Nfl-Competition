"""
Experiment Tracking Module for NFL ML Pipeline
Handles experiment logging, metrics tracking, and results persistence.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ExperimentTracker:
    """
    Simple experiment tracking for ML pipeline.

    Can be extended to integrate with MLflow, Weights & Biases, or other
    experiment tracking frameworks.
    """

    def __init__(self, config):
        """
        Initialize experiment tracker.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.results = {
            'experiment_name': config.experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'parameters': {},
            'metrics': {},
            'artifacts': {}
        }

    def log_config(self, config_dict: Dict[str, Any]):
        """
        Log configuration parameters.

        Args:
            config_dict: Configuration dictionary
        """
        self.results['config'] = config_dict

    def log_params(self, params: Dict[str, Any]):
        """
        Log model or pipeline parameters.

        Args:
            params: Parameter dictionary
        """
        self.results['parameters'].update(params)

    def log_metrics(self, metrics: Dict[str, float], step_name: str = ""):
        """
        Log metrics for a specific step.

        Args:
            metrics: Dictionary of metric name to value
            step_name: Name of the step (e.g., 'training', 'validation', 'test')
        """
        if 'metrics' not in self.results:
            self.results['metrics'] = {}

        key = step_name if step_name else 'final'
        self.results['metrics'][key] = metrics

    def log_metric(self, name: str, value: float, step_name: str = ""):
        """
        Log a single metric.

        Args:
            name: Metric name
            value: Metric value
            step_name: Name of the step
        """
        key = step_name if step_name else 'final'

        if key not in self.results['metrics']:
            self.results['metrics'][key] = {}

        self.results['metrics'][key][name] = value

    def log_artifact(self, name: str, path: Path):
        """
        Log an artifact path (model, plot, etc.).

        Args:
            name: Artifact name
            path: Path to artifact file
        """
        if 'artifacts' not in self.results:
            self.results['artifacts'] = {}

        self.results['artifacts'][name] = str(path)

    def log_model(self, model_name: str, model_path: Path, metrics: Optional[Dict[str, float]] = None):
        """
        Log a trained model.

        Args:
            model_name: Name of the model
            model_path: Path where model is saved
            metrics: Optional metrics associated with the model
        """
        model_info = {
            'path': str(model_path),
            'timestamp': datetime.now().isoformat()
        }

        if metrics:
            model_info['metrics'] = metrics

        if 'models' not in self.results:
            self.results['models'] = {}

        self.results['models'][model_name] = model_info

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the experiment.

        Args:
            tags: Dictionary of tag name to value
        """
        if 'tags' not in self.results:
            self.results['tags'] = {}

        self.results['tags'].update(tags)

    def save_results(self, output_path: Optional[Path] = None):
        """
        Save experiment results to JSON file.

        Args:
            output_path: Optional custom output path
        """
        # Add end time
        self.results['end_time'] = datetime.now().isoformat()

        # Determine save path
        if output_path is None:
            output_path = self.config.output_dir / f"{self.config.experiment_name}_results.json"

        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        return output_path

    def get_summary(self) -> str:
        """
        Get a text summary of the experiment.

        Returns:
            Summary string
        """
        summary = []
        summary.append("=" * 80)
        summary.append(f"EXPERIMENT SUMMARY: {self.results['experiment_name']}")
        summary.append("=" * 80)
        summary.append(f"Start Time: {self.results.get('start_time', 'N/A')}")
        summary.append(f"End Time: {self.results.get('end_time', 'In Progress')}")
        summary.append("")

        # Parameters
        if self.results.get('parameters'):
            summary.append("Parameters:")
            for key, value in self.results['parameters'].items():
                summary.append(f"  {key}: {value}")
            summary.append("")

        # Metrics
        if self.results.get('metrics'):
            summary.append("Metrics:")
            for step, metrics in self.results['metrics'].items():
                summary.append(f"  {step}:")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, float):
                        summary.append(f"    {metric_name}: {metric_value:.4f}")
                    else:
                        summary.append(f"    {metric_name}: {metric_value}")
            summary.append("")

        # Models
        if self.results.get('models'):
            summary.append("Models:")
            for model_name, model_info in self.results['models'].items():
                summary.append(f"  {model_name}: {model_info['path']}")
            summary.append("")

        # Artifacts
        if self.results.get('artifacts'):
            summary.append("Artifacts:")
            for artifact_name, artifact_path in self.results['artifacts'].items():
                summary.append(f"  {artifact_name}: {artifact_path}")

        summary.append("=" * 80)
        return "\n".join(summary)


if __name__ == "__main__":
    # Test experiment tracker
    print("Testing ExperimentTracker...")

    # Create a mock config
    class MockConfig:
        def __init__(self):
            self.experiment_name = "test_experiment"
            self.output_dir = Path("./test_output")

    config = MockConfig()
    tracker = ExperimentTracker(config)

    # Log various items
    tracker.log_params({'learning_rate': 0.001, 'batch_size': 32})
    tracker.log_metrics({'rmse': 2.5, 'mae': 2.0}, 'validation')
    tracker.log_artifact('model', Path('./model.pkl'))
    tracker.set_tags({'version': '1.0', 'status': 'test'})

    # Print summary
    print(tracker.get_summary())

    print("ExperimentTracker test completed successfully")
