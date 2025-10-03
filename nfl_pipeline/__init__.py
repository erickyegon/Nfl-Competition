"""
NFL Player Movement Prediction Pipeline
A modular ML pipeline for predicting NFL player trajectories.
"""

__version__ = '2.1.0'
__author__ = 'NFL ML Team'

from nfl_pipeline.core.pipeline import NFLPipeline
from nfl_pipeline.core.config import (
    PipelineConfig,
    get_quick_config,
    get_lstm_config,
    get_full_config
)

__all__ = [
    'NFLPipeline',
    'PipelineConfig',
    'get_quick_config',
    'get_lstm_config',
    'get_full_config'
]
