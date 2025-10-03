"""
Logging utilities for the NFL Pipeline
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from nfl_pipeline.core.config import PipelineConfig


class PipelineLogger:
    """Enhanced logging for the ML pipeline with singleton pattern"""

    _instance = None
    _logger = None

    def __new__(cls, config: Optional[PipelineConfig] = None):
        if cls._instance is None:
            cls._instance = super(PipelineLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[PipelineConfig] = None):
        # Only setup once
        if self._logger is None and config is not None:
            self.config = config
            self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # Create logger
        self._logger = logging.getLogger('nfl_pipeline')
        self._logger.setLevel(log_level)
        self._logger.propagate = False  # Prevent duplicate logging

        # Clear existing handlers
        self._logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')

        # Console handler (simple format)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        self._logger.addHandler(console_handler)

        # File handler (detailed format)
        if self.config.log_dir:
            self.config.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.config.log_dir / f"pipeline_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(detailed_formatter)
            self._logger.addHandler(file_handler)

            # Create symlink to latest log
            latest_log = self.config.log_dir / 'latest.log'
            if latest_log.exists():
                latest_log.unlink()
            try:
                latest_log.symlink_to(log_file.name)
            except (OSError, NotImplementedError):
                # Symlinks not supported on this system
                pass

            self._logger.info(f"Logging to: {log_file}")

    @property
    def logger(self):
        """Return the logger instance"""
        if self._logger is None:
            # Create a basic logger if config wasn't provided
            self._logger = logging.getLogger('nfl_pipeline')
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)
        return self._logger


# Global logger instance
_global_logger = None


def get_logger(config: Optional[PipelineConfig] = None) -> logging.Logger:
    """
    Get the global pipeline logger.

    Args:
        config: Pipeline configuration (required for first call)

    Returns:
        Logger instance
    """
    global _global_logger

    if _global_logger is None:
        pipeline_logger = PipelineLogger(config)
        _global_logger = pipeline_logger.logger

    return _global_logger


def reset_logger():
    """Reset the global logger (useful for testing)"""
    global _global_logger
    _global_logger = None
    PipelineLogger._instance = None
    PipelineLogger._logger = None
