"""
Setup configuration for NFL Player Movement Prediction Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ''

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name='nfl-pipeline',
    version='2.1.0',
    description='NFL Player Movement Prediction Pipeline with LSTM',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='NFL ML Team',
    author_email='',
    url='https://github.com/nfl/player-movement',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'lstm': ['torch>=1.11.0'],
        'dev': ['pytest>=6.0.0', 'black>=21.0.0', 'flake8>=4.0.0'],
    },
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'nfl-train=train:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
