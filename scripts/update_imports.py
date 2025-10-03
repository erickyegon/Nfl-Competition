"""
Script to update imports in restructured files
"""
import os
import re
from pathlib import Path

# Import mappings: old -> new
IMPORT_MAPPINGS = {
    'from config import': 'from nfl_pipeline.core.config import',
    'from utils import': 'from nfl_pipeline.utils.helpers import',
    'from data_loader import': 'from nfl_pipeline.data.loader import',
    'from feature_engineering import': 'from nfl_pipeline.features.engineer import',
    'from data_preparation import': 'from nfl_pipeline.data.preprocessor import',
    'from models import': 'from nfl_pipeline.models.traditional import',
    'from ensemble import': 'from nfl_pipeline.models.ensemble import',
    'from evaluation import': 'from nfl_pipeline.evaluation.evaluator import',
    'from prediction import': 'from nfl_pipeline.prediction.predictor import',
    'from sequence_models import': 'from nfl_pipeline.models.sequence import',
}

def update_file_imports(filepath):
    """Update imports in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    for old_import, new_import in IMPORT_MAPPINGS.items():
        content = content.replace(old_import, new_import)

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def update_all_imports():
    """Update imports in all Python files in nfl_pipeline"""
    base_path = Path(__file__).parent.parent / 'nfl_pipeline'
    updated_files = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                filepath = Path(root) / file
                if update_file_imports(filepath):
                    updated_files.append(str(filepath))

    return updated_files

if __name__ == '__main__':
    print("Updating imports in restructured files...")
    updated = update_all_imports()
    print(f"\nUpdated {len(updated)} files:")
    for f in updated:
        print(f"  - {f}")
    print("\nDone!")
