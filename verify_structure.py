"""
Structure Verification Script
Verifies that the NFL pipeline restructuring is complete and correct.
"""

import sys
from pathlib import Path


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if filepath.exists():
        print(f"[OK] {description}: {filepath}")
        return True
    else:
        print(f"[MISS] {description}: {filepath}")
        return False


def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    if dirpath.exists() and dirpath.is_dir():
        print(f"[OK] {description}: {dirpath}")
        return True
    else:
        print(f"[MISS] {description}: {dirpath}")
        return False


def main():
    root = Path(__file__).parent
    all_checks_passed = True

    print("=" * 80)
    print("NFL PIPELINE STRUCTURE VERIFICATION")
    print("=" * 80)

    # Check main directories
    print("\n[DIRECTORIES] Main Directories:")
    dirs_to_check = [
        (root / "nfl_pipeline", "Pipeline package"),
        (root / "scripts", "Scripts directory"),
        (root / "tests", "Tests directory"),
        (root / "configs", "Configs directory"),
    ]

    for dirpath, description in dirs_to_check:
        if not check_directory_exists(dirpath, description):
            all_checks_passed = False

    # Check nfl_pipeline structure
    print("\n[MODULES] nfl_pipeline/ modules:")
    nfl_dirs = [
        (root / "nfl_pipeline" / "core", "Core module"),
        (root / "nfl_pipeline" / "data", "Data module"),
        (root / "nfl_pipeline" / "features", "Features module"),
        (root / "nfl_pipeline" / "models", "Models module"),
        (root / "nfl_pipeline" / "evaluation", "Evaluation module"),
        (root / "nfl_pipeline" / "prediction", "Prediction module"),
        (root / "nfl_pipeline" / "utils", "Utils module"),
    ]

    for dirpath, description in nfl_dirs:
        if not check_directory_exists(dirpath, description):
            all_checks_passed = False

    # Check NEW evaluation files
    print("\n[EVALUATION] Evaluation Module (SPLIT):")
    eval_files = [
        (root / "nfl_pipeline" / "evaluation" / "metrics.py", "Metrics module"),
        (root / "nfl_pipeline" / "evaluation" / "selector.py", "Selector module"),
        (root / "nfl_pipeline" / "evaluation" / "evaluator.py", "Evaluator module"),
    ]

    for filepath, description in eval_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check NEW utils files
    print("\n[UTILS] Utils Module (REORGANIZED):")
    utils_files = [
        (root / "nfl_pipeline" / "utils" / "logging.py", "Logging module"),
        (root / "nfl_pipeline" / "utils" / "tracking.py", "Tracking module"),
        (root / "nfl_pipeline" / "utils" / "helpers.py", "Helpers module"),
    ]

    for filepath, description in utils_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check NEW models files
    print("\n[MODELS] Models Module (ENHANCED):")
    model_files = [
        (root / "nfl_pipeline" / "models" / "base.py", "Base model"),
        (root / "nfl_pipeline" / "models" / "traditional.py", "Traditional models"),
        (root / "nfl_pipeline" / "models" / "sequence.py", "Sequence models"),
        (root / "nfl_pipeline" / "models" / "ensemble.py", "Ensemble models"),
    ]

    for filepath, description in model_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check scripts
    print("\n[SCRIPTS] Scripts:")
    script_files = [
        (root / "scripts" / "train.py", "Training script"),
        (root / "scripts" / "predict.py", "Prediction script"),
        (root / "scripts" / "evaluate.py", "Evaluation script"),
    ]

    for filepath, description in script_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check tests
    print("\n[TESTS] Tests:")
    test_files = [
        (root / "tests" / "test_features.py", "Features tests"),
        (root / "tests" / "test_models.py", "Models tests"),
        (root / "tests" / "test_pipeline.py", "Pipeline tests"),
    ]

    for filepath, description in test_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check configs
    print("\n[CONFIGS] Configurations:")
    config_files = [
        (root / "configs" / "default.yaml", "Default config"),
        (root / "configs" / "quick.yaml", "Quick config"),
        (root / "configs" / "lstm.yaml", "LSTM config"),
    ]

    for filepath, description in config_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check root files
    print("\n[ROOT] Root Files:")
    root_files = [
        (root / "setup.py", "Setup file"),
        (root / "README.md", "README"),
        (root / "QUICK_START.md", "Quick start guide"),
        (root / "RESTRUCTURE_SUMMARY.md", "Restructure summary"),
    ]

    for filepath, description in root_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False

    # Check that OLD files are deleted
    print("\n[CLEANUP] Old Files (Should be DELETED):")
    old_files = [
        (root / "main.py", "Old main.py"),
        (root / "config.py", "Old config.py"),
        (root / "data_loader.py", "Old data_loader.py"),
        (root / "models.py", "Old models.py"),
        (root / "evaluation.py", "Old evaluation.py"),
        (root / "train.py", "Old train.py (moved to scripts/)"),
    ]

    for filepath, description in old_files:
        if filepath.exists():
            print(f"[!] STILL EXISTS (should be deleted): {description}")
            all_checks_passed = False
        else:
            print(f"[OK] Correctly deleted: {description}")

    # Final summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("[SUCCESS] ALL CHECKS PASSED - Restructuring is COMPLETE!")
        print("=" * 80)
        print("\nNext steps:")
        print("  1. Run tests: pytest tests/")
        print("  2. Train model: python scripts/train.py --config quick")
        print("  3. Read: RESTRUCTURE_SUMMARY.md for full details")
        return 0
    else:
        print("[FAIL] SOME CHECKS FAILED - Please review the output above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
