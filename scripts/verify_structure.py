"""
Verify that the NFL pipeline structure is correctly set up
"""

from pathlib import Path
import sys

def check_structure():
    """Verify directory structure and key files"""

    project_root = Path(__file__).parent.parent

    print("\n" + "="*70)
    print("NFL PIPELINE STRUCTURE VERIFICATION")
    print("="*70 + "\n")

    # Define expected structure
    structure = {
        'nfl_pipeline': {
            'core': ['__init__.py', 'pipeline.py', 'config.py'],
            'data': ['__init__.py', 'loader.py', 'preprocessor.py', 'manager.py'],
            'features': ['__init__.py', 'base.py', 'physics.py', 'spatial.py', 'temporal.py', 'nfl_domain.py'],
            'models': ['__init__.py', 'base.py', 'traditional.py', 'sequence.py', 'ensemble.py'],
            'evaluation': ['__init__.py', 'metrics.py', 'evaluator.py', 'selector.py'],
            'utils': ['__init__.py', 'logging.py', 'helpers.py', 'tracking.py'],
            'prediction': ['__init__.py', 'predictor.py']
        },
        'data': {
            'raw': ['README.md'],
            'processed': ['README.md'],
            'features': ['README.md']
        },
        'scripts': ['train.py', 'predict.py', 'evaluate.py', 'setup_data_structure.py', 'verify_structure.py'],
        'tests': ['test_features.py', 'test_models.py', 'test_pipeline.py'],
        'configs': ['default.yaml', 'quick.yaml', 'lstm.yaml'],
        'root': ['setup.py', 'README.md', 'QUICK_START.md', 'DATA_PIPELINE.md', 'FINAL_STRUCTURE.md']
    }

    all_checks_passed = True

    # Check nfl_pipeline package
    print("[PACKAGE] Checking nfl_pipeline package...")
    for module, files in structure['nfl_pipeline'].items():
        module_path = project_root / 'nfl_pipeline' / module
        if not module_path.exists():
            print(f"  [X] Missing: nfl_pipeline/{module}/")
            all_checks_passed = False
        else:
            print(f"  [OK] nfl_pipeline/{module}/")
            for file in files:
                file_path = module_path / file
                if not file_path.exists():
                    print(f"     [X] Missing: {file}")
                    all_checks_passed = False
                else:
                    print(f"     [OK] {file}")

    # Check data structure
    print("\n[DATA] Checking data structure...")
    data_dir = project_root / 'data'
    if not data_dir.exists():
        print("  [!] Data directory not found (run setup_data_structure.py)")
    else:
        for folder, files in structure['data'].items():
            folder_path = data_dir / folder
            if not folder_path.exists():
                print(f"  [!] Missing: data/{folder}/ (run setup_data_structure.py)")
            else:
                print(f"  [OK] data/{folder}/")
                for file in files:
                    file_path = folder_path / file
                    if not file_path.exists():
                        print(f"     [!] Missing: {file}")
                    else:
                        print(f"     [OK] {file}")

    # Check scripts
    print("\n[SCRIPTS] Checking scripts...")
    scripts_dir = project_root / 'scripts'
    if not scripts_dir.exists():
        print("  [X] Missing: scripts/")
        all_checks_passed = False
    else:
        print("  [OK] scripts/")
        for file in structure['scripts']:
            file_path = scripts_dir / file
            if not file_path.exists():
                print(f"     [X] Missing: {file}")
                all_checks_passed = False
            else:
                print(f"     [OK] {file}")

    # Check tests
    print("\n[TESTS] Checking tests...")
    tests_dir = project_root / 'tests'
    if not tests_dir.exists():
        print("  [X] Missing: tests/")
        all_checks_passed = False
    else:
        print("  [OK] tests/")
        for file in structure['tests']:
            file_path = tests_dir / file
            if not file_path.exists():
                print(f"     [X] Missing: {file}")
                all_checks_passed = False
            else:
                print(f"     [OK] {file}")

    # Check configs
    print("\n[CONFIG] Checking configs...")
    configs_dir = project_root / 'configs'
    if not configs_dir.exists():
        print("  [X] Missing: configs/")
        all_checks_passed = False
    else:
        print("  [OK] configs/")
        for file in structure['configs']:
            file_path = configs_dir / file
            if not file_path.exists():
                print(f"     [X] Missing: {file}")
                all_checks_passed = False
            else:
                print(f"     [OK] {file}")

    # Check root files
    print("\n[ROOT] Checking root files...")
    for file in structure['root']:
        file_path = project_root / file
        if not file_path.exists():
            print(f"  [X] Missing: {file}")
            all_checks_passed = False
        else:
            print(f"  [OK] {file}")

    # Check if package is importable
    print("\n[PYTHON] Checking Python imports...")
    try:
        sys.path.insert(0, str(project_root))
        import nfl_pipeline
        print("  [OK] nfl_pipeline package importable")

        from nfl_pipeline.core.pipeline import NFLPipeline
        print("  [OK] NFLPipeline importable")

        from nfl_pipeline.data.manager import DataManager
        print("  [OK] DataManager importable")

        from nfl_pipeline.features.engineer import FeatureEngineer
        print("  [OK] FeatureEngineer importable")

    except ImportError as e:
        print(f"  [X] Import error: {e}")
        all_checks_passed = False

    # Final summary
    print("\n" + "="*70)
    if all_checks_passed:
        print("[SUCCESS] ALL CHECKS PASSED! Structure is correct.")
        print("\nNext steps:")
        print("  1. Run: python scripts/setup_data_structure.py  (organize data)")
        print("  2. Run: pip install -e .                         (install package)")
        print("  3. Run: python scripts/train.py --config configs/quick.yaml")
    else:
        print("[FAILED] SOME CHECKS FAILED! Please review the structure.")
    print("="*70 + "\n")

    return all_checks_passed


if __name__ == "__main__":
    success = check_structure()
    sys.exit(0 if success else 1)
