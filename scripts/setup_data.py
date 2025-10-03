"""
Simple script to organize data into raw/processed/features structure
"""

from pathlib import Path
import shutil


def setup_data_structure(data_dir):
    """
    Organize existing data into proper structure:
    - Move train/test folders to raw/
    - Create processed/ and features/ directories
    """
    print("\n" + "="*60)
    print("SETTING UP DATA STRUCTURE")
    print("="*60 + "\n")

    data_dir = Path(data_dir)

    # Create directory structure
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    features_dir = data_dir / 'features'

    print("Creating directories...")
    raw_dir.mkdir(exist_ok=True)
    processed_dir.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)
    print("  - raw/")
    print("  - processed/")
    print("  - features/")

    # Move existing data to raw/
    print("\nMoving existing data to raw/...")

    # Move train folder if it exists at root level
    train_dir = data_dir / 'train'
    raw_train_dir = raw_dir / 'train'
    if train_dir.exists() and not raw_train_dir.exists():
        shutil.move(str(train_dir), str(raw_train_dir))
        print(f"  - Moved train/ to raw/train/")
    elif raw_train_dir.exists():
        print(f"  - raw/train/ already exists")

    # Move test folder if it exists at root level
    test_dir = data_dir / 'test'
    raw_test_dir = raw_dir / 'test'
    if test_dir.exists() and not raw_test_dir.exists():
        shutil.move(str(test_dir), str(raw_test_dir))
        print(f"  - Moved test/ to raw/test/")
    elif raw_test_dir.exists():
        print(f"  - raw/test/ already exists")

    # Move any CSV files at root to raw/
    csv_files = list(data_dir.glob('*.csv'))
    if csv_files:
        print(f"\nMoving {len(csv_files)} CSV files to raw/...")
        for csv_file in csv_files:
            dest = raw_dir / csv_file.name
            if not dest.exists():
                shutil.move(str(csv_file), str(dest))
                print(f"  - Moved {csv_file.name}")

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60 + "\n")

    print("Your data is now organized as:")
    print(f"  {data_dir}/")
    print(f"     |- raw/              # Original CSV files")
    print(f"     |   |- train/")
    print(f"     |   |- test/")
    print(f"     |   |- *.csv")
    print(f"     |- processed/        # Cleaned, merged data")
    print(f"     |- features/         # Engineered features")
    print()


if __name__ == "__main__":
    # Default data directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        exit(1)

    setup_data_structure(data_dir)
