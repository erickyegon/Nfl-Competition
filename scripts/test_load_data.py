"""
Test script to verify load_data function works with actual CSV structure
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_dir, max_files=None, sample_size=None):
    """
    Load and merge input/output CSV files

    Args:
        data_dir: Directory containing train data
        max_files: Maximum number of files to load (None = all)
        sample_size: Sample size for faster testing (None = all)

    Returns:
        input_df, output_df: Input and output dataframes
    """
    train_dir = data_dir / 'train'

    # Get file lists
    input_files = sorted(train_dir.glob('input_*.csv'))
    output_files = sorted(train_dir.glob('output_*.csv'))

    if max_files:
        input_files = input_files[:max_files]
        output_files = output_files[:max_files]

    print(f"Loading {len(input_files)} input files and {len(output_files)} output files...")

    # Load input files
    input_dfs = []
    for file in input_files:
        df = pd.read_csv(file)
        input_dfs.append(df)
        print(f"   - {file.name}: {len(df):,} rows")

    input_df = pd.concat(input_dfs, ignore_index=True)

    # Load output files
    output_dfs = []
    for file in output_files:
        df = pd.read_csv(file)
        output_dfs.append(df)

    output_df = pd.concat(output_dfs, ignore_index=True)

    # Sample if requested
    if sample_size and len(input_df) > sample_size:
        print(f"\nSampling {sample_size:,} rows from {len(input_df):,}...")
        input_df = input_df.sample(n=sample_size, random_state=42)
        # Filter output to match sampled input using proper keys
        sampled_keys = input_df[['game_id', 'play_id', 'nfl_id', 'frame_id']]
        output_df = output_df.merge(sampled_keys, on=['game_id', 'play_id', 'nfl_id', 'frame_id'])
        output_df = output_df.reset_index(drop=True)

    print(f"\nData loaded successfully")
    print(f"   Input shape: {input_df.shape}")
    print(f"   Output shape: {output_df.shape}")

    return input_df, output_df


def test_merge(input_df, output_df):
    """Test merging input and output data"""
    print("\n" + "="*60)
    print("TESTING MERGE OPERATION")
    print("="*60)

    print("\nMerging input and output data...")
    merged_df = input_df.merge(
        output_df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']],
        on=['game_id', 'play_id', 'nfl_id', 'frame_id'],
        suffixes=('', '_target')
    )
    merged_df = merged_df.rename(columns={'x_target': 'target_x', 'y_target': 'target_y'})
    print(f"   Merged shape: {merged_df.shape}")

    print("\nInput columns:")
    print(f"   {input_df.columns.tolist()}")

    print("\nOutput columns:")
    print(f"   {output_df.columns.tolist()}")

    print("\nMerged columns (first 10):")
    print(f"   {merged_df.columns.tolist()[:10]}")

    print("\nMerged columns (last 5):")
    print(f"   {merged_df.columns.tolist()[-5:]}")

    print("\nSample merged data:")
    print(merged_df[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 'target_x', 'target_y']].head())

    # Check for missing values
    print("\nMissing values in target columns:")
    print(f"   target_x: {merged_df['target_x'].isnull().sum()}")
    print(f"   target_y: {merged_df['target_y'].isnull().sum()}")

    return merged_df


if __name__ == "__main__":
    # Test with limited data
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    print("="*60)
    print("TESTING LOAD_DATA FUNCTION")
    print("="*60 + "\n")

    # Load small sample
    input_df, output_df = load_data(
        data_dir,
        max_files=2,
        sample_size=10000
    )

    # Test merge
    merged_df = test_merge(input_df, output_df)

    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
