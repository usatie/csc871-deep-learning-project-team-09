#!/usr/bin/env python3
import os
import pandas as pd
import torch
from typing import Tuple
import argparse


def load_tatoeba_data(file_path: str) -> pd.DataFrame:
    """Load Tatoeba dataset from TSV file."""
    columns = ["sentence_id", "source", "unknown", "target"]
    return pd.read_csv(file_path, sep="\t", names=columns, header=None)


def split_by_source_id(
    df: pd.DataFrame, split_ratio: Tuple[float, float, float], seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by source sentence ID.

    Args:
        df: DataFrame containing the dataset
        split_ratio: Tuple of (train_ratio, val_ratio, test_ratio)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Get unique source IDs
    unique_sources = df["sentence_id"].unique()

    # Calculate split sizes
    total_size = len(unique_sources)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - val_size
    print("=" * 50)
    print("Unique source sentences:")
    print("=" * 50)
    print(f"Total size: {total_size:,}")
    print(f"Train size: {train_size:,} ({split_ratio[0]:.2%})")
    print(f"Val size: {val_size:,} ({split_ratio[1]:.2%})")
    print(f"Test size: {test_size:,} ({split_ratio[2]:.2%})")

    # Split unique source IDs
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator)

    train_ids = unique_sources[indices[:train_size]]
    val_ids = unique_sources[indices[train_size : train_size + val_size]]
    test_ids = unique_sources[indices[train_size + val_size :]]

    # Create splits
    train_df = df[df["sentence_id"].isin(train_ids)]
    val_df = df[df["sentence_id"].isin(val_ids)]
    test_df = df[df["sentence_id"].isin(test_ids)]

    # Print statistics
    print("=" * 50)
    print("Splits:")
    print("=" * 50)
    print(f"Total size: {len(df):,} pairs")
    print(
        f"Train set: {len(train_df):,} pairs ({len(train_df)/len(df):.2%}), {len(train_df['sentence_id'].unique()):,} unique sources"
    )
    print(
        f"Val set: {len(val_df):,} pairs ({len(val_df)/len(df):.2%}), {len(val_df['sentence_id'].unique()):,} unique sources"
    )
    print(
        f"Test set: {len(test_df):,} pairs ({len(test_df)/len(df):.2%}), {len(test_df['sentence_id'].unique()):,} unique sources"
    )
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str
) -> None:
    """Save splits to TSV files.

    Args:
        train_df: Training set DataFrame
        val_df: Validation set DataFrame
        test_df: Test set DataFrame
        output_dir: Directory to save the splits
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
    val_df.to_csv(os.path.join(output_dir, "val.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False)

    print("=" * 50)
    print("Example of train set:")
    print("=" * 50)
    print(train_df.head())

    print("=" * 50)
    print("Example of val set:")
    print("=" * 50)
    print(val_df.head())

    print("=" * 50)
    print("Example of test set:")
    print("=" * 50)
    print(test_df.head())


def main():
    parser = argparse.ArgumentParser(
        description="Split Tatoeba dataset by source sentence ID"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input TSV file"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save the splits"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Ratio of training set"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Ratio of validation set"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Ratio of test set"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate split ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    # Load and split dataset
    df = load_tatoeba_data(args.input)
    train_df, val_df, test_df = split_by_source_id(
        df,
        split_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )

    # Save splits
    save_splits(train_df, val_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()
