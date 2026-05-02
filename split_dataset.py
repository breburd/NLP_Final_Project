import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from models.common import DEFAULT_DATA_PATH, load_data, make_folder, seed_everything


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "preprocess" / "splits"


def validate_input_csv(data_path):
    """
    Validate that the requested CSV is a real labeled dataset file.
    """

    if not Path(data_path).exists():
        raise SystemExit(f"Input file not found: {data_path}")

    header = Path(data_path).read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
    if header and header[0].startswith("version https://git-lfs.github.com/spec/v1"):
        raise SystemExit(
            "The input CSV is a Git LFS pointer, not the real dataset.\n"
            "Fetch the actual file with Git LFS or point --data_path to a real labeled CSV."
        )

    columns = list(pd.read_csv(data_path, nrows=0, engine="python").columns)
    required_columns = {"from", "to", "subject", "body", "label"}
    missing_columns = required_columns.difference(columns)
    if missing_columns:
        raise SystemExit(
            f"Input CSV is missing required columns: {sorted(missing_columns)}.\n"
            f"Found columns: {columns}"
        )


def make_pair_ids(df):
    """
    Build a stable group id for each sender-recipient pair.
    """

    return df.apply(lambda row: "_".join(sorted([row["from"], row["to"]])), axis=1)


def _sorted_split(df):
    """
    Sort rows so saved CSV contents are stable across runs.
    """

    sort_columns = [
        column
        for column in ["from", "to", "subject", "body", "label", "prob_label", "text"]
        if column in df.columns
    ]
    if sort_columns:
        df = df.sort_values(sort_columns, kind="mergesort")
    return df.reset_index(drop=True)


def split_dataframe(df, test_size=0.2, valid_size=0.1):
    """
    Split the labeled dataset into train/validation/test partitions.

    Uses sender-recipient pair ids as groups so the same pair does not
    appear across multiple splits. Reproducibility comes from calling
    seed_everything before splitting.
    """

    working_df = df.copy()
    working_df["pair_id"] = make_pair_ids(working_df)

    train_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=1.0 - test_size,
    )
    train_idx, temp_idx = next(train_splitter.split(working_df, groups=working_df["pair_id"]))

    train_df = working_df.iloc[train_idx].drop(columns="pair_id")
    temp_df = working_df.iloc[temp_idx].copy()

    valid_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=valid_size / (valid_size + test_size),
    )
    valid_idx, test_idx = next(valid_splitter.split(temp_df, groups=temp_df["pair_id"]))

    valid_df = temp_df.iloc[valid_idx].drop(columns="pair_id")
    test_df = temp_df.iloc[test_idx].drop(columns="pair_id")

    return (
        _sorted_split(train_df),
        _sorted_split(valid_df),
        _sorted_split(test_df),
    )


def verify_deterministic_split(df, test_size=0.2, valid_size=0.1):
    """
    Ensure the split function returns identical results when run twice.
    """

    seed_everything()
    first = split_dataframe(df, test_size=test_size, valid_size=valid_size)
    seed_everything()
    second = split_dataframe(df, test_size=test_size, valid_size=valid_size)

    if not all(left.equals(right) for left, right in zip(first, second)):
        raise RuntimeError("Dataset split is not deterministic.")

    return first


def write_split(df, output_path):
    """
    Save one split and return its SHA-256 hash.
    """

    df.to_csv(output_path, index=False)
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic train/validation/test CSV splits from the labeled Enron dataset."
    )
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--valid_size", type=float, default=0.1)
    args = parser.parse_args()

    if args.test_size <= 0 or args.valid_size <= 0:
        raise SystemExit("test_size and valid_size must both be greater than 0.")

    if args.test_size + args.valid_size >= 1:
        raise SystemExit("test_size + valid_size must be less than 1.")

    data_path = Path(args.data_path)
    output_dir = make_folder(args.output_dir)

    validate_input_csv(data_path)
    df = load_data(data_path)
    train_df, valid_df, test_df = verify_deterministic_split(
        df,
        test_size=args.test_size,
        valid_size=args.valid_size,
    )

    train_hash = write_split(train_df, output_dir / "train.csv")
    valid_hash = write_split(valid_df, output_dir / "valid.csv")
    test_hash = write_split(test_df, output_dir / "test.csv")

    manifest = {
        "data_path": str(data_path),
        "seed": 42,
        "test_size": args.test_size,
        "valid_size": args.valid_size,
        "counts": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
        },
        "sha256": {
            "train.csv": train_hash,
            "valid.csv": valid_hash,
            "test.csv": test_hash,
        },
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved deterministic splits to: {output_dir}")
    print(f"train: {len(train_df)} rows ({train_hash})")
    print(f"valid: {len(valid_df)} rows ({valid_hash})")
    print(f"test: {len(test_df)} rows ({test_hash})")


if __name__ == "__main__":
    main()
