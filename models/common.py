import json
import sys
import random
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
import torch


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.

    Ensures consistent results across Python's random module,
    NumPy, and PyTorch.

    Args:
        seed (int, optional): Seed value to use. Defaults to 42.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "preprocess" / "enron_emails_labeled.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"


def print_gpu_memory():
    """
    Print GPU memory usage statistics.

    Displays allocated, reserved, and maximum reserved GPU memory
    for the current process. Also prints output from `nvidia-smi`
    for additional diagnostics.

    Notes:
        Only runs if a CUDA-enabled GPU is available.
    """

    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


def make_folder(folder_path):
    """
    Create a directory if it does not exist.

    Args:
        folder_path (str or pathlib.Path): Path to the directory.

    Returns:
        pathlib.Path: Path object of the created/existing directory.
    """

    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def fix_csv_limit():
    """
    Increase CSV field size limit to handle large files.

    Attempts to set the maximum allowable CSV field size to avoid
    OverflowError when reading large text fields. If the maximum
    value is too large, it progressively reduces the size until
    successful.
    """
    
    size = sys.maxsize
    while size > 0:
        try:
            import csv

            csv.field_size_limit(size)
            return
        except OverflowError:
            size = size // 10


def load_data(data_path):
    """
    Load and preprocess the dataset from a CSV file.

    Cleans missing values, combines subject and body into a single
    text field, and ensures correct data types.

    Args:
        data_path (str or pathlib.Path): Path to the CSV file.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with columns:
            - from (str)
            - to (str)
            - subject (str)
            - body (str)
            - text (str)
            - label (int)
    """

    fix_csv_limit()
    df = pd.read_csv(data_path, encoding="utf-8", engine="python")
    df["from"] = df["from"].fillna("").astype(str)
    df["to"] = df["to"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = (df["subject"].str.strip() + "\n" + df["body"].str.strip()).str.strip()
    df = df[df["text"] != ""].copy()
    df["label"] = df["label"].astype(int)
    return df


def load_pre_split_data(train_path, valid_path, test_path):
    """
    Load already-split train, validation, and test CSV files.

    Args:
        train_path (str or pathlib.Path): Path to training CSV.
        valid_path (str or pathlib.Path): Path to validation CSV.
        test_path (str or pathlib.Path): Path to test CSV.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            Train, validation, and test DataFrames.
    """

    train_df = load_data(train_path)
    valid_df = load_data(valid_path)
    test_df = load_data(test_path)
    return train_df, valid_df, test_df


def split_data(df, test_size=0.2, valid_size=0.1, seed=42):
    """
    Split dataset into train, validation, and test sets.

    Uses group-based splitting to prevent data leakage by ensuring
    emails from the same sender-recipient pair are not split across
    different sets.

    Args:
        df (pandas.DataFrame): Input dataset.
        test_size (float, optional): Proportion for test set.
        valid_size (float, optional): Proportion for validation set.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            Train, validation, and test DataFrames.
    """

    df['pair_id'] = df.apply(lambda x: "_".join(sorted([x['from'], x['to']])), axis=1)

    # Split once for Train/(Validation + Test)
    gs = GroupShuffleSplit(
        n_splits=1,
        train_size=1.0 - test_size,
        random_state=seed,
    )
    train_idx, temp_idx = next(gs.split(df, groups=df['pair_id']))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Split the remainder into Validation/Test
    gs_val = GroupShuffleSplit(
        n_splits=1,
        train_size=valid_size / (valid_size + test_size),
        random_state=seed + 1,
    )
    val_idx, test_idx = next(gs_val.split(temp_df, groups=temp_df['pair_id']))

    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    print("Size of the loaded dataset:")
    print(f" - train: {len(train_df)}")
    print(f" - val: {len(val_df)}")
    print(f" - test: {len(test_df)}")

    return train_df.copy(), val_df.copy(), test_df.copy()


def get_scores(y_true, y_pred):
    """
    Compute classification evaluation metrics.

    Calculates accuracy, precision, recall, F1 score, and a detailed
    classification report.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        dict: Dictionary containing:
            - accuracy (float)
            - precision (float)
            - recall (float)
            - f1 (float)
            - classification_report (str)
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def save_json(data, output_path):
    """
    Save data as a JSON file.

    Ensures the output directory exists before writing the file.

    Args:
        data (dict): Data to save.
        output_path (str or pathlib.Path): Destination file path.
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
