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
    Set random seeds for reproducibility across random, numpy, and torch libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "preprocess" / "enron_emails_labeled.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU

    Source: base_classification.py from Assignment 6
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
    Create a folder at the specified path if it does not already exist.
    This is useful for ensuring that the output directory exists before saving files to it.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def fix_csv_limit():
    """
    Fix the CSV field size limit to avoid errors when loading large CSV files.
    This is necessary because the default field size limit may be too small for some datasets, leading
    to OverflowError when trying to load the data. The function attempts to set the field size limit to the maximum
    possible value, and if that fails due to an OverflowError, it reduces the limit by a factor of 10 and tries again until it succeeds.
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
    Load the dataset from a CSV file, combine the subject and body into a single text field,
    and return a cleaned DataFrame with the necessary columns."""
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


def split_data(df, test_size=0.2, valid_size=0.1):
    """
     Split the dataset into train, validation, and test sets while ensuring that emails from the same user pairs
     (based on 'from' and 'to' fields) are not split across different sets to prevent data leakage.
     This is done by creating a 'pair_id' for each email based on the sorted combination
     of the 'from' and 'to' fields, and then using GroupShuffleSplit to split the data based on these pairs."""
    # Split into train/validation/test -> We want unique users in each split to prevent data leakage.
    # This ensures the model does not learn user-specific patterns that could lead to overfitting
    # and poor generalization on unseen users.
    df['pair_id'] = df.apply(lambda x: "_".join(sorted([x['from'], x['to']])), axis=1)

    # Split once for Train/(Validation + Test)
    gs = GroupShuffleSplit(n_splits=1, train_size=1.0-test_size)
    train_idx, temp_idx = next(gs.split(df, groups=df['pair_id']))

    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # Split the remainder into Validation/Test
    gs_val = GroupShuffleSplit(n_splits=1, train_size=valid_size/(valid_size+test_size))
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
    Calculate and return a dictionary of evaluation metrics including accuracy, precision, recall, F1 score, and a classification report.
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
    Save a dictionary as a JSON file at the specified output path. The function ensures that the output directory exists before saving the file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
