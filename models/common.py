import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "preprocess" / "enron_emails_labeled.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baselines"


def make_folder(folder_path):
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def fix_csv_limit():
    size = sys.maxsize
    while size > 0:
        try:
            import csv

            csv.field_size_limit(size)
            return
        except OverflowError:
            size = size // 10


def load_data(data_path):
    fix_csv_limit()
    df = pd.read_csv(data_path, encoding="utf-8", engine="python")
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)
    df["text"] = (df["subject"].str.strip() + "\n" + df["body"].str.strip()).str.strip()
    df = df[df["text"] != ""].copy()
    df["label"] = df["label"].astype(int)
    return df


def split_data(df, seed=42, test_size=0.2, valid_size=0.1):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    valid_part = valid_size / (1.0 - test_size)
    train_df, valid_df = train_test_split(
        train_df,
        test_size=valid_part,
        random_state=seed,
        stratify=train_df["label"],
    )

    return train_df.copy(), valid_df.copy(), test_df.copy()


def get_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def save_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
