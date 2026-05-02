import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from conftest import load_module_from_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_MODULE = load_module_from_path(
    "split_dataset_for_tests",
    PROJECT_ROOT / "split_dataset.py",
)


def make_sample_dataframe():
    rows = []
    for idx in range(24):
        pair_number = idx % 6
        rows.append(
            {
                "from": f"sender{pair_number}@example.com",
                "to": f"recipient{pair_number}@example.com",
                "subject": f"subject {idx}",
                "body": f"body {idx}",
                "label": idx % 2,
            }
        )
    return pd.DataFrame(rows)


def test_split_dataframe_is_deterministic():
    df = make_sample_dataframe()

    SPLIT_MODULE.seed_everything()
    first = SPLIT_MODULE.split_dataframe(df, test_size=0.25, valid_size=0.25)
    SPLIT_MODULE.seed_everything()
    second = SPLIT_MODULE.split_dataframe(df, test_size=0.25, valid_size=0.25)

    assert all(left.equals(right) for left, right in zip(first, second))


def test_main_writes_identical_files_when_run_twice(monkeypatch, temp_workspace):
    input_path = temp_workspace / "emails.csv"
    output_dir = temp_workspace / "splits"
    make_sample_dataframe().to_csv(input_path, index=False)

    args = argparse.Namespace(
        data_path=str(input_path),
        output_dir=str(output_dir),
        test_size=0.25,
        valid_size=0.25,
    )

    monkeypatch.setattr(SPLIT_MODULE.argparse.ArgumentParser, "parse_args", lambda self: args)
    SPLIT_MODULE.main()

    first_hashes = {
        file_name: hashlib.sha256((output_dir / file_name).read_bytes()).hexdigest()
        for file_name in ["train.csv", "valid.csv", "test.csv"]
    }

    monkeypatch.setattr(SPLIT_MODULE.argparse.ArgumentParser, "parse_args", lambda self: args)
    SPLIT_MODULE.main()

    second_hashes = {
        file_name: hashlib.sha256((output_dir / file_name).read_bytes()).hexdigest()
        for file_name in ["train.csv", "valid.csv", "test.csv"]
    }

    manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))

    assert first_hashes == second_hashes
    assert manifest["sha256"] == second_hashes


def test_validate_input_csv_rejects_git_lfs_pointer(temp_workspace):
    pointer_path = temp_workspace / "pointer.csv"
    pointer_path.write_text(
        "\n".join(
            [
                "version https://git-lfs.github.com/spec/v1",
                "oid sha256:abc123",
                "size 100",
            ]
        ),
        encoding="utf-8",
    )

    try:
        SPLIT_MODULE.validate_input_csv(pointer_path)
    except SystemExit as exc:
        assert "Git LFS pointer" in str(exc)
    else:
        raise AssertionError("Expected validate_input_csv to reject LFS pointer input.")
